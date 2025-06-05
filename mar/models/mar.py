import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='timm')
# Add this new filter for vision_transformer warnings
warnings.filterwarnings('ignore', message='Overwriting .* in registry.*', category=UserWarning)

from functools import partial
from enum import Enum

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat

from models.res_blocks import Block

from models.diffloss import DiffLoss
from models.flowloss import FlowLoss


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


class LossType(Enum):
    DIFF = DiffLoss
    FLOW = FlowLoss

class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1, encoder_embed_dim=1024, encoder_depth=16, 
                 encoder_num_heads=16, decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16, mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, vae_embed_dim=16, mask_ratio_min=0.7, label_drop_prob=0.1, class_num=1000, 
                 attn_dropout=0.1, proj_dropout=0.1, buffer_size=64, diffloss_d=3, diffloss_w=1024, 
                 num_sampling_steps='100', diffusion_batch_mul=4, grad_checkpointing=False, unfold_before_diff=False,
                 unfold_repeat_cond=False, token_mixing=False, shuffle_before_patchify=False, network_type='MLP',
                 mlp_post_norm=False, loss_type='DIFF', ar_post_norm=False, ar_post_norm_before_res=False,
                 qk_norm=False, seq_len=None, attn_norm_layer=nn.LayerNorm, encoder_grad_ckpt=0, decoder_grad_ckpt=0,
                 diffloss_grad_ckpt=0):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        if seq_len is None:
            self.seq_h = self.seq_w = img_size // vae_stride // patch_size
            self.seq_len = self.seq_h * self.seq_w
        else:
            print(f"seq_len is provided: {seq_len} assuming that the vae output is of shape (bsz, seq_len, vae_embed_dim)")
            self.seq_h = self.seq_w = None
            self.seq_len = seq_len
            assert not shuffle_before_patchify, "Shuffle before patchify is not supported when seq_len is provided"
            assert patch_size == 1, "Patch size must be 1 when seq_len is provided"
            assert not unfold_before_diff

        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing
        self.encoder_grad_ckpt = encoder_grad_ckpt
        self.decoder_grad_ckpt = decoder_grad_ckpt
        self.diffloss_grad_ckpt = diffloss_grad_ckpt

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size

        if shuffle_before_patchify:
            # We need to keep the number of position embeddings
            # So that we can fold the tokens with correct position embeddings
            # What need to be changed:
            #  1. Shuffle tokens before patchify, save the order
            #  2. Wherever the encoder_pos_embed_learned is used
            #  3. Wherever the decoder_pos_embed_learned is used
            #  4. Inference
            assert patch_size > 1, "Shuffle before patchify only works with patch_size > 1"
            pos_embed_len_wo_buffer = (img_size // vae_stride) ** 2
        else:
            pos_embed_len_wo_buffer = self.seq_len

        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, pos_embed_len_wo_buffer + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([Block(
            encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, 
            proj_drop=proj_dropout, attn_drop=attn_dropout, post_norm=ar_post_norm,
            post_norm_before_res=ar_post_norm_before_res, qk_norm=qk_norm, attn_norm_layer=attn_norm_layer
        ) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, pos_embed_len_wo_buffer + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([Block(
            decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
            proj_drop=proj_dropout, attn_drop=attn_dropout, post_norm=ar_post_norm,
            post_norm_before_res=ar_post_norm_before_res, qk_norm=qk_norm
        ) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, pos_embed_len_wo_buffer, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.unfold_before_diff = unfold_before_diff
        self.unfold_repeat_cond = unfold_repeat_cond
        if self.unfold_before_diff:
            assert decoder_embed_dim % (patch_size ** 2) == 0
            if not self.unfold_repeat_cond:  # We need projection to match the token dimension
                self.unfold_proj = nn.Linear(decoder_embed_dim // (patch_size ** 2), decoder_embed_dim)
            else:
                self.unfold_pos_embed = nn.Parameter(torch.zeros(1, patch_size ** 2, decoder_embed_dim))
            
        if token_mixing:
            assert self.unfold_before_diff, "We can only mix tokens after the tokens are unfolded"
            token_mixing_num = patch_size ** 2
        else:
            token_mixing_num = 1
        self.network_type = network_type
        self.diffloss = LossType[loss_type].value(
            target_channels=self.vae_embed_dim if self.unfold_before_diff else self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing,
            token_mixing_num=token_mixing_num,
            network_type=network_type,
            mlp_post_norm=mlp_post_norm,
            layers_to_apply_grad_ckpt=diffloss_grad_ckpt,
        )
        self.diffusion_batch_mul = diffusion_batch_mul
        self.shuffle_before_patchify = shuffle_before_patchify

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        if self.seq_h is None:
            if x.dim() == 4:
                if not hasattr(self, '_warned_patchify'):
                    print("Warning: Input shape is 4D but seq_h is None, flattening (B, C, H, W) -> (B, H * W, C)")
                    self._warned_patchify = True
                x = x.flatten(2).transpose(1, 2)
            return x, None

        bsz, c, h, w = x.shape
        if self.shuffle_before_patchify:
            x = x.view(bsz, c, h*w)
            orders = self.sample_orders(bsz, sample_len=h*w)  # bsz, h * w
            gather_idx = orders.unsqueeze(1).expand(-1, c, -1)  # bsz, c, h * w
            x = x.gather(dim=2, index=gather_idx)  # bsz, c, h * w
            x = x.view(bsz, c, h, w)
        else:
            orders = None

        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x, orders


    def unpatchify(self, x, token_shuffle_orders=None):
        if self.seq_h is None:
            return x

        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)

        if self.shuffle_before_patchify:
            assert token_shuffle_orders is not None
            x = x.view(bsz, c, -1)
            inverse_orders = torch.argsort(token_shuffle_orders, dim=1)
            gather_idx = inverse_orders.unsqueeze(1).expand(-1, c, -1)
            x = x.gather(dim=2, index=gather_idx)
            x = x.view(bsz, c, h_ * p , w_ * p)

        return x  # [n, c, h, w]
    
    def fold(self, z):
        # The reverse operation of unfold
        if z.ndim == 3:
            pattern = 'b (l p q) c -> b l (c p q)'
        elif z.ndim == 2:
            pattern = '(l p q) c -> l (c p q)'
        else:
            raise ValueError(f"Invalid input shape {z.shape}")
        
        return rearrange(z, pattern, p=self.patch_size, q=self.patch_size)

    def unfold(self, z, targets, mask):
        # This function unfold each patch into original tokens
        if not self.unfold_repeat_cond:  
            # We split channel into patches, and proj to match the dimension
            z = rearrange(z, 'b l (c p q) -> b (l p q) c', p=self.patch_size, q=self.patch_size)
            z = self.unfold_proj(z)
        else:
            # Repeat the condition for each token, and add position embedding
            l = z.shape[1]
            z = repeat(z, 'b l c -> b (l repeat) c', repeat=self.patch_size**2)
            pos_emb = repeat(self.unfold_pos_embed, 'b t c -> b (repeat t) c', repeat=l)
            z = z + pos_emb
        
        targets = rearrange(targets, 'b l (c p q) -> b (l p q) c', p=self.patch_size, q=self.patch_size)
        mask = mask.repeat_interleave(self.patch_size**2, dim=-1)
        
        return z, targets, mask
    
    def unfold_infer(self, z):
        if not self.unfold_repeat_cond:
            # We split channel into patches, and proj to match the dimension
            z = rearrange(z, 'l (c p q) -> (l p q) c', p=self.patch_size, q=self.patch_size)
            z = self.unfold_proj(z)
        else:
            # Repeat the condition for each token, and add position embedding
            # For network_type MLP, we repeat the condition for each token as they are independent
            l = z.shape[0]
            z = repeat(z, 'l c -> (l repeat) c', repeat=self.patch_size**2)
            pos_emb = repeat(self.unfold_pos_embed, 'b t c -> b (repeat t) c', repeat=l).squeeze(0)
            z = z + pos_emb
        return z
    
    def shuffle_and_merge_pos(self, pos_embed, orders=None, with_buffer=False):
        '''
        Shuffle the position embedding and merge with orders

        Parameters:
            pos_embed (Tensor): position embedding, shape (1, h * p * w * p, c)
            orders (Tensor): orders for token shuffling, shape (bsz, h * p * w * p)
            with_buffer (bool): whether the pos_embed contains buffer
        '''
        if orders is None:
            return pos_embed
        
        pos_embed = pos_embed.repeat(orders.size(0), 1, 1)
        
        if with_buffer:
            pos_embed_buffer, pos_embed_shuffled = (
                pos_embed[:, :self.buffer_size], 
                pos_embed[:, self.buffer_size:]
            )
        else:
            pos_embed_shuffled = pos_embed

        bsz, _, c = pos_embed_shuffled.shape
        gather_idx = orders.unsqueeze(-1).expand(-1, -1, c)
        pos_embed_shuffled = pos_embed_shuffled.gather(dim=1, index=gather_idx)

        pos_embed_shuffled = pos_embed_shuffled.view(
            bsz, self.seq_h * self.seq_w, self.patch_size ** 2, -1).sum(dim=2)  # bsz, h * w, c
        
        if with_buffer:
            pos_embed_shuffled = torch.cat([pos_embed_buffer, pos_embed_shuffled], dim=1)

        return pos_embed_shuffled

    def sample_orders(self, bsz, sample_len=None):
        # generate a batch of random generation orders
        # if sample_len is None:
        #     sample_len = self.seq_len
        # # Generate one large tensor and reshape
        # orders = torch.rand(bsz, sample_len, device='cuda').argsort(dim=1)
        # return orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding, token_shuffle_orders=None):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        # encoder position embedding
        x = x + self.shuffle_and_merge_pos(
            self.encoder_pos_embed_learned, orders=token_shuffle_orders, with_buffer=True
        )
        x = self.z_proj_ln(x)

        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        layers_to_apply_grad_ckpt = self.encoder_grad_ckpt

        for i, block in enumerate(self.encoder_blocks):
            if (len(self.encoder_blocks) - i <= layers_to_apply_grad_ckpt and 
                self.grad_checkpointing and 
                not torch.jit.is_scripting()):
                x = checkpoint(block, x)
            else:
                x = block(x)

        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask, token_shuffle_orders=None):

        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad + self.shuffle_and_merge_pos(
            self.decoder_pos_embed_learned, token_shuffle_orders, with_buffer=True
        )

        layers_to_apply_grad_ckpt = self.decoder_grad_ckpt

        for i, block in enumerate(self.decoder_blocks):
            if (len(self.decoder_blocks) - i <= layers_to_apply_grad_ckpt and 
                self.grad_checkpointing and 
                not torch.jit.is_scripting()):
                x = checkpoint(block, x)
            else:
                x = block(x)

        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.shuffle_and_merge_pos(
            self.diffusion_pos_embed_learned, orders=token_shuffle_orders
        )
        return x

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, imgs, labels):

        # class embed
        class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        x, token_shuffle_orders = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        # mae encoder
        x = self.forward_mae_encoder(x, mask, class_embedding, token_shuffle_orders)

        # mae decoder
        z = self.forward_mae_decoder(x, mask, token_shuffle_orders)

        # diffloss
        if self.unfold_before_diff:  # unfold patch into tokens
            z, gt_latents, mask = self.unfold(z, gt_latents, mask)
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):

        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        if self.shuffle_before_patchify:
            token_shuffle_orders = self.sample_orders(bsz, sample_len=self.seq_len * self.patch_size ** 2)
        else:
            token_shuffle_orders = None

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()
            # print("cur_tokens.std()", cur_tokens.std())

            # class embedding and CFG
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            # print("class_embedding: ", class_embedding)
            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)
            
            # print("tokens.std()", tokens.std())
            # mae encoder
            x = self.forward_mae_encoder(tokens, mask, class_embedding, token_shuffle_orders)
            # print("x.std()", x.std())

            # mae decoder
            z = self.forward_mae_decoder(x, mask, token_shuffle_orders)
            # print("z.std()", z.std())

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))
            
            # print("mask_len, orders, bsz, self.seq_len: ", mask_len, orders, bsz, self.seq_len)
            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            # print("mask_next: ", mask_next)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            # print("mask_to_pred: ", mask_to_pred)
            
            assert mask_to_pred.sum() > 0, (
                "No token to predict",
                "Mask len: ", mask_len[0],
                "Orders: ", orders[0],
                "Batch size: ", bsz,
                "Seq len: ", self.seq_len,
                "Mask next: ", mask_next[0],
                "Mask to pred: ", mask_to_pred[0],
                "step: ", step,
                "num_iter: ", num_iter,
            )

            # print("z before masking: ", z.std())
            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError

            if self.unfold_before_diff:
                z = self.unfold_infer(z)
            # print("z before sample: ", z.std())
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if self.unfold_before_diff:
                sampled_token_latent = self.fold(sampled_token_latent)

            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
            # print("sampled_token_latent.std()", sampled_token_latent.std())

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()
            # print("tokens.std()", tokens.std())

        # unpatchify
        tokens = self.unpatchify(tokens, token_shuffle_orders)
        return tokens


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
