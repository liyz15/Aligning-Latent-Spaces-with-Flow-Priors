import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
from enum import Enum
from einops import rearrange

from diffusion import create_diffusion
from models.pixart.PixArt import PixArtBlock, PixArtBlockwoAttn, T2IFinalLayer
from models.pixart.PixArt import TimestepEmbedder as PixArtTimestepEmbedder



class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False,
                 token_mixing_num=1, network_type='MLP', mlp_post_norm=False, layers_to_apply_grad_ckpt=0):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels
        assert network_type in ['MLP', 'PIXART'], 'network_type must be either mlp or pixart'
        self.network_type = network_type
        self.token_mixing_num = token_mixing_num
        self.net = DiffLossType[network_type].value(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
            token_mixing_num=token_mixing_num,
            use_post_norm=mlp_post_norm,
            layers_to_apply_grad_ckpt=layers_to_apply_grad_ckpt,
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    def forward(self, target, z, mask=None):
        if self.network_type in ['MLP', 'PIXARTWOATTN']:
            # Individual timestep for each sample
            t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        elif self.network_type == 'PIXART':
            assert target.shape[0] % self.token_mixing_num == 0, 'Batch size must be divisible by token_mixing_num'
            t = torch.randint(0, self.train_diffusion.num_timesteps, 
                              (target.shape[0] // self.token_mixing_num,), device=target.device)
            t = t.repeat_interleave(self.token_mixing_num)

        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0):
        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward
        
        # print("sample cfg, model_kwargs: ", cfg, model_kwargs)
        # print("sample noise.std()", noise.std())

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )

        # print("sample sampled_token_latent.std()", sampled_token_latent.std())

        return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TokenMixer(nn.Module):
    """
    A residual block in token dim.
    :param channels: the number of input tokens.
    """
    def __init__(
        self,
        num_tokens
    ):
        super().__init__()
        self.num_tokens = num_tokens

        self.in_ln = nn.LayerNorm(num_tokens, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(num_tokens, num_tokens, bias=True),
            nn.SiLU(),
            nn.Linear(num_tokens, num_tokens, bias=True),
        )

    def forward(self, x, *args, **kwargs):
        # x: [bsz * seq_len, channels]
        x = rearrange(x, '(l p) c -> l c p', p=self.num_tokens)
        h = self.in_ln(x)
        h = self.mlp(h)
        x = x + h
        x = rearrange(x, 'l c p -> (l p) c')
        return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels,
        use_post_norm=False,
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

        self.use_post_norm = use_post_norm
        if use_post_norm:
            self.post_norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x, y):
        # print("resblock x.std() y.std()", x.std(), y.std())
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)

        if self.use_post_norm:
            return x + self.post_norm(gate_mlp * h)

        # print("resblock x + gate_mlp * h.std()", (x + gate_mlp * h).std())

        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
        token_mixing_num=1,
        use_post_norm=False,
        layers_to_apply_grad_ckpt=0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.layers_to_apply_grad_ckpt = layers_to_apply_grad_ckpt

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            if token_mixing_num > 1:
                res_blocks.append(TokenMixer(token_mixing_num))
            res_blocks.append(ResBlock(
                model_channels,
                use_post_norm=use_post_norm,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            if hasattr(block, 'adaLN_modulation'):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        # print("forward x.std() t.std() c.std()", x.std(), t, c.std())
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        # print("forward x.std() t.std() c.std()", x.std(), t.std(), c.std())
        y = t + c
        # print("forward y.std()", y.std())

        layers_to_apply_grad_ckpt = self.layers_to_apply_grad_ckpt

        for i, block in enumerate(self.res_blocks):
            if (len(self.res_blocks) - i <= layers_to_apply_grad_ckpt and 
                self.grad_checkpointing and 
                not torch.jit.is_scripting()):
                x = checkpoint(block, x, y)
            else:
                x = block(x, y)

        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     for block in self.res_blocks:
        #         x = checkpoint(block, x, y)
        # else:
        #     for block in self.res_blocks:
        #         x = block(x, y)
        
        # print("forward x.std() y.std() final_layer(x, y).std()", x.std(), y.std(), self.final_layer(x, y).std())
        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    

class PixArt(nn.Module):
    """
    PixArt model architecture for diffusion modeling.
    Uses transformer blocks with self-attention for token mixing.
    """
    def __init__(
        self,
        in_channels,          # Number of input channels
        model_channels,       # Hidden dimension size
        out_channels,         # Number of output channels 
        z_channels,           # Dimension of conditioning embedding
        num_res_blocks,       # Number of transformer blocks
        grad_checkpointing=False,  # Whether to use gradient checkpointing
        token_mixing_num=1,   # Number of tokens to mix together
        heads = 16,          # Number of attention heads
        *args,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        # Embeddings for timesteps and conditioning
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        # Project input to model dimension
        self.input_proj = nn.Linear(in_channels, model_channels)

        # Time embedding block that outputs shift/scale params
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 6 * model_channels, bias=True)
        )

        # Stochastic depth decay rule
        drop_path = [x.item() for x in torch.linspace(0, 0., self.num_res_blocks)]

        assert model_channels % heads == 0, 'model_channels must be divisible by heads'

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            PixArtBlock(model_channels, heads, mlp_ratio=4.0, drop_path=drop_path[i])
            for i in range(self.num_res_blocks)
        ])

        # Final output layer
        self.final_layer = T2IFinalLayer(
            hidden_size=model_channels, 
            patch_size=1,
            out_channels=out_channels
        )

        assert token_mixing_num > 1, 'Token mixing should be use in PixArt otherwise it is just a MLP'
        self.token_mixing_num = token_mixing_num
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

    def forward(self, x, t, c):
        """
        Forward pass of the model.
        Args:
            x: Input tensor [N x C x ...]
            t: Timestep tensor [N]
            c: Conditioning tensor from AR transformer
        Returns:
            Output tensor [N x C x ...]
        """
        def sanity_check(t, c):
            """Verify timesteps and conditioning are consistent across token groups"""
            t = rearrange(t, '(l p) -> p l', p=self.token_mixing_num)
            c = rearrange(c, '(l p) c -> p l c', p=self.token_mixing_num)
            assert (t[:-1] - t[1:]).float().abs().max() < 1e-6, 'timesteps must be the same'
            assert (c[:-1] - c[1:]).abs().max() < 1e-6, 'conditioning must be the same'

        # Reshape input for token mixing
        x = rearrange(x, '(l p) c -> l p c', p=self.token_mixing_num)

        x = self.input_proj(x)
        assert t.size(0) == x.size(0) * self.token_mixing_num, f'timesteps ({t.size(0)}) must be divisible by token_mixing_num ({self.token_mixing_num})'
        assert c.size(0) == x.size(0) * self.token_mixing_num, f'conditioning must be divisible by token_mixing_num, got c.size(0)={c.size(0)} and token_mixing_num={self.token_mixing_num}'

        # Take first token group's timesteps/conditioning since they're identical
        t = rearrange(t, '(l p) -> p l', p=self.token_mixing_num)[0]
        c = rearrange(c, '(l p) c -> p l c', p=self.token_mixing_num)[0]

        # Get embeddings
        t = self.time_embed(t) # (N, D)
        c = self.cond_embed(c)

        # Get time modulation parameters
        t_0 = self.t_block(t)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c, t_0) 

        # Final output projection
        x = self.final_layer(x, t)
        x = rearrange(x, 'l p c -> (l p) c')
        return x
    
    def forward_with_cfg(self, x, t, c, cfg_scale):
        """
        Forward pass with classifier-free guidance.
        Applies guidance scale to modify prediction based on conditioning.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    

class PixArtwoAttn(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
        token_mixing_num=1,
        heads=16,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        # Embeddings for timesteps and conditioning
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        # Project input to model dimension
        self.input_proj = nn.Linear(in_channels, model_channels)

        # Time embedding block that outputs shift/scale params
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 6 * model_channels, bias=True)
        )

        # Stochastic depth decay rule
        drop_path = [x.item() for x in torch.linspace(0, 0., self.num_res_blocks)]

        # Stack of transformer blocks without attention
        self.blocks = nn.ModuleList([
            PixArtBlockwoAttn(model_channels, heads, mlp_ratio=4.0, drop_path=drop_path[i])
            for i in range(self.num_res_blocks)
        ])

        # Final output layer
        self.final_layer = T2IFinalLayer(
            hidden_size=model_channels, 
            patch_size=1,
            out_channels=out_channels
        )

        assert token_mixing_num == 1
        self.token_mixing_num = token_mixing_num
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

    def forward(self, x, t, c):
        """
        Forward pass of the model.
        Args:
            x: Input tensor [N x C]
            t: Timestep tensor [N]
            c: Conditioning tensor from AR transformer
        Returns:
            Output tensor [N x C]
        """
        x  = x.unsqueeze(1)
        x = self.input_proj(x)

        # Get embeddings
        t = self.time_embed(t)
        c = self.cond_embed(c)

        # Get time modulation parameters
        t_0 = self.t_block(t)

        # Apply transformer blocks without attention
        for block in self.blocks:
            x = block(x, c, t_0)

        # Final output projection
        x = self.final_layer(x, t)
        x = x.squeeze(1)
        return x

    def forward_with_cfg(self, x, t, c, cfg_scale):
        """
        Forward pass with classifier-free guidance.
        Applies guidance scale to modify prediction based on conditioning.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class DiffLossType(Enum):
    MLP = SimpleMLPAdaLN
    PIXART = PixArt
    PIXARTWOATTN = PixArtwoAttn
