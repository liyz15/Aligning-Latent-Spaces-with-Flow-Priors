# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


from modelling.modules import Encoder, Decoder, TimmViTEncoder, TimmViTDecoder
from modelling.quantizers.vq import VectorQuantizer
from modelling.quantizers.kl import DiagonalGaussianDistribution
from modelling.quantizers.softvq import SoftVectorQuantizer
from modelling.flowloss import FlowLoss

from timm import create_model


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


@dataclass
class ModelArgs:
    image_size: int = 256
    base_image_size: int = 256
    
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    vq_loss_ratio: float = 1.0 # for soft vq
    kl_loss_weight: float = 0.000001
    tau: float = 0.1
    num_codebooks: int = 1
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0

    enc_type: str = 'cnn'
    dec_type: str = 'cnn'
    encoder_model: str = 'llamagen_encoder'
    decoder_model: str = 'llamagen_decoder'
    num_latent_tokens: int = 256
    to_pixel: str = 'linear'
    
    # for pre-trained models
    enc_tuning_method: str = 'full'
    dec_tuning_method: str = 'full'
    enc_pretrained: bool = True
    dec_pretrained: bool = False 
    
    # for vit 
    enc_patch_size: int = 16
    dec_patch_size: int = 16
    enc_drop_path_rate: float = 0.0
    dec_drop_path_rate: float = 0.0

    # encoder token drop
    enc_token_drop: float = 0.0
    enc_token_drop_max: float = 0.6
    
    # deocder cls token
    dec_cls_token: bool = True
    
    # rope
    use_ape: bool = True 
    use_rope: bool = False
    rope_mixed: bool = False
    rope_theta: float = 10.0
    
    # repa for vit
    repa: bool = False
    repa_patch_size: int = 16
    repa_model: str = 'vit_base_patch16_224'
    repa_proj_dim: int = 2048
    repa_loss_weight: float = 0.1
    repa_align: str = 'global'
    repa_flow_depth: int = 2
    repa_flow_mul: int = 4
    
    vq_mean: float = 0.0
    vq_std: float = 1.0

    # gradient checkpointing
    grad_ckpt: bool = False

    # flow loss
    # Unlike repa which is applied to any model, flow loss is only applied to FlowAEModel
    # Therefore we do not need to set enable flag
    flow_loss_weight: float = 0.1
    flow_target_channels: int = 32
    flow_z_channels: int = 768
    flow_depth: int = 6
    flow_width: int = 1024
    flow_num_sampling_steps: int = 100
    flow_grad_checkpointing: bool = False
    flow_flow_mul: int = 4
    flow_norm_target: bool = False
    flow_moving_avg_std: bool = False

    std_latents: bool = False


class VQModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs, 
                tags=["arxiv:2412.10958", "image-generation", "32 tokens", "SoftVQ-VAE"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__()
        self.config = config
        self.vq_mean = config.vq_mean
        self.vq_std = config.vq_std
        self.num_latent_tokens = config.num_latent_tokens
        self.codebook_embed_dim = config.codebook_embed_dim
        
        self.repa = config.repa
        self.repa_loss_weight = config.repa_loss_weight
        self.repa_align = config.repa_align
        if config.repa and config.enc_type == 'vit':
            self.repa_model = create_model(config.repa_model, pretrained=True, img_size=config.image_size, patch_size=config.repa_patch_size)
            for param in self.repa_model.parameters():
                param.requires_grad = False
            self.repa_model.eval()
            repa_z_dim = self.repa_model.embed_dim
            self.repa_z_dim = repa_z_dim
            if config.repa_align in ['global', 'avg_1d', 'avg_1d_shuffle', 'repeat']:
                self.projection = build_mlp(config.codebook_embed_dim, config.repa_proj_dim, repa_z_dim)
            else:
                self.projection = FlowLoss(
                    target_channels=repa_z_dim,
                    depth=config.repa_flow_depth,
                    width=config.repa_proj_dim,
                    num_sampling_steps=100,
                    flow_mul=config.repa_flow_mul,
                    z_channels=config.codebook_embed_dim,
                )
            from modelling.lpips.lpips_timm import Normalize, Denormalize
            self.de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if config.grad_ckpt:
                self.repa_model.set_grad_checkpointing(True)
        else:
            repa_z_dim = None
        
        
        if config.enc_type == 'cnn':
            if config.encoder_model == 'llamagen_encoder':
                self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels * 2, dropout=config.dropout_p)
            else:
                raise NotImplementedError
            self.quant_conv = nn.Conv2d(config.z_channels * 2, config.codebook_embed_dim, 1)
        elif config.enc_type == 'vit':
            self.encoder = TimmViTEncoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.encoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  #
                model_kwargs={'img_size': config.image_size, 'patch_size': config.enc_patch_size, 'drop_path_rate': config.enc_drop_path_rate},
                pretrained=config.enc_pretrained,
                tuning_method=config.enc_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                token_drop=config.enc_token_drop,
                token_drop_max=config.enc_token_drop_max,
                base_img_size=config.base_image_size
            )
            self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim)
            # Enable gradient checkpointing for encoder if configured
            if config.grad_ckpt:
                self.encoder.model.set_grad_checkpointing(True)
            
        
        if config.dec_type == 'cnn':
            if config.decoder_model == 'llamagen_decoder':
                self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            else:
                raise NotImplementedError
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)
        elif config.dec_type == 'vit':
            self.decoder = TimmViTDecoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.decoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  #
                model_kwargs={'img_size': config.image_size, 'patch_size': config.dec_patch_size, 'drop_path_rate': config.dec_drop_path_rate, 'latent_dim': config.codebook_embed_dim},
                pretrained=config.dec_pretrained,
                tuning_method=config.dec_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                cls_token=config.dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_pixel=config.to_pixel,
                base_img_size=config.base_image_size
            )
            self.post_quant_conv = nn.Linear(config.codebook_embed_dim, self.decoder.embed_dim)
            # Enable gradient checkpointing for decoder if configured
            if config.grad_ckpt:
                self.decoder.model.set_grad_checkpointing(True)
        # check movq
        if 'movq' in config.decoder_model:
            self.use_movq = True 
        else:
            self.use_movq = False
        
        
        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        
        if self.repa and self.training:
            # get z from repa_encoder
            rescale_x = self.scale(self.de_scale(x))
            z = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]

            # taking average over spatial dimension
            if self.repa_align == 'global':
                z = z.mean(dim=1)
                z_hat = quant.mean(dim=1)
                # calculate repa loss
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d':
                z = F.adaptive_avg_pool1d(z.permute(0, 2, 1), quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d_shuffle':
                # shuffle the length dimension of z and avg
                indices = torch.randperm(z.shape[1])
                z = F.adaptive_avg_pool1d(z[:, indices, :].permute(0, 2, 1) , quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'repeat':
                z_hat = self.projection(quant)
                b, l, d = z_hat.shape
                z_hat = z_hat.unsqueeze(2).expand(-1, -1, z.size(1) // l, -1).reshape(b, -1, d)
            elif self.repa_align == 'repeat_flow':
                b, l, d = quant.shape
                z_hat = quant.unsqueeze(2).expand(-1, -1, z.size(1) // l, -1).reshape(b, -1, d)
                proj_loss = self.projection(target=z, z=z_hat)

            if self.repa_align in ['global', 'avg_1d', 'avg_1d_shuffle', 'repeat']:
                # For repeat flow, the loss is calculated in the flow loss
                z = F.normalize(z, dim=-1)
                z_hat = F.normalize(z_hat, dim=-1)
                proj_loss = mean_flat(-(z * z_hat).sum(dim=-1))
                proj_loss = proj_loss.mean()

            proj_loss *= self.repa_loss_weight
            
            emb_loss += (proj_loss,)
        
        return quant, emb_loss, info

    def decode(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        if self.use_movq:
            dec = self.decoder(quant, tmp_quant, h, w)
        else:
            dec = self.decoder(quant, None, h, w)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, extra_latents=None):
        b, _, h, w = input.size()
        quant, diff, info = self.encode(input)
        self.quant = quant
        if extra_latents is not None:
            quant = torch.cat([quant, extra_latents], dim=0)
        dec = self.decode(quant, x=input, h=h, w=w)
        return dec, diff, info


class SoftVQModel(VQModel, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs, 
                tags=["arxiv:2412.10958", "image-generation", "32 tokens", "SoftVQ-VAE"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__(config)
        self.quantize = SoftVectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                            config.entropy_loss_ratio, 
                                            config.tau,                                   
                                            config.num_codebooks,
                                            config.codebook_l2_norm, config.codebook_show_usage)


class KLModel(VQModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.kl_loss_weight = float(config.kl_loss_weight)
        self.quantize = None
        
        if config.enc_type == 'cnn':
            self.quant_conv = nn.Conv2d(config.z_channels * 2, config.codebook_embed_dim * 2, 1)
        elif config.enc_type == 'vit':
            self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim * 2)
        

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        # quant, emb_loss, info = self.quantize(h)
        h_posterior = DiagonalGaussianDistribution(h)
        return h_posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode_code(self, posterior, shape=None):
        z = posterior.sample()
        dec = self.decode(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        # compute kl loss
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        diff = (kl_loss * self.kl_loss_weight, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = z
        return dec, diff, info


class AEModel(VQModel):
    def __init__(self, config: ModelArgs,
                tags=["arxiv:xxx", "image-generation", "1d-tokenizer", "128 tokens", "MAETok"], 
                repo_url="https://github.com/Hhhhhhao/continuous_tokenizer", 
                license="apache-2.0"):
        super().__init__(config)
        self.quantize = None 
        self.std_latents = config.std_latents

    def encode(self, x):
        h = self.encoder(x)
        quant = self.quant_conv(h)
        if self.std_latents:
            quant = (quant - quant.mean().item()) / (quant.std().item() + 1e-8)

        emb_loss = (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = quant
        return quant, emb_loss, info
    
    @property
    def quant_layer(self):
        return self.quant_conv.weight

    def decode(self, quant, x=None, h=None, w=None):
        z = self.post_quant_conv(quant)
        dec = self.decoder(z)
        return dec


class FlowAEModel(VQModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.quantize = None
        self.flow_loss_weight = config.flow_loss_weight
        self.flow_loss = FlowLoss(
            target_channels=config.flow_target_channels,
            depth=config.flow_depth,
            width=config.flow_width,
            flow_mul=config.flow_flow_mul,
            norm_target=config.flow_norm_target,
            moving_avg_std=config.flow_moving_avg_std,
        )

    def encode(self, x):
        h = self.encoder(x)
        quant = self.quant_conv(h)

        flow_loss = self.flow_loss(torch.cat([quant, quant.detach()], dim=0), reduction='none')

        flow_loss = flow_loss.mean(dim=1)
        flow_loss, flow_loss_detached = flow_loss.chunk(2, dim=0)
        flow_loss, flow_loss_detached = flow_loss.mean(), flow_loss_detached.mean()

        flow_loss = flow_loss * self.flow_loss_weight + flow_loss_detached

        # vq_loss, commit_loss, entropy_loss, codebook_usage, repa_loss, flow_loss
        emb_loss = tuple([torch.tensor(0.) for _ in range(5)]) + (flow_loss,)
        info = quant
        return quant, emb_loss, info
    
    def decode(self, quant, x=None, h=None, w=None):
        z = self.post_quant_conv(quant)
        dec = self.decoder(z)
        return dec



#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def VQ_16(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def KL_8(**kwargs):
    return KLModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def KL_16(**kwargs):
    return KLModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def AE_16(**kwargs):
    return AEModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def SoftVQ(**kwargs):
    return SoftVQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def FlowAE(**kwargs):
    return FlowAEModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

VQ_models = {
    'AE-16': AE_16,
    'VQ-16': VQ_16, 'VQ-8': VQ_8,
    'KL-16': KL_16, 'KL-8': KL_8,
    'SoftVQ': SoftVQ,
    'FlowAE': FlowAE,
    }


if __name__ == '__main__':
    
    # model = VQ_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', repa=True, repa_model='vit_base_patch14_dinov2.lvd142m', repa_align='avg_1d_shuffle', enc_img_res=True, enc_img_align='avg_1d', dec_img_res=True)    
    model = SoftVQ_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', num_codebooks=4, codebook_size=16384, topk=16)
    # model = AE_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m', num_codebooks=4, codebook_size=16384)
    model.train()
    # model = KL_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m')
    # model = GMM_16(codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_base_patch14_dinov2.lvd142m', decoder_model='vit_base_patch14_dinov2.lvd142m')
    x = torch.randn(1, 3, 256, 256)
    y, _, info = model(x)
