from typing import Optional, Union, Tuple
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from einops import rearrange


class FlowLoss(nn.Module):
    """Flow Loss using rectified flow"""
    def __init__(
        self, 
        target_channels, 
        depth, 
        width, 
        num_sampling_steps='100', 
        grad_checkpointing=False,
        flow_mul=1,
        z_channels=None, 
        norm_target=False,
        moving_avg_std=False,
        unit_norm=False,
        norm_v=False,
        pretrained_path=None,
    ):
        super(FlowLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,  # Only need target_channels for flow (no VLB term)
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        )
        
        self.num_sampling_steps = int(num_sampling_steps)
        self.flow_mul = flow_mul
        self.norm_target = norm_target
        self.moving_avg_std = moving_avg_std
        self.unit_norm = unit_norm
        self.norm_v = norm_v
        if self.moving_avg_std:
            self.register_buffer('moving_avg', torch.tensor(0.0))
            self.moving_avg_decay = 0.9997

        self.init_weight(pretrained_path)

    def init_weight(self, pretrained_path=None):
        if pretrained_path is None:
            return
        state_dict = torch.load(pretrained_path, map_location='cpu')
        if "ema" in state_dict:
            print(f"Loading ema for flow loss from {pretrained_path}")
            state_dict = state_dict["ema"]
        elif "model" in state_dict:
            print(f"Loading model for flow loss from {pretrained_path}")
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            print(f"Loading state_dict for flow loss from {pretrained_path}")
            state_dict = state_dict["state_dict"]
        else:
            raise ValueError(f"Unknown state dict: {state_dict.keys()}")
        
        if next(iter(state_dict.keys())).startswith('flow_loss'):
            state_dict = {k[len('flow_loss.'):]: v for k, v in state_dict.items()}
        
        self.load_state_dict(state_dict)

    def forward_fitting(self, target, z=None, reduction='mean'):
        # Loss 2: Match the target
        batch_size = target.shape[0]
        t = torch.rand(batch_size, device=target.device)
        noise = torch.randn_like(target)
        x_t = target * (1 - t[:, None]) + noise * t[:, None]
        v = self.net(x_t, t, z)
        if self.norm_v:
            v = nn.functional.normalize(v, dim=-1) * math.sqrt(self.in_channels)

        ideal_target = noise - v
        loss = torch.nn.functional.mse_loss(ideal_target, target, reduction=reduction)
        return loss

    def forward(self, target, z=None, reduction='mean', detach_pred_v=False, seperate_fitting=False):
        if target.dim() == 3:
            target = target.reshape(-1, target.shape[-1])
        if z is not None and z.dim() == 3:
            z = z.reshape(-1, z.shape[-1])
        
        if self.norm_target:
            if not self.moving_avg_std and not self.unit_norm:
                target_mean = target.mean().detach()
                target_std = target.std().detach()
                target = (target - target_mean) / (target_std + 1e-8)
            elif self.unit_norm:
                target = nn.functional.normalize(target, dim=-1)
            else:
                if self.moving_avg < 1e-8:  # initialize moving avg
                    self.moving_avg = torch.tensor(target.std().detach())
                else:
                    self.moving_avg = self.moving_avg * self.moving_avg_decay + \
                        target.std().detach() * (1 - self.moving_avg_decay)
                target_mean = target.mean(dim=-1, keepdim=True).detach()
                target_std = self.moving_avg
                target = target_mean + target * target_std
        
        if self.flow_mul > 1:
            # target = target.repeat_interleave(self.flow_mul, dim=0)
            target = target.repeat(self.flow_mul, 1)
            if z is not None:
                z = z.repeat(self.flow_mul, 1)
        
        if seperate_fitting:
            return self.forward_fitting(target, z, reduction)

        # Loss 1: Match the field
        t = torch.rand(target.shape[0], device=target.device)

        # Compute noised data at time t
        noise = torch.randn_like(target)
        x_t = target * (1 - t[:, None]) + noise * t[:, None]
        if detach_pred_v:
            x_t = x_t.detach()

        # Predict velocity field
        v = self.net(x_t, t, z)
        if self.norm_v:
            v = nn.functional.normalize(v, dim=-1) * math.sqrt(self.in_channels)
        
        # Compute loss (MSE between predicted and ideal velocity)
        ideal_v = noise - target
        loss = torch.nn.functional.mse_loss(v, ideal_v, reduction=reduction)
        
        return loss


    # @torch.no_grad()
    # def sample(self, z=None, temperature=1.0, cfg=1.0, batch_size=None):
    #     assert not (z is None and batch_size is None), "Either z or batch_size must be provided"
    #     if z is not None:
    #         batch_size = z.shape[0]

    #     # Initialize with random noise
    #     if not cfg == 1.0 and z is not None:
    #         x = torch.randn(batch_size // 2, self.in_channels).cuda() * temperature
    #         x = torch.cat([x, x], dim=0)
    #         model_kwargs = dict(c=z, cfg_scale=cfg)
    #         sample_fn = self.net.forward_with_cfg
    #     else:
    #         x = torch.randn(batch_size, self.in_channels).cuda() * temperature
    #         model_kwargs = dict(c=z)
    #         sample_fn = self.net.forward

    #     # Euler steps for sampling
    #     dt = 1.0 / self.num_sampling_steps
    #     for t in torch.linspace(1.0, dt, self.num_sampling_steps, device=x.device):
    #         # Predict velocity
    #         v = sample_fn(x, t.expand(x.shape[0]), **model_kwargs)
    #         # Euler step
    #         x = x - v * dt

    #     return x


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
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)

        if self.use_post_norm:
            return x + self.post_norm(gate_mlp * h)

        return x + gate_mlp * h


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
        num_res_blocks,
        grad_checkpointing=False,
        z_channels=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        if z_channels is not None:
            self.cond_embed = nn.Linear(z_channels, model_channels)
        else:
            # Unconditional
            self.cond_embed = None

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
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

    def forward(self, x, t, c=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        if self.cond_embed is not None and c is not None:
            c = self.cond_embed(c)
            y = t + c
        else:
            y = t

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

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
