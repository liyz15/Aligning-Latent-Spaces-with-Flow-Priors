import torch
import torch.nn as nn

from torchdiffeq import odeint

from models.diffloss import DiffLossType


class FlowLoss(nn.Module):
    """Flow Loss using rectified flow"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False,
                 token_mixing_num=1, network_type='MLP', mlp_post_norm=False, layers_to_apply_grad_ckpt=0):
        super(FlowLoss, self).__init__()
        self.in_channels = target_channels
        self.network_type = network_type
        self.token_mixing_num = token_mixing_num
        self.net = DiffLossType[network_type].value(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,  # Only need target_channels for flow (no VLB term)
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
            token_mixing_num=token_mixing_num,
            use_post_norm=mlp_post_norm,
            layers_to_apply_grad_ckpt=layers_to_apply_grad_ckpt,
        )
        
        self.num_sampling_steps = int(num_sampling_steps)

    def forward(self, target, z, mask=None):
        # Sample random timesteps uniformly between 0 and 1
        if self.network_type in ['MLP', 'PIXARTWOATTN']:
            t = torch.rand(target.shape[0], device=target.device)
        elif self.network_type == 'PIXART':
            assert target.shape[0] % self.token_mixing_num == 0, 'Batch size must be divisible by token_mixing_num'
            t = torch.rand(target.shape[0] // self.token_mixing_num, device=target.device)
            t = t.repeat_interleave(self.token_mixing_num)

        # Compute noised data at time t
        noise = torch.randn_like(target)
        x_t = target * (1 - t[:, None]) + noise * t[:, None]
        
        # Predict velocity field
        v = self.net(x_t, t, z)
        
        # Compute loss (MSE between predicted and ideal velocity)
        ideal_v = noise - target
        loss = torch.nn.functional.mse_loss(v, ideal_v, reduction='none')
        
        if mask is not None:
            # Mask is already flattened to [bsz * seq_len * diffusion_batch_mul]
            # Loss is [bsz * seq_len * diffusion_batch_mul, channels]
            # Sum over channels first
            loss = loss.mean(dim=-1)  # Average over channels
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)  # Weighted average over batch
            return loss
        return loss.mean()

    # @torch.no_grad()
    # def sample(self, z, temperature=1.0, cfg=1.0):
    #     # Initialize with random noise
    #     if not cfg == 1.0:
    #         x = torch.randn(z.shape[0] // 2, self.in_channels).cuda() * temperature
    #         x = torch.cat([x, x], dim=0)
    #         model_kwargs = dict(c=z, cfg_scale=cfg)
    #         sample_fn = self.net.forward_with_cfg
    #     else:
    #         x = torch.randn(z.shape[0], self.in_channels).cuda() * temperature
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

    @torch.no_grad()
    def sample(self, z, temperature=1.0, cfg=1.0, rtol=1e-5, atol=1e-5, method='euler'):

        # Initialize with random noise
        if not cfg == 1.0:
            x = torch.randn(z.shape[0] // 2, self.in_channels).cuda() * temperature
            x = torch.cat([x, x], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            x = torch.randn(z.shape[0], self.in_channels).cuda() * temperature
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        # Define ODE function
        def ode_fn(t, x):
            if isinstance(t, float):
                t = torch.tensor([t], device=x.device)
            t_batch = t.expand(x.shape[0])
            v = sample_fn(x, t_batch, **model_kwargs)
            return v  # Negative because we're integrating from t=1 to t=0

        # Time span from 1.0 to 0.0
        t = torch.linspace(1.0, 0.0, self.num_sampling_steps, device=x.device)

        # Solve ODE using Dormand-Prince RK45
        x = odeint(
            ode_fn,
            x,
            t,
            method=method,
            rtol=rtol,
            atol=atol,
        )[-1]  # Get the final state at t=0.0

        return x