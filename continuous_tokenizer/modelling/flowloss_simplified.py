from modelling.flowloss import SimpleMLPAdaLN

import torch
import torch.nn as nn


class FlowLoss(nn.Module):
    """Flow Loss using rectified flow"""
    def __init__(
        self, 
        target_channels, 
        depth, 
        width, 
        num_sampling_steps='100', 
    ):
        super(FlowLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,  # Only need target_channels for flow (no VLB term)
            num_res_blocks=depth,
        )
        
        self.num_sampling_steps = int(num_sampling_steps)

    def forward(self, target, reduction='mean'):
        t = torch.rand(target.shape[0], device=target.device)

        # Compute noised data at time t
        noise = torch.randn_like(target)
        x_t = target * (1 - t[:, None]) + noise * t[:, None]
        
        # Predict velocity field
        v = self.net(x_t, t)
        
        # Compute loss (MSE between predicted and ideal velocity)
        ideal_v = noise - target
        loss = torch.nn.functional.mse_loss(v, ideal_v, reduction=reduction)
        
        return loss


    @torch.no_grad()
    def sample(self, batch_size=None):
        # Initialize with random noise
        x = torch.randn(batch_size, self.in_channels).cuda()

        # Euler steps for sampling
        dt = 1.0 / self.num_sampling_steps
        for t in torch.linspace(1.0, dt, self.num_sampling_steps, device=x.device):
            # Predict velocity
            v = self.net(x, t)
            # Euler step
            x = x - v * dt

        return x

