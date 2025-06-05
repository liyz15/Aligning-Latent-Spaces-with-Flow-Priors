import torch
import torch.nn as nn
import torch.nn.functional as F

from modelling.flowloss import FlowLoss

import torch.distributed as tdist

class FlowHeadLoss(nn.Module):
    def __init__(self, 
                 flow_target_channels=32,
                 flow_depth=6,
                 flow_width=1024,
                 flow_flow_mul=4,
                 flow_norm_target=False,
                 flow_moving_avg_std=False,
                 flow_unit_norm=False,
                 flow_detach_pred_v=False,
                 flow_seperate_fitting=False,
                 flow_norm_v=False,
                 **kwargs,
    ):
        super().__init__()
        self.flow_loss = FlowLoss(
            target_channels=flow_target_channels,
            depth=flow_depth,
            width=flow_width,
            flow_mul=flow_flow_mul,
            norm_target=flow_norm_target,
            moving_avg_std=flow_moving_avg_std,
            unit_norm=flow_unit_norm,
            norm_v=flow_norm_v,
        )
        self.flow_detach_pred_v = flow_detach_pred_v
        self.flow_seperate_fitting = flow_seperate_fitting
        self.tensorboard_writer = None

    def set_writer(self, tensorboard_writer=None):
        self.tensorboard_writer = tensorboard_writer

    def forward(self, latents, global_step, logger=None, log_every=100):

        loss = self.flow_loss(latents, detach_pred_v=self.flow_detach_pred_v, seperate_fitting=self.flow_seperate_fitting)

        latent_norm = torch.linalg.norm(latents, dim=-1).mean()

        if global_step % log_every == 0:
            logger.info(f"flow_loss: {loss:.4f}, "
                        f"latent_norm: {latent_norm:.4f}"
            )
            
            if tdist.get_rank() == 0 and self.tensorboard_writer is not None:
                for key, value in {
                    "flow_loss": loss,
                    "latent_norm": latent_norm,
                }.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    self.tensorboard_writer.add_scalar(f"FlowHeadLoss/{key}", value, global_step)
        return loss
