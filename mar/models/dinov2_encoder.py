import torch
import torch.nn as nn
from timm import create_model

import sys
sys.path.append('/path/to/continuous_tokenizer')
from modelling.flowloss import FlowLoss

class Normalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Normalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

class Denormalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Denormalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return x * self.std + self.mean


class DinoV2Encoder(nn.Module):
    def __init__(self, model_name='vit_large_patch14_dinov2.lvd142m', image_size=256, patch_size=16):
        super().__init__()
        self.model = create_model(model_name, pretrained=True, img_size=image_size, patch_size=patch_size)
        self.de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def encode(self, x):
        rescale_x = self.scale(self.de_scale(x))
        return self.model.forward_features(rescale_x)[:, self.model.num_prefix_tokens:]
    
    def decode(self, x):
        raise NotImplementedError("DinoV2Encoder does not support decoding")
    
    def forward(self, x):
        raise NotImplementedError("DinoV2Encoder does not support forward pass")


class RevertDinoV2Encoder(DinoV2Encoder):
    def __init__(self, model_name='vit_large_patch14_dinov2.lvd142m', image_size=256, patch_size=16,
                 flow_head_path=None, vae_scaling_factor=0.6843):
        super().__init__(model_name, image_size, patch_size)
        self.flow_head = FlowLoss(
            target_channels=1024,
            depth=6,
            width=1024,
            num_sampling_steps='100',
            grad_checkpointing=False,
        )
        flow_head_path = '/path/to/flow_head.pt'
        self.load_flow_head(flow_head_path)
        
        self.vae_scaling_factor = vae_scaling_factor
    
    def load_flow_head(self, flow_head_path):
        if flow_head_path is None:
            return
        ckpt = torch.load(flow_head_path, map_location='cpu')
        state_dict = ckpt['model']
        state_dict = {k.replace('flow_loss.', ''): v for k, v in state_dict.items()}
        self.flow_head.load_state_dict(state_dict)
    
    @torch.no_grad()
    def revert_sample(self, z, num_sampling_steps=100):
        bz, seq_len = z.shape[:2]
        z = z.reshape(bz * seq_len, -1)
        dt = 1.0 / num_sampling_steps
        sample_fn = self.flow_head.net.forward
        z = z.mul_(self.vae_scaling_factor)

        for t in torch.linspace(dt, 1.0, num_sampling_steps, device=z.device):
            z = z + sample_fn(z, t.expand(bz * seq_len)) * dt
        z = z.reshape(bz, seq_len, -1)
        return z

    def encode(self, x):
        z = super().encode(x)
        z = self.revert_sample(z)
        return z
