from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
import math
import numpy as np


from modelling.modules import Encoder, Decoder, TimmViTEncoder, TimmViTDecoder
from modelling.quantizers.vq import VectorQuantizer
from modelling.quantizers.kl import DiagonalGaussianDistribution
from modelling.quantizers.softvq import SoftVectorQuantizer
from modelling.flowloss import FlowLoss


class SimpleMLPAdaLN(nn.Module):
    """Simple MLP with Adaptive Layer Norm for flow model"""
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, model_channels)
        
        # Residual blocks with adaptive layer norm
        self.res_blocks = nn.ModuleList([])
        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResBlock(model_channels, time_embed_dim)
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, out_channels)
        )
    
    def forward(self, x, t):
        # Embed time
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x.device)
        elif t.dim() == 0:
            t = t.view(1)
        t = t.float().view(-1, 1)
        temb = self.time_embed(t)
        
        # Input projection
        h = self.input_proj(x)
        
        # Process through residual blocks
        for block in self.res_blocks:
            h = block(h, temb)
        
        # Output projection
        output = self.output_proj(h)
        
        return output


class ResBlock(nn.Module):
    """Residual block with adaptive layer norm"""
    def __init__(self, channels, time_embed_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)
        
        self.time_emb_proj = nn.Linear(time_embed_dim, 2 * channels)
        
        self.conv1 = nn.Linear(channels, channels)
        self.conv2 = nn.Linear(channels, channels)
        
    def forward(self, x, temb):
        h = self.ln1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_emb_proj(temb).unsqueeze(1)
        scale, shift = time_emb.chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        
        h = self.ln2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return x + h


class PatchFlowLoss(nn.Module):
    """Flow Loss using rectified flow for image patches"""
    def __init__(
        self, 
        patch_size=16,
        target_channels=768, 
        depth=6, 
        width=1024, 
        num_sampling_steps=100,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            num_res_blocks=depth,
        )
        
        self.num_sampling_steps = int(num_sampling_steps)
        
    @torch.no_grad()
    def _unfold_img(self, x):
        """Convert image to patches: [B, C, H, W] -> [B*num_patches, C*patch_size*patch_size]"""
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2)  # [B, num_patches, C*patch_size*patch_size]
        return patches.reshape(-1, patches.shape[-1])  # [B*num_patches, C*patch_size*patch_size]
        
    @torch.no_grad()
    def _fold_img(self, patches, batch_size, height, width):
        """Convert patches back to image: [B*num_patches, C*patch_size*patch_size] -> [B, C, H, W]"""
        C = 3  # Assuming RGB images
        patches_per_row = width // self.patch_size
        patches_per_col = height // self.patch_size
        num_patches = patches_per_row * patches_per_col
        
        patches = patches.reshape(batch_size, num_patches, -1)
        patches = patches.transpose(1, 2)  # [B, C*patch_size*patch_size, num_patches]
        
        reconstructed = F.fold(
            patches, 
            output_size=(height, width), 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        return reconstructed

    def forward(self, images, reduction='mean'):
        """
        Compute flow loss for image patches
        Args:
            images: [B, C, H, W] tensor of images
            reduction: Reduction method for loss
        """
        # Convert images to patches
        # patches = self._unfold_img(images)  # [B*num_patches, C*patch_size*patch_size]
        patches = images.reshape(-1, images.shape[-1])

        # Sample random timesteps
        t = torch.rand(patches.shape[0], device=patches.device)
        
        # Compute noised data at time t
        noise = torch.randn_like(patches)
        x_t = patches * t[:, None] + noise * (1 - t[:, None])
        
        # Predict velocity field
        v = self.net(x_t, t)
        
        # Compute loss (MSE between predicted and ideal velocity)
        ideal_v = patches - noise
        loss = F.mse_loss(v, ideal_v, reduction=reduction)
        
        return loss

    @torch.no_grad()
    def encode(self, images):
        """
        Encode images to latent representations by running the flow model backward
        Args:
            images: [B, C, H, W] tensor of images
        Returns:
            latents: [B, num_patches, patch_dim] tensor of latent codes
        """
        B, C, H, W = images.shape
        patches = self._unfold_img(images)  # [B*num_patches, C*patch_size*patch_size]
        
        # Run the flow model backward (data -> noise)
        latents = patches  # Initialize with data
        
        # Euler steps for encoding
        dt = 1.0 / self.num_sampling_steps
        for t in torch.linspace(1.0, dt, self.num_sampling_steps, device=latents.device):
            v = self.net(latents, t)  # Use latents as input
            latents = latents - v * dt  # Update latents directly
        
        # Reshape final encoded latents
        patches_per_img = (H // self.patch_size) * (W // self.patch_size)
        latents = latents.reshape(B, patches_per_img, -1)
        
        return latents

    @torch.no_grad()
    def decode(self, latents, height=None, width=None):
        """
        Decode latent representations to images
        Args:
            latents: [B, num_patches, patch_dim] tensor of latent codes
            height: Height of the output image
            width: Width of the output image
        Returns:
            reconstructed: [B, C, H, W] tensor of reconstructed images
        """
        B, num_patches, patch_dim = latents.shape

        latents = latents.reshape(-1, patch_dim)

        dt = 1.0 / self.num_sampling_steps
        for t in torch.linspace(0.0, 1.0 - dt, self.num_sampling_steps, device=latents.device):
            v = self.net(latents, t)
            latents = latents + v * dt
        
        reconstructed = self._fold_img(latents, B, height, width)
        
        return reconstructed


@dataclass
class FlowTokenizerArgs:
    """Configuration for the Flow Tokenizer"""
    image_size: int = 256
    patch_size: int = 16
    flow_depth: int = 6
    flow_width: int = 1024
    flow_loss_weight: float = 1.0
    flow_num_sampling_steps: int = 100
    dropout_p: float = 0.0
    flow_target_channels: int = 768  # 3 * 16 * 16 (RGB * 16x16 patch)


class FlowTokenizer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: FlowTokenizerArgs,
                 tags=["image-tokenization", "flow-based-model"],
                 repo_url="https://github.com/Hhhhhhao/continuous_tokenizer",
                 license="apache-2.0"):
        super().__init__()
        self.config = config
        
        # Ensure image size is divisible by patch size
        assert config.image_size % config.patch_size == 0, f"Image size {config.image_size} must be divisible by patch size {config.patch_size}"
        
        # Calculate number of patches
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.patch_dim = 3 * config.patch_size * config.patch_size  # RGB * patch_size^2
        
        # Create the flow model for patches
        self.flow_model = PatchFlowLoss(
            patch_size=config.patch_size,
            target_channels=self.patch_dim,
            depth=config.flow_depth,
            width=config.flow_width,
            num_sampling_steps=config.flow_num_sampling_steps,
        )
        
        self.flow_loss_weight = config.flow_loss_weight

    def encode(self, x):
        """
        Encode images to latent representations
        Args:
            x: [B, C, H, W] tensor of images
        Returns:
            latents: [B, num_patches, patch_dim] tensor of latent codes
        """
        return self.flow_model.encode(x)

    def decode(self, latents, height=None, width=None):
        """
        Decode latent representations to images
        Args:
            latents: [B, num_patches, patch_dim] tensor of latent codes
            height: Height of the output image (optional)
            width: Width of the output image (optional)
        Returns:
            reconstructed: [B, C, H, W] tensor of reconstructed images
        """
        return self.flow_model.decode(latents, height, width)

    def forward(self, input_images):
        """
        Forward pass for training
        Args:
            input_images: [B, C, H, W] tensor of images
        Returns:
            None: No reconstruction during training to save computation
            diff: Tuple of loss values
            info: Additional information for logging
        """
        # Compute flow loss
        flow_loss = self.flow_model(input_images)
        flow_loss = flow_loss * self.flow_loss_weight
        
        # Return flow loss as diff for compatibility with training script
        diff = (flow_loss, torch.tensor(0.0, device=input_images.device), torch.tensor(0.0, device=input_images.device), torch.tensor(1.0, device=input_images.device))
        
        # Return flow loss for logging
        info = flow_loss
        
        return None, diff, info


def get_local_tokenizer(tokenizer_type, **kwargs):
    """Factory function for creating tokenizers"""
    if tokenizer_type == "flow":
        return FlowTokenizer(FlowTokenizerArgs(**kwargs))
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


# Register available tokenizers
LocalTokenizers = {
    'flow': get_local_tokenizer,
}

if __name__ == "__main__":
    tokenizer = get_local_tokenizer("flow")
    images = torch.randn(1, 3, 256, 256)
    latents = tokenizer.encode(images)
    reconstructed = tokenizer.decode(latents, 256, 256)
    print(reconstructed.shape)
    print((reconstructed - images).abs().mean())
    print(images.min(), images.max())
    print(reconstructed.min(), reconstructed.max())
