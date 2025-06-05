import torch
import torch.nn as nn
from typing import Optional

from timm.models.vision_transformer import Attention, Mlp, LayerScale, DropPath


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            post_norm: bool = False,
            post_norm_before_res: bool = False,
            attn_norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=attn_norm_layer,
        )
        assert self.attn.fused_attn, "Fused attention is not enabled"
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.post_norm = post_norm
        self.post_norm_before_res = post_norm_before_res
        assert not (post_norm and post_norm_before_res), "post_norm and post_norm_before_res cannot be both True"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.post_norm:
            x = self.norm1(x + self.drop_path1(self.ls1(self.attn(x))))
            x = self.norm2(x + self.drop_path2(self.ls2(self.mlp(x))))
        elif self.post_norm_before_res:
            x = x + self.drop_path1(self.ls1(self.norm1(self.attn(x))))
            x = x + self.drop_path2(self.ls2(self.norm2(self.mlp(x))))
        else:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
