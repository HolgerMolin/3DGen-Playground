from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0


@dataclass(frozen=True)
class GaussianVerseUNetConfig:
    sample_size: int
    in_channels: int
    out_channels: int
    num_classes: int
    class_embedding_dim: int
    down_block_types: tuple[str, ...]
    mid_block_type: str
    up_block_types: tuple[str, ...]
    block_out_channels: tuple[int, ...]
    layers_per_block: int
    attention_head_dim: int | tuple[int, ...]
    norm_num_groups: int
    dropout: float
    spatial_fold_factor: int = 1
    gradient_checkpointing: bool = False


GAUSSIANVERSE_UNET_PRESETS: dict[str, dict[str, object]] = {
    "UNet-S": {
        "block_out_channels": (128, 256, 256, 512),
        "attention_head_dim": 8,
        "layers_per_block": 2,
    },
    "UNet-B": {
        "block_out_channels": (192, 384, 640, 640),
        "attention_head_dim": 8,
        "layers_per_block": 2,
    },
    "UNet-L": {
        "block_out_channels": (320, 640, 1280, 1280),
        "attention_head_dim": 8,
        "layers_per_block": 2,
    },
}


class GaussianVerseUNet(nn.Module):
    """Diffusers-backed UNet wrapper with GaussianVerse class conditioning.

    The wrapper keeps the same diffusion-facing contract as the DiT trainer:
    `forward(x, t, y) -> prediction`.

    Following the reference UNet construction, class labels are first embedded
    into dense vectors, then used in two places:
    - as `encoder_hidden_states` for cross-attention
    - as `class_labels` for diffusers' projection-based class embedding path
    """

    def __init__(self, config: GaussianVerseUNetConfig):
        super().__init__()
        if config.spatial_fold_factor < 1:
            raise ValueError(f"spatial_fold_factor must be >= 1, got {config.spatial_fold_factor}")
        if config.sample_size % config.spatial_fold_factor != 0:
            raise ValueError(
                f"sample_size={config.sample_size} must be divisible by "
                f"spatial_fold_factor={config.spatial_fold_factor}"
            )

        self.config = config
        self.sample_size = config.sample_size
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.spatial_fold_factor = config.spatial_fold_factor
        self.folded_sample_size = config.sample_size // config.spatial_fold_factor
        self.folded_in_channels = config.in_channels * (config.spatial_fold_factor ** 2)
        self.folded_out_channels = config.out_channels * (config.spatial_fold_factor ** 2)
        self.class_embedding = nn.Embedding(config.num_classes, config.class_embedding_dim)
        self.unet = UNet2DConditionModel(
            sample_size=self.folded_sample_size,
            in_channels=self.folded_in_channels,
            out_channels=self.folded_out_channels,
            down_block_types=config.down_block_types,
            mid_block_type=config.mid_block_type,
            up_block_types=config.up_block_types,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            cross_attention_dim=config.class_embedding_dim,
            attention_head_dim=config.attention_head_dim,
            norm_num_groups=config.norm_num_groups,
            dropout=config.dropout,
            class_embed_type="projection",
            projection_class_embeddings_input_dim=config.class_embedding_dim,
        )
        self.unet.set_attn_processor(AttnProcessor2_0())
        if config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    def fold_spatial(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_fold_factor == 1:
            return x
        return F.pixel_unshuffle(x, self.spatial_fold_factor)

    def unfold_spatial(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_fold_factor == 1:
            return x
        return F.pixel_shuffle(x, self.spatial_fold_factor)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if y is None:
            raise ValueError("Class labels `y` must be provided for GaussianVerseUNet.")
        if x.ndim != 4:
            raise ValueError(f"Expected `x` to have shape (N, C, H, W), got {tuple(x.shape)}")
        if self.spatial_fold_factor > 1 and (
            x.shape[-2] % self.spatial_fold_factor != 0 or x.shape[-1] % self.spatial_fold_factor != 0
        ):
            raise ValueError(
                f"Input spatial shape {tuple(x.shape[-2:])} must be divisible by "
                f"spatial_fold_factor={self.spatial_fold_factor}"
            )

        x = self.fold_spatial(x)
        label_embeddings = self.class_embedding(y.long())
        label_embeddings = label_embeddings.to(dtype=x.dtype)
        encoder_hidden_states = label_embeddings.unsqueeze(1)
        model_out = self.unet(
            x,
            t,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=label_embeddings,
            return_dict=False,
        )[0]
        return self.unfold_spatial(model_out)


def build_gaussianverse_unet(
    model_name: str,
    *,
    sample_size: int,
    in_channels: int,
    num_classes: int,
    out_channels: Optional[int] = None,
    class_embedding_dim: int = 768,
    norm_num_groups: int = 32,
    dropout: float = 0.0,
    spatial_fold_factor: int = 1,
    gradient_checkpointing: bool = False,
) -> GaussianVerseUNet:
    if model_name not in GAUSSIANVERSE_UNET_PRESETS:
        available = ", ".join(sorted(GAUSSIANVERSE_UNET_PRESETS))
        raise ValueError(f"Unknown model preset {model_name!r}. Available presets: {available}")

    preset = GAUSSIANVERSE_UNET_PRESETS[model_name]
    config = GaussianVerseUNetConfig(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=in_channels if out_channels is None else out_channels,
        num_classes=num_classes,
        class_embedding_dim=class_embedding_dim,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        block_out_channels=tuple(preset["block_out_channels"]),  # type: ignore[arg-type]
        layers_per_block=int(preset["layers_per_block"]),
        attention_head_dim=preset["attention_head_dim"],  # type: ignore[arg-type]
        norm_num_groups=norm_num_groups,
        dropout=dropout,
        spatial_fold_factor=spatial_fold_factor,
        gradient_checkpointing=gradient_checkpointing,
    )
    return GaussianVerseUNet(config)
