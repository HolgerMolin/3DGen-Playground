"""DINOv2-based perceptual loss for the gsplat render path.

Drop-in replacement for the `lpips.LPIPS` callable wired into
`_compute_render_loss_for_batch`: takes two `[-1, 1]`-normalized RGB tensors
of shape `(N, 3, H, W)` and returns a per-image distance shaped
`(N, 1, 1, 1)` so the existing `.flatten().view(B, num_cam)` reshape works
unchanged.

Distance is the mean cosine distance between L2-normalized DINOv2 patch
tokens. The backbone is frozen and runs in fp32 to avoid NaNs when the
trainer is in mixed-precision (fp16) autocast.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DINOV2_HUB_REPO = "facebookresearch/dinov2"


class DinoV2Perceptual(nn.Module):
    def __init__(self, model_name: str = "dinov2_vitb14", input_size: int = 224):
        super().__init__()
        if input_size % 14 != 0:
            raise ValueError(
                f"input_size must be a multiple of 14 for DINOv2 patch=14, got {input_size}"
            )
        self.backbone = torch.hub.load(_DINOV2_HUB_REPO, model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.input_size = int(input_size)

        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) * 0.5
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return (x - self.imagenet_mean) / self.imagenet_std

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        return feats["x_norm_patchtokens"]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # bf16 autocast: overrides the trainer's outer fp16 autocast so DINOv2
        # runs in bf16 (same memory footprint as fp16, fp32-equivalent dynamic
        # range — avoids fp16 NaNs in ViT pre-norm residuals). Requires Ampere
        # or newer; will fall back to slow emulation on V100/T4.
        with torch.amp.autocast(device_type=pred.device.type, dtype=torch.bfloat16):
            pred_in = self._preprocess(pred)
            tgt_in = self._preprocess(target)
            pred_tok = self._patch_tokens(pred_in)
            with torch.no_grad():
                tgt_tok = self._patch_tokens(tgt_in)
            cos = F.cosine_similarity(pred_tok, tgt_tok, dim=-1)
            dist = (1.0 - cos).mean(dim=-1)
        return dist.view(-1, 1, 1, 1)


__all__ = ["DinoV2Perceptual"]
