"""
Training script for JiT-style large-patch diffusion on 3DGS data (text-conditional).
3DGS data (16384 points x 59 features) on 128x128 grid is the latent space directly — no VAE needed.

Text conditioning uses the frozen CLIP-L/14 EOS-pooled vector precomputed by
object_classification/encode_text_embeddings.py (pooled.npy in the encoder's
output directory, or the legacy text_embeddings.npz). The pooled vector is
projected by an MLP and summed into the AdaLN signal — no cross-attention,
no embedding table. A sibling null_text_token.npz holds the empty-string
encoding used as the unconditional branch for CFG dropout.

Single-GPU:  python jit/train_gsplat.py --obj_list ... --gs_path ... --text_embed_path ...
Multi-GPU:   accelerate launch [--num_processes N] jit/train_gsplat.py ...
Optional:    --config jit/configs/jit_train_gsplat.yaml  (CLI overrides YAML)
Optional:    --overrides_yaml path/to/overrides.yaml  (hot-reload lr_scale, max_grad_norm, render weights, P_mean, …)
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import tarfile
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

# Add repo root to path for imports
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Make gaussian-splatting submodule importable.
GS_ROOT = os.path.join(REPO_ROOT, "submodules", "gaussian-splatting")
if GS_ROOT not in sys.path:
    sys.path.insert(0, GS_ROOT)

from dataloaders.standard_3dgen_loader import Standard3DGenDataset, load_null_text_token
from dataloaders.text_3dgen_loader import (
    Text3DGenDataset, DC_ONLY_FEATURE_INDICES, FULL_3DGS_FEATURE_DIM,
)
from jit.models import JiT_3DGS_models
from jit.diffusion import create_diffusion
from jit.sampling import SAMPLER_CHOICES, resolve_sampling_shape, sample_model
from utils.plane_utils import load_sphere2plane, plane_to_point_cloud
from utils.loss_tracker import LossTracker

try:
    import wandb
except ImportError:
    wandb = None
from utils.gsplat_render_util import (
    _compute_render_loss_for_batch,
    _denormalize_point_cloud,
    _load_reference_cameras,
    _plane_to_point_cloud_batch,
    _point_clouds_to_gsplat_inputs,
    _prepare_train_cameras,
    _render_gsplat_batch,
    _save_training_render_preview,
    _try_import_lpips,
    _try_import_renderer,
    load_rank_transform_payload_torch,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


#################################################################################
#                          LR schedule (warmup / cosine)                        #
#################################################################################

def _compute_lr(
    *,
    schedule: str,
    opt_step: int,
    base_lr: float,
    lr_min: float,
    lr_warmup_steps: int,
    max_opt_steps: int,
) -> float:
    """Learning rate: none (constant), warmup (linear ramp then hold), cosine (warmup + cosine decay)."""
    if schedule == "none":
        return float(base_lr)
    warmup = max(0, int(lr_warmup_steps))
    if schedule == "warmup":
        if warmup <= 0:
            return float(base_lr)
        if opt_step < warmup:
            return base_lr * float(opt_step + 1) / float(warmup)
        return float(base_lr)
    if schedule == "cosine":
        if max_opt_steps < 1:
            max_opt_steps = 1
        if warmup > 0 and opt_step < warmup:
            return base_lr * float(opt_step + 1) / float(warmup)
        if opt_step >= max_opt_steps:
            return float(lr_min)
        cos_steps = max_opt_steps - warmup
        if cos_steps <= 0:
            return float(base_lr)
        cos_pos = opt_step - warmup
        if cos_steps == 1:
            return float(lr_min)
        progress = float(cos_pos) / float(cos_steps - 1)
        return float(
            lr_min + (base_lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
        )
    raise ValueError(f"Unknown lr_schedule: {schedule!r}")


def _default_null_path(text_embed_path: str) -> str:
    """Derive a default null-token path next to the first text-embed shard.

    `text_embed_path` may be a comma-separated list of shards; we look in the
    directory of the first shard. The encoder script writes
    ``null_text_token.npz`` there by default, idempotently across shards.
    """
    first = text_embed_path.split(",")[0].strip()
    return str(Path(first).parent / "null_text_token.npz")


# Default 64-prompt validation pool. Mix of simple class words and compositional
# prompts (multi-attribute, spatial relations) so the cond_signal probe can
# distinguish AdaLN-level conditioning gains from cross-attn-level gains that
# require attending to specific text positions.
_DEFAULT_VAL_PROMPTS: list[str] = [
    # 16 simple objects — comparable with the legacy probe
    "a wooden chair", "a red car", "a small house", "a tree",
    "a dog", "a teapot", "a sword", "a rocket",
    "a guitar", "a cat", "a hammer", "a robot",
    "a hat", "a backpack", "a sailboat", "a vase",
    # 16 single-attribute (color OR material OR size)
    "a blue chair", "a metallic teapot", "a tiny rocket", "a wooden sword",
    "a golden vase", "a striped backpack", "a stone tower", "a glass bottle",
    "a giant mushroom", "a black cat", "a polished helmet", "a tall lamp",
    "a furry rabbit", "a marble statue", "a wooden barrel", "a ceramic mug",
    # 16 multi-attribute (color + material, etc.)
    "a red wooden chair", "a small blue car", "a tall green tree",
    "a tiny golden teapot", "a polished metal sword", "a fluffy white cat",
    "a black leather backpack", "a striped ceramic vase",
    "a tall stone tower", "a glossy red apple", "a rusted iron axe",
    "a glowing crystal orb", "a smooth jade frog", "a checkered wool blanket",
    "a transparent glass bowl", "a wooden barrel filled with apples",
    # 16 compositional / spatial / multi-object
    "a red cube next to a blue sphere",
    "a small chair on top of a large table",
    "a green tree behind a wooden house",
    "a cat sitting next to a dog",
    "a teapot pouring into a cup",
    "a sword resting on a stone pedestal",
    "a robot holding a flower",
    "a bird perched on a branch",
    "a candle burning on a wooden table",
    "a stack of books on a shelf",
    "a wooden boat in a calm pond",
    "a guitar leaning against a chair",
    "a hat hanging on a hook",
    "a flower growing out of a cracked pot",
    "a small house with a red roof and white walls",
    "a knight wearing armor and holding a shield",
]


def _encode_clip_penultimate(
    prompts: list[str],
    device: torch.device,
) -> torch.Tensor:
    """CLIP-L/14 EOS-pool encoder mirroring object_classification/encode_text_embeddings.py.

    Returns ``pooled`` (N, 768) fp32 on ``device`` — the EOS-position token of
    the penultimate hidden state after CLIP's final_layer_norm. Geometry MUST
    match the offline encoder so the val pool lives in the same CLIP space as
    training-time samples.
    """
    from transformers import CLIPTextModel, CLIPTokenizer
    model_id = "openai/clip-vit-large-patch14"
    tok = CLIPTokenizer.from_pretrained(model_id)
    enc = CLIPTextModel.from_pretrained(model_id).to(device).eval()
    inputs = tok(
        prompts, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = enc(**inputs, output_hidden_states=True)
        # Penultimate + CLIP's final LayerNorm — "clip-skip 1" trick used by the
        # offline encoder. final_layer_norm lives directly on CLIPTextModel in
        # this transformers version (no .text_model wrapper).
        penultimate = out.hidden_states[-2]
        tokens = enc.final_layer_norm(penultimate)
    eos_positions = inputs["input_ids"].to(torch.int).argmax(dim=-1)
    batch_idx = torch.arange(inputs["input_ids"].size(0), device=device)
    pooled = tokens[batch_idx, eos_positions].float().detach().clone()
    del enc
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pooled


def _load_val_prompts(
    prompts_path: Optional[str],
    n_tiles: int,
    device: torch.device,
) -> torch.Tensor:
    """Load N validation prompts and encode them with the same CLIP-L/14
    pipeline used offline (penultimate + final_layer_norm + EOS-pool). Returns
    a ``(N, text_dim)`` tensor on device.

    ``prompts_path``: JSON list of strings, OR None to use the built-in
    64-prompt default pool covering simple, multi-attribute, and compositional
    cases.
    """
    if prompts_path is None:
        prompts = list(_DEFAULT_VAL_PROMPTS)
    else:
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError(f"{prompts_path} must contain a JSON list of strings")
    if len(prompts) < n_tiles:
        # Pad by repeating; better than failing on a tiny prompt file.
        prompts = (prompts * ((n_tiles + len(prompts) - 1) // len(prompts)))[:n_tiles]
    prompts = prompts[:n_tiles]
    logger.info(f"Encoding {len(prompts)} validation prompts (CLIP-L/14, penultimate+LN) …")
    return _encode_clip_penultimate(prompts, device)


#################################################################################
#                    Hot-reload training overrides (YAML)                      #
#################################################################################

_OVERRIDABLE_KEYS = frozenset({
    "lr_scale",
    "max_grad_norm",
    "render_loss_weight",
    "alpha_mask_loss_weight",
    "lpips_loss_weight",
    "P_mean",
    "grad_norm_log_every_n_prints",
    "chamfer_loss_weight",
    "recon_loss_weight",
    "mse_hybrid_weight",
    "chamfer_rev_weight",
    "sinkhorn_epsilon",
})


@dataclass
class TrainRuntimeOverrides:
    """Mutable hyperparameters; optionally synced from overrides.yaml on a fixed step interval."""

    lr_scale: float
    max_grad_norm: float
    render_loss_weight: float
    alpha_mask_loss_weight: float
    lpips_loss_weight: float
    P_mean: float
    grad_norm_log_every_n_prints: float  # float so overrides YAML can write it; cast to int on use
    chamfer_loss_weight: float  # scales the Chamfer reconstruction term (no-op when recon_loss=mse)
    recon_loss_weight: float  # outer scalar on the recon return in total_loss (1.0 = on, 0.0 = render-only)
    mse_hybrid_weight: float  # lambda on the index-MSE term added to Chamfer (0 = off)
    chamfer_rev_weight: float  # multiplier on the backward (GT-as-query) Chamfer term (1 = symmetric)
    sinkhorn_epsilon: float  # entropic-OT regularizer for recon_loss=sinkhorn_patch (hard<->soft)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainRuntimeOverrides":
        return cls(
            lr_scale=1.0,
            max_grad_norm=float(args.max_grad_norm),
            render_loss_weight=float(args.render_loss_weight),
            alpha_mask_loss_weight=float(args.alpha_mask_loss_weight),
            lpips_loss_weight=float(args.lpips_loss_weight),
            P_mean=float(args.P_mean),
            grad_norm_log_every_n_prints=float(args.grad_norm_log_every_n_prints),
            chamfer_loss_weight=float(args.chamfer_loss_weight),
            recon_loss_weight=float(args.recon_loss_weight),
            mse_hybrid_weight=float(args.mse_hybrid_weight),
            chamfer_rev_weight=float(args.chamfer_rev_weight),
            sinkhorn_epsilon=float(args.sinkhorn_epsilon),
        )


def _any_render_loss_weight(o: TrainRuntimeOverrides) -> bool:
    return (
        o.render_loss_weight > 0.0
        or o.alpha_mask_loss_weight > 0.0
        or o.lpips_loss_weight > 0.0
    )


def _parse_p_mean_schedule(raw: Any) -> Optional[list[tuple[int, float]]]:
    """Normalize a P_mean curriculum into a sorted list of (step, value) control points.

    Accepts None/empty (returns None), a JSON string (from CLI), or a Python
    list-of-pairs (from YAML merge). Linear interpolation between control
    points; held constant outside the endpoints.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--P_mean_schedule JSON parse error: {exc}") from exc
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        raise ValueError(
            f"--P_mean_schedule must be a non-empty list of [step, value] pairs, got {raw!r}"
        )
    pts: list[tuple[int, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"--P_mean_schedule entry must be [step, value], got {item!r}"
            )
        s, v = item
        pts.append((int(s), float(v)))
    pts.sort(key=lambda p: p[0])
    for (s_prev, _), (s_cur, _) in zip(pts, pts[1:]):
        if s_cur == s_prev:
            raise ValueError(f"--P_mean_schedule has duplicate step {s_cur}")
    if pts[0][0] < 0:
        raise ValueError(f"--P_mean_schedule first step must be >= 0, got {pts[0][0]}")
    return pts


def _p_mean_at_step(schedule: list[tuple[int, float]], step: int) -> float:
    if step <= schedule[0][0]:
        return schedule[0][1]
    if step >= schedule[-1][0]:
        return schedule[-1][1]
    for (s0, v0), (s1, v1) in zip(schedule, schedule[1:]):
        if s0 <= step <= s1:
            frac = (step - s0) / (s1 - s0)
            return v0 + frac * (v1 - v0)
    return schedule[-1][1]


def _parse_lr_scale_schedule(raw: Any) -> Optional[list[tuple[int, float]]]:
    """Normalize an lr_scale ramp into sorted [(step, value)] control points.

    Same shape as P_mean_schedule but values must be > 0 (lr_scale is multiplicative).
    Used to cushion the optimizer when a new loss component engages: configure e.g.
    ``[[22000, 0.2], [24000, 1.0]]`` to linearly ramp lr_scale 0.2 → 1.0 over the 2k
    steps following render-loss engagement. Held constant outside the endpoints.
    When active, overrides any ``lr_scale`` set via ``--overrides_yaml``.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--lr_scale_schedule JSON parse error: {exc}") from exc
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        raise ValueError(
            f"--lr_scale_schedule must be a non-empty list of [step, value] pairs, got {raw!r}"
        )
    pts: list[tuple[int, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"--lr_scale_schedule entry must be [step, value], got {item!r}")
        s, v = item
        if float(v) <= 0:
            raise ValueError(f"--lr_scale_schedule values must be > 0, got {item!r}")
        pts.append((int(s), float(v)))
    pts.sort(key=lambda p: p[0])
    for (s_prev, _), (s_cur, _) in zip(pts, pts[1:]):
        if s_cur == s_prev:
            raise ValueError(f"--lr_scale_schedule has duplicate step {s_cur}")
    if pts[0][0] < 0:
        raise ValueError(f"--lr_scale_schedule first step must be >= 0, got {pts[0][0]}")
    return pts


def _parse_render_weight_schedule(
    raw: Any,
) -> Optional[list[tuple[int, float, float, float]]]:
    """Normalize a render-weight ramp into sorted [(step, rl1, alpha, lpips)] control points.

    Each entry is a 4-tuple ``[step, render_loss_weight, alpha_mask_loss_weight,
    lpips_loss_weight]``; values are linearly interpolated between control points and
    held constant outside the endpoints. Weights must be >= 0. Passing any negative
    value or an item of the wrong arity raises ValueError. Accepts None/empty (→ None),
    a JSON string (from CLI), or a Python list-of-lists (from YAML merge).
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--render_weight_schedule JSON parse error: {exc}") from exc
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        raise ValueError(
            "--render_weight_schedule must be a non-empty list of "
            f"[step, rl1, alpha, lpips] entries, got {raw!r}"
        )
    pts: list[tuple[int, float, float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise ValueError(
                "--render_weight_schedule entry must be [step, rl1, alpha, lpips], "
                f"got {item!r}"
            )
        s, rl1, alpha, lpips = item
        rl1_f, alpha_f, lpips_f = float(rl1), float(alpha), float(lpips)
        if rl1_f < 0 or alpha_f < 0 or lpips_f < 0:
            raise ValueError(
                f"--render_weight_schedule weights must be >= 0, got {item!r}"
            )
        pts.append((int(s), rl1_f, alpha_f, lpips_f))
    pts.sort(key=lambda p: p[0])
    for (s_prev, *_), (s_cur, *_) in zip(pts, pts[1:]):
        if s_cur == s_prev:
            raise ValueError(f"--render_weight_schedule has duplicate step {s_cur}")
    if pts[0][0] < 0:
        raise ValueError(
            f"--render_weight_schedule first step must be >= 0, got {pts[0][0]}"
        )
    return pts


def _render_weights_at_step(
    schedule: list[tuple[int, float, float, float]], step: int
) -> tuple[float, float, float]:
    """Return (rl1, alpha, lpips) weights linearly interpolated at the given step."""
    if step <= schedule[0][0]:
        return schedule[0][1], schedule[0][2], schedule[0][3]
    if step >= schedule[-1][0]:
        return schedule[-1][1], schedule[-1][2], schedule[-1][3]
    for pt0, pt1 in zip(schedule, schedule[1:]):
        s0, rl1_0, a0, l0 = pt0
        s1, rl1_1, a1, l1 = pt1
        if s0 <= step <= s1:
            frac = (step - s0) / (s1 - s0)
            return (
                rl1_0 + frac * (rl1_1 - rl1_0),
                a0 + frac * (a1 - a0),
                l0 + frac * (l1 - l0),
            )
    return schedule[-1][1], schedule[-1][2], schedule[-1][3]


def _load_and_apply_overrides_yaml(
    path: Optional[str],
    state: TrainRuntimeOverrides,
    *,
    is_main: bool,
) -> None:
    if not path:
        return
    p = Path(path).expanduser()
    if not p.is_file():
        if is_main:
            logger.warning("[overrides] file not found (skipping): %s", p)
        return
    try:
        with open(p) as f:
            raw = yaml.safe_load(f)
    except Exception as exc:
        if is_main:
            logger.warning("[overrides] failed to read %s: %s", p, exc)
        return
    if raw is None:
        return
    if not isinstance(raw, dict):
        if is_main:
            logger.warning("[overrides] top level must be a mapping, got %s", type(raw).__name__)
        return
    updates: list[str] = []
    for k, v in raw.items():
        if k not in _OVERRIDABLE_KEYS:
            if is_main:
                logger.warning("[overrides] ignoring unknown key: %s", k)
            continue
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            if is_main:
                logger.warning("[overrides] %s=%r is not numeric, skipping", k, v)
            continue
        if k == "lr_scale" and fv <= 0.0:
            if is_main:
                logger.warning("[overrides] lr_scale must be > 0, got %s — skipping", fv)
            continue
        if k == "max_grad_norm" and fv < 0.0:
            if is_main:
                logger.warning("[overrides] max_grad_norm must be >= 0, got %s — skipping", fv)
            continue
        setattr(state, k, fv)
        updates.append(f"{k}={fv}")
    if updates and is_main:
        logger.info("[overrides] applied from %s: %s", p, " ".join(updates))


#################################################################################
#                          Rendering Loss Helpers                               #
#################################################################################

def _sample_jit_timesteps(
    batch_size: int,
    num_timesteps: int,
    device: torch.device,
    p_mean: float,
    p_std: float,
    dist: str = "logitnormal",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample JiT-style timesteps (``dist``: logit-normal or uniform).

    Returns ``(t_value, t_discrete)``:

      * ``t_value`` is continuous in ``(0, 1)`` and drives the flow-matching
        interpolation ``x_t = t · x_0 + (1 − t) · ε`` (``t=0`` → noise,
        ``t=1`` → clean — matches ``jit.sampling._jit_velocity_from_xstart``).
      * ``t_discrete`` is the integer timestep fed to the model's timestep
        embedding, using the same ``round(t_value · (T-1))`` mapping as the
        heun/euler samplers so training and inference share a grid.
    """
    # Keep t_value strictly inside (0, 1) to avoid the (1 - t) singularity in
    # the velocity used by heun/euler at the clean end.
    eps = 1e-4
    if dist == "uniform":
        # Equal mass on every noise level; P_mean/P_std are ignored.
        t_value = torch.rand(batch_size, device=device).clamp(min=eps, max=1.0 - eps)
    elif dist == "logitnormal":
        probs = torch.sigmoid(torch.randn(batch_size, device=device) * p_std + p_mean)
        t_value = probs.clamp(min=eps, max=1.0 - eps)
    else:
        raise ValueError(f"Unknown timestep_dist {dist!r}; choices: logitnormal, uniform")
    t_discrete = torch.clamp(
        (t_value * (num_timesteps - 1)).round().long(),
        min=0,
        max=num_timesteps - 1,
    )
    return t_value, t_discrete

def _tensor_debug_summary(tensor: torch.Tensor) -> dict[str, Any]:
    """Summarize a tensor for non-finite debugging without dumping full contents."""
    t = tensor.detach()
    finite_mask = torch.isfinite(t)
    finite_count = int(finite_mask.sum().item())
    summary: dict[str, Any] = {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
        "nonfinite": int(t.numel() - finite_count),
    }
    if finite_count > 0:
        finite_vals = t[finite_mask].float()
        summary.update({
            "min": float(finite_vals.min().item()),
            "max": float(finite_vals.max().item()),
            "mean": float(finite_vals.mean().item()),
        })
    return summary


def _debug_nonfinite_mse(
    *,
    args,
    diffusion,
    model,
    x: torch.Tensor,
    y_pooled: torch.Tensor,
    x_full: Optional[torch.Tensor],
    t: torch.Tensor,
    t_value: torch.Tensor,
    noise: torch.Tensor,
    sample_losses: torch.Tensor,
    step: int,
    epoch: int,
    hash_keys: list[str],
    is_main: bool,
) -> None:
    """Fail fast with enough context to localize the first non-finite MSE."""
    bad_positions = (~torch.isfinite(sample_losses)).nonzero(as_tuple=False).flatten().tolist()
    if not bad_positions:
        bad_positions = list(range(min(1, x.shape[0])))

    with torch.no_grad():
        x_t_debug = diffusion.flow_matching_q_sample(x, t_value, noise=noise)
        model_out_debug = model(x_t_debug, t, y_pooled)
        target_debug = x

    bad_param_summaries = []
    total_bad_param_tensors = 0
    total_bad_param_values = 0
    for name, param in model.named_parameters():
        bad_count = int((~torch.isfinite(param)).sum().item())
        if bad_count == 0:
            continue
        total_bad_param_tensors += 1
        total_bad_param_values += bad_count
        bad_param_summaries.append({
            "name": name,
            "shape": tuple(param.shape),
            "nonfinite": bad_count,
        })
        if len(bad_param_summaries) >= 16:
            break

    per_sample_debug = []
    for pos in bad_positions[:4]:
        per_sample_debug.append({
            "batch_pos": int(pos),
            "hash_key": hash_keys[pos],
            "y_pooled_rms": float(y_pooled[pos].detach().float().square().mean().sqrt().cpu().item()),
            "loss": float(sample_losses[pos].detach().float().cpu().item()),
            "x": _tensor_debug_summary(x[pos]),
            "x_t": _tensor_debug_summary(x_t_debug[pos]),
            "model_output": _tensor_debug_summary(model_out_debug[pos]),
            "target": _tensor_debug_summary(target_debug[pos]),
        })

    debug_payload = {
        "step": int(step),
        "epoch": int(epoch),
        "predict_xstart": bool(args.predict_xstart),
        "bad_positions": bad_positions,
        "hash_keys": [hash_keys[pos] for pos in bad_positions[:16]],
        "timesteps": [int(v) for v in t.detach().cpu().tolist()],
        "t_values": [float(v) for v in t_value.detach().cpu().tolist()],
        "sample_loss_isfinite": torch.isfinite(sample_losses).detach().cpu(),
        "sample_losses": sample_losses.detach().cpu(),
        "bad_param_tensors": total_bad_param_tensors,
        "bad_param_values": total_bad_param_values,
        "bad_param_summaries": bad_param_summaries,
        "per_sample_debug": per_sample_debug,
        "x_bad_samples": x[bad_positions[:4]].detach().cpu(),
        "x_t_bad_samples": x_t_debug[bad_positions[:4]].detach().cpu(),
        "x_full_bad_samples": None if x_full is None else x_full[bad_positions[:4]].detach().cpu(),
        "model_output_bad_samples": model_out_debug[bad_positions[:4]].detach().cpu(),
        "noise_bad_samples": noise[bad_positions[:4]].detach().cpu(),
    }
    debug_path = os.path.join(args.results_dir, f"nonfinite_step_{step:07d}.pt")
    if is_main:
        torch.save(debug_payload, debug_path)
        logger.error(
            "[nonfinite] step=%d epoch=%d bad_positions=%s bad_hash_keys=%s debug_dump=%s",
            step,
            epoch,
            bad_positions,
            [hash_keys[pos] for pos in bad_positions[:16]],
            debug_path,
        )
        logger.error(
            "[nonfinite] bad_param_tensors=%d bad_param_values=%d bad_params=%s",
            total_bad_param_tensors,
            total_bad_param_values,
            bad_param_summaries,
        )
        for sample_info in per_sample_debug:
            logger.error("[nonfinite] sample_debug=%s", sample_info)
    raise FloatingPointError(f"Non-finite diffusion MSE detected at step {step}; debug dump saved to {debug_path}")


@torch.no_grad()
def _measure_conditioning_signal(
    model: nn.Module,
    cond_pool: torch.Tensor,
    in_channels: int,
    diffusion_num_timesteps: int,
    device: torch.device,
    t_value: float = 0.3,
    batch_size: int = 8,
    seed: int = 0,
    num_cond_probes: int = 64,
    num_cond_pairs: int = 64,
) -> dict:
    """Probe text conditioning at a fixed t on a fixed random batch.

    ``cond_pool`` is the ``(N, text_dim)`` pooled-CLIP tensor returned by
    ``_load_val_prompts``.

    Computes, as fractions of ``‖pred(y=k)‖_RMS``:
      - ``cfg_signal_k`` = ``‖pred(x_t, y=k) − pred(x_t, null)‖_RMS``
      - ``cond_signal_(a,b)`` = ``‖pred(x_t, y=a) − pred(x_t, y=b)‖_RMS``

    ``cfg_signal < 0.01`` → conditioning collapsed, CFG is a no-op.
    ``cond_signal ≈ 0`` with non-zero ``cfg_signal`` → model distinguishes
    cond-vs-null but not between prompts.
    """
    pooled_pool = cond_pool
    model_was_training = model.training
    model.eval()
    g = torch.Generator(device=device).manual_seed(int(seed))

    shape = resolve_sampling_shape(model=model, batch_size=batch_size, in_channels=in_channels)
    x_t = torch.randn(*shape, generator=g, device=device)
    t_disc = torch.full(
        (batch_size,),
        int(round(t_value * (diffusion_num_timesteps - 1))),
        dtype=torch.long, device=device,
    )

    pool_size = int(pooled_pool.shape[0])
    K = max(1, min(num_cond_probes, pool_size))
    eps = 1e-8

    def _expand(idx: int):
        return pooled_pool[idx:idx + 1].expand(batch_size, -1).to(device)

    # Null forward: pooled vector replaced with cached null via force_drop_ids=1.
    p0 = _expand(0)
    force_drop = torch.ones(batch_size, device=device, dtype=torch.long)
    pred_null = model(x_t, t_disc, p0, force_drop_ids=force_drop).float()

    cond_preds: list[torch.Tensor] = []
    cond_norms: list[float] = []
    cfg_signals: list[float] = []
    for k in range(K):
        p = _expand(k)
        pred_c = model(x_t, t_disc, p).float()
        norm_c = float(pred_c.square().mean().sqrt().item())
        cfg_c = float((pred_c - pred_null).square().mean().sqrt().item()) / (norm_c + eps)
        cond_preds.append(pred_c)
        cond_norms.append(norm_c)
        cfg_signals.append(cfg_c)

    cond_signals: list[float] = []
    if K >= 2 and num_cond_pairs > 0:
        max_pairs = min(num_cond_pairs, K * (K - 1) // 2)
        g_cpu = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        seen: set[tuple[int, int]] = set()
        attempts = 0
        while len(cond_signals) < max_pairs and attempts < 20 * max_pairs:
            attempts += 1
            i = int(torch.randint(0, K, (1,), generator=g_cpu).item())
            j = int(torch.randint(0, K, (1,), generator=g_cpu).item())
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            pa, pb = cond_preds[i], cond_preds[j]
            denom = 0.5 * (cond_norms[i] + cond_norms[j]) + eps
            cond_signals.append(float((pa - pb).square().mean().sqrt().item()) / denom)

    cfg_mean = sum(cfg_signals) / len(cfg_signals) if cfg_signals else 0.0
    cfg_min = min(cfg_signals) if cfg_signals else 0.0
    cfg_max = max(cfg_signals) if cfg_signals else 0.0
    cond_mean = sum(cond_signals) / len(cond_signals) if cond_signals else 0.0
    cond_min = min(cond_signals) if cond_signals else 0.0
    cond_max = max(cond_signals) if cond_signals else 0.0
    pred_rms_mean = sum(cond_norms) / len(cond_norms) if cond_norms else 0.0

    if model_was_training:
        model.train()

    return {
        "cfg_signal": cfg_mean,
        "cfg_signal_min": cfg_min,
        "cfg_signal_max": cfg_max,
        "cond_signal": cond_mean,
        "cond_signal_min": cond_min,
        "cond_signal_max": cond_max,
        "pred_rms": pred_rms_mean,
        "num_cond_probes": K,
        "num_cond_pairs": len(cond_signals),
    }


def _run_validation_render(
    model: nn.Module,
    plane_to_sphere: torch.Tensor,
    norm_mean: Optional[torch.Tensor],
    norm_std: Optional[torch.Tensor],
    train_cameras: list,
    renderer_tuple: tuple,
    output_dir: str,
    epoch: int,
    step: int,
    device: torch.device,
    in_channels: int,
    cond_pool: torch.Tensor,  # (N, text_dim) pooled CLIP vectors
    dc_only: bool = False,
    predict_xstart: bool = False,
    noise_schedule: str = "linear",
    diffusion_steps: int = 1000,
    val_sampling_steps: int = 50,
    val_sampler: str = "heun",
    dpm_solver_order: int = 2,
    dpm_algorithm_type: str = "dpmsolver++",
    dpm_solver_type: str = "midpoint",
    dpm_timestep_spacing: str = "trailing",
    dpm_use_karras_sigmas: bool = False,
    ddim_eta: float = 0.0,
    cfg_scale: float = 1.0,
    P_mean: float = 0.0,
    P_std: float = 1.0,
    rank_transform_tables: Optional[dict] = None,
) -> None:
    """Generate a validation sample, render it, and save the result.

    Picks one random pooled-CLIP vector from ``cond_pool`` to condition on.
    """
    pool_idx = random.randrange(int(cond_pool.shape[0]))
    cond_embeds = cond_pool[pool_idx:pool_idx + 1].to(device)

    shape = resolve_sampling_shape(model=model, batch_size=1, in_channels=in_channels)
    sample = sample_model(
        sampler=val_sampler,
        model=model,
        shape=shape,
        cond_embeds=cond_embeds,
        num_inference_steps=val_sampling_steps,
        device=device,
        predict_xstart=predict_xstart,
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule,
        solver_order=dpm_solver_order,
        algorithm_type=dpm_algorithm_type,
        solver_type=dpm_solver_type,
        timestep_spacing=dpm_timestep_spacing,
        use_karras_sigmas=dpm_use_karras_sigmas,
        ddim_eta=ddim_eta,
        cfg_scale=cfg_scale,
        P_mean=P_mean,
        P_std=P_std,
    )

    # Build GS inputs from generated sample.
    pred_pc = _plane_to_point_cloud_batch(sample.float(), plane_to_sphere)
    pred_pc_raw = _denormalize_point_cloud(pred_pc, norm_mean, norm_std)
    pred_gaussians = _point_clouds_to_gsplat_inputs(
        pred_pc_raw.to(device),
        dc_only=dc_only,
        detach_input=True,
        rank_transform_tables=rank_transform_tables,
    )

    cam_indices = [random.randrange(int(train_cameras["viewmats"].shape[0]))]
    with torch.no_grad():
        pred_img = _render_gsplat_batch(renderer_tuple, pred_gaussians, train_cameras, cam_indices, device)[0, 0]

    # Convert to numpy HWC and save
    pred_np = pred_img.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
    img_uint8 = (pred_np * 255.0).astype(np.uint8)

    val_dir = os.path.join(output_dir, "dit_validation")
    os.makedirs(val_dir, exist_ok=True)
    out_path = os.path.join(val_dir, f"epoch_{epoch:03d}_step_{step:07d}_p{pool_idx:03d}.png")
    Image.fromarray(img_uint8).save(out_path)
    logger.info(
        "[validation] saved: %s (prompt_idx=%d, sampler=%s, steps=%d)",
        out_path,
        pool_idx,
        val_sampler,
        val_sampling_steps,
    )
    return out_path


def _run_validation_grid(
    model: nn.Module,
    plane_to_sphere: torch.Tensor,
    norm_mean: Optional[torch.Tensor],
    norm_std: Optional[torch.Tensor],
    train_cameras: dict,
    renderer_tuple: tuple,
    output_dir: str,
    epoch: int,
    step: int,
    device: torch.device,
    in_channels: int,
    cond_pool: torch.Tensor,  # (N, text_dim) pooled CLIP vectors
    grid_seed: int,
    camera_idx: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
    dc_only: bool = False,
    predict_xstart: bool = False,
    noise_schedule: str = "linear",
    diffusion_steps: int = 1000,
    val_sampling_steps: int = 50,
    val_sampler: str = "heun",
    dpm_solver_order: int = 2,
    dpm_algorithm_type: str = "dpmsolver++",
    dpm_solver_type: str = "midpoint",
    dpm_timestep_spacing: str = "trailing",
    dpm_use_karras_sigmas: bool = False,
    ddim_eta: float = 0.0,
    cfg_scale: float = 1.0,
    P_mean: float = 0.0,
    P_std: float = 1.0,
    rank_transform_tables: Optional[dict] = None,
) -> None:
    """Render a fixed grid of `grid_rows x grid_cols` samples with deterministic
    per-prompt seeds. Tile ``i`` uses ``cond_pool[i]`` and initial noise drawn
    from a CPU torch.Generator seeded with ``grid_seed + i`` — so a given
    (grid_seed, prompt_idx) pair always produces the same noise.

    Only supports the heun/euler/x0_renoise samplers (other samplers don't accept
    injected initial noise — see jit/sampling.py).
    """
    if val_sampler not in {"heun", "euler", "x0_renoise"}:
        logger.warning(
            "[validation-grid] sampler=%s not supported (only heun/euler/x0_renoise accept "
            "initial_noise); skipping grid",
            val_sampler,
        )
        return

    n_tiles = grid_rows * grid_cols
    if int(cond_pool.shape[0]) < n_tiles:
        raise ValueError(
            f"cond_pool has {int(cond_pool.shape[0])} prompts but grid expects {n_tiles}"
        )

    shape = resolve_sampling_shape(model=model, batch_size=n_tiles, in_channels=in_channels)
    # Per-tile deterministic noise: seed depends on the tile index.
    initial_noise = torch.empty(shape, dtype=torch.float32)
    cpu_gen = torch.Generator(device="cpu")
    for i in range(n_tiles):
        cpu_gen.manual_seed(int(grid_seed) + int(i))
        initial_noise[i] = torch.randn(shape[1:], generator=cpu_gen, dtype=torch.float32)
    initial_noise = initial_noise.to(device)

    cond_embeds = cond_pool[:n_tiles].to(device)

    sample = sample_model(
        sampler=val_sampler,
        model=model,
        shape=shape,
        cond_embeds=cond_embeds,
        num_inference_steps=val_sampling_steps,
        device=device,
        predict_xstart=predict_xstart,
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule,
        solver_order=dpm_solver_order,
        algorithm_type=dpm_algorithm_type,
        solver_type=dpm_solver_type,
        timestep_spacing=dpm_timestep_spacing,
        use_karras_sigmas=dpm_use_karras_sigmas,
        ddim_eta=ddim_eta,
        cfg_scale=cfg_scale,
        P_mean=P_mean,
        P_std=P_std,
        initial_noise=initial_noise,
    )

    pred_pc = _plane_to_point_cloud_batch(sample.float(), plane_to_sphere)
    pred_pc_raw = _denormalize_point_cloud(pred_pc, norm_mean, norm_std)
    pred_gaussians = _point_clouds_to_gsplat_inputs(
        pred_pc_raw.to(device),
        dc_only=dc_only,
        detach_input=True,
        rank_transform_tables=rank_transform_tables,
    )

    num_cams_avail = int(train_cameras["viewmats"].shape[0])
    cam_idx = int(camera_idx) % max(num_cams_avail, 1)
    with torch.no_grad():
        # rendered shape: (B, num_cam=1, 3, H, W)
        rendered = _render_gsplat_batch(
            renderer_tuple, pred_gaussians, train_cameras, [cam_idx], device
        )
    rendered = rendered[:, 0]  # (B, 3, H, W)

    tile_h = int(rendered.shape[-2])
    tile_w = int(rendered.shape[-1])
    grid_np = (rendered.clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

    canvas = np.zeros((grid_rows * tile_h, grid_cols * tile_w, 3), dtype=np.uint8)
    for i in range(n_tiles):
        r = i // grid_cols
        c = i % grid_cols
        canvas[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = grid_np[i]

    val_dir = os.path.join(output_dir, "dit_validation")
    os.makedirs(val_dir, exist_ok=True)
    out_path = os.path.join(
        val_dir, f"epoch_{epoch:03d}_step_{step:07d}_grid.png"
    )
    Image.fromarray(canvas).save(out_path)
    logger.info(
        "[validation-grid] saved: %s (%dx%d, prompts=0..%d, seed=%d, cam=%d, sampler=%s, steps=%d)",
        out_path,
        grid_rows,
        grid_cols,
        n_tiles - 1,
        grid_seed,
        cam_idx,
        val_sampler,
        val_sampling_steps,
    )
    return out_path


def _save_overfit_gt_renders(
    *,
    dataset,
    has_full_for_render: bool,
    plane_to_sphere: torch.Tensor,
    norm_mean: Optional[torch.Tensor],
    norm_std: Optional[torch.Tensor],
    norm_mean_full: Optional[torch.Tensor],
    norm_std_full: Optional[torch.Tensor],
    train_cameras: dict,
    renderer_tuple,
    output_dir: str,
    device: torch.device,
    dc_only: bool,
    rank_transform_tables: Optional[dict],
    num_views: int = 4,
) -> None:
    """For --overfit runs: render each held sample from `num_views` cameras
    spread across the reference set and save a side-by-side strip per sample,
    so the user has a GT baseline to compare validation renders against."""
    out_dir = os.path.join(output_dir, "overfit_gt")
    os.makedirs(out_dir, exist_ok=True)

    n_cams = int(train_cameras["viewmats"].shape[0])
    if n_cams <= 0:
        return
    k = max(1, min(int(num_views), n_cams))
    cam_indices = (
        [int(round(i * (n_cams - 1) / max(1, k - 1))) for i in range(k)]
        if k > 1 else [0]
    )

    for i in range(len(dataset)):
        item = dataset[i]
        if has_full_for_render:
            pc, _pooled, pc_full, hash_key = item
        else:
            pc, _pooled, hash_key = item
            pc_full = None

        if pc_full is not None and norm_mean_full is not None and norm_std_full is not None:
            x = pc_full.float().unsqueeze(0).to(device)
            pc_mean, pc_std, render_dc_only = norm_mean_full, norm_std_full, False
        else:
            x = pc.float().unsqueeze(0).to(device)
            pc_mean, pc_std, render_dc_only = norm_mean, norm_std, dc_only

        gt_pc = _plane_to_point_cloud_batch(x, plane_to_sphere)
        gt_pc_raw = _denormalize_point_cloud(gt_pc, pc_mean, pc_std)
        gt_gaussians = _point_clouds_to_gsplat_inputs(
            gt_pc_raw.to(device),
            dc_only=render_dc_only,
            detach_input=True,
            rank_transform_tables=rank_transform_tables,
        )
        with torch.no_grad():
            imgs = _render_gsplat_batch(
                renderer_tuple, gt_gaussians, train_cameras, cam_indices, device,
            )
        tiles = imgs[0].permute(0, 2, 3, 1).clamp(0.0, 1.0).cpu().numpy()
        strip = np.concatenate(list(tiles), axis=1)
        strip_u8 = (strip * 255.0).astype(np.uint8)

        safe_hash = str(hash_key).replace('/', '_')
        cam_tag = '-'.join(str(c) for c in cam_indices)
        out_path = os.path.join(
            out_dir,
            f"gt_idx{i:02d}_{safe_hash}_cams{cam_tag}.png",
        )
        Image.fromarray(strip_u8).save(out_path)
        logger.info(
            "[overfit-gt] saved: %s (sample %d/%d, cams=%s, dc_only=%s)",
            out_path, i + 1, len(dataset), cam_indices, render_dc_only,
        )


#################################################################################
#                           Weights & Biases helpers                            #
#################################################################################

def _wandb_run_id_from_results_dir(results_dir: str) -> str:
    """Deterministic 32-char wandb id so --resume continues the same run.

    Why: a fresh `wandb.init` makes a new run on every resume, fragmenting curves.
    Hashing results_dir gives a stable id without leaking absolute paths.
    """
    import hashlib
    h = hashlib.sha1(os.path.abspath(results_dir).encode()).hexdigest()
    return h[:32]


def _init_wandb(args) -> Optional[Any]:
    """Initialize wandb on rank-0. Returns the run handle, or None if disabled/failed."""
    if not getattr(args, "wandb", True):
        return None
    if getattr(args, "wandb_mode", "online") == "disabled":
        return None
    if wandb is None:
        logger.warning("[wandb] package not installed; skipping (pip install wandb)")
        return None

    run_name = args.wandb_run_name or os.path.basename(os.path.normpath(args.results_dir))
    tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()]
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            id=_wandb_run_id_from_results_dir(args.results_dir),
            resume="allow",
            mode=args.wandb_mode,
            tags=tags or None,
            group=args.wandb_group,
            config=vars(args),
            dir=args.results_dir,
        )
        logger.info("[wandb] run=%s id=%s mode=%s project=%s",
                    run.name, run.id, args.wandb_mode, args.wandb_project)
        return run
    except Exception as exc:
        logger.warning("[wandb] init failed: %s — continuing without wandb", exc)
        return None


def _wandb_log(run, payload: dict, step: int) -> None:
    """Safe wandb.log; never raises into the training loop."""
    if run is None:
        return
    try:
        run.log(payload, step=step)
    except Exception as exc:
        logger.warning("[wandb] log failed at step %d: %s", step, exc)


#################################################################################
#                             EMA Utilities                                     #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Update EMA model parameters. `model` should be the unwrapped model."""
    ema_ps = list(ema_model.parameters())
    model_ps = [p.data for p in model.parameters()]
    torch._foreach_mul_(ema_ps, decay)
    torch._foreach_add_(ema_ps, model_ps, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def _measure_per_loss_grad_norms(
    model: nn.Module,
    weighted_losses: dict,
) -> dict:
    """Compute per-loss gradient norms for logging without affecting training.

    Iterates over each (loss, weight) pair, does a retain_graph backward,
    measures the parameter gradient norm, then restores the original gradients
    so the caller's main backward() can still proceed normally.

    Args:
        model: The unwrapped model (parameters to measure).
        weighted_losses: Dict of name -> (loss_tensor, scalar_weight).

    Returns:
        Dict of name -> float grad norm (0.0 if loss not applicable).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    # Save existing grad buffers (may contain accumulated grads from earlier micro-batches)
    saved = [p.grad.clone() if p.grad is not None else None for p in params]

    norms = {}
    try:
        for name, (loss_val, weight) in weighted_losses.items():
            if weight == 0.0 or not torch.isfinite(loss_val):
                norms[name] = 0.0
                continue
            weighted = weight * loss_val
            if not weighted.requires_grad:
                norms[name] = 0.0
                continue
            # Zero grads so we measure only this loss's contribution
            for p in params:
                p.grad = None
            # retain_graph=True so subsequent backwards (including the main one) still work
            weighted.backward(retain_graph=True)
            sq = sum(
                p.grad.detach().float().norm().item() ** 2
                for p in params if p.grad is not None
            )
            norms[name] = sq ** 0.5
    except RuntimeError:
        # Graph already freed or other issue — skip silently
        norms = {name: 0.0 for name in weighted_losses}
    finally:
        # Restore saved grads so the main backward can accumulate on top of them
        for p, g in zip(params, saved):
            p.grad = g

    return norms


#################################################################################
#                             Training Loop                                     #
#################################################################################

def main(args):
    # ── Accelerator ──────────────────────────────────────────────────────
    # gradient_as_bucket_view=True: gradients alias DDP's bucket tensors
    # directly, avoiding the layout copy that triggers the "Grad strides do
    # not match bucket view strides" warning when torch.compile produces
    # channels-last grads for 1×1 Conv2d weights. Restores comm/compute
    # overlap on those layers.
    ddp_kwargs = DistributedDataParallelKwargs(gradient_as_bucket_view=True)
    accelerator = Accelerator(
        mixed_precision="no" if args.mixed_precision == "none" else args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        logger.info(f"Accelerator: num_processes={accelerator.num_processes}, "
                     f"mixed_precision={accelerator.mixed_precision}, device={device}")
        logger.info("Validation sampler: %s", args.val_sampler)

    # Seed for reproducibility (accelerate handles per-process offset)
    set_seed(args.seed)

    # Create results directory (main process only to avoid race)
    if is_main:
        os.makedirs(args.results_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Create base dataset (Standard, with text embeddings attached)
    if is_main:
        logger.info("Creating base dataset...")
    if not args.text_embed_path:
        raise ValueError("--text_embed_path is required for text-conditioned training")
    base_dataset = Standard3DGenDataset(
        obj_list=[args.obj_list],
        gs_path=args.gs_path,
        caption_path=None,
        mean_file=args.mean_file,
        std_file=args.std_file,
        sphere2plane_path=args.sphere2plane_path,
        exclude_keys_file=args.exclude_keys_file,
        rank_transform_file=args.rank_transform_file,
        clip_thresholds_file=args.clip_thresholds_file,
        text_embed_path=args.text_embed_path,
    )
    text_dim = int(base_dataset.text_pooled.shape[1])
    if is_main:
        logger.info(f"Text-conditioning: pooled-AdaLN, text_dim={text_dim}")

    # Resolve feature indices for sh_degree0_only
    if args.sh_degree0_only:
        feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
        in_channels = len(DC_ONLY_FEATURE_INDICES)
        if is_main:
            logger.info(f"sh_degree0_only: selecting {in_channels} features from {FULL_3DGS_FEATURE_DIM}")
    else:
        feature_indices = None
        in_channels = FULL_3DGS_FEATURE_DIM

    # Load sphere2plane permutation
    point_cloud_shape = tuple(base_dataset[0]['point_cloud'].shape)
    num_points = (
        int(point_cloud_shape[-2] * point_cloud_shape[-1])
        if len(point_cloud_shape) == 3
        else int(point_cloud_shape[0])
    )
    plane_to_sphere = load_sphere2plane(args.sphere2plane_path, num_points)
    if is_main:
        logger.info(f"Loaded sphere2plane permutation: {num_points} points")

    # Render-related features
    render_loss_requested = (
        args.render_loss_weight > 0.0
        or args.alpha_mask_loss_weight > 0.0
        or args.lpips_loss_weight > 0.0
    )
    use_render_loss = render_loss_requested and args.enable_render_loss_after >= 0
    enable_train_render_log = args.train_render_log_every > 0
    if render_loss_requested and not use_render_loss and is_main:
        logger.info("[render-loss] disabled because enable_render_loss_after < 0")

    # Wrap with text-conditional dataset. PC grids are mmap-cached on first
    # access (lazy) or up front (eager); text embeddings are not cached here
    # because base_dataset.text_{pooled,tokens,mask} are single COW-shared arrays.
    dataset = Text3DGenDataset(
        base_dataset,
        feature_indices=feature_indices,
        return_full_for_render=(
            (use_render_loss or enable_train_render_log)
            and feature_indices is None
        ),
        preload_to_cpu=args.preload_to_cpu,
        lazy_cache_to_cpu=args.lazy_cache_to_cpu,
        cache_dtype=(torch.bfloat16 if args.mixed_precision == 'bf16' else torch.float32),
        preload_max_samples=args.preload_max_samples,
        preload_workers=args.preload_workers,
    )

    # Overfit modes: --overfit N takes the first N samples.
    if args.overfit > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(args.overfit, len(dataset))))
        if is_main:
            logger.info(f"[overfit] Restricting to {len(dataset)} samples (log_every={args.log_every})")

    # DataLoader — accelerate will inject DistributedSampler automatically
    overfitting = args.overfit > 0
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=not overfitting,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=not overfitting,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        if args.prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_kwargs)
    if is_main:
        logger.info(f"Dataset size: {len(dataset)}, Per-GPU batch size: {args.batch_size}")

    # Create model
    if is_main:
        logger.info(f"Creating model: {args.model}")
    model = JiT_3DGS_models[args.model](
        input_size=128,
        in_channels=in_channels,
        text_dim=text_dim,
        class_dropout_prob=args.class_dropout_prob,
        learn_sigma=False,
        gradient_checkpointing=args.gradient_checkpointing,
        bottleneck=args.bottleneck,
    )

    # Populate the pooled-AdaLN null buffer from the cached empty-string CLIP
    # encoding so the unconditional branch lives in the same geometry as
    # conditional inputs. Default path is the encoder's sibling output next to
    # text_embed_path's first shard.
    null_path = args.null_text_token_path or _default_null_path(args.text_embed_path)
    null_token_np = load_null_text_token(null_path)
    model.load_null_embeddings(torch.from_numpy(null_token_np.astype(np.float32)))
    if is_main:
        logger.info(f"[null] loaded null text token from {null_path} (shape={null_token_np.shape})")
    if is_main:
        logger.info(
            "[patch-embed] %s",
            "bottleneck (proj1→bottleneck_dim→proj2)" if args.bottleneck
            else "single-conv (in_chans→embed_dim, no rank reduction below embed_dim)",
        )
    spatial_fold_factor = int(getattr(model, "spatial_fold_factor", 1))
    if spatial_fold_factor != 1:
        raise ValueError(
            f"JiT training expects no spatial folding; expected spatial_fold_factor=1, got {spatial_fold_factor}"
        )
    if is_main:
        logger.info("JiT spatial folding: disabled (factor=%d)", spatial_fold_factor)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    if is_main:
        logger.info(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Create EMA model (lives on device, not wrapped by accelerate).
    # Keep on CPU until after accelerator.prepare() moves the main model to GPU,
    # so both don't compete for device memory during initialization.
    # EMA is copied before torch.compile so it stays an eager model (used only for
    # checkpointing / inference, not forward passes during training).
    ema = deepcopy(model)
    requires_grad(ema, False)
    ema.eval()

    # Conv2d weights → channels_last so DDP captures bucket strides matching
    # the channels-last grads torch.compile/Inductor produces for 1×1 Conv2d.
    # Must run before torch.compile (so compile sees the new layout) and
    # before accelerator.prepare (so DDP captures it). Pairs with
    # DistributedDataParallelKwargs(gradient_as_bucket_view=True) to enable
    # true grad/bucket aliasing — no per-step copy, restores comm/compute
    # overlap. EMA was deep-copied above and stays NCHW (not part of DDP;
    # update_ema copies values, not memory format).
    #
    # For 1×1 kernels (H=W=1), .contiguous(memory_format=channels_last) is a
    # no-op — PyTorch treats NCHW and NHWC as equivalent because spatial
    # indexing is degenerate, so stride metadata stays NCHW. But Inductor's
    # grad allocator uses explicit NHWC stride (C*H*W, 1, W*C, C), which is
    # what DDP needs to see in the param. Force it via as_strided — safe
    # because for H=W=1 the physical memory layout is identical.
    n_cl = 0
    for m in model.modules():
        if not isinstance(m, nn.Conv2d):
            continue
        w = m.weight.data
        N, C, H, W = w.shape
        nhwc_stride = (C * H * W, 1, W * C, C)
        if (H, W) == (1, 1):
            m.weight.data = w.as_strided(w.shape, nhwc_stride)
        else:
            m.weight.data = w.contiguous(memory_format=torch.channels_last)
        n_cl += 1
    if is_main:
        logger.info("[layout] converted %d Conv2d weights to channels_last", n_cl)

    # Optional torch.compile (PyTorch 2.0+).  Apply before accelerator.prepare so
    # the compiled forward is wrapped by DDP, not the other way around.
    if args.compile:
        if is_main:
            logger.info("torch.compile: enabled (mode=%s)", args.compile_mode)
        model = torch.compile(model, mode=args.compile_mode)

    # Create diffusion
    diffusion = create_diffusion(
        timestep_respacing="",  # use all 1000 timesteps for training
        noise_schedule=args.noise_schedule,
        learn_sigma=False,
        predict_xstart=args.predict_xstart,
    )
    if is_main:
        logger.info(f"Diffusion timesteps: {diffusion.num_timesteps}, "
                     f"predict={'x0' if args.predict_xstart else 'eps'}, "
                     f"schedule={args.noise_schedule}")
        if args.timestep_dist == "uniform":
            logger.info(
                "Training mode: DDPM (sampler=%s), timestep sampling: uniform U(0,1) "
                "mapped to discrete steps [0, %d]",
                args.val_sampler,
                diffusion.num_timesteps - 1,
            )
        else:
            logger.info(
                "Training mode: DDPM (sampler=%s), timestep sampling: sigmoid(N(%.3f, %.3f)) "
                "mapped to discrete steps [0, %d]",
                args.val_sampler,
                args.P_mean,
                args.P_std,
                diffusion.num_timesteps - 1,
            )

    # Optimizer — split params so the text-projection MLP can have its own LR/betas.
    # The projection is a small set of params at the conditioning input; a larger LR
    # often helps the conditioning signal converge faster than the trunk.
    main_betas = tuple(float(b) for b in args.betas)
    if len(main_betas) != 2:
        raise ValueError(f"--betas must be two floats; got {args.betas!r}")
    text_proj_lr = float(args.text_proj_lr) if args.text_proj_lr is not None else float(args.lr)
    text_proj_betas = (
        tuple(float(b) for b in args.text_proj_betas)
        if args.text_proj_betas is not None
        else main_betas
    )
    if len(text_proj_betas) != 2:
        raise ValueError(f"--text_proj_betas must be two floats; got {args.text_proj_betas!r}")
    # The pooled-CLIP → AdaLN projector gets its own LR group. The null vector
    # is a non-persistent buffer, not a Parameter, so .parameters() correctly
    # returns only the projection MLP.
    text_proj_params = list(model.y_embedder.parameters())
    text_proj_ids = {id(p) for p in text_proj_params}
    main_params = [p for p in model.parameters() if id(p) not in text_proj_ids]
    opt = torch.optim.AdamW(
        [
            {
                'name': 'main',
                'params': main_params,
                'lr': float(args.lr),
                'betas': main_betas,
            },
            {
                'name': 'text_proj',
                'params': text_proj_params,
                'lr': text_proj_lr,
                'betas': text_proj_betas,
            },
        ],
        weight_decay=0,
        fused=True,
    )
    if is_main:
        logger.info(
            "Optimizer: AdamW | main lr=%.3g betas=(%.3g, %.3g) | text_proj lr=%.3g betas=(%.3g, %.3g)",
            float(args.lr), main_betas[0], main_betas[1],
            text_proj_lr, text_proj_betas[0], text_proj_betas[1],
        )

    # ── Let accelerate prepare model, optimizer, dataloader ──────────────
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Move EMA to device now that the main model has claimed its GPU memory.
    ema = ema.to(device)
    # Initialize EMA from the (now device-placed) unwrapped model
    update_ema(ema, accelerator.unwrap_model(model), decay=0)

    max_opt_steps = 1
    if args.lr_schedule == 'cosine':
        if args.lr_cosine_total_steps > 0:
            max_opt_steps = max(1, int(args.lr_cosine_total_steps))
        else:
            # len(loader) is per-process iterations per epoch after accelerate.prepare
            # has sharded the batch sampler (BatchSamplerShard with even_batches=True
            # by default), so it already accounts for the real dataset size (including
            # --overfit), num_processes, and drop_last. Accelerate forces sync_gradients
            # at end-of-dataloader (sync_with_dataloader=True default) so a final
            # partial-accumulation opt step happens on the last batch of every epoch
            # → opt_steps_per_epoch = ceil(iters_per_epoch / grad_accum).
            ga = max(1, int(args.gradient_accumulation_steps))
            try:
                iters_per_epoch = len(loader)
            except TypeError as e:
                raise RuntimeError(
                    "lr_cosine_total_steps=auto requires a sized DataLoader, "
                    "but len(loader) raised TypeError. Set lr_cosine_total_steps explicitly."
                ) from e
            if iters_per_epoch <= 0:
                raise RuntimeError(
                    f"lr_cosine_total_steps=auto got iters_per_epoch={iters_per_epoch}; "
                    "dataset/sampler is empty or batch_size > num_samples."
                )
            opt_steps_per_epoch = (iters_per_epoch + ga - 1) // ga  # ceil
            max_opt_steps = max(1, int(args.epochs) * opt_steps_per_epoch)
        if is_main:
            logger.info(
                "LR schedule: cosine | warmup=%d opt steps | max_opt_steps=%d | lr_min=%g",
                args.lr_warmup_steps,
                max_opt_steps,
                args.lr_min,
            )
    elif args.lr_schedule == 'warmup' and is_main:
        logger.info(
            "LR schedule: warmup only | warmup=%d opt steps (then constant lr=%g)",
            args.lr_warmup_steps,
            args.lr,
        )

    # Load Gaussian rank-transform tables (used to invert the dataloader's
    # forward rank transform on opacity / scale channels before sigmoid/exp).
    rank_transform_tables = load_rank_transform_payload_torch(
        args.rank_transform_file, device=device
    )

    # Load normalization stats for render loss denormalization
    norm_mean = None
    norm_std = None
    norm_mean_full = None
    norm_std_full = None
    if args.mean_file and args.std_file:
        # Load directly to device so _denormalize_point_cloud's .to(device=...) is a no-op.
        norm_mean_full = torch.load(args.mean_file, weights_only=True).float().to(device)
        norm_std_full = torch.load(args.std_file, weights_only=True).float().to(device)
        if rank_transform_tables is not None:
            # Rank-transformed channels are N(0,1) by construction; force their
            # stats to (0,1) so the destandardize is a no-op for them and the
            # inverse rank transform inside _constrain owns the round-trip.
            rank_idx = torch.tensor(
                rank_transform_tables["channels"], dtype=torch.long, device=device
            )
            norm_mean_full = norm_mean_full.clone()
            norm_std_full = norm_std_full.clone()
            norm_mean_full[rank_idx] = 0.0
            norm_std_full[rank_idx] = 1.0
        if feature_indices is not None:
            norm_mean = norm_mean_full[feature_indices]
            norm_std = norm_std_full[feature_indices]
        else:
            norm_mean = norm_mean_full
            norm_std = norm_std_full

    # Per-channel MSE weighting. Audit via data/audit_norm_stats.py: after
    # global per-channel normalization, some channels (e.g. opacity/logit-scale)
    # have per-object spatial std << 1, so their MSE contribution is ~var²
    # smaller than unit-variance channels. Per-channel weights compensate.
    channel_loss_weights = None
    if args.channel_loss_weights:
        raw = json.loads(args.channel_loss_weights)
        if not isinstance(raw, list) or not all(isinstance(v, (int, float)) for v in raw):
            raise ValueError(
                "--channel_loss_weights must be a JSON list of numbers"
            )
        if len(raw) != in_channels:
            raise ValueError(
                f"--channel_loss_weights length {len(raw)} != in_channels {in_channels}"
            )
        channel_loss_weights = torch.tensor(raw, dtype=torch.float32, device=device)
        if is_main:
            mean_w = float(channel_loss_weights.mean().item())
            logger.info(
                "[channel_loss_weights] active: %d values, mean=%.4f, min=%.4f, max=%.4f",
                len(raw), mean_w,
                float(channel_loss_weights.min().item()),
                float(channel_loss_weights.max().item()),
            )

    # Reconstruction loss selection. Fail fast if a kNN-based Chamfer mode is requested
    # but pytorch3d is missing, rather than crashing mid-training on the first step.
    # chamfer_patch is pure-torch (no KeOps/pytorch3d), so it's exempt from the gate.
    if args.recon_loss not in ("mse", "chamfer_patch", "sinkhorn_patch", "sinkhorn_patch_hard"):
        try:
            import pytorch3d  # noqa: F401
            from pytorch3d.loss import chamfer_distance  # noqa: F401
            from pytorch3d.ops import knn_gather, knn_points  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"--recon_loss={args.recon_loss} requires the pytorch3d package, "
                f"which failed to import: {exc}"
            )
        if is_main:
            sub = int(args.chamfer_subsample)
            logger.info(
                "[recon_loss] %s active (chamfer_loss_weight=%.4g, chamfer_subsample=%s)",
                args.recon_loss, float(args.chamfer_loss_weight),
                f"{sub} query pts/dir" if sub > 0 else "off (all points)",
            )

    if args.recon_loss == "chamfer_patch" and is_main:
        logger.info(
            "[recon_loss] chamfer_patch active (chamfer_loss_weight=%.4g, patch_size=%s)",
            float(args.chamfer_loss_weight),
            args.chamfer_patch_size if args.chamfer_patch_size > 0 else "model",
        )

    if args.recon_loss in ("sinkhorn_patch", "sinkhorn_patch_hard") and is_main:
        logger.info(
            "[recon_loss] %s active — %s "
            "(chamfer_loss_weight=%.4g, patch_size=%s, sinkhorn_eps=%.4g, sinkhorn_iters=%d, compile=%s). "
            "chamfer_rev_weight is a no-op for this mode.",
            args.recon_loss,
            "argmax-HARD assignment (MSE to single matched target; monitor SinkColl%)"
            if args.recon_loss == "sinkhorn_patch_hard"
            else "optimal-assignment EMD (soft plan-weighted)",
            float(args.chamfer_loss_weight),
            args.chamfer_patch_size if args.chamfer_patch_size > 0 else "model",
            float(args.sinkhorn_epsilon), int(args.sinkhorn_iters),
            "on (~8x, fused logsumexp; ~1min first-step compile)" if args.compile_sinkhorn else "off",
        )

    if args.chamfer_rev_weight != 1.0 and args.recon_loss not in ("sinkhorn_patch", "sinkhorn_patch_hard"):
        if args.recon_loss not in ("chamfer_feature", "chamfer_geometric", "chamfer_patch"):
            raise ValueError(
                "--chamfer_rev_weight only applies to a Chamfer recon_loss "
                f"(got recon_loss={args.recon_loss!r}; it is a no-op for sinkhorn_patch)."
            )
        if is_main:
            logger.info(
                "[recon_loss] backward (GT-as-query) Chamfer term weighted x%.4g "
                "(coverage/recall upweight, hot-reloadable)",
                float(args.chamfer_rev_weight),
            )

    if args.mse_hybrid_weight > 0.0:
        if args.recon_loss not in ("chamfer_feature", "chamfer_geometric", "chamfer_patch",
                                   "sinkhorn_patch", "sinkhorn_patch_hard"):
            raise ValueError(
                "--mse_hybrid_weight only applies on top of a Chamfer/assignment recon_loss "
                f"(got recon_loss={args.recon_loss!r}; for plain MSE use --recon_loss=mse)."
            )
        if is_main:
            if int(args.mse_hybrid_warmup_steps) > 0:
                logger.info(
                    "[recon_loss] Chamfer+MSE hybrid as COLD-START BOOTSTRAP: index-MSE λ "
                    "decays linearly %.4g -> 0 over %d steps, then 0 (schedule governs; "
                    "hot-reload ignored while active)",
                    float(args.mse_hybrid_weight), int(args.mse_hybrid_warmup_steps),
                )
            else:
                logger.info(
                    "[recon_loss] Chamfer+MSE hybrid active (mse_hybrid_weight=%.4g, hot-reloadable)",
                    float(args.mse_hybrid_weight),
                )

    if float(args.recon_loss_weight) != 1.0 and is_main:
        rlw = float(args.recon_loss_weight)
        if rlw == 0.0:
            logger.info(
                "[recon_loss] recon_loss_weight=0.0 — RENDER-ONLY mode: recon forward "
                "(Sinkhorn/Chamfer/MSE) SKIPPED entirely. SinkResid/SinkColl/MSE_hyb "
                "diagnostics will not log. Flip recon_loss_weight nonzero via overrides.yaml "
                "to re-enable next step."
            )
        else:
            logger.info(
                "[recon_loss] recon_loss_weight=%.4g (outer scalar on recon contribution to "
                "total_loss; hot-reloadable)",
                rlw,
            )

    if args.permute_atlas != "none":
        if args.recon_loss == "mse":
            raise ValueError(
                "--permute_atlas requires a Chamfer recon_loss; index-aligned MSE is "
                "unlearnable under input permutation (the ordering signal vanishes in noise)."
            )
        if is_main:
            _model_patch = int(args.model.split("/")[-1])
            if args.permute_atlas == "patch" and args.recon_loss in (
                "chamfer_patch", "sinkhorn_patch", "sinkhorn_patch_hard"
            ):
                _pp = int(args.chamfer_patch_size) or _model_patch
                aligned = f"patch_size={_pp} (aligned to chamfer_patch_size)"
            elif args.permute_atlas == "patch":
                aligned = f"patch_size={_model_patch} (model patch)"
            else:
                aligned = "global"
            logger.info(
                "[permute_atlas] %s permutation active, %s (fresh per step, Chamfer-only)",
                args.permute_atlas, aligned,
            )

    # Render / validation setup (per-process; gsplat renderer is local)
    renderer_for_train = None
    lpips_fn_for_train = None
    train_cameras = None
    per_sample_zoom_table = None    # (N_obj, num_cam) float zoom factors; loaded below if --per_sample_zoom_file
    hash_to_zoom_row: dict[str, int] = {}
    enable_val = args.val_every > 0
    needs_renderer = (
        use_render_loss or enable_val or enable_train_render_log or args.overrides_yaml is not None
    )
    if needs_renderer and device.type != "cuda":
        if is_main:
            logger.info("[renderer] disabled: CUDA required for gsplat")
        needs_renderer = False
        enable_val = False
    if needs_renderer:
        ref_cameras = _load_reference_cameras(args.ref_camera_tar)
        if is_main:
            logger.info(f"Loaded {len(ref_cameras)} reference cameras from {args.ref_camera_tar}")
        renderer_probe = _try_import_renderer()
        if not isinstance(renderer_probe, Exception):
            renderer_for_train = renderer_probe
            train_cameras = _prepare_train_cameras(
                ref_cameras, args.train_render_size, device,
                zoom_factor=float(args.render_zoom_factor),
            )
            if is_main and float(args.render_zoom_factor) != 1.0:
                import math as _m
                _old_fov = _m.degrees(float(ref_cameras[0]["fovx"]))
                _new_fov = _m.degrees(
                    2.0 * _m.atan(_m.tan(float(ref_cameras[0]["fovx"]) / 2.0) / float(args.render_zoom_factor))
                )
                logger.info(
                    "[renderer] zoom_factor=%.2gx active; FOV %.1f° -> %.1f° (intrinsics narrowed, "
                    "camera position unchanged) — object fills more of the rendered frame so render "
                    "L1/LPIPS aren't diluted by background pixels.",
                    float(args.render_zoom_factor), _old_fov, _new_fov,
                )
            # Per-sample zoom table (data/build_per_sample_render_fov.py). When provided
            # each (sample, chosen_cam) pair gets its own (fx, fy) scale at render time —
            # a tighter, per-object framing than the global --render_zoom_factor allows.
            # Composes multiplicatively with the global zoom (per-sample × global).
            per_sample_zoom_table = None
            hash_to_zoom_row: dict[str, int] = {}
            if args.per_sample_zoom_file:
                _zoom_path = Path(args.per_sample_zoom_file)
                if not _zoom_path.is_file():
                    raise FileNotFoundError(
                        f"--per_sample_zoom_file does not exist: {args.per_sample_zoom_file}"
                    )
                _payload = torch.load(_zoom_path, map_location="cpu", weights_only=False)
                _zoom_t = _payload["zoom_factors"].float()
                _hashes = list(_payload["hash_keys"])
                if _zoom_t.shape[0] != len(_hashes):
                    raise ValueError(
                        f"per_sample_zoom_file: zoom_factors row count {_zoom_t.shape[0]} "
                        f"!= hash_keys length {len(_hashes)}"
                    )
                if _zoom_t.shape[1] != len(ref_cameras):
                    raise ValueError(
                        f"per_sample_zoom_file has {_zoom_t.shape[1]} cameras but training "
                        f"loaded {len(ref_cameras)} ref cameras — table was built for a "
                        "different ref_camera_tar; rebuild it."
                    )
                hash_to_zoom_row = {h: i for i, h in enumerate(_hashes)}
                per_sample_zoom_table = _zoom_t.to(device)
                if is_main:
                    _mean = float(_zoom_t.mean())
                    _p50, _p95, _p99 = (float(torch.quantile(_zoom_t.reshape(-1), q)) for q in (0.5, 0.95, 0.99))
                    _meta = _payload.get("meta", {})
                    logger.info(
                        "[renderer] per_sample_zoom: loaded %s — %d objects × %d cameras "
                        "(zoom mean=%.3f p50=%.3f p95=%.3f p99=%.3f, target_fill=%s, "
                        "zoom_range=[%s, %s])",
                        args.per_sample_zoom_file, _zoom_t.shape[0], _zoom_t.shape[1],
                        _mean, _p50, _p95, _p99,
                        _meta.get("target_bbox_fill", "?"),
                        _meta.get("zoom_min", "?"), _meta.get("zoom_max", "?"),
                    )
            if use_render_loss and args.lpips_loss_weight > 0.0:
                if args.perceptual_backend == 'dinov2':
                    # torch.hub.load downloads on first call; serialize ranks to keep
                    # 4 processes from racing on the same cache file.
                    try:
                        from utils.dinov2_perceptual import DinoV2Perceptual
                        if is_main:
                            lpips_fn_for_train = DinoV2Perceptual().to(device).eval()
                        accelerator.wait_for_everyone()
                        if not is_main:
                            lpips_fn_for_train = DinoV2Perceptual().to(device).eval()
                        for p in lpips_fn_for_train.parameters():
                            p.requires_grad_(False)
                        if is_main:
                            logger.info("[render-loss] perceptual backend = dinov2_vitb14 (cosine, 224x224)")
                    except Exception as exc:
                        if is_main:
                            logger.warning(
                                f"[render-loss] DINOv2 perceptual disabled: init failed: {exc}"
                            )
                        lpips_fn_for_train = None
                else:
                    lpips_probe = _try_import_lpips()
                    if isinstance(lpips_probe, Exception):
                        if is_main:
                            logger.warning(f"[render-loss] LPIPS disabled: import failed: {lpips_probe}")
                    else:
                        lpips_fn_for_train = lpips_probe.LPIPS(net=args.lpips_net).to(device).eval()
                        for p in lpips_fn_for_train.parameters():
                            p.requires_grad_(False)
        else:
            if is_main:
                logger.warning(f"[renderer] disabled: import failed: {renderer_probe}")
            needs_renderer = False
            enable_val = False

    # Overfit GT renders: dump N-camera strips of each held sample so the user
    # has a baseline to eyeball validation outputs against. Cheap; runs once.
    if (
        args.overfit > 0
        and is_main
        and renderer_for_train is not None
        and train_cameras is not None
    ):
        _underlying_for_full = (
            dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
        )
        _save_overfit_gt_renders(
            dataset=dataset,
            has_full_for_render=bool(getattr(_underlying_for_full, 'return_full_for_render', False)),
            plane_to_sphere=plane_to_sphere,
            norm_mean=norm_mean,
            norm_std=norm_std,
            norm_mean_full=norm_mean_full,
            norm_std_full=norm_std_full,
            train_cameras=train_cameras,
            renderer_tuple=renderer_for_train,
            output_dir=args.results_dir,
            device=device,
            dc_only=args.sh_degree0_only,
            rank_transform_tables=rank_transform_tables,
            num_views=4,
        )

    # Resume from checkpoint if provided.
    # `step` now means optimizer steps (post-refactor). Legacy checkpoints saved
    # `step` as micro-batch count and a separate `opt_step` as the optim count;
    # detect that format via the presence of `opt_step` and use it directly.
    start_step = 0
    start_epoch = 0
    ga = max(1, int(args.gradient_accumulation_steps))
    optim_steps_per_epoch = max(1, (len(loader) + ga - 1) // ga)
    if args.resume:
        if is_main:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        # strict=False so checkpoints from neighboring branches with a different
        # conditioning surface still load; mismatches are logged so accidental
        # architecture drift doesn't pass silently.
        missing, unexpected = accelerator.unwrap_model(model).load_state_dict(
            ckpt['model'], strict=False
        )
        if is_main and (missing or unexpected):
            logger.info(
                "[resume] load_state_dict non-strict: missing=%s unexpected=%s",
                list(missing), list(unexpected),
            )
        ema_missing, ema_unexpected = ema.load_state_dict(ckpt['ema'], strict=False)
        if is_main and (ema_missing or ema_unexpected):
            logger.info(
                "[resume] EMA load_state_dict non-strict: missing=%s unexpected=%s",
                list(ema_missing), list(ema_unexpected),
            )
        ema.eval()
        try:
            opt.load_state_dict(ckpt['opt'])
        except ValueError as e:
            if is_main:
                logger.warning(
                    "[resume] optimizer state_dict mismatch (%s) — "
                    "starting optimizer from scratch. This is expected when "
                    "resuming into a model with added/removed params.",
                    e,
                )
        del ckpt['model'], ckpt['ema'], ckpt['opt']
        torch.cuda.empty_cache()
        if 'opt_step' in ckpt:
            # Legacy: 'step' = micro-batch count, 'opt_step' = optim count.
            start_step = int(ckpt['opt_step'])
        else:
            start_step = int(ckpt['step'])
        start_epoch = start_step // optim_steps_per_epoch
        if is_main:
            logger.info(f"Resumed at optim step {start_step}, epoch {start_epoch}")

    # Per-t-bucket MSE accumulators: t in (0,1) split into custom-width bins
    # to give finer resolution near t=1 (clean end) where loss falls fastest.
    # t→0 is noise, t→1 is clean (FM convention).
    t_bucket_edges = (0.0, 0.5, 0.85, 0.95, 1.0)
    num_t_buckets = len(t_bucket_edges) - 1
    # Inner boundaries fed to torch.bucketize: t < 0.5 → 0, [0.5,0.85) → 1, ...
    t_bucket_boundaries = torch.tensor(
        t_bucket_edges[1:-1], device=device, dtype=torch.float32
    )

    # Loss tracker (rank-0 only; resume appends to existing loss_log.csv)
    loss_tracker = LossTracker(
        output_dir=args.results_dir,
        num_t_buckets=num_t_buckets,
        t_bucket_edges=list(t_bucket_edges),
        enabled=is_main,
        resume=bool(args.resume),
    )

    # wandb (rank-0 only; deterministic id so --resume continues the same run)
    wandb_run = _init_wandb(args) if is_main else None

    # Training
    model.train()
    step = start_step
    # Accumulate as GPU scalar tensors; .item() is deferred to the log block to
    # avoid forcing a GPU–CPU sync (and DDP barrier) on every training step.
    log_loss = torch.zeros([], device=device)
    log_mse_hybrid = torch.zeros([], device=device)  # raw MSE component (Chamfer+MSE hybrid)
    log_sinkhorn_resid = torch.zeros([], device=device)  # sinkhorn plan marginal residual (convergence monitor)
    log_sinkhorn_coll = torch.zeros([], device=device)   # sinkhorn_patch_hard argmax collision fraction (bijection monitor)
    log_render_kept = 0.0  # sum of render-kept sample counts (host int; folded into Step line)
    log_render_kept_steps = 0  # micro-batches that ran render (for averaging the kept count)
    log_render_l1 = torch.zeros([], device=device)
    log_render_alpha_l1 = torch.zeros([], device=device)
    log_render_lpips = torch.zeros([], device=device)
    # Accumulate on-device — `.item()` once per print window instead of per
    # optim step. Non-finite grad norms contribute 0 to the sum and 0 to the
    # count (preserving the original "average over finite steps" semantics).
    log_grad_norm = torch.zeros([], device=device)
    log_grad_steps = torch.zeros([], device=device)
    log_steps = 0  # micro-batches contributing to running loss averages
    log_optim_steps = 0  # optim steps in window (used for steps_per_sec)
    log_mse_gn = 0.0
    log_rl1_gn = 0.0
    log_alpha_gn = 0.0
    log_lpips_gn = 0.0
    log_per_loss_gn_steps = 0
    log_t_bucket_sum = torch.zeros(num_t_buckets, device=device)
    log_t_bucket_cnt = torch.zeros(num_t_buckets, device=device)
    start_time = time.time()
    last_lr_val: Optional[float] = None
    last_text_proj_lr_val: Optional[float] = None
    # Pre-allocated zero tensor used as a no-op sentinel for render losses when
    # render loss is disabled or hasn't warmed up yet.
    _zero_render_loss = torch.zeros([], dtype=torch.float32, device=device)

    if is_main:
        logger.info(f"Starting training from epoch {start_epoch}, step {start_step}...")

    # ── Step-time profiling (opt-in via --profile_step_times) ────────────
    # When on, inserts CUDA syncs around each phase to attribute time
    # accurately. This disables DDP all-reduce / next-step-fwd overlap, so
    # numbers are larger than the non-profile run — use the *breakdown*,
    # not absolute totals, to identify bottlenecks. Zero overhead when off.
    _prof = bool(args.profile_step_times) and device.type == "cuda"
    if _prof:
        _PROF_WIN = 50
        _PROF_SKIP = 20
        _PROF_PRINT_EVERY = 100
        _prof_data_q: deque[float] = deque(maxlen=_PROF_WIN)
        _prof_h2d_q: deque[float] = deque(maxlen=_PROF_WIN)
        _prof_fwd_q: deque[float] = deque(maxlen=_PROF_WIN)
        _prof_bwd_q: deque[float] = deque(maxlen=_PROF_WIN)
        _prof_opt_q: deque[float] = deque(maxlen=_PROF_WIN)
        _prof_ev_fwd_start = torch.cuda.Event(enable_timing=True)
        _prof_ev_bwd_start = torch.cuda.Event(enable_timing=True)
        _prof_ev_opt_start = torch.cuda.Event(enable_timing=True)
        _prof_ev_step_end = torch.cuda.Event(enable_timing=True)
        _prof_t_step_start = time.perf_counter()
        _prof_data_ms = 0.0
        _prof_h2d_ms = 0.0
        if is_main:
            logger.info(
                "[profile] step-time breakdown enabled "
                "(window=%d, skip first %d, print every %d steps)",
                _PROF_WIN, _PROF_SKIP, _PROF_PRINT_EVERY,
            )

    runtime = TrainRuntimeOverrides.from_args(args)

    lr_scale_schedule = _parse_lr_scale_schedule(args.lr_scale_schedule)
    if lr_scale_schedule is not None and is_main:
        pretty = ", ".join(f"({s}, x{v:.3f})" for s, v in lr_scale_schedule)
        _initial = _p_mean_at_step(lr_scale_schedule, start_step)
        logger.info(
            "[lr_scale_schedule] active with %d control points: %s | "
            "initial lr_scale at step %d = x%.4f (overrides any lr_scale in --overrides_yaml)",
            len(lr_scale_schedule), pretty, start_step, _initial,
        )

    p_mean_schedule = _parse_p_mean_schedule(args.P_mean_schedule)
    if p_mean_schedule is not None:
        runtime.P_mean = _p_mean_at_step(p_mean_schedule, start_step)
        if is_main:
            pretty = ", ".join(f"({s}, {v:+.3f})" for s, v in p_mean_schedule)
            logger.info(
                "[P_mean_schedule] active with %d control points: %s | "
                "initial P_mean at step %d = %+.4f",
                len(p_mean_schedule), pretty, start_step, runtime.P_mean,
            )

    render_weight_schedule = _parse_render_weight_schedule(args.render_weight_schedule)
    if render_weight_schedule is not None:
        rl1_0, alpha_0, lpips_0 = _render_weights_at_step(render_weight_schedule, start_step)
        runtime.render_loss_weight = rl1_0
        runtime.alpha_mask_loss_weight = alpha_0
        runtime.lpips_loss_weight = lpips_0
        if is_main:
            pretty = ", ".join(
                f"({s}, rl1={rl1:.3f}, a={a:.3f}, lp={lp:.4f})"
                for s, rl1, a, lp in render_weight_schedule
            )
            logger.info(
                "[render_weight_schedule] active with %d control points: %s | "
                "initial weights at step %d: rl1=%.4f, alpha=%.4f, lpips=%.4f",
                len(render_weight_schedule), pretty, start_step,
                runtime.render_loss_weight,
                runtime.alpha_mask_loss_weight,
                runtime.lpips_loss_weight,
            )

    dc_only = args.sh_degree0_only
    # Subset wraps the underlying dataset — look through it for the attribute
    _underlying = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    has_full_for_render = getattr(_underlying, 'return_full_for_render', False)

    # Validation / probe conditioning pool: encode a fixed prompt set once with
    # the same CLIP-L/14 pipeline used offline (penultimate + final_layer_norm
    # + EOS-pooled), so validation renders and the [cond] probe live in the
    # same text-embedding space as training samples. Pool size includes both
    # the val grid AND the cond probe (default 64 prompts).
    val_grid_n_tiles = int(args.val_grid_rows) * int(args.val_grid_cols)
    pool_size = max(val_grid_n_tiles, 64)
    cond_pool = _load_val_prompts(args.val_prompts_file, pool_size, device)
    if is_main:
        logger.info(
            "[cond-pool] %d prompts encoded, pooled_dim=%d (%s)",
            int(cond_pool.shape[0]), int(cond_pool.shape[1]),
            args.val_prompts_file or "default 64-prompt set",
        )

    for epoch in range(start_epoch, args.epochs):
        for batch in loader:
            if _prof:
                _prof_t_data_end = time.perf_counter()
                _prof_data_ms = (_prof_t_data_end - _prof_t_step_start) * 1000.0

            # Pre-step work that uses `step` as a gate. `step` is constant
            # across the `ga` micro-batches of an accumulation cycle, so any
            # `step % N == 0` gate would otherwise fire `ga` times in a row.
            # Gate on sync_gradients so each runs once per optim step. Schedules
            # and LR computation are also gated even though they only read
            # `step` (no modulo): values they push are only consumed at the
            # next opt.step(), which itself fires only on sync.
            if accelerator.sync_gradients:
                if args.overrides_yaml and step >= args.enable_render_loss_after and (
                    step % max(1, int(args.overrides_every)) == 0 or step == start_step
                ):
                    _load_and_apply_overrides_yaml(args.overrides_yaml, runtime, is_main=is_main)

                if p_mean_schedule is not None:
                    runtime.P_mean = _p_mean_at_step(p_mean_schedule, step)

                # Cold-start bootstrap: linearly decay the index-MSE hybrid term
                # from its base (args.mse_hybrid_weight, at step 0) to 0 (at
                # mse_hybrid_warmup_steps), then hold 0. Governs the term entirely
                # while active (hot-reload of mse_hybrid_weight is gated after
                # enable_render_loss_after and stays 0 here). Breaks the symmetric
                # mean-collapse that stalls sinkhorn_patch trained from scratch.
                if int(args.mse_hybrid_warmup_steps) > 0:
                    _decay = max(0.0, 1.0 - step / float(args.mse_hybrid_warmup_steps))
                    runtime.mse_hybrid_weight = float(args.mse_hybrid_weight) * _decay

                if render_weight_schedule is not None:
                    rl1_s, alpha_s, lpips_s = _render_weights_at_step(
                        render_weight_schedule, step
                    )
                    runtime.render_loss_weight = rl1_s
                    runtime.alpha_mask_loss_weight = alpha_s
                    runtime.lpips_loss_weight = lpips_s

                if args.lr_schedule in ('cosine', 'warmup', 'none'):
                    lr_val = _compute_lr(
                        schedule=args.lr_schedule,
                        opt_step=step,
                        base_lr=args.lr,
                        lr_min=args.lr_min,
                        lr_warmup_steps=args.lr_warmup_steps,
                        max_opt_steps=max_opt_steps,
                    )
                    # text_proj group follows the same warmup+cosine shape but
                    # anchored to its own peak (text_proj_lr) and proportional floor.
                    tp_lr_min = (
                        args.lr_min * (text_proj_lr / args.lr) if args.lr > 0 else args.lr_min
                    )
                    lr_val_tp = _compute_lr(
                        schedule=args.lr_schedule,
                        opt_step=step,
                        base_lr=text_proj_lr,
                        lr_min=tp_lr_min,
                        lr_warmup_steps=args.lr_warmup_steps,
                        max_opt_steps=max_opt_steps,
                    )
                    # Schedule wins over the override-hot-reload value when configured,
                    # so step-1 (and all subsequent steps) see the right lr_scale even
                    # before the override-yaml reload tick fires.
                    if lr_scale_schedule is not None:
                        runtime.lr_scale = _p_mean_at_step(lr_scale_schedule, step)
                    eff_lr = lr_val * float(runtime.lr_scale)
                    eff_lr_tp = lr_val_tp * float(runtime.lr_scale)
                    for pg in opt.param_groups:
                        if pg.get('name') == 'text_proj':
                            pg['lr'] = eff_lr_tp
                        else:
                            pg['lr'] = eff_lr
                    last_lr_val = eff_lr
                    last_text_proj_lr_val = eff_lr_tp

            if has_full_for_render:
                x, y_pooled, x_full, hash_keys = batch
            else:
                x, y_pooled, hash_keys = batch
                x_full = None
            # Pooled CLIP vectors are stored fp16 on disk → fp32 here so
            # downstream arithmetic (loss/grad) stays accurate.
            y_pooled = y_pooled.float()
            hash_keys = list(hash_keys)

            if _prof:
                # Drain pending non_blocking H2D copies issued by accelerate's
                # wrapped loader. perf_counter delta = wait time on H2D + any
                # tail of the previous step's GPU work.
                torch.cuda.synchronize()
                _prof_h2d_ms = (time.perf_counter() - _prof_t_data_end) * 1000.0

            with accelerator.accumulate(model):
                # JiT-style logit-normal timestep sampling. ``t_value`` drives
                # the flow-matching interpolation x_t = t·x_0 + (1−t)·ε and
                # ``t`` (discrete) conditions the model — same mapping as the
                # heun/euler samplers.
                t_value, t = _sample_jit_timesteps(
                    x.shape[0], diffusion.num_timesteps, device, runtime.P_mean, args.P_std,
                    dist=args.timestep_dist,
                )
                noise = torch.randn_like(x)

                # CFG drop is drawn once inside DiT.forward at the model level.
                unwrapped_model = accelerator.unwrap_model(model)
                model_kwargs = dict(y_pooled=y_pooled)
                # Resolved chamfer patch side (0 => model patch size). Permute aligns to it.
                _cps = int(args.chamfer_patch_size) or int(unwrapped_model.patch_size)

                if _prof:
                    _prof_ev_fwd_start.record()

                # Forward pass (accelerate handles autocast).
                loss_dict = diffusion.flow_matching_training_losses(
                    model,
                    x,
                    t_value,
                    t,
                    model_kwargs=model_kwargs,
                    noise=noise,
                    channel_loss_weights=channel_loss_weights,
                    recon_loss=args.recon_loss,
                    chamfer_loss_weight=float(runtime.chamfer_loss_weight),
                    chamfer_subsample=int(args.chamfer_subsample),
                    chamfer_patch_size=_cps,
                    chamfer_rev_weight=float(runtime.chamfer_rev_weight),
                    mse_hybrid_weight=float(runtime.mse_hybrid_weight),
                    mse_hybrid_lownoise_mult=float(args.mse_hybrid_lownoise_mult),
                    sinkhorn_eps=float(runtime.sinkhorn_epsilon),
                    sinkhorn_iters=int(args.sinkhorn_iters),
                    compile_sinkhorn=bool(args.compile_sinkhorn),
                    permute_mode=args.permute_atlas,
                    # Permute must stay within chamfer patches or the per-patch target
                    # becomes non-stationary, so for chamfer_patch/sinkhorn_patch we align the
                    # permute granularity to chamfer_patch_size; otherwise use the model patch.
                    permute_patch_size=(
                        _cps if args.recon_loss in ("chamfer_patch", "sinkhorn_patch", "sinkhorn_patch_hard")
                        else int(unwrapped_model.patch_size)
                    ),
                    # When recon_loss_weight==0, the recon term contributes nothing to backward —
                    # skip the OT/Chamfer/MSE forward entirely (hot-reloadable via overrides.yaml).
                    skip_recon=(float(runtime.recon_loss_weight) == 0.0),
                )
                # ``mse_loss`` is the per-step reconstruction term (index-aligned MSE
                # or Chamfer, depending on --recon_loss). Name/log keys kept as "mse"
                # for dashboard continuity; the Chamfer weight is already folded in.
                sample_losses = loss_dict["loss"]
                mse_loss = sample_losses.mean()
                x0_pred = loss_dict.get("pred_xstart")

                # `torch.isfinite(mse_loss)` is a CPU↔GPU sync (Python branch on a
                # device tensor). Gate behind log_every so we still catch non-finite
                # losses periodically without stalling every micro-batch on the main
                # stream (which serializes against NCCL/prefetch).
                if step % args.log_every == 0 and not torch.isfinite(mse_loss):
                    _debug_nonfinite_mse(
                        args=args,
                        diffusion=diffusion,
                        model=model,
                        x=x,
                        y_pooled=y_pooled,
                        x_full=x_full,
                        t=t,
                        t_value=t_value,
                        noise=noise,
                        sample_losses=sample_losses,
                        step=step,
                        epoch=epoch,
                        hash_keys=hash_keys,
                        is_main=is_main,
                    )

                # Render loss (computed in fp32 outside autocast for GS renderer compatibility)
                render_l1_loss = _zero_render_loss
                render_alpha_l1_loss = _zero_render_loss
                render_lpips_loss = _zero_render_loss
                train_render_preview_due = (
                    args.train_render_log_every > 0
                    and renderer_for_train is not None
                    and train_cameras is not None
                    and step % args.train_render_log_every == 0
                )
                should_compute_render = (
                    renderer_for_train is not None
                    and train_cameras is not None
                    and _any_render_loss_weight(runtime)
                    and step >= args.enable_render_loss_after
                )
                if x0_pred is not None:
                    x0_pred = x0_pred.float()
                x_gt_for_render = x_full if x_full is not None else x
                if should_compute_render and x0_pred is None:
                    noise_for_render = torch.randn_like(x)
                    x_t = diffusion.flow_matching_q_sample(x, t_value, noise=noise_for_render)
                    model_out = model(x_t, t, y_pooled)
                    x0_pred = model_out.float()

                if should_compute_render and x0_pred is not None:
                    if (
                        runtime.lpips_loss_weight > 0.0
                        and lpips_fn_for_train is None
                        and args.perceptual_backend == 'lpips'
                    ):
                        lpips_probe = _try_import_lpips()
                        if isinstance(lpips_probe, Exception):
                            if is_main:
                                logger.warning(
                                    "[overrides/render-loss] LPIPS requested but import failed: %s",
                                    lpips_probe,
                                )
                        else:
                            lpips_fn_for_train = lpips_probe.LPIPS(net=args.lpips_net).to(device).eval()
                            for p in lpips_fn_for_train.parameters():
                                p.requires_grad_(False)

                    # Per-sample mask on the flow-matching t: keep samples
                    # whose clean-fraction t_value ≥ cutoff (i.e. low-noise),
                    # since render loss at high noise produces useless gradients.
                    # Slice the batch here so the masked samples never enter
                    # gsplat or LPIPS — the cutoff is the memory cap.
                    x0_pred_render = x0_pred
                    x_gt_render = x_gt_for_render
                    kept_hash_keys = hash_keys
                    noise_cutoff = float(args.render_loss_noise_cutoff)
                    if noise_cutoff > 0.0:
                        keep_mask = (t_value >= noise_cutoff)
                        # Boolean indexing already syncs once to allocate the
                        # output. Read the count from the resulting shape rather
                        # than calling `.item()` on a separate sum (which would
                        # add a redundant sync per micro-batch).
                        x0_pred_render = x0_pred[keep_mask]
                        x_gt_render = x_gt_for_render[keep_mask]
                        # Filter hash_keys with the same mask so per-sample zoom lookups stay aligned.
                        keep_mask_cpu = keep_mask.detach().cpu().tolist()
                        kept_hash_keys = [h for h, k in zip(hash_keys, keep_mask_cpu) if k]
                        n_kept = x0_pred_render.shape[0]
                        if n_kept == 0:
                            should_compute_render = False

                    if should_compute_render:
                        # Track the render-kept count for the periodic log line (folded
                        # into the main "Step" row instead of a separate per-microbatch print).
                        log_render_kept += x0_pred_render.shape[0]
                        log_render_kept_steps += 1
                        # Per-sample camera zooms — gather rows for the kept-batch's hashes.
                        # Composes multiplicatively with --render_zoom_factor (already baked
                        # into train_cameras' Ks), so the effective per-(sample, cam) zoom is
                        # render_zoom_factor × per_sample_cam_zooms[b, c].
                        per_sample_cam_zooms = None
                        if per_sample_zoom_table is not None:
                            rows = torch.tensor(
                                [hash_to_zoom_row[h] for h in kept_hash_keys],
                                device=per_sample_zoom_table.device, dtype=torch.long,
                            )
                            per_sample_cam_zooms = per_sample_zoom_table.index_select(0, rows)
                        render_l1_loss, render_alpha_l1_loss, render_lpips_loss = _compute_render_loss_for_batch(
                            x0_pred=x0_pred_render,
                            x_gt_full=x_gt_render,
                            norm_mean_pred=norm_mean,
                            norm_std_pred=norm_std,
                            norm_mean_full=norm_mean_full if x_full is not None else norm_mean,
                            norm_std_full=norm_std_full if x_full is not None else norm_std,
                            train_cameras=train_cameras,
                            renderer_tuple=renderer_for_train,
                            lpips_fn=lpips_fn_for_train,
                            num_cam=args.render_loss_num_cam,
                            device=device,
                            dc_only=dc_only,
                            plane_to_sphere=plane_to_sphere,
                            sample_weights=None,
                            rank_transform_tables=rank_transform_tables,
                            per_sample_cam_zooms=per_sample_cam_zooms,
                        )

                if train_render_preview_due:
                    # Keep all ranks aligned before and after main-process-only preview rendering.
                    accelerator.wait_for_everyone()
                    if is_main:
                        preview_idx = random.randrange(max(1, x.shape[0]))
                        preview_slice = slice(preview_idx, preview_idx + 1)
                        preview_x_gt = x_gt_for_render[preview_slice]
                        preview_t = t[preview_slice]
                        preview_t_value = t_value[preview_slice]
                        preview_pooled = y_pooled[preview_slice]

                        if x0_pred is not None:
                            preview_x0_pred = x0_pred.detach()[preview_slice]
                        else:
                            preview_x = x[preview_slice]
                            noise_for_preview = torch.randn_like(preview_x)
                            x_t_preview = diffusion.flow_matching_q_sample(
                                preview_x, preview_t_value, noise=noise_for_preview
                            )
                            with torch.no_grad():
                                model_out_preview = model(
                                    x_t_preview, preview_t, preview_pooled,
                                )
                                preview_x0_pred = model_out_preview.float()

                        _save_training_render_preview(
                            x0_pred=preview_x0_pred,
                            x_gt_full=preview_x_gt,
                            norm_mean_pred=norm_mean,
                            norm_std_pred=norm_std,
                            norm_mean_full=norm_mean_full if x_full is not None else norm_mean,
                            norm_std_full=norm_std_full if x_full is not None else norm_std,
                            train_cameras=train_cameras,
                            renderer_tuple=renderer_for_train,
                            output_dir=args.results_dir,
                            epoch=epoch,
                            step=step,
                            timesteps=preview_t,
                            labels=None,
                            device=device,
                            num_cam=args.train_render_log_num_cam,
                            dc_only=dc_only,
                            plane_to_sphere=plane_to_sphere,
                            rank_transform_tables=rank_transform_tables,
                        )
                        loss_tracker.flush_plots()
                    accelerator.wait_for_everyone()

                total_loss = (
                    float(runtime.recon_loss_weight) * mse_loss
                    + float(runtime.render_loss_weight) * render_l1_loss
                    + float(runtime.alpha_mask_loss_weight) * render_alpha_l1_loss
                    + float(runtime.lpips_loss_weight) * render_lpips_loss
                )

                # Per-loss gradient norm measurement (single-GPU only).
                # Fires on every sync step within log intervals that will print grad norms,
                # i.e. when the upcoming print index is a multiple of grad_norm_log_every_n_prints.
                # Disabled under DDP: the helper does multiple retain_graph backwards on one
                # forward, and DDP's reducer marks each parameter ready on every backward —
                # even inside accelerator.no_sync — which corrupts reducer state and makes
                # the subsequent main backward crash with "marked as ready twice".
                _gnl_n = max(1, int(runtime.grad_norm_log_every_n_prints))
                _print_idx = step // args.log_every + 1  # index of the upcoming print
                if (
                    is_main
                    and accelerator.num_processes == 1
                    and accelerator.sync_gradients
                    and int(runtime.grad_norm_log_every_n_prints) > 0
                    and _print_idx % _gnl_n == 0
                ):
                    with accelerator.no_sync(model):
                        _per_loss_norms = _measure_per_loss_grad_norms(
                            model=accelerator.unwrap_model(model),
                            weighted_losses={
                                "mse": (mse_loss, 1.0),
                                "render_l1": (render_l1_loss, float(runtime.render_loss_weight)),
                                "alpha_l1": (render_alpha_l1_loss, float(runtime.alpha_mask_loss_weight)),
                                "lpips": (render_lpips_loss, float(runtime.lpips_loss_weight)),
                            },
                        )
                    log_mse_gn += _per_loss_norms.get("mse", 0.0)
                    log_rl1_gn += _per_loss_norms.get("render_l1", 0.0)
                    log_alpha_gn += _per_loss_norms.get("alpha_l1", 0.0)
                    log_lpips_gn += _per_loss_norms.get("lpips", 0.0)
                    log_per_loss_gn_steps += 1

                # Backward pass (accelerate handles scaling + sync)
                if _prof:
                    _prof_ev_bwd_start.record()

                accelerator.backward(total_loss)

                if _prof:
                    # Wait for DDP all-reduces (issued on the NCCL side stream
                    # during backward) so bwd_time absorbs comm cost. Without
                    # this sync the all-reduce wait leaks into opt_time when
                    # clip_grad_norm reads grads.
                    torch.cuda.synchronize()
                    _prof_ev_opt_start.record()

                clip_cap = float(runtime.max_grad_norm)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(),
                        clip_cap if clip_cap > 0.0 else float('inf'),
                    )
                    if is_main:
                        finite = torch.isfinite(grad_norm)
                        log_grad_norm += torch.where(
                            finite, grad_norm, torch.zeros_like(grad_norm)
                        )
                        log_grad_steps += finite.to(log_grad_steps.dtype)
                opt.step()
                opt.zero_grad()

            # `step` is the optim-step counter — advance it (and update EMA)
            # only when accelerate actually fired opt.step(). All post-step
            # work below is gated on the same condition so its `step % N == 0`
            # gates fire once per optim step rather than once per micro-batch.
            if accelerator.sync_gradients:
                update_ema(ema, accelerator.unwrap_model(model), decay=args.ema_decay)
                step += 1
                log_optim_steps += 1

            if _prof:
                _prof_ev_step_end.record()
                _prof_ev_step_end.synchronize()
                _prof_fwd_ms = _prof_ev_fwd_start.elapsed_time(_prof_ev_bwd_start)
                _prof_bwd_ms = _prof_ev_bwd_start.elapsed_time(_prof_ev_opt_start)
                _prof_opt_ms = _prof_ev_opt_start.elapsed_time(_prof_ev_step_end)
                if step >= start_step + _PROF_SKIP:
                    _prof_data_q.append(_prof_data_ms)
                    _prof_h2d_q.append(_prof_h2d_ms)
                    _prof_fwd_q.append(_prof_fwd_ms)
                    _prof_bwd_q.append(_prof_bwd_ms)
                    _prof_opt_q.append(_prof_opt_ms)
                _prof_t_step_start = time.perf_counter()

            # Logging
            log_loss += mse_loss.detach()
            _mh = loss_dict.get("mse_hybrid")
            if _mh is not None:
                log_mse_hybrid += _mh.detach().mean()
            _sr = loss_dict.get("sinkhorn_marginal_resid")
            if _sr is not None:
                log_sinkhorn_resid += _sr.detach()
            _sc = loss_dict.get("sinkhorn_collision_frac")
            if _sc is not None:
                log_sinkhorn_coll += _sc.detach()
            log_render_l1 += render_l1_loss.detach()
            log_render_alpha_l1 += render_alpha_l1_loss.detach()
            log_render_lpips += render_lpips_loss.detach()
            # Bucket per-sample MSE by continuous t_value ∈ (0,1) using the
            # custom edges defined above.
            with torch.no_grad():
                bucket_idx = torch.bucketize(
                    t_value.detach().to(t_bucket_boundaries.dtype),
                    t_bucket_boundaries,
                    right=True,
                )
                per_sample = sample_losses.detach()
                log_t_bucket_sum.scatter_add_(0, bucket_idx, per_sample)
                log_t_bucket_cnt.scatter_add_(
                    0, bucket_idx, torch.ones_like(per_sample)
                )
            log_steps += 1

            # All `step % N == 0` gates below run only on optim-step boundaries
            # — `step` doesn't change across the `ga` micro-batches of an
            # accumulation cycle, so without the sync_gradients guard each
            # block would fire `ga` times in a row on the boundary cycle.
            if _prof and is_main and accelerator.sync_gradients and step % _PROF_PRINT_EVERY == 0 and len(_prof_data_q) > 0:
                _d = sum(_prof_data_q) / len(_prof_data_q)
                _h = sum(_prof_h2d_q) / len(_prof_h2d_q)
                _f = sum(_prof_fwd_q) / len(_prof_fwd_q)
                _b = sum(_prof_bwd_q) / len(_prof_bwd_q)
                _o = sum(_prof_opt_q) / len(_prof_opt_q)
                _tot = _d + _h + _f + _b + _o
                _sps = (1000.0 / _tot) if _tot > 0 else 0.0
                logger.info(
                    "[step %d] data: %.1fms (%.0f%%) | h2d: %.1fms (%.0f%%) | "
                    "fwd: %.1fms (%.0f%%) | bwd: %.1fms (%.0f%%) | opt: %.1fms (%.0f%%) | "
                    "total: %.1fms | sps: %.2f",
                    step, _d, _d / _tot * 100, _h, _h / _tot * 100,
                    _f, _f / _tot * 100, _b, _b / _tot * 100, _o, _o / _tot * 100,
                    _tot, _sps,
                )

            if step % args.log_every == 0 and is_main and accelerator.sync_gradients:
                avg_loss = log_loss.item() / log_steps
                # Sinkhorn convergence monitor (accumulated for both sinkhorn recon modes);
                # collision fraction is hard-mode only.
                avg_sink_resid = (
                    log_sinkhorn_resid.item() / log_steps
                    if args.recon_loss in ("sinkhorn_patch", "sinkhorn_patch_hard") else None
                )
                avg_sink_coll = (
                    log_sinkhorn_coll.item() / log_steps
                    if args.recon_loss == "sinkhorn_patch_hard" else None
                )
                elapsed = time.time() - start_time
                # Optim steps / sec — log_optim_steps counts sync micro-batches
                # only, while log_steps counts every micro-batch (used for
                # averaging losses across the window).
                steps_per_sec = log_optim_steps / elapsed if elapsed > 0 else 0.0
                msg = (
                    f"Step {step:>7d} | Epoch {epoch:>3d} | "
                    f"MSE: {avg_loss:.4f} | "
                    f"Steps/sec: {steps_per_sec:.2f}"
                )
                if args.lr_schedule in ('cosine', 'warmup', 'none') and last_lr_val is not None:
                    msg += f" | LR: {last_lr_val:.2e}"
                    if (
                        last_text_proj_lr_val is not None
                        and last_text_proj_lr_val != last_lr_val
                    ):
                        msg += f" (txt {last_text_proj_lr_val:.2e})"
                if p_mean_schedule is not None:
                    msg += f" | P_mean: {runtime.P_mean:+.3f}"
                if float(runtime.mse_hybrid_weight) > 0.0:
                    avg_mse_hyb = log_mse_hybrid.item() / log_steps
                    msg += f" | MSE_hyb: {avg_mse_hyb:.4f} (λ{float(runtime.mse_hybrid_weight):.3g})"
                if avg_sink_resid is not None:
                    msg += f" | SinkResid: {avg_sink_resid:.2e}"
                if avg_sink_coll is not None:
                    msg += f" | SinkColl: {avg_sink_coll*100:.2f}%"
                if _any_render_loss_weight(runtime):
                    avg_rl1 = log_render_l1.item() / log_steps
                    avg_alpha_rl1 = log_render_alpha_l1.item() / log_steps
                    avg_rlpips = log_render_lpips.item() / log_steps
                    msg += (
                        f" | Render_L1: {avg_rl1:.4f}"
                        f" | Alpha_L1: {avg_alpha_rl1:.4f}"
                        f" | Render_LPIPS: {avg_rlpips:.4f}"
                    )
                    if log_render_kept_steps > 0:
                        msg += f" | kept: {log_render_kept / log_render_kept_steps:.0f}/{x.shape[0]}"
                grad_steps_int = int(log_grad_steps.item())
                if grad_steps_int > 0:
                    avg_grad_norm = log_grad_norm.item() / grad_steps_int
                    msg += f" | GradNorm: {avg_grad_norm:.4f}"
                else:
                    avg_grad_norm = None
                if log_per_loss_gn_steps > 0:
                    n = log_per_loss_gn_steps
                    msg += f" | GN[mse]: {log_mse_gn / n:.4f}"
                    if log_rl1_gn > 0.0:
                        msg += f" | GN[rl1]: {log_rl1_gn / n:.4f}"
                    if log_alpha_gn > 0.0:
                        msg += f" | GN[alpha]: {log_alpha_gn / n:.4f}"
                    if log_lpips_gn > 0.0:
                        msg += f" | GN[lpips]: {log_lpips_gn / n:.4f}"
                # Per-t-bucket MSE: low-t = noisy, high-t = clean. Empty buckets
                # print NaN rather than crash — happens only on a pathological
                # P_mean/P_std where some bin is never sampled in the window.
                bucket_cnt = log_t_bucket_cnt.clamp(min=1)
                bucket_avg = (log_t_bucket_sum / bucket_cnt).tolist()
                bucket_hits = log_t_bucket_cnt.tolist()
                bucket_str = " ".join(
                    f"[{t_bucket_edges[i]:.2f}-{t_bucket_edges[i + 1]:.2f}]"
                    f"{bucket_avg[i]:.3f}(n={int(bucket_hits[i])})"
                    for i in range(num_t_buckets)
                )
                msg += f" | MSE/t: {bucket_str}"
                logger.info(msg)

                # Persist this print's averages to the loss tracker. Values are
                # already host-side floats, so this is a cheap dict append +
                # CSV line write — no extra GPU syncs.
                render_active = _any_render_loss_weight(runtime)
                grad_norm_per_loss = None
                if log_per_loss_gn_steps > 0:
                    n = log_per_loss_gn_steps
                    grad_norm_per_loss = {
                        "mse": log_mse_gn / n,
                        "render_l1": (log_rl1_gn / n) if log_rl1_gn > 0.0 else None,
                        "alpha_l1": (log_alpha_gn / n) if log_alpha_gn > 0.0 else None,
                        "lpips": (log_lpips_gn / n) if log_lpips_gn > 0.0 else None,
                    }
                _render_l1 = (log_render_l1.item() / log_steps) if render_active else None
                _alpha_l1 = (log_render_alpha_l1.item() / log_steps) if render_active else None
                _lpips = (log_render_lpips.item() / log_steps) if render_active else None
                _grad_norm = avg_grad_norm

                loss_tracker.record(
                    step=step,
                    mse=avg_loss,
                    render_l1=_render_l1,
                    alpha_l1=_alpha_l1,
                    lpips=_lpips,
                    grad_norm=_grad_norm,
                    grad_norm_per_loss=grad_norm_per_loss,
                    lr=last_lr_val,
                    p_mean=runtime.P_mean,
                    steps_per_sec=steps_per_sec,
                    bucket_means=bucket_avg,
                    bucket_counts=[int(h) for h in bucket_hits],
                )

                if wandb_run is not None:
                    payload = {
                        "train/mse": avg_loss,
                        "train/grad_norm": _grad_norm,
                        "train/lr": last_lr_val,
                        "train/lr_text_proj": last_text_proj_lr_val,
                        "train/p_mean": runtime.P_mean,
                        "train/steps_per_sec": steps_per_sec,
                    }
                    if float(runtime.mse_hybrid_weight) > 0.0:
                        payload["train/mse_hybrid"] = log_mse_hybrid.item() / log_steps
                        payload["train/mse_hybrid_weight"] = float(runtime.mse_hybrid_weight)
                    if avg_sink_resid is not None:
                        payload["train/sinkhorn_marginal_resid"] = avg_sink_resid
                    if avg_sink_coll is not None:
                        payload["train/sinkhorn_collision_frac"] = avg_sink_coll
                    if render_active:
                        payload["train/render_l1"] = _render_l1
                        payload["train/alpha_l1"] = _alpha_l1
                        payload["train/lpips"] = _lpips
                    if grad_norm_per_loss is not None:
                        for k, v in grad_norm_per_loss.items():
                            if v is not None:
                                payload[f"train/grad_norm_{k}"] = v
                    if bucket_avg is not None:
                        for i, m in enumerate(bucket_avg):
                            payload[f"train/bucket{i}_mse"] = m
                            payload[f"train/bucket{i}_count"] = int(bucket_hits[i])
                    payload = {k: v for k, v in payload.items() if v is not None}
                    _wandb_log(wandb_run, payload, step=step)

                log_loss.zero_()
                log_mse_hybrid.zero_()
                log_sinkhorn_resid.zero_()
                log_sinkhorn_coll.zero_()
                log_render_kept = 0.0
                log_render_kept_steps = 0
                log_render_l1.zero_()
                log_render_alpha_l1.zero_()
                log_render_lpips.zero_()
                log_grad_norm.zero_()
                log_grad_steps.zero_()
                log_steps = 0
                log_optim_steps = 0
                log_mse_gn = 0.0
                log_rl1_gn = 0.0
                log_alpha_gn = 0.0
                log_lpips_gn = 0.0
                log_per_loss_gn_steps = 0
                log_t_bucket_sum.zero_()
                log_t_bucket_cnt.zero_()
                start_time = time.time()

            checkpoint_due = accelerator.sync_gradients and step % args.ckpt_every == 0
            if checkpoint_due:
                accelerator.wait_for_everyone()
                if is_main:
                    ckpt_path = os.path.join(args.results_dir, f"{step:07d}.pt")
                    torch.save({
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'ema': ema.state_dict(),
                        'opt': opt.state_dict(),
                        'args': vars(args),
                        'step': step,
                    }, ckpt_path)
                    logger.info(f"Saved checkpoint to {ckpt_path}")
                accelerator.wait_for_everyone()

            validation_due = (
                enable_val and accelerator.sync_gradients and step % args.val_every == 0
            )
            if validation_due:
                accelerator.wait_for_everyone()
                if is_main:
                    val_render_path = _run_validation_render(
                        model=ema,
                        plane_to_sphere=plane_to_sphere,
                        norm_mean=norm_mean,
                        norm_std=norm_std,
                        train_cameras=train_cameras,
                        renderer_tuple=renderer_for_train,
                        output_dir=args.results_dir,
                        epoch=epoch,
                        step=step,
                        device=device,
                        in_channels=in_channels,
                        cond_pool=cond_pool,
                        dc_only=dc_only,
                        predict_xstart=args.predict_xstart,
                        noise_schedule=args.noise_schedule,
                        diffusion_steps=diffusion.num_timesteps,
                        val_sampling_steps=args.val_sampling_steps,
                        val_sampler=args.val_sampler,
                        dpm_solver_order=args.dpm_solver_order,
                        dpm_algorithm_type=args.dpm_algorithm_type,
                        dpm_solver_type=args.dpm_solver_type,
                        dpm_timestep_spacing=args.dpm_timestep_spacing,
                        dpm_use_karras_sigmas=args.dpm_use_karras_sigmas,
                        ddim_eta=args.ddim_eta,
                        cfg_scale=args.val_cfg_scale,
                        P_mean=runtime.P_mean,
                        P_std=args.P_std,
                        rank_transform_tables=rank_transform_tables,
                    )
                    val_grid_path = None
                    if args.val_grid_enabled:
                        val_grid_path = _run_validation_grid(
                            model=ema,
                            plane_to_sphere=plane_to_sphere,
                            norm_mean=norm_mean,
                            norm_std=norm_std,
                            train_cameras=train_cameras,
                            renderer_tuple=renderer_for_train,
                            output_dir=args.results_dir,
                            epoch=epoch,
                            step=step,
                            device=device,
                            in_channels=in_channels,
                            cond_pool=cond_pool,
                            grid_seed=args.val_grid_seed,
                            camera_idx=args.val_grid_camera_idx,
                            grid_rows=args.val_grid_rows,
                            grid_cols=args.val_grid_cols,
                            dc_only=dc_only,
                            predict_xstart=args.predict_xstart,
                            noise_schedule=args.noise_schedule,
                            diffusion_steps=diffusion.num_timesteps,
                            val_sampling_steps=args.val_sampling_steps,
                            val_sampler=args.val_sampler,
                            dpm_solver_order=args.dpm_solver_order,
                            dpm_algorithm_type=args.dpm_algorithm_type,
                            dpm_solver_type=args.dpm_solver_type,
                            dpm_timestep_spacing=args.dpm_timestep_spacing,
                            dpm_use_karras_sigmas=args.dpm_use_karras_sigmas,
                            ddim_eta=args.ddim_eta,
                            cfg_scale=args.val_cfg_scale,
                            P_mean=runtime.P_mean,
                            P_std=args.P_std,
                            rank_transform_tables=rank_transform_tables,
                        )
                    loss_tracker.flush_plots()
                    if wandb_run is not None and args.wandb_log_images:
                        img_payload = {}
                        if val_render_path and os.path.exists(val_render_path):
                            img_payload["val/render"] = wandb.Image(val_render_path)
                        if val_grid_path and os.path.exists(val_grid_path):
                            img_payload["val/grid"] = wandb.Image(val_grid_path)
                        if img_payload:
                            _wandb_log(wandb_run, img_payload, step=step)

                    # Conditioning-signal probe — measured only alongside the
                    # validation images, not on every train log line.
                    if args.class_dropout_prob > 0:
                        # FM convention: t_value=0 is max noise; pure-noise probe
                        # input is then in-distribution at t_discrete=0. See
                        # CLAUDE.md "Diffusion / timestep convention".
                        sig = _measure_conditioning_signal(
                            model=accelerator.unwrap_model(model),
                            cond_pool=cond_pool,
                            in_channels=in_channels,
                            diffusion_num_timesteps=diffusion.num_timesteps,
                            device=device,
                            t_value=0.0,
                            batch_size=args.batch_size,
                            seed=args.seed,
                        )
                        logger.info(
                            "[cond] cfg_signal=%.4f (min=%.4f max=%.4f, n=%d) | "
                            "cond_signal=%.4f (min=%.4f max=%.4f, n=%d) | pred_rms=%.4f",
                            sig["cfg_signal"], sig["cfg_signal_min"], sig["cfg_signal_max"],
                            sig["num_cond_probes"],
                            sig["cond_signal"], sig["cond_signal_min"], sig["cond_signal_max"],
                            sig["num_cond_pairs"], sig["pred_rms"],
                        )
                        _wandb_log(
                            wandb_run,
                            {f"cond/{k}": v for k, v in sig.items()
                             if isinstance(v, (int, float))},
                            step=step,
                        )
                accelerator.wait_for_everyone()

            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            if is_main:
                logger.info(f"[max_steps] Reached --max_steps={args.max_steps}, stopping.")
            break

    # Save final checkpoint
    accelerator.wait_for_everyone()
    if is_main:
        ckpt_path = os.path.join(args.results_dir, f"{step:07d}.pt")
        torch.save({
            'model': accelerator.unwrap_model(model).state_dict(),
            'ema': ema.state_dict(),
            'opt': opt.state_dict(),
            'args': vars(args),
            'step': step,
        }, ckpt_path)
        logger.info(f"Training complete. Final checkpoint: {ckpt_path}")
        loss_tracker.flush_plots()
        loss_tracker.close()
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as exc:
                logger.warning("[wandb] finish failed: %s", exc)
    accelerator.wait_for_everyone()


def build_train_gsplat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train JiT for 3DGS generation')

    # Model
    parser.add_argument('--model', type=str, default='JiT-B/8',
                        choices=list(JiT_3DGS_models.keys()))
    parser.add_argument(
        '--bottleneck',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use the two-stage BottleneckPatchEmbed (in_chans→bottleneck_dim→embed_dim). '
             'Disable with --no-bottleneck to use a single-conv patch embed '
             '(in_chans→embed_dim) with no rank reduction below embed_dim.',
    )
    parser.add_argument('--predict_xstart', action=argparse.BooleanOptionalAction, default=True,
                        help='Model predicts x0 directly (default: True). Disable with --no-predict_xstart.')
    parser.add_argument(
        '--noise_schedule',
        type=str,
        default='linear',
        choices=['linear', 'squaredcos_cap_v2'],
        help='Beta schedule for diffusion noise',
    )
    parser.add_argument(
        '--class_dropout_prob',
        type=float,
        default=0.1,
        help='Conditioning-dropout probability for classifier-free guidance. '
             'Name kept for YAML/CLI backwards compatibility; applies to the '
             'text conditioning vector (replaced with the learned null embedding).',
    )

    # Data
    parser.add_argument('--obj_list', type=str, required=True,
                        help='Path to obj_list JSON file')
    parser.add_argument('--gs_path', type=str, required=True,
                        help='Path to 3DGS data directory')
    parser.add_argument('--mean_file', type=str, default=None,
                        help='Path to normalization mean file')
    parser.add_argument('--std_file', type=str, default=None,
                        help='Path to normalization std file')
    parser.add_argument('--rank_transform_file', type=str, default=None,
                        help='Path to Gaussian rank-transform tables built by '
                             'data/build_rank_transform.py. Listed channels are '
                             'mapped to N(0,1) at load time and the inverse is '
                             'applied before the renderer\'s sigmoid/exp. '
                             'mean/std for these channels are forced to (0,1) '
                             'so the standardize round-trip is a no-op.')
    parser.add_argument('--clip_thresholds_file', type=str, default=None,
                        help='Path to per-channel hard-clip thresholds built by '
                             'data/build_clip_thresholds.py. Listed channels are '
                             'clipped to [lower, upper] at load time, BEFORE the '
                             'rank transform. When clipped channels are also in the '
                             'rank set, the rank tables must have been built on the '
                             'clipped stream (build_rank_transform.py '
                             '--clip_thresholds_file).')
    parser.add_argument('--text_embed_path', type=str, default=None,
                        help='Path to precomputed pooled CLIP vectors. Accepts a '
                             'directory containing keys.json + pooled.npy (the '
                             'mmap-friendly format produced by '
                             'object_classification/encode_text_embeddings.py — '
                             'tokens.npy / mask.npy are ignored if present), or a '
                             '.npz with keys + pooled (or legacy embeddings) field. '
                             'Comma-separated shard list also accepted. '
                             'Required for text-conditioned training.')
    parser.add_argument('--null_text_token_path', type=str, default=None,
                        help='Path to null_text_token.npz produced by the encoder '
                             '(EOS token of empty-string penultimate-norm encoding). '
                             'If omitted, defaults to null_text_token.npz next to the '
                             'first --text_embed_path shard.')
    parser.add_argument('--val_prompts_file', type=str, default=None,
                        help='Optional JSON list of validation prompts to encode '
                             'with CLIP-L/14 at startup. If None, uses the built-in 64-prompt '
                             'mix (simple, multi-attribute, compositional/spatial).')
    parser.add_argument('--sphere2plane_path', type=str, default='data/sphere2plane.npy',
                        help='Path to sphere2plane.npy permutation file')
    parser.add_argument('--exclude_keys_file', type=str, default=None,
                        help='Optional JSON list of hash_keys to drop from the dataset (e.g. data/outlier_keys_8sigma.json)')
    parser.add_argument('--sh_degree0_only', action=argparse.BooleanOptionalAction, default=False,
                        help='Keep only SH degree-0 / DC coefficients, reducing from 59 to 14 channels')

    # Render loss
    parser.add_argument('--render_loss_weight', type=float, default=0.0,
                        help='Weight for render L1 photometric loss term')
    parser.add_argument('--alpha_mask_loss_weight', type=float, default=0.0,
                        help='Weight for render alpha-mask L1 loss term')
    parser.add_argument('--lpips_loss_weight', type=float, default=0.0,
                        help='Weight for render perceptual photometric loss term '
                             '(applied to whichever --perceptual_backend is active)')
    parser.add_argument('--lpips_net', type=str, default='vgg', choices=('vgg', 'alex', 'squeeze'),
                        help='LPIPS backbone (only used when --perceptual_backend=lpips)')
    parser.add_argument('--perceptual_backend', type=str, default='lpips',
                        choices=('lpips', 'dinov2'),
                        help='Perceptual backend for the render perceptual loss term. '
                             'lpips = lpips package; dinov2 = frozen DINOv2 ViT-B/14 cosine-distance '
                             'over patch tokens. Set at launch only — not hot-reloadable.')
    parser.add_argument('--render_loss_num_cam', type=int, default=1,
                        help='Number of cameras to randomly sample per render loss step')
    parser.add_argument('--train_render_size', type=int, default=128,
                        help='Train-time rendering resolution for render loss')
    parser.add_argument('--render_zoom_factor', type=float, default=1.0,
                        help='Zoom factor applied to the rendering FOV (>1 = zoomed in; equivalent '
                             'to increasing focal length). 1.0 = no change (default). At the '
                             'gaussianverse fovx=39.6° the objects fill only ~20%% of the frame so '
                             '~80%% of every render L1/LPIPS pixel is background that dilutes the '
                             'loss-mean. 1.6-2.0 narrows the FOV to make the object fill the frame, '
                             'boosting per-pixel signal density ~4x. Camera position unchanged; '
                             'only intrinsics narrow. Set at launch only (cameras built once).')
    parser.add_argument('--ref_camera_tar', type=str, default='/home/tiangexiang/gen3d/ref_camera.tar.gz',
                        help='Path to reference camera tar.gz for render loss')
    parser.add_argument('--per_sample_zoom_file', type=str, default=None,
                        help='Optional .pt produced by data/build_per_sample_render_fov.py: '
                             'per-(sample, camera) zoom factor used to scale the rendering '
                             'fx/fy at training time so each render fills the frame with the '
                             "object. When unset, falls back to the global --render_zoom_factor "
                             '(default 1.0). Composes multiplicatively with --render_zoom_factor.')
    parser.add_argument('--enable_render_loss_after', type=int, default=0,
                        help='Number of training steps before enabling render loss (-1 disables render loss)')
    parser.add_argument('--render_loss_noise_cutoff', type=float, default=0.4,
                        help='Minimum alpha_bar (SNR proxy) for a sample to contribute to render loss. '
                             'Samples below this threshold are masked out. 0.0 = no masking.')
    parser.add_argument('--train_render_log_every', type=int, default=0,
                        help='Save side-by-side train-time 2D render previews every N steps (0 = disabled)')
    parser.add_argument('--train_render_log_num_cam', type=int, default=2,
                        help='Number of camera views per train-time render preview')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=0,
                        help='Stop training after this many optimizer steps (0 = disabled). '
                             'Used by jit/sweep.py to cap each hyperparameter trial.')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument(
        '--betas',
        type=float,
        nargs=2,
        default=[0.9, 0.999],
        metavar=('BETA1', 'BETA2'),
        help='AdamW betas for the main parameter group.',
    )
    parser.add_argument(
        '--text_proj_lr',
        type=float,
        default=None,
        help='Optional learning rate for the text-projection MLP '
             '(y_embedder.proj feeding the AdaLN signal). Defaults to --lr. '
             'Follows the same warmup+cosine shape as --lr, anchored at this peak.',
    )
    parser.add_argument(
        '--text_proj_betas',
        type=float,
        nargs=2,
        default=None,
        metavar=('BETA1', 'BETA2'),
        help='Optional AdamW betas for the text-projection MLP. Defaults to --betas.',
    )
    parser.add_argument(
        '--lr_schedule',
        type=str,
        default='none',
        choices=['none', 'warmup', 'cosine'],
        help='LR schedule: none (constant), warmup (linear ramp to --lr then hold), cosine (warmup + cosine decay)',
    )
    parser.add_argument(
        '--lr_warmup_steps',
        type=int,
        default=0,
        help='Optimizer steps for linear LR warmup to --lr (0 = no warmup). Used with --lr_schedule warmup or cosine',
    )
    parser.add_argument(
        '--lr_min',
        type=float,
        default=0.0,
        help='Minimum LR at end of cosine decay. Only used with --lr_schedule cosine',
    )
    parser.add_argument(
        '--lr_cosine_total_steps',
        type=int,
        default=0,
        help='Total optimizer steps for cosine schedule (0 = auto: epochs * len(loader) // gradient_accumulation_steps, computed after accelerate.prepare so it accounts for real dataset size, num_processes, and drop_last). '
        'Only used with --lr_schedule cosine',
    )
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                        choices=['fp16', 'bf16', 'none'])
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--gradient_checkpointing', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable gradient checkpointing to save memory (reduces speed). '
                             'Disable with --no-gradient_checkpointing to use more memory but train faster.')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable torch.compile on the model for faster training (requires PyTorch 2.0+). '
                             'First step is slow (compilation); subsequent steps are faster.')
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=('default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'),
                        help='torch.compile mode. default=safe baseline, max-autotune-no-cudagraphs=Triton '
                             'tile-size autotuning (same numerical path), reduce-overhead=CUDA Graphs '
                             '(removes per-kernel launch overhead, requires stable strides under DDP), '
                             'max-autotune=both autotune+cudagraphs.')
    parser.add_argument('--profile_step_times', action=argparse.BooleanOptionalAction, default=False,
                        help='Print rolling per-phase step time breakdown (data/h2d/fwd/bwd/opt) every '
                             '100 steps on rank 0. Adds explicit GPU syncs around each phase, so step '
                             'times in profile mode are higher than normal — use only for diagnostics.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (0 = disabled)')
    parser.add_argument(
        '--overrides_yaml',
        type=str,
        default=None,
        help='Optional YAML of hot-reloaded overrides (lr_scale, max_grad_norm, render loss weights, P_mean, …). '
        'Re-read every --overrides_every training steps.',
    )
    parser.add_argument(
        '--overrides_every',
        type=int,
        default=1000,
        help='Re-load --overrides_yaml every N optim steps (same counter as logging/ckpt/val).',
    )
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--preload_to_cpu', action=argparse.BooleanOptionalAction, default=False,
                        help='Preload the transformed class-conditioned training dataset into a shared RAM cache in /dev/shm at startup. '
                             'All local GPU processes attach to the same in-memory cache; this does not fall back to disk.')
    parser.add_argument('--lazy_cache_to_cpu', action=argparse.BooleanOptionalAction, default=False,
                        help='Cache samples into the shared /dev/shm CPU cache on first access so training speeds up progressively instead of paying the full preload cost up front.')
    parser.add_argument('--preload_max_samples', type=int, default=0,
                        help='Cap eager CPU preloading to the first N samples (0 = preload the full dataset). '
                             'When used with --preload_to_cpu, training is restricted to that cached subset.')
    parser.add_argument('--preload_workers', type=int, default=0,
                        help='Worker processes used to build the shared preload cache (0 = auto, uses all available CPU workers).')
    parser.add_argument('--persistent_workers', action=argparse.BooleanOptionalAction, default=True,
                        help='Keep DataLoader worker processes alive across epochs when num_workers > 0.')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches each DataLoader worker prefetches ahead when num_workers > 0. '
                             'Set <= 0 to disable the explicit override.')
    parser.add_argument('--overfit', type=int, default=0,
                        help='Overfit to the first N samples (0 = disabled). Disables shuffling, '
                             'drops the last incomplete batch, and logs every step.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--P_mean', type=float, default=-0.8,
                        help='Mean of the JiT logit-normal timestep sampler before sigmoid.')
    parser.add_argument('--P_std', type=float, default=0.8,
                        help='Stddev of the JiT logit-normal timestep sampler before sigmoid.')
    parser.add_argument('--timestep_dist', type=str, default='logitnormal',
                        choices=['logitnormal', 'uniform'],
                        help='Training t_value distribution. logitnormal: sigmoid(N(P_mean,P_std)) '
                             '(default). uniform: t_value ~ U(0,1) — equal mass on every noise level '
                             '(P_mean/P_std ignored).')
    parser.add_argument(
        '--P_mean_schedule',
        type=str,
        default=None,
        help='Optional P_mean curriculum: list of [step, P_mean] control points. '
             'Linear interpolation between points; held constant outside the endpoints. '
             'On CLI pass as JSON, e.g. --P_mean_schedule "[[0,-0.5],[20000,0.0],[40000,0.3],[70000,0.5]]". '
             'When active, overrides both --P_mean and any P_mean in --overrides_yaml.',
    )
    parser.add_argument(
        '--lr_scale_schedule',
        type=str,
        default=None,
        help='Optional lr_scale curriculum: list of [step, lr_scale] control points (values > 0). '
             'Linear interpolation between points; held constant outside the endpoints. Used to '
             'cushion the optimizer when introducing a new loss (engagement warmup), e.g. '
             '--lr_scale_schedule "[[22000,0.2],[24000,1.0]]" linearly ramps LR from 20%% to 100%% '
             'over the 2k steps after render-loss engagement. When active, overrides any '
             'lr_scale in --overrides_yaml.',
    )
    parser.add_argument(
        '--render_weight_schedule',
        type=str,
        default=None,
        help='Optional render-weight ramp: list of [step, rl1, alpha, lpips] control points. '
             'Linear interpolation between points; held constant outside the endpoints. '
             'On CLI pass as JSON, e.g. '
             '--render_weight_schedule "[[0,0.1,0.1,0.01],[100000,0.1,0.1,0.01],[130000,0.3,0.15,0.02]]". '
             'When active, overrides the static weights (CLI / overrides.yaml) for render_loss_weight, '
             'alpha_mask_loss_weight, and lpips_loss_weight. Engagement is still gated by '
             '--enable_render_loss_after; the schedule only supplies the weight values once engaged.',
    )
    parser.add_argument(
        '--channel_loss_weights',
        type=str,
        default=None,
        help='Optional per-channel MSE weighting as a JSON list of floats of length == '
             'in_channels (e.g. 14 for --sh_degree0_only). Compensates for channels whose '
             'per-object spatial std is << 1 after normalization (low-spatial-variance '
             'channels otherwise get near-zero gradient signal). Computed from '
             'data/audit_norm_stats.py. Normalize to mean=1 so the scalar loss '
             'magnitude is preserved.',
    )
    parser.add_argument(
        '--recon_loss',
        type=str,
        default='mse',
        choices=['mse', 'chamfer_feature', 'chamfer_geometric', 'chamfer_patch', 'sinkhorn_patch',
                 'sinkhorn_patch_hard'],
        help='Reconstruction loss on the predicted x0. "mse" (default) is the '
             'index-aligned per-Gaussian MSE. "chamfer_feature" uses '
             'pytorch3d.loss.chamfer_distance over all channels (each Gaussian is a '
             'point in C-dim feature space). "chamfer_geometric" matches Gaussians by '
             'xyz via pytorch3d knn_points, then MSE on all matched channels (both '
             'directions). "chamfer_patch" is bidirectional Chamfer restricted to within '
             'each --chamfer_patch_size patch (pure-torch, no pytorch3d; a middle ground '
             'that bounds matches to a patch and composes with --permute_atlas=patch). '
             'The Chamfer modes are permutation-invariant (globally, or per-patch for '
             'chamfer_patch), removing the dependence on the atlas ordering. '
             'channel_loss_weights are reused inside the distance metric. feature/'
             'geometric require the pytorch3d package; chamfer_patch does not. '
             '"sinkhorn_patch" reuses the chamfer_patch per-patch cost but matches via OPTIMAL '
             'ASSIGNMENT (entropic-OT EMD via log-Sinkhorn) instead of nearest-neighbour, forcing '
             'a within-patch bijection (collision -> 0); see --sinkhorn_epsilon / --sinkhorn_iters. '
             '"sinkhorn_patch_hard" uses the SAME Sinkhorn plan but HARD-rounds it '
             '(argmax per pred) to MSE against a single matched target (DETR-style): a crisp '
             'per-point gradient with no soft blend / mean-pull. The argmax converges in far '
             'fewer --sinkhorn_iters than the soft plan, but is only ~bijective (watch the '
             'logged SinkColl%% collision monitor).',
    )
    parser.add_argument(
        '--chamfer_loss_weight',
        type=float,
        default=1.0,
        help='Scalar weight on the Chamfer reconstruction term (no-op when '
             '--recon_loss=mse). Overridable live via overrides.yaml. NOTE: stock '
             'chamfer_feature magnitude is ~2*C larger than MSE (~118x for C=59); '
             'expect a small value (~0.005-0.02) for feature mode. chamfer_geometric '
             'is already ~MSE scale.',
    )
    parser.add_argument(
        '--recon_loss_weight',
        type=float,
        default=1.0,
        help='Outer scalar on the reconstruction loss (mse / Chamfer / Sinkhorn) in the '
             'total-loss combination. 1.0 (default) = unchanged behaviour; 0.0 = zero recon '
             'gradient (recon still computes for diagnostics like SinkResid/SinkColl, but '
             'contributes nothing to backward). Use for render-only or render-dominant '
             'experiments without ripping the recon plumbing out. Overridable live via '
             'overrides.yaml. Independent of --chamfer_loss_weight, which only scales the '
             'Chamfer term INSIDE the recon return.',
    )
    parser.add_argument(
        '--chamfer_subsample',
        type=int,
        default=0,
        help='Cap the Chamfer kNN search at this many *query* points per direction '
             '(no-op when --recon_loss=mse). A fresh uniform subset is drawn each step; '
             'the full target cloud is kept so each query gets its exact nearest '
             'neighbour, and since the point reduction is a mean this is an unbiased '
             'estimate of the full Chamfer (trades a little gradient variance for speed; '
             'cost ~linear in this value). 0 (default) or >= N (16384) uses all points. '
             'e.g. 4096 ~= 2.2x faster, 8192 ~= 1.4x.',
    )
    parser.add_argument(
        '--chamfer_patch_size',
        type=int,
        default=0,
        help='Patch side length for --recon_loss=chamfer_patch (Chamfer is restricted '
             'within each patch_size x patch_size patch; no cross-patch matching). 0 '
             '(default) resolves to the model patch_size at the call site so it aligns '
             'with the Conv2d patchify and --permute_atlas=patch. No-op for other '
             'recon_loss modes; --chamfer_subsample does not apply to chamfer_patch.',
    )
    parser.add_argument(
        '--chamfer_rev_weight',
        type=float,
        default=1.0,
        help='Multiplier on the BACKWARD (GT-as-query) Chamfer term in all chamfer modes '
             '(1.0 = symmetric). The backward term is the coverage/recall direction: every '
             'GT Gaussian must have a nearby prediction. >1 upweights coverage to fight '
             'mode-collapse and push the model to represent every GT Gaussian somewhere — '
             'a set-level lever (no per-cell/ordering constraint). Overridable live via '
             'overrides.yaml. No-op when recon_loss=mse.',
    )
    parser.add_argument(
        '--mse_hybrid_weight',
        type=float,
        default=0.0,
        help='Lambda on an index-aligned MSE term added on top of a Chamfer recon_loss '
             '(Chamfer+MSE hybrid; no-op for recon_loss=mse or 0). Chamfer is permutation-'
             'degenerate so it gives no gradient toward copying the clean low-noise input; '
             'this MSE term supplies it, dragging the low-noise loss down and steepening '
             'loss-vs-t. Under --permute_atlas it rewards equivariant copying (target is the '
             'permuted x_start), so it is compatible with the permutation augmentation. '
             'Overridable live via overrides.yaml. Start small (~0.05-0.2).',
    )
    parser.add_argument(
        '--mse_hybrid_warmup_steps',
        type=int,
        default=0,
        help='If >0, linearly DECAY the index-MSE hybrid term from --mse_hybrid_weight (at step 0) '
             'to 0 (at this step), then keep it 0. This is the cold-start BOOTSTRAP for '
             'optimal-assignment recon (sinkhorn_patch): index-MSE binds cell->target to break the '
             'symmetric mean-collapse trap that stalls sinkhorn from scratch, then fades out leaving '
             'pure assignment. Use --mse_hybrid_weight ~1.0 as the starting magnitude. Overrides any '
             'live mse_hybrid_weight (governs the term entirely while >0). 0 = disabled.',
    )
    parser.add_argument(
        '--mse_hybrid_lownoise_mult',
        type=float,
        default=1.0,
        help='Linearly UPWEIGHT the index-MSE hybrid term toward low noise: per-sample weight '
             'w(t) = 1 + (mult-1)*t_value, i.e. w=1 at the noisy end (t=0) ramping to w=mult at '
             'the clean end (t=1). mult=1 = uniform across t (back-compat). Unlike a t**p downweight '
             'this keeps the t=0 baseline and AMPLIFIES clean (effective clean weight = '
             'mse_hybrid_weight*mult). Pairs with --mse_hybrid_weight / --mse_hybrid_warmup_steps.',
    )
    parser.add_argument(
        '--sinkhorn_epsilon',
        type=float,
        default=0.05,
        help='Entropic-OT regularizer for --recon_loss=sinkhorn_patch (the optimal-assignment '
             'EMD via log-Sinkhorn). Controls hard<->soft: smaller => sharper plan closer to an '
             'exact within-patch bijection (lower collision, stiffer gradient); larger => softer / '
             'blurrier plan that re-collapses toward the mean. No-op for other recon_loss modes. '
             'Overridable live via overrides.yaml.',
    )
    parser.add_argument(
        '--sinkhorn_iters',
        type=int,
        default=50,
        help='Fixed number of log-domain Sinkhorn iterations for --recon_loss=sinkhorn_patch. '
             'Fixed (not hot-reloadable) so the loss stays torch.compile-safe (changing it forces '
             'a recompile). No-op for other recon_loss modes.',
    )
    parser.add_argument(
        '--compile_sinkhorn',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='torch.compile the log-Sinkhorn loop (sinkhorn_patch / sinkhorn_patch_hard). The '
             'eager loop is memory-bound (materializes the (B,nP,M,M) broadcast to HBM each '
             'logsumexp); Inductor fuses it for ~8x on the Sinkhorn term (numerically identical '
             'fp32), measured ~2x whole-step throughput. One-time ~1min compile at first step; '
             'shapes are static (drop_last=True). No-op for non-sinkhorn recon modes.',
    )
    parser.add_argument(
        '--permute_atlas',
        type=str,
        default='none',
        choices=['none', 'patch', 'global'],
        help='Randomize the per-Gaussian ordering the model sees each step (Chamfer '
             'recon_loss ONLY; raises on mse). Fixes the iterative-sampling collapse where '
             'the model, trained only on the canonical sphere2plane ordering, sees its own '
             'non-canonical output fed back at sampling time (OOD). "patch" permutes within '
             'each patch_size² patch (keeps patch-level pos-embed meaningful — targets local '
             'permutation); "global" permutes all 16384 points (also voids the pos-embed). '
             'Fresh permutation per step, ~sub-ms cost. See diffusion.permute_atlas.',
    )

    # Logging / Checkpoints / Validation
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--grad_norm_log_every_n_prints', type=int, default=1,
                        help='Log per-loss gradient norms every N prints (1 = every print, '
                             '0 = disabled). Overridable via overrides.yaml.')
    parser.add_argument('--ckpt_every', type=int, default=10000)
    parser.add_argument('--val_every', type=int, default=0,
                        help='Steps between validation renders (0 = disabled)')
    parser.add_argument('--val_sampling_steps', type=int, default=50,
                        help='Number of sampling steps for validation generation')
    parser.add_argument('--val_sampler', type=str, default='heun',
                        choices=SAMPLER_CHOICES,
                        help='Sampler used for validation generation')
    parser.add_argument('--dpm_solver_order', type=int, default=2, choices=[1, 2, 3],
                        help='Diffusers DPM solver order')
    parser.add_argument('--dpm_algorithm_type', type=str, default='dpmsolver++',
                        choices=['dpmsolver', 'dpmsolver++', 'sde-dpmsolver', 'sde-dpmsolver++'],
                        help='Diffusers DPM algorithm variant')
    parser.add_argument('--dpm_solver_type', type=str, default='midpoint',
                        choices=['midpoint', 'heun'],
                        help='Diffusers DPM solver type')
    parser.add_argument('--dpm_timestep_spacing', type=str, default='trailing',
                        choices=['linspace', 'leading', 'trailing'],
                        help='Diffusers timestep spacing for DPM sampling')
    parser.add_argument('--dpm_use_karras_sigmas', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable Karras sigmas in the diffusers DPM scheduler')
    parser.add_argument('--ddim_eta', type=float, default=0.0,
                        help='DDIM eta: 0.0 = deterministic, 1.0 ≈ DDPM. Only used when --val_sampler ddim')
    parser.add_argument('--val_cfg_scale', type=float, default=1.0,
                        help='Classifier-free guidance scale for validation sampling (1.0 = disabled). '
                             'Requires class_dropout_prob > 0 during training.')
    parser.add_argument('--val_grid_enabled', action=argparse.BooleanOptionalAction, default=True,
                        help='In addition to the single-random-sample validation render, also produce a '
                             'fixed-seed `val_grid_rows x val_grid_cols` class grid every val_every steps. '
                             'Requires val_sampler in {heun,euler}.')
    parser.add_argument('--val_grid_rows', type=int, default=4,
                        help='Rows in the fixed-seed validation grid.')
    parser.add_argument('--val_grid_cols', type=int, default=4,
                        help='Cols in the fixed-seed validation grid.')
    parser.add_argument('--val_grid_seed', type=int, default=1234,
                        help='Base seed for the fixed-seed validation grid. Per-tile noise is seeded '
                             'with (val_grid_seed + tile_idx), so a given (seed, tile_idx) pair is stable '
                             'across runs.')
    parser.add_argument('--val_grid_camera_idx', type=int, default=0,
                        help='Camera index (into ref_camera_tar) used for every tile in the validation '
                             'grid. Held fixed so cross-step diffs reflect model changes, not camera changes.')
    parser.add_argument('--results_dir', type=str, default='output/dit_results')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='YAML file with hyperparameters; merged before CLI (CLI overrides).',
    )

    # Weights & Biases logging
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable wandb logging on rank-0. --no-wandb to disable.')
    parser.add_argument('--wandb_project', type=str, default='3dgen-jit',
                        help='wandb project name.')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='wandb entity (team or user). Defaults to WANDB_ENTITY env / wandb default.')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='wandb run name. Defaults to basename(results_dir).')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=('online', 'offline', 'disabled'),
                        help='wandb mode. "disabled" matches --no-wandb.')
    parser.add_argument('--wandb_tags', type=str, default='',
                        help='Comma-separated tags applied to the wandb run.')
    parser.add_argument('--wandb_group', type=str, default=None,
                        help='Optional wandb group (useful for sweep aggregation).')
    parser.add_argument('--wandb_log_images', action=argparse.BooleanOptionalAction, default=True,
                        help='Upload validation PNGs to wandb (in addition to saving locally).')
    return parser


def _merge_yaml_into_parser_defaults(parser: argparse.ArgumentParser, config_path: str) -> None:
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return
    if not isinstance(cfg, dict):
        raise ValueError("--config must contain a YAML mapping at the top level")
    allowed = {a.dest for a in parser._actions if a.dest not in ('help', 'config')}
    merged: dict[str, Any] = {}
    for k, v in cfg.items():
        if k not in allowed:
            logger.warning("Ignoring unknown config key: %s", k)
            continue
        if v is None:
            continue
        merged[k] = v
    parser.set_defaults(**merged)


if __name__ == '__main__':
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    parser = build_train_gsplat_parser()
    if pre_args.config:
        _merge_yaml_into_parser_defaults(parser, pre_args.config)
    args = parser.parse_args()
    main(args)
