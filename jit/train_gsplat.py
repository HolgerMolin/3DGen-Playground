"""
Training script for JiT-style large-patch diffusion on 3DGS data (class-conditional).
3DGS data (16384 points x 59 features) on 128x128 grid is the latent space directly — no VAE needed.

Single-GPU:  python jit/train_gsplat.py --obj_list ... --gs_path ...
Multi-GPU:   accelerate launch [--num_processes N] jit/train_gsplat.py --obj_list ... --gs_path ...
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

from dataloaders.standard_3dgen_loader import Standard3DGenDataset
from dataloaders.class_3dgen_loader import (
    Class3DGenDataset, DC_ONLY_FEATURE_INDICES, FULL_3DGS_FEATURE_DIM,
)
from jit.models import JiT_3DGS_models
from jit.diffusion import create_diffusion
from jit.sampling import SAMPLER_CHOICES, resolve_sampling_shape, sample_model
from utils.plane_utils import load_sphere2plane, plane_to_point_cloud
from utils.loss_tracker import LossTracker
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


def _build_class_balanced_sampler(dataset, seed: int, rank: int):
    """Build a WeightedRandomSampler with inverse class-frequency weights.

    Each rank gets its own generator (seed + rank) so draws are independent
    across processes. Replacement is True so rare classes can appear in most
    batches; `num_samples` matches dataset length to keep epoch cadence.
    """
    if not hasattr(dataset, "valid_labels"):
        raise ValueError(
            "class_balanced_sampler requires Class3DGenDataset "
            "(needs .valid_labels); got " + type(dataset).__name__
        )
    labels = np.asarray(dataset.valid_labels, dtype=np.int64)
    counts = np.bincount(labels)
    class_weights = np.zeros_like(counts, dtype=np.float64)
    nonzero = counts > 0
    class_weights[nonzero] = 1.0 / counts[nonzero]
    sample_weights = class_weights[labels]
    g = torch.Generator()
    g.manual_seed(int(seed) + int(rank))
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
        generator=g,
    )


#################################################################################
#                    Hot-reload training overrides (YAML)                      #
#################################################################################

_OVERRIDABLE_KEYS = frozenset({
    "lr_scale",
    "max_grad_norm",
    "render_loss_weight",
    "alpha_mask_loss_weight",
    "lpips_loss_weight",
    "aux_classifier_weight",
    "null_repel_weight",
    "P_mean",
    "grad_norm_log_every_n_prints",
})


@dataclass
class TrainRuntimeOverrides:
    """Mutable hyperparameters; optionally synced from overrides.yaml on a fixed step interval."""

    lr_scale: float
    max_grad_norm: float
    render_loss_weight: float
    alpha_mask_loss_weight: float
    lpips_loss_weight: float
    aux_classifier_weight: float
    null_repel_weight: float
    P_mean: float
    grad_norm_log_every_n_prints: float  # float so overrides YAML can write it; cast to int on use

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainRuntimeOverrides":
        return cls(
            lr_scale=1.0,
            max_grad_norm=float(args.max_grad_norm),
            render_loss_weight=float(args.render_loss_weight),
            alpha_mask_loss_weight=float(args.alpha_mask_loss_weight),
            lpips_loss_weight=float(args.lpips_loss_weight),
            aux_classifier_weight=float(args.aux_classifier_weight),
            null_repel_weight=float(args.null_repel_weight),
            P_mean=float(args.P_mean),
            grad_norm_log_every_n_prints=float(args.grad_norm_log_every_n_prints),
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
#                        Class-from-null repulsion loss                         #
#################################################################################

def _compute_null_repel_loss(
    embedding_table: torch.Tensor,
    num_classes: int,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push class embeddings away from the CFG null token on the cosine sphere.

    ``embedding_table`` is ``LabelEmbedder.embedding_table.weight`` — shape
    ``(num_classes + 1, D)`` with the null embedding at row ``num_classes``.

    Returns ``(repel_loss, class_null_cos_mean)``:

      * ``repel_loss`` — hinge-squared on ``cos(e_c, stopgrad(e_null))`` above
        ``margin``, averaged over classes. The null row is detached so only
        class rows receive gradient; null continues to learn from its normal
        CFG-dropout training signal.
      * ``class_null_cos_mean`` — the raw (pre-hinge) mean cosine between
        classes and null, detached for diagnostic logging. This is the
        geometric quantity the regularizer moves; watch it drop over training.
    """
    null_vec = embedding_table[num_classes].detach().unsqueeze(0)  # (1, D), stopgrad
    class_vecs = embedding_table[:num_classes]                      # (C, D)
    cos = F.cosine_similarity(class_vecs, null_vec.expand_as(class_vecs), dim=-1)
    hinge = (cos - margin).clamp(min=0.0)
    repel_loss = (hinge * hinge).mean()
    return repel_loss, cos.detach().mean()


#################################################################################
#                          Rendering Loss Helpers                               #
#################################################################################

def _sample_jit_timesteps(
    batch_size: int,
    num_timesteps: int,
    device: torch.device,
    p_mean: float,
    p_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample JiT-style logit-normal timesteps.

    Returns ``(t_value, t_discrete)``:

      * ``t_value`` is continuous in ``(0, 1)`` and drives the flow-matching
        interpolation ``x_t = t · x_0 + (1 − t) · ε`` (``t=0`` → noise,
        ``t=1`` → clean — matches ``jit.sampling._jit_velocity_from_xstart``).
      * ``t_discrete`` is the integer timestep fed to the model's timestep
        embedding, using the same ``round(t_value · (T-1))`` mapping as the
        heun/euler samplers so training and inference share a grid.
    """
    probs = torch.sigmoid(torch.randn(batch_size, device=device) * p_std + p_mean)
    # Keep t_value strictly inside (0, 1) to avoid the (1 - t) singularity in
    # the velocity used by heun/euler at the clean end.
    eps = 1e-4
    t_value = probs.clamp(min=eps, max=1.0 - eps)
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
    y: torch.Tensor,
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
        model_out_debug = model(x_t_debug, t, y)
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
            "label": int(y[pos].detach().cpu().item()),
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
        "labels": [int(v) for v in y.detach().cpu().tolist()],
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
    num_classes: int,
    in_channels: int,
    diffusion_num_timesteps: int,
    device: torch.device,
    t_value: float = 0.3,
    batch_size: int = 8,
    seed: int = 0,
    num_class_probes: int = 16,
    num_class_pairs: int = 16,
    return_per_class: bool = False,
) -> dict:
    """Probe class conditioning at a fixed t on a fixed random batch.

    Computes per-class, as fractions of ``‖pred(y=k)‖_RMS``:
      - ``cfg_signal_k`` = ``‖pred(x_t, y=k) − pred(x_t, null)‖_RMS``
    and per-pair:
      - ``class_signal_(a,b)`` = ``‖pred(x_t, y=a) − pred(x_t, y=b)‖_RMS``

    Returns the **mean** over a random sample of ``num_class_probes`` classes
    (cfg_signal) and ``num_class_pairs`` random class pairs (class_signal),
    plus min/max for spread. The same noise sample ``x_t`` is reused across
    all conditional/null forwards, so the differences are class-only.

    ``cfg_signal < 0.01`` → conditioning collapsed, CFG is a no-op.
    ``class_signal ≈ 0`` with non-zero ``cfg_signal`` → model uses "some class
    vs null" but doesn't discriminate between classes.

    Uses a fixed RNG seed so the numbers are comparable across checkpoints.
    """
    model_was_training = model.training
    model.eval()
    g = torch.Generator(device=device).manual_seed(int(seed))
    g_cpu = torch.Generator(device="cpu").manual_seed(int(seed) + 1)

    shape = resolve_sampling_shape(model=model, batch_size=batch_size, in_channels=in_channels)
    x_t = torch.randn(*shape, generator=g, device=device)
    t_disc = torch.full(
        (batch_size,),
        int(round(t_value * (diffusion_num_timesteps - 1))),
        dtype=torch.long, device=device,
    )

    K = max(1, min(num_class_probes, num_classes))
    class_pool = torch.randperm(num_classes, generator=g_cpu)[:K].tolist()

    eps = 1e-8
    y_null = torch.full((batch_size,), num_classes, dtype=torch.long, device=device)
    pred_null = model(x_t, t_disc, y_null).float()

    cond_preds: dict[int, torch.Tensor] = {}
    cond_norms: dict[int, float] = {}
    cfg_signals: dict[int, float] = {}
    for cls in class_pool:
        y = torch.full((batch_size,), int(cls), dtype=torch.long, device=device)
        pred_c = model(x_t, t_disc, y).float()
        norm_c = float(pred_c.square().mean().sqrt().item())
        cfg_c = float((pred_c - pred_null).square().mean().sqrt().item()) / (norm_c + eps)
        cond_preds[int(cls)] = pred_c
        cond_norms[int(cls)] = norm_c
        cfg_signals[int(cls)] = cfg_c

    class_signals: list[float] = []
    if K >= 2 and num_class_pairs > 0:
        max_pairs = min(num_class_pairs, K * (K - 1) // 2)
        seen: set[tuple[int, int]] = set()
        attempts = 0
        while len(class_signals) < max_pairs and attempts < 20 * max_pairs:
            attempts += 1
            i = int(torch.randint(0, K, (1,), generator=g_cpu).item())
            j = int(torch.randint(0, K, (1,), generator=g_cpu).item())
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            a, b = class_pool[i], class_pool[j]
            pa, pb = cond_preds[a], cond_preds[b]
            denom = 0.5 * (cond_norms[a] + cond_norms[b]) + eps
            cs = float((pa - pb).square().mean().sqrt().item()) / denom
            class_signals.append(cs)

    cfg_vals = list(cfg_signals.values())
    cfg_mean = sum(cfg_vals) / len(cfg_vals) if cfg_vals else 0.0
    cfg_min = min(cfg_vals) if cfg_vals else 0.0
    cfg_max = max(cfg_vals) if cfg_vals else 0.0
    class_mean = sum(class_signals) / len(class_signals) if class_signals else 0.0
    class_min = min(class_signals) if class_signals else 0.0
    class_max = max(class_signals) if class_signals else 0.0
    pred_rms_mean = sum(cond_norms.values()) / len(cond_norms) if cond_norms else 0.0

    if model_was_training:
        model.train()

    result = {
        "cfg_signal": cfg_mean,
        "cfg_signal_min": cfg_min,
        "cfg_signal_max": cfg_max,
        "class_signal": class_mean,
        "class_signal_min": class_min,
        "class_signal_max": class_max,
        "pred_rms": pred_rms_mean,
        "num_class_probes": K,
        "num_class_pairs": len(class_signals),
    }
    if return_per_class:
        result["cfg_signal_per_class"] = cfg_signals
    return result


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
    num_classes: int,
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
    """Generate a validation sample, render it, and save the result."""
    y_label = random.randrange(num_classes)
    y = torch.tensor([y_label], dtype=torch.long, device=device)

    shape = resolve_sampling_shape(model=model, batch_size=1, in_channels=in_channels)
    sample = sample_model(
        sampler=val_sampler,
        model=model,
        shape=shape,
        class_labels=y,
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
    out_path = os.path.join(val_dir, f"epoch_{epoch:03d}_step_{step:07d}_class{y_label:03d}.png")
    Image.fromarray(img_uint8).save(out_path)
    logger.info(
        "[validation] saved: %s (class=%d, sampler=%s, steps=%d)",
        out_path,
        y_label,
        val_sampler,
        val_sampling_steps,
    )



#################################################################################
#                             EMA Utilities                                     #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Update EMA model parameters. `model` should be the unwrapped model."""
    for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(model_p.data, alpha=1 - decay)


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

    # Load class map
    if is_main:
        logger.info(f"Loading class map from {args.class_map}")
    with open(args.class_map, 'r') as f:
        class_map = json.load(f)
    num_classes = max(v for v in class_map.values() if v >= 0) + 1
    if is_main:
        logger.info(f"Number of classes: {num_classes}")

    # Create base dataset
    if is_main:
        logger.info("Creating base dataset...")
    base_dataset = Standard3DGenDataset(
        obj_list=[args.obj_list],
        gs_path=args.gs_path,
        caption_path=None,
        mean_file=args.mean_file,
        std_file=args.std_file,
        sphere2plane_path=args.sphere2plane_path,
        exclude_keys_file=args.exclude_keys_file,
        rank_transform_file=args.rank_transform_file,
    )

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

    # Wrap with class-conditional dataset
    dataset = Class3DGenDataset(
        base_dataset, class_map,
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

    # Overfit mode: restrict dataset to first N samples
    if args.overfit > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(args.overfit, len(dataset))))
        args.log_every = 1
        if is_main:
            logger.info(f"[overfit] Restricting to {len(dataset)} samples, log_every forced to 1")

    # DataLoader — accelerate will inject DistributedSampler automatically
    overfitting = args.overfit > 0
    balanced_sampler = None
    if args.class_balanced_sampler and not overfitting:
        balanced_sampler = _build_class_balanced_sampler(
            dataset, seed=args.seed, rank=accelerator.process_index,
        )
        if is_main:
            label_counts = np.bincount(np.asarray(dataset.valid_labels, dtype=np.int64))
            nz = int((label_counts > 0).sum())
            logger.info(
                "[class-balanced] sampling with inverse-frequency weights over %d classes "
                "(min=%d, max=%d, mean=%.1f samples/class)",
                nz, int(label_counts[label_counts > 0].min()), int(label_counts.max()),
                float(label_counts[label_counts > 0].mean()),
            )
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=(not overfitting) and balanced_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=not overfitting,
    )
    if balanced_sampler is not None:
        loader_kwargs["sampler"] = balanced_sampler
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
        num_classes=num_classes,
        class_dropout_prob=args.class_dropout_prob,
        learn_sigma=False,
        gradient_checkpointing=args.gradient_checkpointing,
        bottleneck=args.bottleneck,
        aux_classifier=args.aux_classifier,
        label_embed_init_std=args.label_embed_init_std,
    )
    if is_main:
        logger.info(
            "[patch-embed] %s",
            "bottleneck (proj1→bottleneck_dim→proj2)" if args.bottleneck
            else "single-conv (in_chans→embed_dim, no rank reduction below embed_dim)",
        )
    if is_main and args.aux_classifier:
        logger.info(
            "[aux-classifier] enabled: linear head → %d classes, initial weight=%.4g, "
            "label_embed_init_std=%.3g",
            num_classes, args.aux_classifier_weight, args.label_embed_init_std,
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
            logger.info("torch.compile: enabled (mode=default)")
        model = torch.compile(model)

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
        logger.info(
            "Training mode: DDPM (sampler=%s), timestep sampling: sigmoid(N(%.3f, %.3f)) "
            "mapped to discrete steps [0, %d]",
            args.val_sampler,
            args.P_mean,
            args.P_std,
            diffusion.num_timesteps - 1,
        )

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

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

    # Render / validation setup (per-process; gsplat renderer is local)
    renderer_for_train = None
    lpips_fn_for_train = None
    train_cameras = None
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
            train_cameras = _prepare_train_cameras(ref_cameras, args.train_render_size, device)
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
        # strict=False so older checkpoints without aux_classifier keys still load; we log
        # any mismatches so accidental architecture drift doesn't pass silently.
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

    # Training
    model.train()
    step = start_step
    # Accumulate as GPU scalar tensors; .item() is deferred to the log block to
    # avoid forcing a GPU–CPU sync (and DDP barrier) on every training step.
    log_loss = torch.zeros([], device=device)
    log_render_l1 = torch.zeros([], device=device)
    log_render_alpha_l1 = torch.zeros([], device=device)
    log_render_lpips = torch.zeros([], device=device)
    log_aux_loss = torch.zeros([], device=device)
    log_repel_loss = torch.zeros([], device=device)
    log_class_null_cos = torch.zeros([], device=device)
    log_grad_norm = 0.0
    log_grad_steps = 0
    log_steps = 0  # micro-batches contributing to running loss averages
    log_optim_steps = 0  # optim steps in window (used for steps_per_sec)
    log_mse_gn = 0.0
    log_rl1_gn = 0.0
    log_alpha_gn = 0.0
    log_lpips_gn = 0.0
    log_aux_gn = 0.0
    log_repel_gn = 0.0
    log_per_loss_gn_steps = 0
    log_t_bucket_sum = torch.zeros(num_t_buckets, device=device)
    log_t_bucket_cnt = torch.zeros(num_t_buckets, device=device)
    start_time = time.time()
    last_lr_val: Optional[float] = None
    # Pre-allocated zero tensors used as no-op sentinels for render losses when
    # render loss is disabled or hasn't warmed up yet.  Avoids 3 GPU allocations
    # + kernel launches every step for the common case (render loss disabled).
    _zero_render_loss = torch.zeros([], dtype=torch.float32, device=device)
    _zero_aux_loss = torch.zeros([], dtype=torch.float32, device=device)

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
                    eff_lr = lr_val * float(runtime.lr_scale)
                    for pg in opt.param_groups:
                        pg['lr'] = eff_lr
                    last_lr_val = eff_lr

            if has_full_for_render:
                x, y, x_full, hash_keys = batch
            else:
                x, y, hash_keys = batch
                x_full = None
            y = y.long()  # (B,)
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
                    x.shape[0], diffusion.num_timesteps, device, runtime.P_mean, args.P_std
                )
                noise = torch.randn_like(x)

                # Pre-sample the CFG drop mask ourselves so the aux classifier
                # can skip dropped rows (their class label has been replaced by
                # the unconditional slot). When the aux head is off we still
                # pass force_drop_ids=None and let LabelEmbedder resample
                # internally to preserve the original stochastic behavior.
                unwrapped_model = accelerator.unwrap_model(model)
                use_aux_head = unwrapped_model.aux_classifier is not None
                if use_aux_head and args.class_dropout_prob > 0.0:
                    drop_mask = (
                        torch.rand(x.shape[0], device=device) < args.class_dropout_prob
                    )
                    force_drop_ids = drop_mask.long()
                    model_kwargs = dict(y=y, force_drop_ids=force_drop_ids)
                else:
                    drop_mask = None
                    model_kwargs = dict(y=y)

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
                )
                sample_losses = loss_dict["loss"]
                mse_loss = sample_losses.mean()
                x0_pred = loss_dict.get("pred_xstart")

                # Aux classification: cross-entropy over un-dropped rows only.
                # The trunk's pooled logits were stashed on the unwrapped model
                # during forward (see DiT.forward); reading them here avoids
                # changing the diffusion API's forward return contract.
                aux_loss = _zero_aux_loss
                if use_aux_head:
                    aux_logits = unwrapped_model._aux_logits
                    if aux_logits is not None:
                        if drop_mask is not None:
                            keep = ~drop_mask
                            n_keep = int(keep.sum().item())
                            if n_keep > 0:
                                aux_loss = F.cross_entropy(aux_logits[keep], y[keep])
                        else:
                            aux_loss = F.cross_entropy(aux_logits, y)

                if not torch.isfinite(mse_loss):
                    _debug_nonfinite_mse(
                        args=args,
                        diffusion=diffusion,
                        model=model,
                        x=x,
                        y=y,
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
                    model_out = model(x_t, t, y)
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
                    noise_cutoff = float(args.render_loss_noise_cutoff)
                    if noise_cutoff > 0.0:
                        keep_mask = (t_value >= noise_cutoff)
                        n_kept = int(keep_mask.sum().item())
                        if n_kept < x.shape[0] and is_main and step % args.log_every == 0:
                            logger.info(
                                "[render-loss] step=%d kept %d/%d samples (t_value >= %.2f)",
                                step, n_kept, x.shape[0], noise_cutoff,
                            )
                        if n_kept == 0:
                            should_compute_render = False
                        else:
                            x0_pred_render = x0_pred[keep_mask]
                            x_gt_render = x_gt_for_render[keep_mask]

                    if should_compute_render:
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
                        preview_y = y[preview_slice]

                        if x0_pred is not None:
                            preview_x0_pred = x0_pred.detach()[preview_slice]
                        else:
                            preview_x = x[preview_slice]
                            noise_for_preview = torch.randn_like(preview_x)
                            x_t_preview = diffusion.flow_matching_q_sample(
                                preview_x, preview_t_value, noise=noise_for_preview
                            )
                            with torch.no_grad():
                                model_out_preview = model(x_t_preview, preview_t, preview_y)
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
                            labels=preview_y,
                            device=device,
                            num_cam=args.train_render_log_num_cam,
                            dc_only=dc_only,
                            plane_to_sphere=plane_to_sphere,
                            rank_transform_tables=rank_transform_tables,
                        )
                        loss_tracker.flush_plots()
                    accelerator.wait_for_everyone()

                # Class-from-null repulsion: cosine hinge on class vs null
                # embedding. Pure additive scalar; stopgrad on null row inside
                # helper. Computed every step so the diagnostic mean cosine is
                # always logged, even when the weight is 0. No-op when CFG
                # dropout is disabled (no null row exists).
                if args.class_dropout_prob > 0.0:
                    repel_loss, class_null_cos_mean = _compute_null_repel_loss(
                        unwrapped_model.y_embedder.embedding_table.weight,
                        num_classes,
                        float(args.null_repel_margin),
                    )
                else:
                    repel_loss = torch.zeros([], device=device)
                    class_null_cos_mean = torch.zeros([], device=device)

                total_loss = (
                    mse_loss
                    + float(runtime.render_loss_weight) * render_l1_loss
                    + float(runtime.alpha_mask_loss_weight) * render_alpha_l1_loss
                    + float(runtime.lpips_loss_weight) * render_lpips_loss
                    + float(runtime.aux_classifier_weight) * aux_loss
                    + float(runtime.null_repel_weight) * repel_loss
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
                                "aux": (aux_loss, float(runtime.aux_classifier_weight)),
                                "repel": (repel_loss, float(runtime.null_repel_weight)),
                            },
                        )
                    log_mse_gn += _per_loss_norms.get("mse", 0.0)
                    log_rl1_gn += _per_loss_norms.get("render_l1", 0.0)
                    log_alpha_gn += _per_loss_norms.get("alpha_l1", 0.0)
                    log_lpips_gn += _per_loss_norms.get("lpips", 0.0)
                    log_aux_gn += _per_loss_norms.get("aux", 0.0)
                    log_repel_gn += _per_loss_norms.get("repel", 0.0)
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
                    if is_main and torch.isfinite(grad_norm):
                        log_grad_norm += grad_norm.item()
                        log_grad_steps += 1
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
            log_render_l1 += render_l1_loss.detach()
            log_render_alpha_l1 += render_alpha_l1_loss.detach()
            log_render_lpips += render_lpips_loss.detach()
            log_aux_loss += aux_loss.detach()
            log_repel_loss += repel_loss.detach()
            log_class_null_cos += class_null_cos_mean
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
                if p_mean_schedule is not None:
                    msg += f" | P_mean: {runtime.P_mean:+.3f}"
                if _any_render_loss_weight(runtime):
                    avg_rl1 = log_render_l1.item() / log_steps
                    avg_alpha_rl1 = log_render_alpha_l1.item() / log_steps
                    avg_rlpips = log_render_lpips.item() / log_steps
                    msg += (
                        f" | Render_L1: {avg_rl1:.4f}"
                        f" | Alpha_L1: {avg_alpha_rl1:.4f}"
                        f" | Render_LPIPS: {avg_rlpips:.4f}"
                    )
                aux_active = use_aux_head and float(runtime.aux_classifier_weight) > 0.0
                avg_aux = (log_aux_loss.item() / log_steps) if aux_active else None
                if aux_active:
                    msg += f" | Aux: {avg_aux:.4f}"
                # Always log the class-to-null cosine diagnostic when CFG dropout
                # is on (null exists); log repel loss only when the regularizer
                # is active.
                repel_active = float(runtime.null_repel_weight) > 0.0
                null_exists = args.class_dropout_prob > 0.0
                avg_repel = (log_repel_loss.item() / log_steps) if repel_active else None
                avg_class_null_cos = (
                    (log_class_null_cos.item() / log_steps) if null_exists else None
                )
                if repel_active:
                    msg += f" | Repel: {avg_repel:.4f}"
                if avg_class_null_cos is not None:
                    msg += f" | cos(c,null): {avg_class_null_cos:+.4f}"
                if log_grad_steps > 0:
                    msg += f" | GradNorm: {log_grad_norm / log_grad_steps:.4f}"
                if log_per_loss_gn_steps > 0:
                    n = log_per_loss_gn_steps
                    msg += f" | GN[mse]: {log_mse_gn / n:.4f}"
                    if log_rl1_gn > 0.0:
                        msg += f" | GN[rl1]: {log_rl1_gn / n:.4f}"
                    if log_alpha_gn > 0.0:
                        msg += f" | GN[alpha]: {log_alpha_gn / n:.4f}"
                    if log_lpips_gn > 0.0:
                        msg += f" | GN[lpips]: {log_lpips_gn / n:.4f}"
                    if log_aux_gn > 0.0:
                        msg += f" | GN[aux]: {log_aux_gn / n:.4f}"
                    if log_repel_gn > 0.0:
                        msg += f" | GN[repel]: {log_repel_gn / n:.4f}"
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

                # ──── TEMP: per-print conditioning-signal probe ─────────────
                # Mirrors the [cond] line that normally prints at checkpoint
                # time; moved up to give granular feedback while tuning the
                # null-repel regularizer. Runs 17 extra forward passes per
                # invocation; gated to every 1000 steps to keep overhead <0.5%.
                # To remove: delete this entire block (keep markers for grep).
                if args.class_dropout_prob > 0 and step % 1000 == 0:
                    _cond_sig_print = _measure_conditioning_signal(
                        model=accelerator.unwrap_model(model),
                        num_classes=num_classes,
                        in_channels=in_channels,
                        diffusion_num_timesteps=diffusion.num_timesteps,
                        device=device,
                        t_value=0.3,
                        batch_size=8,
                        seed=args.seed,
                    )
                    logger.info(
                        "[cond] cfg_signal=%.4f (min=%.4f max=%.4f, n=%d) | "
                        "class_signal=%.4f (min=%.4f max=%.4f, n=%d) | pred_rms=%.4f",
                        _cond_sig_print["cfg_signal"],
                        _cond_sig_print["cfg_signal_min"],
                        _cond_sig_print["cfg_signal_max"],
                        _cond_sig_print["num_class_probes"],
                        _cond_sig_print["class_signal"],
                        _cond_sig_print["class_signal_min"],
                        _cond_sig_print["class_signal_max"],
                        _cond_sig_print["num_class_pairs"],
                        _cond_sig_print["pred_rms"],
                    )
                # ──── TEMP end ───────────────────────────────────────────────

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
                        "aux": (log_aux_gn / n) if log_aux_gn > 0.0 else None,
                        "repel": (log_repel_gn / n) if log_repel_gn > 0.0 else None,
                    }
                loss_tracker.record(
                    step=step,
                    mse=avg_loss,
                    render_l1=(log_render_l1.item() / log_steps) if render_active else None,
                    alpha_l1=(log_render_alpha_l1.item() / log_steps) if render_active else None,
                    lpips=(log_render_lpips.item() / log_steps) if render_active else None,
                    aux=avg_aux,
                    repel=avg_repel,
                    class_null_cos=avg_class_null_cos,
                    grad_norm=(log_grad_norm / log_grad_steps) if log_grad_steps > 0 else None,
                    grad_norm_per_loss=grad_norm_per_loss,
                    lr=last_lr_val,
                    p_mean=runtime.P_mean,
                    steps_per_sec=steps_per_sec,
                    bucket_means=bucket_avg,
                    bucket_counts=[int(h) for h in bucket_hits],
                )

                log_loss.zero_()
                log_render_l1.zero_()
                log_render_alpha_l1.zero_()
                log_render_lpips.zero_()
                log_aux_loss.zero_()
                log_repel_loss.zero_()
                log_class_null_cos.zero_()
                log_grad_norm = 0.0
                log_grad_steps = 0
                log_steps = 0
                log_optim_steps = 0
                log_mse_gn = 0.0
                log_rl1_gn = 0.0
                log_alpha_gn = 0.0
                log_lpips_gn = 0.0
                log_aux_gn = 0.0
                log_repel_gn = 0.0
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
                    if args.class_dropout_prob > 0:
                        sig = _measure_conditioning_signal(
                            model=accelerator.unwrap_model(model),
                            num_classes=num_classes,
                            in_channels=in_channels,
                            diffusion_num_timesteps=diffusion.num_timesteps,
                            device=device,
                            t_value=0.3,
                            batch_size=8,
                            seed=args.seed,
                        )
                        logger.info(
                            "[cond] cfg_signal=%.4f (min=%.4f max=%.4f, n=%d) | "
                            "class_signal=%.4f (min=%.4f max=%.4f, n=%d) | pred_rms=%.4f",
                            sig["cfg_signal"], sig["cfg_signal_min"], sig["cfg_signal_max"],
                            sig["num_class_probes"],
                            sig["class_signal"], sig["class_signal_min"], sig["class_signal_max"],
                            sig["num_class_pairs"], sig["pred_rms"],
                        )
                accelerator.wait_for_everyone()

            validation_due = (
                enable_val and accelerator.sync_gradients and step % args.val_every == 0
            )
            if validation_due:
                accelerator.wait_for_everyone()
                if is_main:
                    _run_validation_render(
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
                        num_classes=num_classes,
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
                accelerator.wait_for_everyone()

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
        help='Label dropout probability for classifier-free guidance (LabelEmbedder)',
    )
    parser.add_argument(
        '--class_balanced_sampler',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Sample with inverse class-frequency weights (WeightedRandomSampler) '
             'to counter class imbalance. Ignored in --overfit mode.',
    )
    parser.add_argument(
        '--aux_classifier',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Attach a linear classification head on top of mean-pooled transformer tokens. '
             'Trained with cross-entropy on un-dropped class labels; gives the transformer trunk '
             'direct class-discriminative signal alongside the flow-matching MSE.',
    )
    parser.add_argument(
        '--aux_classifier_weight',
        type=float,
        default=0.0,
        help='Weight on the auxiliary cross-entropy classification loss. Overridable via overrides.yaml.',
    )
    parser.add_argument(
        '--label_embed_init_std',
        type=float,
        default=0.02,
        help='Init std for LabelEmbedder.embedding_table. Default 0.02 matches the original DiT recipe; '
             'increase (e.g. 0.1) to amplify class conditioning at init.',
    )
    parser.add_argument(
        '--null_repel_weight',
        type=float,
        default=0.0,
        help='Weight on the class-from-null cosine repulsion regularizer. '
             'Pushes class embeddings away from the CFG null token to strengthen '
             'classifier-free guidance. 0.0 (default) = disabled; override via '
             'overrides.yaml for hot-reload fine-tuning.',
    )
    parser.add_argument(
        '--null_repel_margin',
        type=float,
        default=0.5,
        help='Cosine margin for the null repulsion hinge. No penalty when '
             'cos(e_c, e_null) <= margin; quadratic penalty above. Static (set at '
             'start of training); not hot-reloadable.',
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
    parser.add_argument('--class_map', type=str, default='object_labels/object_to_class.json',
                        help='Path to object-to-class mapping JSON')
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
    parser.add_argument('--ref_camera_tar', type=str, default='/home/tiangexiang/gen3d/ref_camera.tar.gz',
                        help='Path to reference camera tar.gz for render loss')
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
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
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
