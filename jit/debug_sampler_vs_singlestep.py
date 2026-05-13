"""Debug whether bad samples are caused by the sampler trajectory or by the
model's per-step x0 predictions.

For N random objects from the dataset (no held-out split exists in this repo,
just any N), conditioned on each object's true class label:

  1. Partial sampling: noise the GT atlas to t_start in {0.1, 0.3, 0.5} via the
     trainer's flow_matching_q_sample, then run heun from t_start → 1. Three
     partial-full trajectories per object — they all target the same GT, so
     deviations between them and the GT isolate sampler-trajectory drift.
  2. Single-step x0 prediction: noise GT to each t in {0.9..0.1} and run one
     model forward to get its raw x0 (no Euler step, no v-prediction).

Outputs (in --output_dir):

  * One 5x5 grid PNG per object: trajectory snapshots for each warm-start ts,
    GT, and the 9 single-step renders.
  * Per-channel histograms across all N objects comparing GT vs
    partial@{0.1, 0.3, 0.5} vs single@{0.9, 0.5, 0.1}, in both standardized
    and physical-units space (post unstandardization, post sigmoid/exp).
  * Summary CSV with max and 99th-percentile per channel in physical units.
  * v_per_group.png: ||v||_RMS at each Heun step, one subplot per channel
    group (positions / scales / rotations / opacity / SH DC), one line per
    (object, t_start).

Reuses, without reimplementation:

  * Noising:    GaussianDiffusion.flow_matching_q_sample
  * Sampling:   jit.sampling._jit_heun_step / _jit_euler_step (partial loop)
  * Forward:    JiT model under bf16 autocast, eval(), no_grad
  * Rendering:  utils.gsplat_render_util._render_gsplat_batch and helpers
  * Dataset:    Standard3DGenDataset + Class3DGenDataset (sh_degree0_only)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
GS_ROOT = os.path.join(REPO_ROOT, "submodules", "gaussian-splatting")
if GS_ROOT not in sys.path:
    sys.path.insert(0, GS_ROOT)

from dataloaders.class_3dgen_loader import (
    Class3DGenDataset,
    Standard3DGenDataset,
    DC_ONLY_FEATURE_INDICES,
    FULL_3DGS_FEATURE_DIM,
)
from jit.diffusion import create_diffusion
from jit.models import JiT_3DGS_models
from jit.sampling import resolve_sampling_shape
from utils.plane_utils import load_sphere2plane
from utils.gsplat_render_util import (
    RENDER_OPACITY_RAW_MAX,
    RENDER_OPACITY_RAW_MIN,
    RENDER_SCALE_RAW_MAX,
    RENDER_SCALE_RAW_MIN,
    _denormalize_point_cloud,
    _load_reference_cameras,
    _plane_to_point_cloud_batch,
    _point_clouds_to_gsplat_inputs,
    _prepare_train_cameras,
    _render_gsplat_batch,
    _try_import_renderer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("debug_sampler")

# Channel layout produced by Class3DGenDataset under sh_degree0_only.
# DC_ONLY_FEATURE_INDICES = (0,1,2, 3, 4,20,36, 52,53,54, 55,56,57,58)
# After feature selection, atlas channels are:
CHANNEL_NAMES = [
    "x", "y", "z",          # 0..2
    "opacity_raw",          # 3       → sigmoid for physical
    "dc_r", "dc_g", "dc_b", # 4..6    → raw SH DC, no transform
    "scale_x", "scale_y", "scale_z",  # 7..9 → exp for physical
    "quat_w", "quat_x", "quat_y", "quat_z",  # 10..13
]
OPACITY_IDX = 3
SCALE_IDXS = (7, 8, 9)
QUAT_IDXS = (10, 11, 12, 13)

# Channel groups for the per-step ||v|| plot.
DIAG_GROUPS: dict[str, tuple[int, ...]] = {
    "positions": (0, 1, 2),
    "scales": (7, 8, 9),
    "rotations": (10, 11, 12, 13),
    "opacity": (3,),
    "sh_dc": (4, 5, 6),
}
DIAG_GROUP_ORDER: list[str] = ["positions", "scales", "rotations", "opacity", "sh_dc"]


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _ckpt_arg(ckpt: dict, key: str, default=None):
    return (ckpt.get("args") or {}).get(key, default)


def _load_model_from_ckpt(
    ckpt: dict,
    *,
    in_channels: int,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    model_name = _ckpt_arg(ckpt, "model", "JiT-B/8")
    bottleneck = bool(_ckpt_arg(ckpt, "bottleneck", False))
    aux_classifier = bool(_ckpt_arg(ckpt, "aux_classifier", False))
    label_embed_init_std = float(_ckpt_arg(ckpt, "label_embed_init_std", 0.02))
    class_dropout_prob = float(_ckpt_arg(ckpt, "class_dropout_prob", 0.1))

    model = JiT_3DGS_models[model_name](
        input_size=128,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=class_dropout_prob,
        learn_sigma=False,
        bottleneck=bottleneck,
        aux_classifier=aux_classifier,
        label_embed_init_std=label_embed_init_std,
    ).to(device)

    if "ema" not in ckpt:
        raise RuntimeError("Checkpoint has no 'ema' key — cannot load latest EMA weights")
    model.load_state_dict(ckpt["ema"])
    model.eval()
    logger.info("Loaded EMA weights from step %s of %s", ckpt.get("step"), model_name)
    return model


def _physical_atlas(atlas_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Unstandardize and apply sigmoid/exp where the renderer would.

    Input shape (B, C, H, W) standardized, output same shape but in physical units.
    Quaternions are returned normalized (renderer's identity-fallback).
    """
    flat = atlas_norm.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
    expand = (1,) * (flat.ndim - 1) + (flat.shape[-1],)
    raw = flat * (std.view(expand) + 1e-8) + mean.view(expand)

    out = raw.clone()
    out[..., OPACITY_IDX] = torch.sigmoid(
        raw[..., OPACITY_IDX].clamp(RENDER_OPACITY_RAW_MIN, RENDER_OPACITY_RAW_MAX)
    )
    for s in SCALE_IDXS:
        out[..., s] = torch.exp(raw[..., s].clamp(RENDER_SCALE_RAW_MIN, RENDER_SCALE_RAW_MAX))
    quats = raw[..., QUAT_IDXS[0]:QUAT_IDXS[-1] + 1]
    norms = quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    out[..., QUAT_IDXS[0]:QUAT_IDXS[-1] + 1] = quats / norms

    return out.permute(0, 3, 1, 2).contiguous()  # back to (B, C, H, W)


def _render_atlas(
    atlas_norm: torch.Tensor,           # (1, C, H, W) standardized
    *,
    plane_to_sphere: torch.Tensor,
    norm_mean: Optional[torch.Tensor],
    norm_std: Optional[torch.Tensor],
    train_cameras: dict,
    renderer,
    cam_indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Run the same plane→pc→denorm→gsplat path validation uses."""
    pc_norm = _plane_to_point_cloud_batch(atlas_norm.float(), plane_to_sphere)
    pc_raw = _denormalize_point_cloud(pc_norm, norm_mean, norm_std)
    gaussians = _point_clouds_to_gsplat_inputs(pc_raw.to(device), dc_only=True, detach_input=True)
    rendered = _render_gsplat_batch(renderer, gaussians, train_cameras, cam_indices, device)
    if rendered.ndim == 5:
        rendered = rendered[0]  # (num_cam, C, H, W)
    return rendered


def _to_uint8_chw(img_chw: torch.Tensor) -> np.ndarray:
    arr = img_chw.permute(1, 2, 0).clamp(0.0, 1.0).float().cpu().numpy()
    return (arr * 255.0).astype(np.uint8)


def _make_grid(
    tiles: list[Optional[np.ndarray]],
    labels: list[Optional[str]],
    *,
    n_cols: int,
    pad: int = 4,
) -> np.ndarray:
    """Stack tiles into a (n_rows × n_cols) grid with per-tile label bands.

    Pass ``None`` in ``tiles`` (and matching ``None`` in ``labels``) for blank
    cells used as padding when the cell count isn't a perfect multiple of
    ``n_cols``.
    """
    sample = next(t for t in tiles if t is not None)
    h, w, c = sample.shape
    label_band = 16
    cell_h = h + label_band
    n_rows = (len(tiles) + n_cols - 1) // n_cols
    out_h = n_rows * cell_h + (n_rows - 1) * pad
    out_w = n_cols * w + (n_cols - 1) * pad
    out = np.full((out_h, out_w, c), 255, dtype=np.uint8)

    for i, tile in enumerate(tiles):
        if tile is None:
            continue
        r, col = divmod(i, n_cols)
        y0 = r * (cell_h + pad)
        x0 = col * (w + pad)
        out[y0 + label_band:y0 + label_band + h, x0:x0 + w] = tile

    img = Image.fromarray(out)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    for i, label in enumerate(labels):
        if label is None:
            continue
        r, col = divmod(i, n_cols)
        y0 = r * (cell_h + pad)
        x0 = col * (w + pad)
        draw.text((x0 + 2, y0 + 2), label, fill=(0, 0, 0))
    return np.asarray(img)


def _build_timesteps(
    *,
    t_start: float,
    end_t: float,
    n_partial: int,
    schedule: str,
    device: torch.device,
) -> torch.Tensor:
    """Build inference timesteps from t_start → end_t with n_partial intervals.

    schedule="linear":       evenly spaced in t.
    schedule="logit_normal": evenly spaced in the quantile of the
        sigmoid(N(0,1)) distribution that the trainer samples from
        (jit/train_gsplat.py:432). Density is high near t=0.5 and low at the
        endpoints — matches training-time t density.
    """
    if schedule == "linear":
        return torch.linspace(t_start, end_t, n_partial + 1, device=device, dtype=torch.float32)
    if schedule == "logit_normal":
        eps = 1e-6
        def t_to_q(t: float) -> float:
            t_clamped = max(eps, min(1.0 - eps, t))
            z = math.log(t_clamped / (1.0 - t_clamped))
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        q_start = t_to_q(t_start)
        q_end = t_to_q(end_t)
        quantiles = torch.linspace(
            q_start, q_end, n_partial + 1, device=device, dtype=torch.float32,
        ).clamp(eps, 1.0 - eps)
        z = torch.erfinv(2 * quantiles - 1) * math.sqrt(2.0)
        return torch.sigmoid(z)
    raise ValueError(f"Unknown timestep schedule: {schedule!r}")


@torch.no_grad()
def _partial_heun_with_v_log(
    *,
    model: torch.nn.Module,
    x_t_start: torch.Tensor,         # (1, C, H, W) at t = t_start
    t_start: float,
    class_labels: torch.Tensor,
    full_num_steps: int,             # total steps a from-noise run would use
    diffusion_steps: int,
    device: torch.device,
    t_eps: float = 0.05,
    n_snapshots: int = 5,
    obj_pos: int,
    euler_only: bool = False,
    truncate_at: Optional[float] = None,
    noise_inject: float = 0.0,
    timestep_schedule: str = "linear",
    noise_seed: Optional[int] = None,
) -> tuple[list[tuple[float, torch.Tensor]], list[tuple[float, torch.Tensor]], list[dict]]:
    """Heun from x_{t_start} to x_1, returning equally-spaced iterate
    snapshots, x0_pred snapshots taken at the same (t, sample) points, and a
    log of ||v||_RMS per channel group at each step.

    Mirrors ``jit.sampling._jit_heun_step``/``_jit_euler_step`` (cfg_scale=1)
    but inlines the x0-prediction so we can read out the velocity actually
    used in each update (Heun average for non-final steps, Euler v_t for the
    last step).

    The x0_pred snapshots answer test 1 of the integrator-vs-OOD diagnosis:
    if x0_pred stays clean at every iterate, the bug is in the integrator;
    if x0_pred itself develops spikes as the iterate drifts, the model is
    collapsing on OOD inputs.
    """
    end_t = 1.0 if truncate_at is None else float(truncate_at)
    n_partial = max(1, round((end_t - t_start) * full_num_steps))
    timesteps = _build_timesteps(
        t_start=t_start,
        end_t=end_t,
        n_partial=n_partial,
        schedule=timestep_schedule,
        device=device,
    )

    inject_gen: Optional[torch.Generator] = None
    if noise_inject > 0.0 and noise_seed is not None:
        inject_gen = torch.Generator(device=device).manual_seed(int(noise_seed))

    raw = np.linspace(0, n_partial, n_snapshots).round().astype(int).tolist()
    snap_set = set(int(i) for i in raw)
    snapshots: list[tuple[float, torch.Tensor]] = []
    v_rows: list[dict] = []

    sample = x_t_start.float()
    if 0 in snap_set:
        snapshots.append((float(timesteps[0].item()), sample.clone()))

    model_dtype = next(model.parameters()).dtype

    def _x0_pred(x: torch.Tensor, t_val: torch.Tensor) -> torch.Tensor:
        t_disc = (t_val * (diffusion_steps - 1)).round().clamp(0, diffusion_steps - 1).long()
        t_batch = t_disc.expand(x.shape[0]).to(device=x.device)
        return model(x.to(dtype=model_dtype), t_batch, class_labels).float()

    for i in range(n_partial):
        t_val = timesteps[i]
        t_next = timesteps[i + 1]
        is_last = (i == n_partial - 1)

        x0_t = _x0_pred(sample, t_val)
        denom_t = (1.0 - t_val).clamp_min(t_eps)
        v_t = (x0_t - sample) / denom_t
        step = (t_next - t_val)

        if is_last or euler_only:
            v_used = v_t
        else:
            sample_euler = sample + step * v_t
            x0_t_next = _x0_pred(sample_euler, t_next)
            denom_t_next = (1.0 - t_next).clamp_min(t_eps)
            v_t_next = (x0_t_next - sample_euler) / denom_t_next
            v_used = 0.5 * (v_t + v_t_next)

        t_scalar = float(t_val.item())
        for grp_name in DIAG_GROUP_ORDER:
            idxs = list(DIAG_GROUPS[grp_name])
            v_rms = float(v_used[:, idxs].float().pow(2).mean().sqrt().item())
            v_rows.append({
                "object_pos": obj_pos,
                "ts": float(t_start),
                "t": t_scalar,
                "group": grp_name,
                "v_rms": v_rms,
            })

        sample = sample + step * v_used
        # Stochastic / Langevin step: inject fresh Gaussian noise scaled by
        # sqrt(|step|) so the per-time-unit variance is constant. Disabled on
        # the last step (we don't want noise on the final emit). Helps drift
        # off the noise-perturbed-GT manifold rejoin it.
        if not is_last and noise_inject > 0.0:
            noise = torch.randn(
                sample.shape, device=sample.device, dtype=sample.dtype, generator=inject_gen,
            )
            sample = sample + float(noise_inject) * torch.sqrt(step.abs()) * noise
        if (i + 1) in snap_set:
            snapshots.append((float(t_next.item()), sample.clone()))

    # Post-hoc: x0_pred at each snapshot. One extra fwd per snapshot (~5 per
    # trajectory) — negligible vs the ~2 * n_partial fwds in the loop.
    x0_snapshots: list[tuple[float, torch.Tensor]] = []
    for t_at, sample_at in snapshots:
        t_tensor = torch.tensor([t_at], device=device, dtype=torch.float32)
        x0_pred = _x0_pred(sample_at, t_tensor)
        x0_snapshots.append((t_at, x0_pred.float()))

    # Test 5: replace the final iterate with x0_pred(sample, truncate_at).
    # The integrator is allowed to drift up to t=truncate_at, then we snap.
    if truncate_at is not None and snapshots:
        t_at_last, _ = snapshots[-1]
        snap_x0 = x0_snapshots[-1][1]
        snapshots[-1] = (t_at_last, snap_x0.clone())

    return snapshots, x0_snapshots, v_rows


def _plot_v_per_group(
    v_rows: list[dict],
    *,
    t_starts: list[float],
    out_dir: Path,
) -> None:
    if not v_rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as e:
        logger.error("matplotlib import failed: %s; skipping v plot", e)
        return

    palette = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    ts_colors = {ts: palette[i % len(palette)] for i, ts in enumerate(sorted(t_starts))}
    obj_positions = sorted({r["object_pos"] for r in v_rows})

    n_groups = len(DIAG_GROUP_ORDER)
    fig, axes = plt.subplots(1, n_groups, figsize=(4.0 * n_groups, 4.0), squeeze=False)
    axes = axes.ravel()
    for ax, grp_name in zip(axes, DIAG_GROUP_ORDER):
        for obj_pos in obj_positions:
            for ts in t_starts:
                sub = [
                    r for r in v_rows
                    if r["group"] == grp_name and r["object_pos"] == obj_pos and r["ts"] == ts
                ]
                sub.sort(key=lambda r: r["t"])
                if not sub:
                    continue
                xs = [r["t"] for r in sub]
                ys = [r["v_rms"] for r in sub]
                ax.plot(xs, ys, color=ts_colors[ts], alpha=0.5, linewidth=0.9)
        ax.set_title(grp_name)
        ax.set_xlabel("t")
        ax.set_ylabel("||v||_RMS")
        ax.grid(True, alpha=0.3)
    legend_handles = [
        Line2D([0], [0], color=ts_colors[ts], label=f"t_start={ts:.1f}")
        for ts in sorted(t_starts)
    ]
    fig.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(0.99, 0.02))
    fig.suptitle("||v||_RMS at each Heun step — per channel group")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = out_dir / "v_per_group.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    logger.info("Saved %s", out_path.name)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("gsplat rendering requires CUDA")
    _seed_all(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cargs = ckpt.get("args") or {}

    # Resolve required paths from checkpoint args (CLI can override).
    obj_list = args.obj_list or cargs.get("obj_list")
    gs_path = args.gs_path or cargs.get("gs_path")
    mean_file = args.mean_file or cargs.get("mean_file")
    std_file = args.std_file or cargs.get("std_file")
    sphere2plane_path = args.sphere2plane_path or cargs.get("sphere2plane_path")
    ref_camera_tar = args.ref_camera_tar or cargs.get("ref_camera_tar")
    class_map_path = args.class_map or cargs.get("class_map")
    sh_degree0_only = bool(cargs.get("sh_degree0_only", True))
    predict_xstart = bool(cargs.get("predict_xstart", True))
    noise_schedule = str(cargs.get("noise_schedule", "squaredcos_cap_v2"))
    diffusion_steps = int(cargs.get("diffusion_steps", 1000))

    if not sh_degree0_only:
        raise NotImplementedError(
            "This debug script targets sh_degree0_only checkpoints (14-channel atlas)."
        )

    feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
    in_channels = len(DC_ONLY_FEATURE_INDICES)

    # ── Class map / num_classes ──────────────────────────────────────────────
    with open(class_map_path, "r") as f:
        class_map = json.load(f)
    num_classes = max(v for v in class_map.values() if v >= 0) + 1
    logger.info("num_classes=%d, in_channels=%d", num_classes, in_channels)

    # ── Dataset ──────────────────────────────────────────────────────────────
    base = Standard3DGenDataset(
        obj_list=[obj_list],
        gs_path=gs_path,
        caption_path=None,
        mean_file=mean_file,
        std_file=std_file,
        sphere2plane_path=sphere2plane_path,
    )
    dataset = Class3DGenDataset(
        base,
        class_map,
        feature_indices=feature_indices,
        return_full_for_render=False,
        preload_to_cpu=False,
        lazy_cache_to_cpu=False,
    )
    logger.info("Dataset size: %d", len(dataset))

    # ── Normalization stats (selected to 14 dims) ────────────────────────────
    norm_mean_full = torch.load(mean_file, weights_only=True).float().cpu()
    norm_std_full = torch.load(std_file, weights_only=True).float().cpu()
    norm_mean = norm_mean_full[feature_indices]
    norm_std = norm_std_full[feature_indices]

    # ── Sphere↔plane permutation ─────────────────────────────────────────────
    plane_to_sphere = load_sphere2plane(sphere2plane_path, 128 * 128)

    # ── Model ────────────────────────────────────────────────────────────────
    model = _load_model_from_ckpt(
        ckpt, in_channels=in_channels, num_classes=num_classes, device=device,
    )

    # ── Diffusion (only used here for flow_matching_q_sample) ────────────────
    diffusion = create_diffusion(
        timestep_respacing="",
        noise_schedule=noise_schedule,
        learn_sigma=False,
        predict_xstart=predict_xstart,
        diffusion_steps=diffusion_steps,
    )

    # ── Renderer ─────────────────────────────────────────────────────────────
    renderer = _try_import_renderer()
    if isinstance(renderer, Exception):
        raise RuntimeError(f"gsplat import failed: {renderer}") from renderer
    ref_cameras = _load_reference_cameras(ref_camera_tar)
    train_cameras = _prepare_train_cameras(ref_cameras, args.render_size, device)
    num_cams_available = int(train_cameras["viewmats"].shape[0])
    cam_indices = [args.camera_index % num_cams_available]
    logger.info(
        "Renderer ready (size=%d, fixed cam_index=%d of %d)",
        args.render_size, cam_indices[0], num_cams_available,
    )

    # ── Pick N objects ───────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), args.num_objects)
    logger.info("Selected %d object indices (seed=%d)", len(indices), args.seed)

    # ── Output dir ───────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grids_dir = out_dir / "grids"
    grids_dir.mkdir(exist_ok=True)

    t_values = list(args.t_values)              # single-step probe ts: 0.9..0.1
    t_starts = list(args.partial_t_starts)       # partial-full warm-start ts: 0.1, 0.3, 0.5
    n_snapshots = int(args.num_partial_snapshots)

    flat_hw = 128 * 128
    gt_pool = []
    partial_pool = {ts: [] for ts in t_starts}   # final-state only, for histograms
    single_pool = {t: [] for t in t_values}
    all_v_rows: list[dict] = []

    # validate sample shape (fails early if model/in_channels mismatch)
    resolve_sampling_shape(model=model, batch_size=1, in_channels=in_channels)

    for obj_pos, ds_idx in enumerate(indices):
        x_gt, y_label, hash_key = dataset[ds_idx]   # (C,H,W), int, str
        x_gt = x_gt.float().to(device).unsqueeze(0)  # (1,C,H,W)
        y = torch.tensor([int(y_label)], dtype=torch.long, device=device)

        # ── Partial-full trajectories: warm-start from GT noised to t_start ──
        # snapshots[ts]    = list of (t, iterate)   along the trajectory.
        # x0_snapshots[ts] = list of (t, x0_pred)   at the same (t, iterate).
        partial_snapshots: dict[float, list[tuple[float, torch.Tensor]]] = {}
        partial_x0_snapshots: dict[float, list[tuple[float, torch.Tensor]]] = {}
        for ts_idx, t_start in enumerate(t_starts):
            noise_gen = torch.Generator(device=device).manual_seed(
                args.seed * 100003 + obj_pos * 7919 + ts_idx + 1
            )
            noise = torch.randn(x_gt.shape, device=device, dtype=torch.float32, generator=noise_gen)
            t_start_t = torch.tensor([t_start], device=device, dtype=torch.float32)
            x_t_start = diffusion.flow_matching_q_sample(x_gt, t_start_t, noise=noise)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                snaps, x0_snaps, v_rows = _partial_heun_with_v_log(
                    model=model,
                    x_t_start=x_t_start,
                    t_start=float(t_start),
                    class_labels=y,
                    full_num_steps=args.num_inference_steps,
                    diffusion_steps=diffusion_steps,
                    device=device,
                    n_snapshots=n_snapshots,
                    obj_pos=obj_pos,
                    euler_only=bool(args.euler_only),
                    truncate_at=args.truncate_at,
                    noise_inject=float(args.noise_inject),
                    timestep_schedule=str(args.timestep_schedule),
                    noise_seed=args.seed * 100003 + obj_pos * 41 + ts_idx + 1,
                )
            # store sample as float on device for rendering, last snapshot is x_1
            partial_snapshots[t_start] = [(t, s.float()) for (t, s) in snaps]
            partial_x0_snapshots[t_start] = [(t, x.float()) for (t, x) in x0_snaps]
            all_v_rows.extend(v_rows)

        # ── Single-step preds at each t ──────────────────────────────────────
        single_preds = {}
        for t_idx, t in enumerate(t_values):
            noise_gen = torch.Generator(device=device).manual_seed(
                args.seed * 100003 + obj_pos * 1009 + t_idx + 1
            )
            noise = torch.randn(x_gt.shape, device=device, dtype=torch.float32, generator=noise_gen)
            t_value = torch.tensor([t], device=device, dtype=torch.float32)
            x_t = diffusion.flow_matching_q_sample(x_gt, t_value, noise=noise)
            t_discrete = (t_value * (diffusion_steps - 1)).round().clamp(0, diffusion_steps - 1).long()
            t_batch = t_discrete.expand(1)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(x_t.to(dtype=next(model.parameters()).dtype), t_batch, y).float()
            single_preds[t] = pred

        # ── Render everything (one fixed view for alignment) ─────────────────
        def _r(atlas):
            return _render_atlas(
                atlas,
                plane_to_sphere=plane_to_sphere,
                norm_mean=norm_mean,
                norm_std=norm_std,
                train_cameras=train_cameras,
                renderer=renderer,
                cam_indices=cam_indices,
                device=device,
            )[0]

        # ── Render all snapshots + GT + single-step probes (one fixed view) ──
        with torch.no_grad():
            tile_gt = _to_uint8_chw(_r(x_gt))
            traj_tiles: dict[float, list[tuple[float, np.ndarray]]] = {}
            x0_tiles: dict[float, list[tuple[float, np.ndarray]]] = {}
            for ts in t_starts:
                traj_tiles[ts] = [
                    (t_at, _to_uint8_chw(_r(sample)))
                    for t_at, sample in partial_snapshots[ts]
                ]
                x0_tiles[ts] = [
                    (t_at, _to_uint8_chw(_r(x0)))
                    for t_at, x0 in partial_x0_snapshots[ts]
                ]
            single_tiles = [_to_uint8_chw(_r(single_preds[t])) for t in t_values]

        # ── n_snapshots × n_rows grid ────────────────────────────────────────
        # n_cols = n_snapshots (5 by default).
        # Per t_start, two stacked rows: iterate (top), x0_pred (bottom).
        #   "iter" row = sample along the trajectory at the snapshot t.
        #   "x0"   row = model's x0 prediction at that same (t, sample).
        # If x0 looks clean while iter has spikes ⇒ integrator bug.
        # If x0 itself is spiky                  ⇒ model collapses on OOD input.
        # Last rows: GT followed by 9 single-step probes, packed L-to-R, padded.
        n_cols = n_snapshots
        ordered_tiles: list[Optional[np.ndarray]] = []
        ordered_labels: list[Optional[str]] = []
        for ts in t_starts:
            iter_row = traj_tiles[ts]
            x0_row = x0_tiles[ts]
            for k in range(n_cols):
                if k < len(iter_row):
                    t_at, tile = iter_row[k]
                    ordered_tiles.append(tile)
                    ordered_labels.append(f"ts{ts:.1f} t={t_at:.2f} iter")
                else:
                    ordered_tiles.append(None)
                    ordered_labels.append(None)
            for k in range(n_cols):
                if k < len(x0_row):
                    t_at, tile = x0_row[k]
                    ordered_tiles.append(tile)
                    ordered_labels.append(f"ts{ts:.1f} t={t_at:.2f} x0_pred")
                else:
                    ordered_tiles.append(None)
                    ordered_labels.append(None)

        bottom_tiles = [tile_gt, *single_tiles]
        bottom_labels = ["GT"] + [f"single t={t:.1f}" for t in t_values]
        for tile, lab in zip(bottom_tiles, bottom_labels):
            ordered_tiles.append(tile)
            ordered_labels.append(lab)
        while len(ordered_tiles) % n_cols != 0:
            ordered_tiles.append(None)
            ordered_labels.append(None)

        grid = _make_grid(ordered_tiles, ordered_labels, n_cols=n_cols)
        out_path = grids_dir / f"obj{obj_pos:02d}_idx{ds_idx}_class{int(y_label):03d}_{hash_key}.png"
        Image.fromarray(grid).save(out_path)

        # ── Pool standardized samples for histograms (final state of each
        # trajectory only — snapshots are visual aid; histograms compare the
        # converged x_1 against GT and single-step probes). ─────────────────
        gt_pool.append(x_gt.detach().cpu().reshape(in_channels, flat_hw))
        for ts in t_starts:
            x_final = partial_snapshots[ts][-1][1]
            partial_pool[ts].append(
                x_final.detach().cpu().reshape(in_channels, flat_hw)
            )
        for t in t_values:
            single_pool[t].append(
                single_preds[t].detach().cpu().reshape(in_channels, flat_hw)
            )

        logger.info(
            "[%d/%d] idx=%d class=%d → %s",
            obj_pos + 1, len(indices), ds_idx, int(y_label), out_path.name,
        )

    # ── Build histograms ─────────────────────────────────────────────────────
    # Stack into (N*flat_hw,) per channel for each category.
    def _stack(pool_list):
        # pool_list: list of (C, flat_hw) tensors
        return torch.stack(pool_list, dim=0)  # (N, C, flat_hw)

    gt_std = _stack(gt_pool)
    partial_std = {ts: _stack(partial_pool[ts]) for ts in t_starts}
    single_std = {t: _stack(single_pool[t]) for t in t_values}

    # Convert to physical-unit space (apply unstandardize + sigmoid/exp).
    def _to_physical(stacked: torch.Tensor) -> torch.Tensor:
        atlas = stacked.view(stacked.shape[0], in_channels, 128, 128)
        phys = _physical_atlas(atlas, norm_mean, norm_std)
        return phys.reshape(stacked.shape[0], in_channels, flat_hw)

    gt_phys = _to_physical(gt_std)
    partial_phys = {ts: _to_physical(partial_std[ts]) for ts in t_starts}
    single_phys = {t: _to_physical(single_std[t]) for t in t_values}

    # Plot per-channel histograms.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.error("matplotlib import failed: %s; skipping histograms", e)
        plt = None

    hist_t_keys = [t for t in (0.9, 0.5, 0.1) if t in t_values]
    partial_colors = ["tab:blue", "tab:cyan", "tab:purple"]
    single_colors = ["tab:orange", "tab:green", "tab:red"]
    if plt is not None:
        for unit_name, gt_arr, partial_arr, single_arr in [
            ("standardized", gt_std, partial_std, single_std),
            ("physical", gt_phys, partial_phys, single_phys),
        ]:
            fig, axes = plt.subplots(in_channels, 1, figsize=(10, 2.0 * in_channels))
            if in_channels == 1:
                axes = [axes]
            for c in range(in_channels):
                ax = axes[c]
                series = [("GT", gt_arr[:, c].flatten().numpy(), "black")]
                for color, ts in zip(partial_colors, t_starts):
                    series.append((
                        f"partial t_start={ts:.1f}",
                        partial_arr[ts][:, c].flatten().numpy(),
                        color,
                    ))
                for color, t in zip(single_colors, hist_t_keys):
                    series.append((
                        f"single t={t:.1f}",
                        single_arr[t][:, c].flatten().numpy(),
                        color,
                    ))

                # Robust shared range: 0.5–99.5 percentile across categories.
                concat = np.concatenate([s[1] for s in series])
                lo, hi = np.percentile(concat, [0.5, 99.5])
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo, hi = float(concat.min()), float(concat.max() + 1e-6)
                bins = np.linspace(lo, hi, 80)
                for label, data, color in series:
                    ax.hist(
                        np.clip(data, lo, hi),
                        bins=bins, density=True, histtype="step",
                        label=label, color=color, linewidth=1.1,
                    )
                ax.set_title(f"ch{c} — {CHANNEL_NAMES[c]} ({unit_name})")
                ax.set_yscale("log")
                ax.legend(fontsize=7, loc="upper right")
            fig.tight_layout()
            fig.savefig(out_dir / f"histograms_{unit_name}.png", dpi=110)
            plt.close(fig)
            logger.info("Saved histograms_%s.png", unit_name)

    # ── Summary CSV: max + p99 per channel in physical units ─────────────────
    def _stats(arr_NCflat):
        flat = arr_NCflat.reshape(arr_NCflat.shape[0] * arr_NCflat.shape[2], in_channels) \
            if arr_NCflat.ndim == 3 else arr_NCflat
        # arr is (N, C, K) → flatten to (N*K, C) for per-channel stats
        flat = arr_NCflat.permute(0, 2, 1).reshape(-1, in_channels).numpy()
        p99 = np.percentile(flat, 99.0, axis=0)
        mx = flat.max(axis=0)
        return p99, mx

    gt_p99, gt_max = _stats(gt_phys)
    partial_stats = {ts: _stats(partial_phys[ts]) for ts in t_starts}
    single_stats = {t: _stats(single_phys[t]) for t in hist_t_keys}

    csv_path = out_dir / "summary_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["channel", "name", "gt_p99", "gt_max"]
        for ts in t_starts:
            header += [f"partial_ts{ts:.1f}_p99", f"partial_ts{ts:.1f}_max"]
        for t in hist_t_keys:
            header += [f"single_t{t:.1f}_p99", f"single_t{t:.1f}_max"]
        w.writerow(header)
        for c in range(in_channels):
            row = [c, CHANNEL_NAMES[c], f"{gt_p99[c]:.4g}", f"{gt_max[c]:.4g}"]
            for ts in t_starts:
                p99v, mxv = partial_stats[ts]
                row += [f"{p99v[c]:.4g}", f"{mxv[c]:.4g}"]
            for t in hist_t_keys:
                p99v, mxv = single_stats[t]
                row += [f"{p99v[c]:.4g}", f"{mxv[c]:.4g}"]
            w.writerow(row)
    logger.info("Saved %s", csv_path)

    # Pretty-print scale + opacity tail summary to stdout.
    focus = [OPACITY_IDX, *SCALE_IDXS]
    print("\nTail summary (physical units) — focus channels:")
    print(f"{'ch':<3} {'name':<10} {'src':<22} {'p99':>10} {'max':>10}")
    for c in focus:
        rows = [("GT", gt_p99[c], gt_max[c])]
        for ts in t_starts:
            p99v, mxv = partial_stats[ts]
            rows.append((f"partial t_start={ts:.1f}", p99v[c], mxv[c]))
        for t in hist_t_keys:
            p99v, mxv = single_stats[t]
            rows.append((f"single t={t:.1f}", p99v[c], mxv[c]))
        for src_name, p99v, mxv in rows:
            print(f"{c:<3} {CHANNEL_NAMES[c]:<10} {src_name:<22} {p99v:>10.4g} {mxv:>10.4g}")
        print()

    # ── ||v||-per-group plot ─────────────────────────────────────────────────
    _plot_v_per_group(all_v_rows, t_starts=t_starts, out_dir=out_dir)

    logger.info("Done. Outputs in: %s", out_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--output_dir", default="output/debug_sampler_vs_singlestep")
    p.add_argument("--num_objects", type=int, default=5)
    p.add_argument("--num_inference_steps", type=int, default=100,
                   help="heun denoising steps (matches val_sampling_steps default)")
    p.add_argument("--render_size", type=int, default=128,
                   help="Atlas-native render resolution; do not upsample")
    p.add_argument("--camera_index", type=int, default=0,
                   help="Fixed camera index used for all objects (deterministic alignment)")
    p.add_argument("--t_values", type=float, nargs="+",
                   default=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p.add_argument("--partial_t_starts", type=float, nargs="+",
                   default=[0.1, 0.3, 0.5],
                   help="Warm-start t for partial-full heun trajectories")
    p.add_argument("--num_partial_snapshots", type=int, default=5,
                   help="Number of (t, sample) snapshots saved per partial "
                        "trajectory; first is x_{t_start}, last is x_1. "
                        "Also sets the grid column count.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--euler_only", action="store_true",
                   help="Skip Heun corrector; use first-order Euler at every "
                        "step (test 4: isolates the corrector's contribution "
                        "to drift)")
    p.add_argument("--truncate_at", type=float, default=None,
                   help="If set (e.g. 0.95), integrate only over [t_start, "
                        "truncate_at] and emit x0_pred(sample, truncate_at) as "
                        "the final state (test 5: avoids the unstable tail "
                        "where the model collapses on drifted iterates)")
    p.add_argument("--noise_inject", type=float, default=0.0,
                   help="If > 0, after each non-final step add "
                        "noise_inject * sqrt(|dt|) * randn (Langevin / SDE "
                        "stochasticity). Helps drifted iterates rejoin the "
                        "training manifold. Try 0.05–0.2.")
    p.add_argument("--timestep_schedule", choices=["linear", "logit_normal"],
                   default="linear",
                   help="Inference timestep spacing. 'logit_normal' matches "
                        "the trainer's t-density (sigmoid(N(0,1))), placing "
                        "more steps near t=0.5 and fewer at the endpoints.")

    # Path overrides — default to checkpoint args.
    p.add_argument("--obj_list", default=None)
    p.add_argument("--gs_path", default=None)
    p.add_argument("--mean_file", default=None)
    p.add_argument("--std_file", default=None)
    p.add_argument("--sphere2plane_path", default=None)
    p.add_argument("--ref_camera_tar", default=None)
    p.add_argument("--class_map", default=None)
    return p


if __name__ == "__main__":
    main(_build_parser().parse_args())
