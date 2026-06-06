#!/usr/bin/env python
"""Per-CHANNEL Sinkhorn difficulty probe — which GT *channels* (and which atlas
*pixels*) does the assignment loss the model was actually trained on stay worst on?

This is the per-channel decomposition of ``sinkhorn_atlas_probe.py``. That probe sums
the matched cost over channels and asks "which GT cell is hard"; here we keep the cost
*resolved per channel* and ask "which GT channel is consistently hard, and where on the
atlas". It is faithful to ``recon_loss=sinkhorn_patch_hard`` (the loss this run trained
on), NOT index-MSE:

  sq[i,j]   = ‖pred_i − gt_j‖²                     (sum over channels, within a P×P patch)
  sq_h      = huber(sq, huber_delta)               (the run's robust ground cost)
  Π[i,j]    = log-Sinkhorn(sq_h, eps, iters)       (entropic-OT plan)
  σ(i)      = argmax_j Π[i,j]                       (HARD assignment, the trained matching)
  loss      = mean_i sq_h[i, σ(i)]                  (DETR-style hard MSE to the single match)

identical to jit/diffusion/gaussian_diffusion.py::_chamfer_recon_loss(sinkhorn_patch_hard).
The HARD assignment σ is computed from the exact huber'd Sinkhorn plan with the run's
knobs, so the matching is the trained one. We then decompose the *matched residual* per
channel (in the model's normalized feature space, where every DC channel is ~unit-variance,
so channels are directly comparable):

  res_c[i, c] = (pred_i,c − gt_σ(i),c)²            Σ_c res_c[i,c] = sq[i,σ(i)] (raw, pre-huber)

and localize it to the GT cell σ(i) it was matched to (so "GT pixel j is hard" = the
residual of whichever prediction got assigned to GT cell j). Near-bijective (collision
~3% at iters=100), so this is well-defined; cells that receive 2 preds are averaged.

Outputs (FM convention, see CLAUDE.md: t=0 noise → t=1 clean):
  * perchannel_cost_atlas.png — 14 channel rows × t cols, per-panel autoscale (each panel
    its own colorbar, like sinkhorn_atlas_probe's "per-panel scale" row). The headline
    "which GT pixels are worse" map, resolved per channel.
  * pergroup_cost_atlas.png   — 5 channel-GROUP rows (xyz / opacity / color / scale /
    rotation) × t cols, per-panel autoscale. Mirrors the 5-panel reference layout.
  * perchannel_mean_vs_t.png  — mean matched residual per channel vs t (+ grouped). The
    direct answer to "are certain channels consistently more difficult".
  * cost_atlas_summed.png     — channel-summed per-GT-cell HARD cost (huber'd), shared +
    per-panel rows — the literal trained loss localized to the atlas (the reference figure).
  * maps.npz / metadata.json  — raw arrays + the run's exact knobs.

Class-conditioned (LabelEmbedder) by default to match this run; pass --class_map_path.
Same checkpoint / preprocessing / EMA plumbing as sinkhorn_atlas_probe.py. No rendering.

Single-GPU example (defaults match the classcond_hier1000 run's knobs):

    CUDA_VISIBLE_DEVICES=1 python jit/sinkhorn_perchannel_probe.py \
        --resume output/<run>/0103000.pt \
        --obj_list /path/all_obj_list_filtered.json --gs_path /path/gaussianverse \
        --mean_file data/stats/all_mean_postfix.pt --std_file data/stats/all_std_postfix.pt \
        --rank_transform_file data/stats/rank_quantiles_8ch_clipped.pt \
        --clip_thresholds_file data/stats/clip_thresholds_opacity_scales.pt \
        --class_map_path object_labels/hier_uniform_k1000/object_to_class.json \
        --sphere2plane_path /path/sphere2plane.npy --sh_degree0_only \
        --patch_size 8 --sinkhorn_eps 0.05 --sinkhorn_iters 100 --huber_delta 2.0 \
        --t_values 0.1,0.3,0.5,0.7,0.9 --num_samples 64 --batch_size 4 --weights ema
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from dataloaders.standard_3dgen_loader import Standard3DGenDataset  # noqa: E402
from dataloaders.text_3dgen_loader import (  # noqa: E402
    Text3DGenDataset,
    DC_ONLY_FEATURE_INDICES,
    FULL_3DGS_FEATURE_DIM,
)
from jit.models import JiT_3DGS_models  # noqa: E402
from jit.diffusion import create_diffusion  # noqa: E402
from jit.diffusion.gaussian_diffusion import _sinkhorn_log  # noqa: E402  (exact training kernel)


# ── Channel identity for the 14 DC-only features (DC_ONLY_FEATURE_INDICES) ───────────
# 59-dim layout: xyz(0-2), opacity(3), SH(4-51: 3 colors × 16 coeffs), scale(52-54),
# rot(55-58). DC_ONLY keeps xyz, opacity, the 3 DC SH coeffs (4,20,36 = R,G,B DC),
# the 3 log-scales (52-54) and the quaternion (55-58) → these 14, in this order:
CHANNEL_NAMES = [
    "x", "y", "z", "opacity",
    "color_dc_R", "color_dc_G", "color_dc_B",
    "log_scale_0", "log_scale_1", "log_scale_2",
    "quat_w", "quat_x", "quat_y", "quat_z",
]
CHANNEL_GROUPS = [
    ("xyz", [0, 1, 2]),
    ("opacity", [3]),
    ("color", [4, 5, 6]),
    ("scale", [7, 8, 9]),
    ("rotation", [10, 11, 12, 13]),
]
GROUP_OF_CHANNEL = {c: g for g, idxs in CHANNEL_GROUPS for c in idxs}


def find_latest_checkpoint(search_root: str) -> str | None:
    cands = [
        c
        for c in glob.glob(os.path.join(search_root, "**", "*.pt"), recursive=True)
        if Path(c).stem.isdigit()
    ]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def parse_t_values(raw: str) -> list[float]:
    vals = [float(v) for v in raw.split(",") if v.strip() != ""]
    if not vals:
        raise ValueError(f"--t_values produced an empty list: {raw!r}")
    for v in vals:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"t_value {v} out of range [0, 1]")
    return vals


def to_patches(t: torch.Tensor, P: int) -> torch.Tensor:
    """(B,C,H,W) -> (B, nP, M, C). IDENTICAL grouping to _chamfer_recon_loss / permute_atlas."""
    B, C, H, W = t.shape
    nH, nW = H // P, W // P
    M = P * P
    return (
        t.float()
        .reshape(B, C, nH, P, nW, P)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(B, nH * nW, C, M)
        .transpose(-1, -2)
        .contiguous()
    )


def unpatchify_perchannel(v: torch.Tensor, nH: int, nW: int, P: int) -> torch.Tensor:
    """Inverse of to_patches for a per-cell vector: (B, nP, M, K) -> (B, H, W, K)."""
    B, _, _, K = v.shape
    return (
        v.reshape(B, nH, nW, P, P, K)   # [b, ph, pw, pr, pc, k]
        .permute(0, 1, 3, 2, 4, 5)      # [b, ph, pr, pw, pc, k]
        .reshape(B, nH * P, nW * P, K)
    )


def huber_sq(sq: torch.Tensor, delta: float) -> torch.Tensor:
    """Pseudo-Huber on the per-pair squared distance, IDENTICAL to the training transform
    (gaussian_diffusion.py): 2δ²(√(1+sq/δ²)−1), rationalized to avoid fp32 cancellation.
    delta<=0 → identity (squared-L2)."""
    if delta and delta > 0.0:
        d2 = delta * delta
        return 2.0 * sq / ((1.0 + sq / d2).sqrt() + 1.0)
    return sq


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Model / checkpoint
    p.add_argument("--model", type=str, default="JiT-B/8", choices=list(JiT_3DGS_models.keys()))
    p.add_argument("--bottleneck", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint .pt. If omitted, newest NNNNNNN.pt under --ckpt_search_root.")
    p.add_argument("--ckpt_search_root", type=str, default="output")
    p.add_argument("--weights", type=str, default="ema", choices=("ema", "model"))
    p.add_argument("--noise_schedule", type=str, default="squaredcos_cap_v2")
    # Conditioning — class (LabelEmbedder) for this run; text path kept for reuse.
    p.add_argument("--class_map_path", type=str, default=None,
                   help="JSON {hash: class_id}. Set → discrete class conditioning (LabelEmbedder).")
    p.add_argument("--num_classes", type=int, default=None,
                   help="Override; default = max(class id)+1 from --class_map_path.")
    p.add_argument("--text_embed_path", type=str, default=None,
                   help="Only for text-conditioned checkpoints (leave unset for class runs).")
    # Data (mirror training)
    p.add_argument("--obj_list", type=str, required=True)
    p.add_argument("--gs_path", type=str, required=True)
    p.add_argument("--mean_file", type=str, default=None)
    p.add_argument("--std_file", type=str, default=None)
    p.add_argument("--rank_transform_file", type=str, default=None)
    p.add_argument("--clip_thresholds_file", type=str, default=None)
    p.add_argument("--sphere2plane_path", type=str, required=True)
    p.add_argument("--exclude_keys_file", type=str, default=None)
    p.add_argument("--sh_degree0_only", action=argparse.BooleanOptionalAction, default=True)
    # Sinkhorn / patch knobs — DEFAULTS MATCH this run (sinkhorn_patch_hard).
    p.add_argument("--patch_size", type=int, default=8, help="P (chamfer_patch_size); M=P² cells/patch.")
    p.add_argument("--sinkhorn_eps", type=float, default=0.05)
    p.add_argument("--sinkhorn_iters", type=int, default=100)
    p.add_argument("--huber_delta", type=float, default=2.0,
                   help="Robust ground cost (matches the run). 0 = squared-L2.")
    # Probe params
    p.add_argument("--t_values", type=str, default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4, help="Keep low to coexist with training.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=("bf16", "fp16", "none"))
    p.add_argument("--output_dir", type=str, default=None,
                   help="Default: jit/sinkhorn_perchannel_probe_out/run_<timestamp>.")
    p.add_argument("--cmap", type=str, default="magma")
    return p


def _atlas_row_grid(maps, row_labels, row_scalars, t_values, title, cmap, out_path):
    """Grid: one row per channel/group, one col per t. Per-panel autoscale (each panel its
    own colorbar) — the 'per-panel scale' style of the reference image."""
    nrow, ncol = len(row_labels), len(t_values)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.55 * nrow), squeeze=False)
    for r, lbl in enumerate(row_labels):
        for c, t in enumerate(t_values):
            hm = maps[c, r]   # maps is (n_t, nrow, H, W) → panel (row r, col t=c)
            im = axes[r, c].imshow(hm, cmap=cmap)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f"t={t}", fontsize=10)
            if c == 0:
                axes[r, c].set_ylabel(f"{lbl}\nμ={row_scalars[r]:.3f}", fontsize=8)
            fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _summed_atlas_grid(maps, t_values, loss_per_t, title, cmap, out_path):
    """Channel-summed cost: shared-scale row + per-panel-scale row (reproduces the
    sinkhorn_atlas_probe reference figure)."""
    n = len(t_values)
    gvmax = max(float(np.nanmax(maps[c])) for c in range(n))
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 6.6), squeeze=False)
    im0 = None
    for c, t in enumerate(t_values):
        hm = maps[c]
        im0 = axes[0, c].imshow(hm, cmap=cmap, vmin=0.0, vmax=gvmax)
        axes[0, c].set_title(f"t={t}\nhard_loss={loss_per_t[c]:.4f}", fontsize=9)
        axes[0, c].set_xticks([]); axes[0, c].set_yticks([])
        im1 = axes[1, c].imshow(hm, cmap=cmap)
        axes[1, c].set_xticks([]); axes[1, c].set_yticks([])
        fig.colorbar(im1, ax=axes[1, c], fraction=0.046, pad=0.04)
    axes[0, 0].set_ylabel("shared scale", fontsize=10)
    axes[1, 0].set_ylabel("per-panel scale", fontsize=10)
    fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.025, pad=0.02)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _mean_vs_t_figure(perchan_mean, t_values, title, out_path):
    """perchan_mean: (n_t, C) mean matched residual. Left: per-channel lines colored by
    group. Right: per-group mean lines. The 'which channel is consistently harder' answer."""
    C = perchan_mean.shape[1]
    group_names = [g for g, _ in CHANNEL_GROUPS]
    cmap = plt.get_cmap("tab10")
    gcolor = {g: cmap(i) for i, g in enumerate(group_names)}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
    for c in range(C):
        g = GROUP_OF_CHANNEL[c]
        axL.plot(t_values, perchan_mean[:, c], "o-", color=gcolor[g], alpha=0.85, lw=1.5)
        axL.annotate(CHANNEL_NAMES[c], (t_values[-1], perchan_mean[-1, c]),
                     fontsize=7, color=gcolor[g], xytext=(3, 0), textcoords="offset points")
    axL.set_xlabel("t_value  (0 = noise, 1 = clean)")
    axL.set_ylabel("mean matched residual  (pred−gt)²  [normalized units]")
    axL.set_title("Per-channel difficulty vs t")
    axL.set_yscale("log")
    axL.grid(True, alpha=0.3)
    axL.legend(handles=[Line2D([0], [0], color=gcolor[g], lw=2, label=g) for g in group_names],
               fontsize=8, title="group")
    for g, idxs in CHANNEL_GROUPS:
        gm = perchan_mean[:, idxs].mean(axis=1)
        axR.plot(t_values, gm, "s-", color=gcolor[g], lw=2, label=g)
    axR.set_xlabel("t_value  (0 = noise, 1 = clean)")
    axR.set_ylabel("group-mean matched residual")
    axR.set_title("Per-group difficulty vs t")
    axR.set_yscale("log")
    axR.grid(True, alpha=0.3)
    axR.legend(fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    t_values = parse_t_values(args.t_values)
    P = int(args.patch_size)

    ckpt_path = args.resume or find_latest_checkpoint(args.ckpt_search_root)
    if ckpt_path is None:
        raise FileNotFoundError(f"No --resume and none found under {args.ckpt_search_root!r}.")
    print(f"[ckpt] {ckpt_path}  (weights={args.weights})")

    out_dir = args.output_dir or os.path.join(
        REPO_ROOT, "jit", "sinkhorn_perchannel_probe_out",
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out]  {out_dir}")
    print(f"[sink] mode=sinkhorn_patch_hard  P={P} (M={P*P}/patch)  eps={args.sinkhorn_eps}  "
          f"iters={args.sinkhorn_iters}  huber_delta={args.huber_delta}")

    # ── Feature selection (must match training) ───────────────────────────
    if args.sh_degree0_only:
        feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
        in_channels = len(DC_ONLY_FEATURE_INDICES)
    else:
        feature_indices = None
        in_channels = FULL_3DGS_FEATURE_DIM
    if in_channels != len(CHANNEL_NAMES):
        raise ValueError(
            f"Per-channel labels assume the {len(CHANNEL_NAMES)} DC-only channels; got "
            f"in_channels={in_channels}. Run with --sh_degree0_only (the run's setting)."
        )
    print(f"[feat] in_channels={in_channels} (sh_degree0_only={args.sh_degree0_only})")

    # ── Conditioning mode ─────────────────────────────────────────────────
    class_conditioned = bool(args.class_map_path)
    class_map = None
    num_classes = None
    if class_conditioned:
        with open(args.class_map_path, "r", encoding="utf-8") as f:
            class_map = json.load(f)
        num_classes = args.num_classes or (max(int(v) for v in class_map.values()) + 1)
        print(f"[cond] class-conditioned (LabelEmbedder), num_classes={num_classes}, "
              f"{len(class_map):,} labeled objects")
    else:
        print("[cond] text-conditioned (pooled CLIP)")

    # ── Dataset / loader ──────────────────────────────────────────────────
    base = Standard3DGenDataset(
        obj_list=[args.obj_list],
        gs_path=args.gs_path,
        caption_path=None,
        mean_file=args.mean_file,
        std_file=args.std_file,
        sphere2plane_path=args.sphere2plane_path,
        exclude_keys_file=args.exclude_keys_file,
        rank_transform_file=args.rank_transform_file,
        clip_thresholds_file=args.clip_thresholds_file,
        text_embed_path=(None if class_conditioned else args.text_embed_path),
    )
    if class_conditioned:
        # Keep only objects with a class label (mirrors train_gsplat.py's class filter).
        before = len(base.obj_data)
        base.obj_data = {h: p for h, p in base.obj_data.items()
                         if p.split('.tar.gz')[0] in class_map}
        base.keys = list(base.obj_data.keys())
        print(f"[class] {before:,} -> {len(base.obj_data):,} objects with a class label")
        text_dim = None
    else:
        text_dim = int(base.text_pooled.shape[1])

    dataset = Text3DGenDataset(
        base,
        feature_indices=feature_indices,
        return_full_for_render=False,
        preload_to_cpu=False,
        lazy_cache_to_cpu=False,
        class_map=class_map,
    )
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False, generator=loader_gen,
    )
    print(f"[data] dataset={len(dataset)} samples, probing {args.num_samples} @ batch {args.batch_size}")

    # ── Model + checkpoint ────────────────────────────────────────────────
    model = JiT_3DGS_models[args.model](
        input_size=128,
        in_channels=in_channels,
        text_dim=(text_dim or 768),
        num_classes=num_classes,  # None -> TextEmbedder; int -> LabelEmbedder
        class_dropout_prob=0.0,
        learn_sigma=False,
        gradient_checkpointing=False,
        bottleneck=args.bottleneck,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt[args.weights]
    # torch.compile saves the live 'model' with an `_orig_mod.` prefix on every key (the
    # 'ema' copy is clean). Strip it so `--weights model` actually loads (else strict=False
    # silently runs a random net — symptom: collision ~98%, cost ~8). See memory.
    pfx = "_orig_mod."
    if any(k.startswith(pfx) for k in state):
        state = {(k[len(pfx):] if k.startswith(pfx) else k): v for k, v in state.items()}
        print(f"[load] stripped '{pfx}' prefix (torch.compile) from {args.weights} state_dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] non-strict: {len(missing)} missing, {len(unexpected)} unexpected")
        if len(missing) > 5 or len(unexpected) > 5:
            raise RuntimeError(
                f"Large state_dict mismatch ({len(missing)} missing / {len(unexpected)} unexpected) — "
                f"weights did NOT load. Check --model/--bottleneck/--sh_degree0_only/--class_map_path/--weights. "
                f"missing(head)={list(missing)[:4]} unexpected(head)={list(unexpected)[:4]}")
    step = int(ckpt.get("step", ckpt.get("opt_step", -1)))
    print(f"[load] checkpoint step={step}")
    model.to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)

    diffusion = create_diffusion(
        timestep_respacing="", noise_schedule=args.noise_schedule,
        learn_sigma=False, predict_xstart=True,
    )
    T = diffusion.num_timesteps
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.mixed_precision]

    # ── Accumulators ──────────────────────────────────────────────────────
    H = W = 128
    if H % P or W % P:
        raise ValueError(f"patch_size={P} must divide 128")
    nH, nW = H // P, W // P
    M = P * P
    C = in_channels

    # Per-GT-cell per-channel matched residual: sum + count over the dataset (divide at end).
    cost_c_sum = {t: torch.zeros(H, W, C, dtype=torch.float64) for t in t_values}
    count_sum = {t: torch.zeros(H, W, 1, dtype=torch.float64) for t in t_values}
    # Per-GT-cell channel-SUMMED hard cost (the literal trained loss, huber'd).
    hard_sum = {t: torch.zeros(H, W, 1, dtype=torch.float64) for t in t_values}
    # Headline: mean matched residual per channel (over all matched preds).
    res_chan_sum = {t: torch.zeros(C, dtype=torch.float64) for t in t_values}
    n_preds = {t: 0 for t in t_values}
    loss_sum = {t: 0.0 for t in t_values}     # mean per-sample sinkhorn_patch_hard loss
    collision_sum = {t: 0.0 for t in t_values}
    n_seen = 0

    for batch_idx, batch in enumerate(loader):
        if n_seen >= args.num_samples:
            break
        x, y_cond, hash_keys = batch
        remaining = args.num_samples - n_seen
        if x.shape[0] > remaining:
            x = x[:remaining]; y_cond = y_cond[:remaining]
        bsz = x.shape[0]

        x = x.to(device, non_blocking=True).float()
        y_cond = y_cond.to(device, non_blocking=True)
        y_cond = y_cond.long() if class_conditioned else y_cond.float()

        noise_gen = torch.Generator(device=device)
        noise_gen.manual_seed(args.seed * 1_000_003 + batch_idx)
        noise = torch.randn(x.shape, generator=noise_gen, device=device, dtype=x.dtype)

        yp = to_patches(x, P)             # canonical GT patches (B, nP, M, C)
        bb = yp.pow(2).sum(-1)            # (B, nP, M)

        for t in t_values:
            t_value = torch.full((bsz,), float(t), device=device)
            t_discrete = torch.clamp((t_value * (T - 1)).round().long(), 0, T - 1)
            x_t = diffusion.flow_matching_q_sample(x, t_value, noise=noise)
            with torch.no_grad():
                if amp_dtype is not None and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        pred = model(x_t, t_discrete, y_cond)
                else:
                    pred = model(x_t, t_discrete, y_cond)

            xp = to_patches(pred.float(), P)              # (B, nP, M, C)
            aa = xp.pow(2).sum(-1)                         # (B, nP, M)
            ab = xp @ yp.transpose(-1, -2)                # (B, nP, M, M)  [i=pred, j=gt]
            sq = (aa.unsqueeze(-1) + bb.unsqueeze(-2) - 2 * ab).clamp_min(0)  # raw (B,nP,M,M)
            sq_h = huber_sq(sq, args.huber_delta)          # robust ground cost (the run's)
            Pi = _sinkhorn_log(sq_h, args.sinkhorn_eps, args.sinkhorn_iters)  # (B,nP,M,M)
            sigma = Pi.argmax(dim=-1)                      # (B, nP, M)  pred i -> gt cell σ(i)

            # Per-channel matched residual at the trained hard assignment (raw, pre-huber).
            idxC = sigma.unsqueeze(-1).expand(-1, -1, -1, C)       # (B,nP,M,C)
            yp_matched = yp.gather(-2, idxC)                       # (B,nP,M,C)
            res_c = (xp - yp_matched).pow(2)                       # (B,nP,M,C)  Σ_c = sq[i,σ(i)]
            hard_c = sq_h.gather(-1, sigma.unsqueeze(-1))          # (B,nP,M,1)  huber'd hard cost

            # Headline scalar: mean matched residual per channel (over all preds).
            res_chan_sum[t] += res_c.sum(dim=(0, 1, 2)).double().cpu()
            n_preds[t] += bsz * nH * nW * M
            loss_sum[t] += float(hard_c.squeeze(-1).mean(dim=(1, 2)).sum().cpu())  # sum over batch

            # Collision fraction (how far the rounded assignment is from a bijection).
            col_counts = torch.zeros_like(sigma, dtype=res_c.dtype)
            col_counts.scatter_add_(-1, sigma, torch.ones_like(sigma, dtype=res_c.dtype))
            collision_sum[t] += float(((col_counts - 1).clamp_min(0).sum(-1) / M).mean().cpu()) * bsz

            # Localize to the GT cell σ(i): scatter the matched residual into the gt-cell slot.
            cell_sum = torch.zeros(bsz, nH * nW, M, C, device=device, dtype=res_c.dtype)
            cell_sum.scatter_add_(2, idxC, res_c)
            cell_hard = torch.zeros(bsz, nH * nW, M, 1, device=device, dtype=res_c.dtype)
            cell_hard.scatter_add_(2, sigma.unsqueeze(-1), hard_c)
            cell_cnt = torch.zeros(bsz, nH * nW, M, 1, device=device, dtype=res_c.dtype)
            cell_cnt.scatter_add_(2, sigma.unsqueeze(-1), torch.ones_like(hard_c))

            cost_c_sum[t] += unpatchify_perchannel(cell_sum, nH, nW, P).sum(0).double().cpu()
            hard_sum[t] += unpatchify_perchannel(cell_hard, nH, nW, P).sum(0).double().cpu()
            count_sum[t] += unpatchify_perchannel(cell_cnt, nH, nW, P).sum(0).double().cpu()

        n_seen += bsz
        print(f"  [{n_seen}/{args.num_samples}] batch {batch_idx} done")

    if n_seen == 0:
        raise RuntimeError("No samples were processed.")

    # ── Reduce ────────────────────────────────────────────────────────────
    eps_div = 1e-12
    # per-channel atlas: (n_t, C, H, W) mean matched residual per GT cell
    perchan_maps = np.stack(
        [(cost_c_sum[t] / count_sum[t].clamp_min(eps_div)).permute(2, 0, 1).numpy() for t in t_values],
        axis=0,
    )
    # per-group atlas: sum the member channels' per-cell residual → (n_t, G, H, W)
    group_maps = np.stack(
        [np.stack([perchan_maps[ti, idxs].sum(axis=0) for _, idxs in CHANNEL_GROUPS], axis=0)
         for ti in range(len(t_values))],
        axis=0,
    )
    # channel-summed hard cost per GT cell (the literal trained loss): (n_t, H, W)
    summed_maps = np.stack(
        [(hard_sum[t] / count_sum[t].clamp_min(eps_div)).squeeze(-1).numpy() for t in t_values],
        axis=0,
    )
    # headline: (n_t, C) mean matched residual per channel
    perchan_mean = np.stack([(res_chan_sum[t] / max(n_preds[t], 1)).numpy() for t in t_values], axis=0)
    loss_per_t = np.asarray([loss_sum[t] / n_seen for t in t_values], dtype=np.float64)
    collision_per_t = np.asarray([collision_sum[t] / n_seen for t in t_values], dtype=np.float64)
    perchan_atlas_mean = perchan_maps.mean(axis=(2, 3))   # (n_t, C) mean over atlas
    group_atlas_mean = group_maps.mean(axis=(2, 3))       # (n_t, G)

    # ── Print ranking ─────────────────────────────────────────────────────
    print("\n[result] sinkhorn_patch_hard  (FM: t=0 noise → t=1 clean)")
    print(f"  {'t':>5} {'hard_loss':>10} {'collision%':>10}")
    for ti, t in enumerate(t_values):
        print(f"  {t:>5} {loss_per_t[ti]:>10.4f} {collision_per_t[ti]*100:>9.2f}%")
    print("\n[per-channel mean matched residual] (normalized feature units; lower=easier)")
    header = "  channel".ljust(16) + "group".ljust(10) + "".join(f"t={t:<6}" for t in t_values) + "  mean"
    print(header)
    chan_overall = perchan_mean.mean(axis=0)
    order = np.argsort(-chan_overall)
    for c in order:
        row = "  " + CHANNEL_NAMES[c].ljust(14) + GROUP_OF_CHANNEL[c].ljust(10)
        row += "".join(f"{perchan_mean[ti, c]:<8.4f}" for ti in range(len(t_values)))
        row += f"  {chan_overall[c]:.4f}"
        print(row)
    print("\n[per-group mean matched residual]")
    for g, idxs in CHANNEL_GROUPS:
        gm = perchan_mean[:, idxs].mean()
        print(f"  {g:<10} {gm:.4f}")

    # ── Save raw arrays + metadata ────────────────────────────────────────
    np.savez(
        os.path.join(out_dir, "maps.npz"),
        t_values=np.asarray(t_values, dtype=np.float64),
        channel_names=np.asarray(CHANNEL_NAMES),
        perchannel_atlas=perchan_maps,        # (n_t, C, H, W)
        group_atlas=group_maps,               # (n_t, G, H, W)
        summed_hard_atlas=summed_maps,         # (n_t, H, W)
        perchannel_mean_vs_t=perchan_mean,     # (n_t, C)
        hard_loss_per_t=loss_per_t,
        collision_per_t=collision_per_t,
    )
    meta = {
        "checkpoint": ckpt_path, "checkpoint_step": step, "weights": args.weights,
        "model": args.model, "in_channels": in_channels, "sh_degree0_only": bool(args.sh_degree0_only),
        "recon_loss": "sinkhorn_patch_hard",
        "patch_size": P, "sinkhorn_eps": args.sinkhorn_eps, "sinkhorn_iters": args.sinkhorn_iters,
        "huber_delta": args.huber_delta,
        "class_conditioned": class_conditioned, "num_classes": num_classes,
        "t_values": t_values, "num_samples": n_seen, "batch_size": args.batch_size,
        "mixed_precision": args.mixed_precision, "seed": args.seed,
        "channel_names": CHANNEL_NAMES,
        "channel_groups": {g: idxs for g, idxs in CHANNEL_GROUPS},
        "perchannel_mean_vs_t": {CHANNEL_NAMES[c]: [float(perchan_mean[ti, c]) for ti in range(len(t_values))]
                                 for c in range(C)},
        "note": "Per-GT-cell, per-channel matched residual (pred−gt_σ(i))² under the HARD "
                "argmax assignment σ of the huber'd Sinkhorn plan (the trained "
                "sinkhorn_patch_hard matching). Σ_channel = raw matched sq (pre-huber). "
                "summed_hard_atlas = channel-summed huber'd hard cost = the literal loss, "
                "localized to the GT cell. Canonical (unpermuted) GT; atlas = sphere→plane "
                "OT-sorted 128×128 grid, each pixel one Gaussian.",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Figures ───────────────────────────────────────────────────────────
    base_title = (f"{args.model} {args.weights} step {step} | sinkhorn_patch_hard P={P} "
                  f"eps={args.sinkhorn_eps} iters={args.sinkhorn_iters} huber={args.huber_delta} | "
                  f"N={n_seen} (FM: t=0 noise → t=1 clean)")
    _atlas_row_grid(
        perchan_maps, CHANNEL_NAMES, perchan_atlas_mean.mean(axis=0), t_values,
        "Per-CHANNEL matched-residual atlas (per-panel scale)\n" + base_title,
        args.cmap, os.path.join(out_dir, "perchannel_cost_atlas.png"),
    )
    _atlas_row_grid(
        group_maps, [g for g, _ in CHANNEL_GROUPS], group_atlas_mean.mean(axis=0), t_values,
        "Per-GROUP matched-residual atlas (per-panel scale)\n" + base_title,
        args.cmap, os.path.join(out_dir, "pergroup_cost_atlas.png"),
    )
    _summed_atlas_grid(
        summed_maps, t_values, loss_per_t,
        "Channel-summed per-GT-cell HARD cost (the trained loss)\n" + base_title,
        args.cmap, os.path.join(out_dir, "cost_atlas_summed.png"),
    )
    _mean_vs_t_figure(
        perchan_mean, t_values,
        "Which GT channels are consistently harder?  " + base_title,
        os.path.join(out_dir, "perchannel_mean_vs_t.png"),
    )

    print(f"\n[done] wrote figures + maps.npz + metadata.json to:\n  {out_dir}")


if __name__ == "__main__":
    main()
