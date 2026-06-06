#!/usr/bin/env python
"""Sinkhorn atlas probe — where in the Gaussian atlas does the *assignment* loss
the model was actually trained on (recon_loss=sinkhorn_patch) stay high?

This is the faithful counterpart to ``mse_atlas_probe.py``. Index-aligned MSE
punishes the model for permuting Gaussians *within* a patch — exactly what the
sinkhorn (optimal-assignment EMD) loss forgives — so the MSE heatmap conflates
"predicted a bad Gaussian" with "predicted a good Gaussian in a different cell".
Here we decompose the real training loss per GT cell instead.

For each ``patch_size × patch_size`` patch (M = P² Gaussians) the sinkhorn loss is
    sq[i,j] = ‖pred_i − gt_j‖²                       (sum over channels)
    Π[i,j]  = log-Sinkhorn(sq, eps, iters)           (doubly-stochastic, a=b=1/M)
    patch_loss = Σ_{i,j} Π[i,j] · sq[i,j]
identical to jit/diffusion/gaussian_diffusion.py::_chamfer_recon_loss(sinkhorn_patch).
It decomposes EXACTLY per GT cell j:
    cost_gt[j] = Σ_i Π[i,j] · sq[i,j]   ("expected sq-distance to GT cell j's OT match")
with  Σ_j cost_gt[j] = patch_loss.  Mapped back to the canonical 128×128 atlas this
IS the training loss, localized to each GT Gaussian — high = even after the optimal
within-patch assignment, GT Gaussian j has no good match (genuinely hard).

Two heatmaps per t (FM convention, see CLAUDE.md: t=0 noise → t=1 clean):
  1. cost_gt   — per-GT-cell sinkhorn transport cost (the headline "struggle" map).
  2. displ_gt  — expected within-patch spatial displacement of GT cell j's matched
                 prediction (E_i[‖pos(i)−pos(j)‖] under the column-normalized plan).
                 Disentangles failure modes:
                   low cost + high displ  = reproduced fine, just permuted (what MSE
                                            wrongly flagged as error).
                   high cost + low displ  = right slot, wrong features (true failure).

Uses the run's exact knobs (patch_size=4, sinkhorn_eps=0.05, sinkhorn_iters=50) and
the canonical (unpermuted) GT so every cost keeps its atlas-cell identity. Same
checkpoint / preprocessing / EMA plumbing as mse_atlas_probe.py. No rendering.

Single-GPU example (paths usually come from jit/sinkhorn_atlas_probe.sh + .env):

    python jit/sinkhorn_atlas_probe.py \
        --obj_list /path/all_obj_list_filtered.json --gs_path /path/gaussianverse \
        --mean_file data/stats/all_mean_postfix.pt --std_file data/stats/all_std_postfix.pt \
        --rank_transform_file data/stats/rank_quantiles_8ch_clipped.pt \
        --clip_thresholds_file data/stats/clip_thresholds_opacity_scales.pt \
        --text_embed_path object_classification/text_tokens \
        --null_text_token_path object_classification/null_text_token.npz \
        --sphere2plane_path /path/sphere2plane.npy --sh_degree0_only \
        --patch_size 4 --sinkhorn_eps 0.05 --sinkhorn_iters 50 \
        --t_values 0.1,0.3,0.5,0.7,0.9 --num_samples 64 --batch_size 8 --weights ema
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

from dataloaders.standard_3dgen_loader import (  # noqa: E402
    Standard3DGenDataset,
    load_null_text_token,
)
from dataloaders.text_3dgen_loader import (  # noqa: E402
    Text3DGenDataset,
    DC_ONLY_FEATURE_INDICES,
    FULL_3DGS_FEATURE_DIM,
)
from jit.models import JiT_3DGS_models  # noqa: E402
from jit.diffusion import create_diffusion  # noqa: E402
from jit.diffusion.gaussian_diffusion import _sinkhorn_log  # noqa: E402  (exact training kernel)


def find_latest_checkpoint(search_root: str) -> str | None:
    """Newest step-checkpoint (``NNNNNNN.pt``) by mtime under ``search_root``."""
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


def unpatchify_scalar(v: torch.Tensor, nH: int, nW: int, P: int) -> torch.Tensor:
    """Inverse of to_patches for a per-cell scalar: (B, nP, M) -> (B, H, W)."""
    B = v.shape[0]
    return (
        v.reshape(B, nH, nW, P, P)   # [b, ph, pw, pr, pc]
        .permute(0, 1, 3, 2, 4)      # [b, ph, pr, pw, pc]
        .reshape(B, nH * P, nW * P)
    )


def within_patch_dist(P: int, device) -> torch.Tensor:
    """(M, M) Euclidean distance between cells inside one P×P patch (m = pr*P + pc)."""
    idx = torch.arange(P * P, device=device)
    pr, pc = idx // P, idx % P
    dr = pr.view(-1, 1) - pr.view(1, -1)
    dc = pc.view(-1, 1) - pc.view(1, -1)
    return torch.sqrt((dr.float() ** 2 + dc.float() ** 2))  # (M, M)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Model / checkpoint
    p.add_argument("--model", type=str, default="JiT-B/8", choices=list(JiT_3DGS_models.keys()))
    p.add_argument("--bottleneck", action=argparse.BooleanOptionalAction, default=False,
                   help="Must match the checkpoint's patch-embed (config default: False).")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint .pt. If omitted, newest NNNNNNN.pt under --ckpt_search_root.")
    p.add_argument("--ckpt_search_root", type=str, default="output")
    p.add_argument("--weights", type=str, default="ema", choices=("ema", "model"))
    p.add_argument("--noise_schedule", type=str, default="squaredcos_cap_v2")
    # Data (mirror training)
    p.add_argument("--obj_list", type=str, required=True)
    p.add_argument("--gs_path", type=str, required=True)
    p.add_argument("--mean_file", type=str, default=None)
    p.add_argument("--std_file", type=str, default=None)
    p.add_argument("--rank_transform_file", type=str, default=None)
    p.add_argument("--clip_thresholds_file", type=str, default=None)
    p.add_argument("--text_embed_path", type=str, required=True)
    p.add_argument("--null_text_token_path", type=str, default=None)
    p.add_argument("--sphere2plane_path", type=str, required=True)
    p.add_argument("--exclude_keys_file", type=str, default=None)
    p.add_argument("--sh_degree0_only", action=argparse.BooleanOptionalAction, default=True)
    # Sinkhorn / patch knobs — DEFAULTS MATCH jit_train_gsplat.yaml (sinkhorn_patch run)
    p.add_argument("--patch_size", type=int, default=4, help="P (chamfer_patch_size); M=P² cells/patch.")
    p.add_argument("--sinkhorn_eps", type=float, default=0.05)
    p.add_argument("--sinkhorn_iters", type=int, default=50)
    # Probe params
    p.add_argument("--t_values", type=str, default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8, help="Keep low to coexist with training.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_example_samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=("bf16", "fp16", "none"))
    p.add_argument("--output_dir", type=str, default=None,
                   help="Default: jit/sinkhorn_atlas_probe_out/run_<timestamp>.")
    p.add_argument("--cost_cmap", type=str, default="magma")
    p.add_argument("--displ_cmap", type=str, default="viridis")
    return p


def _atlas_grid_figure(maps, t_values, scalar_per_t, scalar_label, title, cmap,
                       out_path, shared_vmax=None):
    """Two-row atlas grid: shared-scale row + per-panel-scale row, one column per t."""
    n = len(t_values)
    gvmax = shared_vmax if shared_vmax is not None else max(float(maps[t].max()) for t in t_values)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 6.6), squeeze=False)
    im0 = None
    for j, t in enumerate(t_values):
        hm = maps[t]
        im0 = axes[0, j].imshow(hm, cmap=cmap, vmin=0.0, vmax=gvmax)
        axes[0, j].set_title(f"t={t}\n{scalar_label}={scalar_per_t[t]:.4f}", fontsize=9)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        im1 = axes[1, j].imshow(hm, cmap=cmap)  # per-panel autoscale
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        fig.colorbar(im1, ax=axes[1, j], fraction=0.046, pad=0.04)
    axes[0, 0].set_ylabel("shared scale", fontsize=10)
    axes[1, 0].set_ylabel("per-panel scale", fontsize=10)
    fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.025, pad=0.02)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _per_sample_figure(examples, t_values, keys, title, cmap, out_path):
    if examples[t_values[0]] is None:
        return
    k = examples[t_values[0]].shape[0]
    n = len(t_values)
    fig, axes = plt.subplots(k, n, figsize=(3.0 * n, 3.0 * k), squeeze=False)
    for r in range(k):
        for j, t in enumerate(t_values):
            im = axes[r, j].imshow(examples[t][r], cmap=cmap)
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            if r == 0:
                axes[r, j].set_title(f"t={t}", fontsize=9)
            if j == 0:
                lbl = keys[r] if r < len(keys) else f"sample {r}"
                axes[r, j].set_ylabel(str(lbl), fontsize=8)
            fig.colorbar(im, ax=axes[r, j], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    t_values = parse_t_values(args.t_values)
    P = int(args.patch_size)

    # ── Resolve checkpoint ────────────────────────────────────────────────
    ckpt_path = args.resume or find_latest_checkpoint(args.ckpt_search_root)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint given and none found under {args.ckpt_search_root!r}. Pass --resume."
        )
    print(f"[ckpt] {ckpt_path}  (weights={args.weights})")

    out_dir = args.output_dir or os.path.join(
        REPO_ROOT, "jit", "sinkhorn_atlas_probe_out",
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out]  {out_dir}")
    print(f"[sink] patch_size={P} (M={P*P}/patch)  eps={args.sinkhorn_eps}  iters={args.sinkhorn_iters}")

    # ── Feature selection (must match training) ───────────────────────────
    if args.sh_degree0_only:
        feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
        in_channels = len(DC_ONLY_FEATURE_INDICES)
    else:
        feature_indices = None
        in_channels = FULL_3DGS_FEATURE_DIM
    print(f"[feat] in_channels={in_channels} (sh_degree0_only={args.sh_degree0_only})")

    # ── Dataset / loader (uncached: small probe, no preload) ──────────────
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
        text_embed_path=args.text_embed_path,
    )
    text_dim = int(base.text_pooled.shape[1])
    dataset = Text3DGenDataset(
        base,
        feature_indices=feature_indices,
        return_full_for_render=False,
        preload_to_cpu=False,
        lazy_cache_to_cpu=False,
    )
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=loader_gen,
    )
    print(f"[data] dataset={len(dataset)} samples, text_dim={text_dim}, "
          f"probing {args.num_samples} @ batch {args.batch_size}")

    # ── Model + checkpoint ────────────────────────────────────────────────
    model = JiT_3DGS_models[args.model](
        input_size=128,
        in_channels=in_channels,
        text_dim=text_dim,
        class_dropout_prob=0.0,
        learn_sigma=False,
        gradient_checkpointing=False,
        bottleneck=args.bottleneck,
    )
    if args.null_text_token_path and os.path.exists(args.null_text_token_path):
        null_np = load_null_text_token(args.null_text_token_path)
        model.load_null_embeddings(torch.from_numpy(null_np.astype(np.float32)))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt[args.weights]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] non-strict: {len(missing)} missing, {len(unexpected)} unexpected")
        if len(missing) > 5 or len(unexpected) > 5:
            print(f"       missing(head)={list(missing)[:5]}")
            print(f"       unexpected(head)={list(unexpected)[:5]}")
            print("       ^ large mismatch — check --model/--bottleneck/--sh_degree0_only.")
    step = int(ckpt.get("step", ckpt.get("opt_step", -1)))
    print(f"[load] checkpoint step={step}")
    model.to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)

    diffusion = create_diffusion(
        timestep_respacing="",
        noise_schedule=args.noise_schedule,
        learn_sigma=False,
        predict_xstart=True,
    )
    T = diffusion.num_timesteps
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.mixed_precision]

    # ── Accumulators ──────────────────────────────────────────────────────
    H = W = 128
    if H % P or W % P:
        raise ValueError(f"patch_size={P} must divide 128")
    nH, nW = H // P, W // P
    M = P * P
    D = within_patch_dist(P, device)  # (M, M) within-patch spatial distance
    max_displ = float(D.max())        # for a shared, interpretable displacement scale

    cost_sum = {t: torch.zeros(H, W, dtype=torch.float64) for t in t_values}
    displ_sum = {t: torch.zeros(H, W, dtype=torch.float64) for t in t_values}
    loss_sum = {t: 0.0 for t in t_values}          # mean per-sample sinkhorn loss (the real objective)
    displ_scalar_sum = {t: 0.0 for t in t_values}  # mean expected displacement over cells
    cost_examples = {t: None for t in t_values}
    displ_examples = {t: None for t in t_values}
    example_keys: list[str] = []
    n_seen = 0

    for batch_idx, batch in enumerate(loader):
        if n_seen >= args.num_samples:
            break
        x, y_pooled, hash_keys = batch
        remaining = args.num_samples - n_seen
        if x.shape[0] > remaining:
            x = x[:remaining]; y_pooled = y_pooled[:remaining]; hash_keys = list(hash_keys)[:remaining]
        bsz = x.shape[0]

        x = x.to(device, non_blocking=True).float()
        y_pooled = y_pooled.to(device, non_blocking=True).float()

        noise_gen = torch.Generator(device=device)
        noise_gen.manual_seed(args.seed * 1_000_003 + batch_idx)
        noise = torch.randn(x.shape, generator=noise_gen, device=device, dtype=x.dtype)

        yp = to_patches(x, P)  # canonical GT patches (B, nP, M, C) — reused across t
        bb = yp.pow(2).sum(-1)  # (B, nP, M)

        for t in t_values:
            t_value = torch.full((bsz,), float(t), device=device)
            t_discrete = torch.clamp((t_value * (T - 1)).round().long(), 0, T - 1)
            x_t = diffusion.flow_matching_q_sample(x, t_value, noise=noise)
            with torch.no_grad():
                if amp_dtype is not None and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        pred = model(x_t, t_discrete, y_pooled)
                else:
                    pred = model(x_t, t_discrete, y_pooled)

            xp = to_patches(pred.float(), P)            # (B, nP, M, C)
            aa = xp.pow(2).sum(-1)                       # (B, nP, M)
            ab = xp @ yp.transpose(-1, -2)              # (B, nP, M, M)  [i=pred, j=gt]
            sq = (aa.unsqueeze(-1) + bb.unsqueeze(-2) - 2 * ab).clamp_min(0)  # (B,nP,M,M)
            Pi = _sinkhorn_log(sq, args.sinkhorn_eps, args.sinkhorn_iters)    # (B,nP,M,M)

            # Per-GT-cell (sum over i=pred, dim -2). Σ_j cost_gt[j] = patch loss.
            cost_gt = (Pi * sq).sum(dim=-2)             # (B, nP, M)
            # Expected within-patch displacement of GT cell j's match. Column marginal
            # Σ_i Π[i,j] = 1/M, so ×M renormalizes to a proper expectation over i.
            displ_gt = (Pi * D.view(1, 1, M, M)).sum(dim=-2) * M   # (B, nP, M)

            cost_map = unpatchify_scalar(cost_gt, nH, nW, P)       # (B, H, W)
            displ_map = unpatchify_scalar(displ_gt, nH, nW, P)     # (B, H, W)

            cost_sum[t] += cost_map.sum(dim=0).double().cpu()
            displ_sum[t] += displ_map.sum(dim=0).double().cpu()
            loss_sum[t] += float((Pi * sq).sum(dim=(-1, -2)).mean(dim=1).sum().cpu())
            displ_scalar_sum[t] += float(displ_gt.mean(dim=(1, 2)).sum().cpu())

            if batch_idx == 0:
                k = min(args.num_example_samples, bsz)
                cost_examples[t] = cost_map[:k].detach().cpu().numpy()
                displ_examples[t] = displ_map[:k].detach().cpu().numpy()
        if batch_idx == 0:
            example_keys = list(hash_keys)[: min(args.num_example_samples, bsz)]

        n_seen += bsz
        print(f"  [{n_seen}/{args.num_samples}] batch {batch_idx} done")

    if n_seen == 0:
        raise RuntimeError("No samples were processed.")

    cost_maps = {t: (cost_sum[t] / n_seen).numpy() for t in t_values}
    displ_maps = {t: (displ_sum[t] / n_seen).numpy() for t in t_values}
    loss_per_t = {t: loss_sum[t] / n_seen for t in t_values}
    displ_per_t = {t: displ_scalar_sum[t] / n_seen for t in t_values}

    print("\n[result] (FM: t=0 noise → t=1 clean)")
    print(f"  {'t':>5}  {'sinkhorn_loss':>14}  {'mean_displ':>11}  (max possible displ={max_displ:.3f})")
    for t in t_values:
        print(f"  {t:>5}  {loss_per_t[t]:>14.6f}  {displ_per_t[t]:>11.4f}")

    # ── Save raw arrays + metadata ────────────────────────────────────────
    np.savez(
        os.path.join(out_dir, "maps.npz"),
        t_values=np.asarray(t_values, dtype=np.float64),
        cost_maps=np.stack([cost_maps[t] for t in t_values], axis=0),
        displ_maps=np.stack([displ_maps[t] for t in t_values], axis=0),
        sinkhorn_loss_per_t=np.asarray([loss_per_t[t] for t in t_values], dtype=np.float64),
        mean_displ_per_t=np.asarray([displ_per_t[t] for t in t_values], dtype=np.float64),
        max_displ=np.asarray(max_displ),
    )
    meta = {
        "checkpoint": ckpt_path, "checkpoint_step": step, "weights": args.weights,
        "model": args.model, "in_channels": in_channels,
        "sh_degree0_only": bool(args.sh_degree0_only),
        "patch_size": P, "sinkhorn_eps": args.sinkhorn_eps, "sinkhorn_iters": args.sinkhorn_iters,
        "t_values": t_values, "num_samples": n_seen, "batch_size": args.batch_size,
        "mixed_precision": args.mixed_precision, "seed": args.seed, "max_displ": max_displ,
        "sinkhorn_loss_per_t": {str(t): loss_per_t[t] for t in t_values},
        "mean_displ_per_t": {str(t): displ_per_t[t] for t in t_values},
        "example_keys": example_keys,
        "note": "cost = per-GT-cell sinkhorn transport cost Σ_i Π[i,j]·‖pred_i−gt_j‖² "
                "(Σ over patch = training loss). displ = expected within-patch spatial "
                "displacement of GT cell j's OT-matched prediction. Canonical (unpermuted) "
                "GT; atlas = sphere→plane OT-sorted 128×128 grid, each pixel one Gaussian.",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Figures ───────────────────────────────────────────────────────────
    _atlas_grid_figure(
        cost_maps, t_values, loss_per_t, "loss",
        f"Per-GT-cell SINKHORN transport cost  |  {args.model} {args.weights} step {step}  |  "
        f"N={n_seen} P={P}  (FM: t=0 noise → t=1 clean)",
        args.cost_cmap, os.path.join(out_dir, "sinkhorn_cost_atlas.png"),
    )
    _atlas_grid_figure(
        displ_maps, t_values, displ_per_t, "mean",
        f"GT-cell correspondent DISPLACEMENT (within-patch cells)  |  {args.model} {args.weights} "
        f"step {step}  |  N={n_seen} P={P}",
        args.displ_cmap, os.path.join(out_dir, "displacement_atlas.png"),
        shared_vmax=max_displ,
    )

    # vs-t curves (twin axis)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(t_values, [loss_per_t[t] for t in t_values], "o-", color="C3", label="sinkhorn loss")
    ax.set_xlabel("t_value  (0 = noise, 1 = clean)")
    ax.set_ylabel("per-sample sinkhorn loss", color="C3")
    ax.tick_params(axis="y", labelcolor="C3")
    ax2 = ax.twinx()
    ax2.plot(t_values, [displ_per_t[t] for t in t_values], "s--", color="C0", label="mean displacement")
    ax2.set_ylabel("mean correspondent displacement (cells)", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")
    ax2.axhline(max_displ, color="C0", ls=":", alpha=0.4)
    ax.set_title(f"Sinkhorn loss & match displacement vs t  ({args.model} {args.weights} step {step}, N={n_seen})")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "sinkhorn_vs_t.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    _per_sample_figure(
        cost_examples, t_values, example_keys,
        "Per-sample per-GT-cell sinkhorn cost (first batch, per-panel scale)",
        args.cost_cmap, os.path.join(out_dir, "sinkhorn_cost_per_sample.png"),
    )
    _per_sample_figure(
        displ_examples, t_values, example_keys,
        "Per-sample GT-cell correspondent displacement (first batch, per-panel scale)",
        args.displ_cmap, os.path.join(out_dir, "displacement_per_sample.png"),
    )

    print(f"\n[done] wrote figures + maps.npz + metadata.json to:\n  {out_dir}")


if __name__ == "__main__":
    main()
