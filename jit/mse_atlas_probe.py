#!/usr/bin/env python
"""MSE atlas probe — where in the Gaussian atlas is x0-prediction error highest?

Loads a trained JiT checkpoint (EMA by default), feeds a small set of real
objects through the flow-matching forward at several fixed ``t_value`` levels,
and computes the per-splat squared error of the x0 prediction. Because the
128x128 input grid is a sphere->plane (OT-sorted) arrangement of the 16,384
Gaussians, the per-spatial-location MSE *is* per-splat MSE: averaging the
squared error over channels gives a [128,128] heatmap of "which splats are hard
to predict" at each noise level.

Convention (flow matching, NOT DDPM — see CLAUDE.md):
    x_t = t_value * x_0 + (1 - t_value) * eps
    t_value = 0 -> pure noise, t_value = 1 -> clean data.
    t_discrete = round(t_value * (T - 1)) is the integer step fed to t_embedder.

The same noise eps is reused across all t_value levels within a batch so the
heatmaps differ only because of t, not because of a different noise draw.

MSE is computed in the *normalized training space* (exactly the tensor the
dataloader hands to the trainer, after clip + rank-transform + per-channel
standardization), i.e. the actual training objective — no rendering involved.

Designed to run alongside a full training job: keep --batch_size small.

Single-GPU example (paths usually come from jit/mse_atlas_probe.sh + .env):

    python jit/mse_atlas_probe.py \
        --obj_list /path/all_obj_list_filtered.json --gs_path /path/gaussianverse \
        --mean_file data/stats/all_mean_postfix.pt --std_file data/stats/all_std_postfix.pt \
        --rank_transform_file data/stats/rank_quantiles_8ch_clipped.pt \
        --clip_thresholds_file data/stats/clip_thresholds_opacity_scales.pt \
        --text_embed_path object_classification/text_tokens \
        --null_text_token_path object_classification/null_text_token.npz \
        --sphere2plane_path /path/sphere2plane.npy --sh_degree0_only \
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Model / checkpoint
    p.add_argument("--model", type=str, default="JiT-B/8", choices=list(JiT_3DGS_models.keys()))
    p.add_argument("--bottleneck", action=argparse.BooleanOptionalAction, default=False,
                   help="Must match the checkpoint's patch-embed (config default: False).")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint .pt. If omitted, newest NNNNNNN.pt under --ckpt_search_root.")
    p.add_argument("--ckpt_search_root", type=str, default="output",
                   help="Where to auto-detect the latest checkpoint when --resume is unset.")
    p.add_argument("--weights", type=str, default="ema", choices=("ema", "model"),
                   help="Which state_dict to load from the checkpoint.")
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
    p.add_argument("--sh_degree0_only", action=argparse.BooleanOptionalAction, default=True,
                   help="DC-only 14-channel feature set (config default: True).")
    # Probe params
    p.add_argument("--t_values", type=str, default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8, help="Keep low to coexist with training.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_example_samples", type=int, default=4,
                   help="Per-sample heatmaps to render from the first batch.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seeds the sample subset and the fixed noise draw.")
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=("bf16", "fp16", "none"),
                   help="Autocast for the model forward (bf16 matches training).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Default: jit/mse_atlas_probe_out/run_<timestamp>.")
    p.add_argument("--cmap", type=str, default="magma")
    return p


def main() -> None:
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    t_values = parse_t_values(args.t_values)

    # ── Resolve checkpoint ────────────────────────────────────────────────
    ckpt_path = args.resume or find_latest_checkpoint(args.ckpt_search_root)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint given and none found under {args.ckpt_search_root!r}. "
            "Pass --resume explicitly."
        )
    print(f"[ckpt] {ckpt_path}  (weights={args.weights})")

    # ── Output dir ────────────────────────────────────────────────────────
    out_dir = args.output_dir or os.path.join(
        REPO_ROOT, "jit", "mse_atlas_probe_out",
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out]  {out_dir}")

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
        class_dropout_prob=0.0,   # eval: forward never drops conditioning
        learn_sigma=False,
        gradient_checkpointing=False,
        bottleneck=args.bottleneck,
    )
    # Ensure the null buffer exists with the right shape before loading state
    # (the trained null is then restored by load_state_dict).
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
            print("       ^ large mismatch — check --model/--bottleneck/--sh_degree0_only "
                  "match this checkpoint.")
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
    sumsq = {t: torch.zeros(H, W, dtype=torch.float64) for t in t_values}  # per-pixel chan-mean sq err
    mse_sum = {t: 0.0 for t in t_values}                                   # sum of per-sample scalar MSE
    examples = {t: None for t in t_values}                                 # [k,H,W] from first batch
    example_keys: list[str] = []
    n_seen = 0

    for batch_idx, batch in enumerate(loader):
        if n_seen >= args.num_samples:
            break
        x, y_pooled, hash_keys = batch
        remaining = args.num_samples - n_seen
        if x.shape[0] > remaining:
            x = x[:remaining]
            y_pooled = y_pooled[:remaining]
            hash_keys = list(hash_keys)[:remaining]
        bsz = x.shape[0]

        x = x.to(device, non_blocking=True).float()
        y_pooled = y_pooled.to(device, non_blocking=True).float()

        # Fixed noise for this batch, reused across every t (deterministic per batch).
        noise_gen = torch.Generator(device=device)
        noise_gen.manual_seed(args.seed * 1_000_003 + batch_idx)
        noise = torch.randn(x.shape, generator=noise_gen, device=device, dtype=x.dtype)

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
            sq = (x - pred.float()) ** 2          # [B, C, H, W]
            chmean = sq.mean(dim=1)               # [B, H, W] — per-splat error
            sumsq[t] += chmean.sum(dim=0).double().cpu()
            mse_sum[t] += float(chmean.mean(dim=(1, 2)).sum().cpu())
            if batch_idx == 0:
                k = min(args.num_example_samples, bsz)
                examples[t] = chmean[:k].detach().cpu().numpy()
        if batch_idx == 0:
            example_keys = list(hash_keys)[: min(args.num_example_samples, bsz)]

        n_seen += bsz
        print(f"  [{n_seen}/{args.num_samples}] batch {batch_idx} done")

    if n_seen == 0:
        raise RuntimeError("No samples were processed.")

    heatmaps = {t: (sumsq[t] / n_seen).numpy() for t in t_values}   # mean per-pixel chan-mean sq err
    mse_per_t = {t: mse_sum[t] / n_seen for t in t_values}

    print("\n[result] mean MSE per t (FM: t=0 noise, t=1 clean):")
    for t in t_values:
        print(f"  t={t:>4}  mean_MSE={mse_per_t[t]:.6f}")

    # ── Save raw arrays + metadata ────────────────────────────────────────
    np.savez(
        os.path.join(out_dir, "heatmaps.npz"),
        t_values=np.asarray(t_values, dtype=np.float64),
        heatmaps=np.stack([heatmaps[t] for t in t_values], axis=0),  # [n_t, 128, 128]
        mse_per_t=np.asarray([mse_per_t[t] for t in t_values], dtype=np.float64),
    )
    meta = {
        "checkpoint": ckpt_path,
        "checkpoint_step": step,
        "weights": args.weights,
        "model": args.model,
        "in_channels": in_channels,
        "sh_degree0_only": bool(args.sh_degree0_only),
        "t_values": t_values,
        "num_samples": n_seen,
        "batch_size": args.batch_size,
        "mixed_precision": args.mixed_precision,
        "seed": args.seed,
        "mse_per_t": {str(t): mse_per_t[t] for t in t_values},
        "example_keys": example_keys,
        "note": "Heatmap = mean over channels of (x0 - pred)^2 per atlas pixel, "
                "averaged over samples. Atlas = sphere->plane OT-sorted 128x128 grid; "
                "each pixel is one Gaussian.",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Figure 1: atlas heatmaps (shared scale row + per-panel scale row) ──
    n = len(t_values)
    gvmax = max(float(heatmaps[t].max()) for t in t_values)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 6.6), squeeze=False)
    for j, t in enumerate(t_values):
        hm = heatmaps[t]
        im0 = axes[0, j].imshow(hm, cmap=args.cmap, vmin=0.0, vmax=gvmax)
        axes[0, j].set_title(f"t={t}\nMSE={mse_per_t[t]:.4f}", fontsize=9)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        im1 = axes[1, j].imshow(hm, cmap=args.cmap)  # per-panel autoscale
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        fig.colorbar(im1, ax=axes[1, j], fraction=0.046, pad=0.04)
    axes[0, 0].set_ylabel("shared scale", fontsize=10)
    axes[1, 0].set_ylabel("per-panel scale", fontsize=10)
    fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.025, pad=0.02)
    fig.suptitle(
        f"Atlas per-splat x0-pred MSE  |  {args.model} {args.weights} step {step}  |  "
        f"N={n_seen}  (FM: t=0 noise → t=1 clean)",
        fontsize=11,
    )
    fig.savefig(os.path.join(out_dir, "mse_atlas_heatmaps.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: mean MSE vs t ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_values, [mse_per_t[t] for t in t_values], "o-")
    ax.set_xlabel("t_value  (0 = noise, 1 = clean)")
    ax.set_ylabel("mean per-splat MSE")
    ax.set_title(f"x0-pred MSE vs t  ({args.model} {args.weights} step {step}, N={n_seen})")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "mse_vs_t.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: per-sample examples (rows=samples, cols=t; per-panel scale) ──
    if examples[t_values[0]] is not None:
        k = examples[t_values[0]].shape[0]
        fig, axes = plt.subplots(k, n, figsize=(3.0 * n, 3.0 * k), squeeze=False)
        for r in range(k):
            for j, t in enumerate(t_values):
                im = axes[r, j].imshow(examples[t][r], cmap=args.cmap)
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
                if r == 0:
                    axes[r, j].set_title(f"t={t}", fontsize=9)
                if j == 0:
                    lbl = example_keys[r] if r < len(example_keys) else f"sample {r}"
                    axes[r, j].set_ylabel(str(lbl), fontsize=8)
                fig.colorbar(im, ax=axes[r, j], fraction=0.046, pad=0.04)
        fig.suptitle("Per-sample atlas MSE (first batch, per-panel scale)", fontsize=11)
        fig.savefig(os.path.join(out_dir, "mse_atlas_per_sample.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"\n[done] wrote figures + heatmaps.npz + metadata.json to:\n  {out_dir}")


if __name__ == "__main__":
    main()
