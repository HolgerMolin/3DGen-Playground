#!/usr/bin/env python
"""Loss DISTRIBUTION probe — visualize the distribution of the exact recon loss a
sinkhorn_patch_hard run was trained on, on PURE recon (no render), for one checkpoint.

Reproduces the training objective bit-for-bit (jit/diffusion/gaussian_diffusion.py
::_chamfer_recon_loss(sinkhorn_patch_hard) + flow_matching_training_losses):

  patches    : (B,C,128,128) -> (B,nP,M,C), P=8 -> M=64 Gaussians/patch (within-patch match)
  sq[i,j]    : Σ_c (pred_i,c - gt_j,c)²                              (channel-summed squared L2)
  sq_h       : pseudo-Huber(sq, huber_delta)                         (the run's robust ground cost)
  Π          : log-Sinkhorn(sq_h, eps, iters)                        (entropic-OT plan, unif marginals)
  σ(i)       : argmax_j Π[i,j]                                       (HARD assignment)
  per-Gauss  : sq_h[i, σ(i)]                                         (the per-element loss term)
  per-sample : chamfer_loss_weight · mean_{patch,i} sq_h[i,σ(i)]     (what enters total_loss)

t is drawn EXACTLY as in training: t_value = sigmoid(N(P_mean, P_std)) per sample (logit-normal),
then x_t = t·x0 + (1-t)·ε and t_discrete = round(t·(T-1)) (FM: t=0 noise → t=1 clean). permute_atlas
is a no-op on the loss VALUE for sinkhorn (within-patch permutation-invariant) so canonical GT is used.
Conditional forward (no class-dropout) in eval. No rendering — this is the pure recon term only.

Outputs:
  loss_dist_per_sample.png        — histogram of the per-sample recon loss (the scalar that's optimized)
  loss_dist_per_gaussian.png      — histogram (log-x) of the per-Gaussian matched cost + Lorenz/tail
  loss_vs_t.png                   — per-sample loss vs its sampled t (+ binned mean)
  loss_dist_by_tband.png          — per-Gaussian cost distribution split by t-band
  dist.npz / metadata.json

Defaults match huber_d1.7 (class-cond sinkhorn_patch_hard P=8 eps0.05 iters100 huber1.7 cw0.07).
"""
from __future__ import annotations

import argparse, glob, json, math, os, sys
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

from dataloaders.standard_3dgen_loader import Standard3DGenDataset  # noqa: E402
from dataloaders.text_3dgen_loader import (  # noqa: E402
    Text3DGenDataset, DC_ONLY_FEATURE_INDICES, FULL_3DGS_FEATURE_DIM,
)
from jit.models import JiT_3DGS_models  # noqa: E402
from jit.diffusion import create_diffusion  # noqa: E402
from jit.diffusion.gaussian_diffusion import _sinkhorn_log  # noqa: E402  (exact training kernel)


def find_latest_checkpoint(root):
    cands = [c for c in glob.glob(os.path.join(root, "**", "*.pt"), recursive=True) if Path(c).stem.isdigit()]
    return max(cands, key=os.path.getmtime) if cands else None


def to_patches(t, P):
    B, C, H, W = t.shape
    nH, nW = H // P, W // P
    return (t.float().reshape(B, C, nH, P, nW, P).permute(0, 2, 4, 1, 3, 5)
            .reshape(B, nH * nW, C, P * P).transpose(-1, -2).contiguous())


def huber_sq(sq, delta):
    """Pseudo-Huber on the squared distance — IDENTICAL to training. delta<=0 → identity."""
    if delta and delta > 0.0:
        d2 = delta * delta
        return 2.0 * sq / ((1.0 + sq / d2).sqrt() + 1.0)
    return sq


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, default="JiT-B/8", choices=list(JiT_3DGS_models.keys()))
    p.add_argument("--bottleneck", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--ckpt_search_root", type=str, default="output")
    p.add_argument("--weights", type=str, default="model", choices=("ema", "model"),
                   help="'model' = the live trained weights the training loss was computed on (default).")
    p.add_argument("--noise_schedule", type=str, default="squaredcos_cap_v2")
    # Conditioning
    p.add_argument("--class_map_path", type=str, default=None)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--text_embed_path", type=str, default=None)
    # Data
    p.add_argument("--obj_list", type=str, required=True)
    p.add_argument("--gs_path", type=str, required=True)
    p.add_argument("--mean_file", type=str, default=None)
    p.add_argument("--std_file", type=str, default=None)
    p.add_argument("--rank_transform_file", type=str, default=None)
    p.add_argument("--clip_thresholds_file", type=str, default=None)
    p.add_argument("--sphere2plane_path", type=str, required=True)
    p.add_argument("--exclude_keys_file", type=str, default=None)
    p.add_argument("--sh_degree0_only", action=argparse.BooleanOptionalAction, default=True)
    # EXACT trained loss knobs
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--sinkhorn_eps", type=float, default=0.05)
    p.add_argument("--sinkhorn_iters", type=int, default=100)
    p.add_argument("--huber_delta", type=float, default=1.7)
    p.add_argument("--chamfer_loss_weight", type=float, default=0.07,
                   help="Outer scalar on the recon loss (what enters total_loss).")
    # Training timestep distribution (logit-normal) — drawn exactly as training
    p.add_argument("--p_mean", type=float, default=0.0)
    p.add_argument("--p_std", type=float, default=1.5)
    # Probe
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=("bf16", "fp16", "none"))
    p.add_argument("--max_pergauss", type=int, default=3_000_000,
                   help="Cap on stored per-Gaussian (cost,t) pairs for the histograms.")
    p.add_argument("--output_dir", type=str, default=None)
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    P = int(args.patch_size)

    ckpt_path = args.resume or find_latest_checkpoint(args.ckpt_search_root)
    if ckpt_path is None:
        raise FileNotFoundError("No --resume and none found under --ckpt_search_root.")
    out_dir = args.output_dir or os.path.join(
        REPO_ROOT, "jit", "sinkhorn_loss_distribution_out",
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[ckpt] {ckpt_path}  (weights={args.weights})")
    print(f"[out]  {out_dir}")
    print(f"[loss] sinkhorn_patch_hard P={P} eps={args.sinkhorn_eps} iters={args.sinkhorn_iters} "
          f"huber={args.huber_delta} cw={args.chamfer_loss_weight} | t~sigmoid(N({args.p_mean},{args.p_std}))")

    # Features
    if args.sh_degree0_only:
        feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
        in_channels = len(DC_ONLY_FEATURE_INDICES)
    else:
        feature_indices = None
        in_channels = FULL_3DGS_FEATURE_DIM

    # Conditioning
    class_conditioned = bool(args.class_map_path)
    class_map = None
    num_classes = None
    if class_conditioned:
        with open(args.class_map_path) as f:
            class_map = json.load(f)
        num_classes = args.num_classes or (max(int(v) for v in class_map.values()) + 1)
        print(f"[cond] class-conditioned, num_classes={num_classes}, {len(class_map):,} labeled")
    else:
        print("[cond] text-conditioned")

    base = Standard3DGenDataset(
        obj_list=[args.obj_list], gs_path=args.gs_path, caption_path=None,
        mean_file=args.mean_file, std_file=args.std_file,
        sphere2plane_path=args.sphere2plane_path, exclude_keys_file=args.exclude_keys_file,
        rank_transform_file=args.rank_transform_file, clip_thresholds_file=args.clip_thresholds_file,
        text_embed_path=(None if class_conditioned else args.text_embed_path),
    )
    if class_conditioned:
        before = len(base.obj_data)
        base.obj_data = {h: p for h, p in base.obj_data.items() if p.split('.tar.gz')[0] in class_map}
        base.keys = list(base.obj_data.keys())
        print(f"[class] {before:,} -> {len(base.obj_data):,} objects with a class label")
        text_dim = None
    else:
        text_dim = int(base.text_pooled.shape[1])

    dataset = Text3DGenDataset(base, feature_indices=feature_indices, return_full_for_render=False,
                               preload_to_cpu=False, lazy_cache_to_cpu=False, class_map=class_map)
    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), drop_last=False, generator=g)
    print(f"[data] dataset={len(dataset)}, probing {args.num_samples} @ batch {args.batch_size}")

    model = JiT_3DGS_models[args.model](
        input_size=128, in_channels=in_channels, text_dim=(text_dim or 768),
        num_classes=num_classes, class_dropout_prob=0.0, learn_sigma=False,
        gradient_checkpointing=False, bottleneck=args.bottleneck)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt[args.weights]
    # torch.compile saves the live model with an `_orig_mod.` prefix on every key
    # (the `ema` copy is clean). Strip it so `--weights model` loads correctly.
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
                f"weights did NOT load. Check --model/--bottleneck/--class_map_path/--weights. "
                f"missing(head)={list(missing)[:4]} unexpected(head)={list(unexpected)[:4]}")
    step = int(ckpt.get("step", ckpt.get("opt_step", -1)))
    print(f"[load] checkpoint step={step}")
    model.to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)

    diffusion = create_diffusion(timestep_respacing="", noise_schedule=args.noise_schedule,
                                 learn_sigma=False, predict_xstart=True)
    T = diffusion.num_timesteps
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.mixed_precision]
    M = P * P

    per_sample_loss = []   # weighted (×cw), what training optimizes
    per_sample_raw = []    # unweighted mean matched huber cost
    per_sample_t = []
    pg_cost = []           # per-Gaussian matched huber cost (capped reservoir)
    pg_t = []
    collision = []
    n_seen = 0
    t_gen = torch.Generator(device=device); t_gen.manual_seed(args.seed * 7 + 13)

    for bidx, batch in enumerate(loader):
        if n_seen >= args.num_samples:
            break
        x, y_cond, _ = batch
        rem = args.num_samples - n_seen
        if x.shape[0] > rem:
            x = x[:rem]; y_cond = y_cond[:rem]
        bsz = x.shape[0]
        x = x.to(device).float()
        y_cond = (y_cond.to(device).long() if class_conditioned else y_cond.to(device).float())

        # t EXACTLY as training: logit-normal per sample.
        logit = args.p_mean + args.p_std * torch.randn(bsz, generator=t_gen, device=device)
        t_value = torch.sigmoid(logit)                                  # (B,) in (0,1)
        t_discrete = torch.clamp((t_value * (T - 1)).round().long(), 0, T - 1)
        noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=t_gen)
        x_t = diffusion.flow_matching_q_sample(x, t_value, noise=noise)
        with torch.no_grad():
            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = model(x_t, t_discrete, y_cond)
            else:
                pred = model(x_t, t_discrete, y_cond)

        xp = to_patches(pred.float(), P)            # (B,nP,M,C)
        yp = to_patches(x, P)
        aa = xp.pow(2).sum(-1); bb = yp.pow(2).sum(-1)
        ab = xp @ yp.transpose(-1, -2)
        sq = (aa.unsqueeze(-1) + bb.unsqueeze(-2) - 2 * ab).clamp_min(0)   # raw
        sq_h = huber_sq(sq, args.huber_delta)                             # the run's ground cost
        Pi = _sinkhorn_log(sq_h, args.sinkhorn_eps, args.sinkhorn_iters)
        sigma = Pi.argmax(dim=-1)                                          # (B,nP,M)
        hard = sq_h.gather(-1, sigma.unsqueeze(-1)).squeeze(-1)            # (B,nP,M) per-Gaussian loss term

        ls_raw = hard.mean(dim=(1, 2))                                    # (B,) per-sample mean matched cost
        per_sample_raw.extend(ls_raw.cpu().tolist())
        per_sample_loss.extend((args.chamfer_loss_weight * ls_raw).cpu().tolist())
        per_sample_t.extend(t_value.cpu().tolist())

        # collision fraction (bijection monitor)
        cc = torch.zeros_like(sigma, dtype=sq.dtype)
        cc.scatter_add_(-1, sigma, torch.ones_like(sigma, dtype=sq.dtype))
        collision.extend(((cc - 1).clamp_min(0).sum(-1) / M).mean(-1).cpu().tolist())

        if sum(len(a) for a in pg_cost) < args.max_pergauss:
            flat = hard.reshape(bsz, -1)                                   # (B, 16384)
            pg_cost.append(flat.cpu().numpy().ravel())
            pg_t.append(np.repeat(t_value.cpu().numpy(), flat.shape[1]))

        n_seen += bsz
        if bidx % 8 == 0:
            print(f"  [{n_seen}/{args.num_samples}] batch {bidx}")

    per_sample_loss = np.asarray(per_sample_loss)
    per_sample_raw = np.asarray(per_sample_raw)
    per_sample_t = np.asarray(per_sample_t)
    pg_cost = np.concatenate(pg_cost) if pg_cost else np.zeros(0)
    pg_t = np.concatenate(pg_t) if pg_t else np.zeros(0)
    collision = float(np.mean(collision)) if collision else 0.0

    qs = np.percentile(per_sample_loss, [5, 25, 50, 75, 95])
    print("\n[result] PURE-RECON loss distribution (training-sampled t, weighted ×cw)")
    print(f"  per-sample loss: mean={per_sample_loss.mean():.5f}  median={qs[2]:.5f}  "
          f"p5={qs[0]:.5f}  p95={qs[4]:.5f}  std={per_sample_loss.std():.5f}")
    print(f"  per-sample RAW (unweighted matched cost): mean={per_sample_raw.mean():.4f}")
    print(f"  per-Gaussian matched cost: mean={pg_cost.mean():.4f}  median={np.median(pg_cost):.4f}  "
          f"p99={np.percentile(pg_cost,99):.4f}  max={pg_cost.max():.4f}  (n={pg_cost.size:,})")
    print(f"  mean collision (bijection gap): {collision*100:.2f}%")
    # tail concentration: fraction of total per-Gaussian loss carried by the worst k%
    sc = np.sort(pg_cost)[::-1]
    tot = sc.sum()
    for k in (1, 5, 10):
        topk = sc[:max(1, int(len(sc) * k / 100))].sum() / tot
        print(f"  worst {k:>2d}% of Gaussians carry {topk*100:.1f}% of the recon loss")

    np.savez(os.path.join(out_dir, "dist.npz"),
             per_sample_loss=per_sample_loss, per_sample_raw=per_sample_raw, per_sample_t=per_sample_t,
             pergauss_cost=pg_cost, pergauss_t=pg_t)
    meta = dict(checkpoint=ckpt_path, checkpoint_step=step, weights=args.weights, model=args.model,
                recon_loss="sinkhorn_patch_hard", patch_size=P, sinkhorn_eps=args.sinkhorn_eps,
                sinkhorn_iters=args.sinkhorn_iters, huber_delta=args.huber_delta,
                chamfer_loss_weight=args.chamfer_loss_weight, p_mean=args.p_mean, p_std=args.p_std,
                class_conditioned=class_conditioned, num_classes=num_classes, num_samples=n_seen,
                per_sample_loss_mean=float(per_sample_loss.mean()),
                per_sample_loss_median=float(qs[2]), collision_frac=collision)
    json.dump(meta, open(os.path.join(out_dir, "metadata.json"), "w"), indent=2)

    title = (f"{args.model} {args.weights} step {step} | sinkhorn_patch_hard P={P} "
             f"huber={args.huber_delta} cw={args.chamfer_loss_weight} | t~logit-N({args.p_mean},{args.p_std}) "
             f"| N={n_seen} | PURE RECON")

    # 1) per-sample loss histogram
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(per_sample_loss, bins=60, color="C0", alpha=0.85, edgecolor="white", lw=0.3)
    for q, lab, c in [(qs[2], "median", "C3"), (per_sample_loss.mean(), "mean", "C1")]:
        ax.axvline(q, color=c, ls="--", lw=1.5, label=f"{lab}={q:.4f}")
    ax.axvspan(qs[0], qs[4], color="gray", alpha=0.12, label="p5–p95")
    ax.set_xlabel("per-sample recon loss  (cw · mean_Gaussians huber-matched cost)")
    ax.set_ylabel("# objects"); ax.set_title("Per-sample recon-loss distribution\n" + title, fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_dist_per_sample.png"), dpi=130); plt.close(fig)

    # 2) per-Gaussian cost histogram (log-x) + tail
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pos = pg_cost[pg_cost > 0]
    bins = np.logspace(np.log10(max(pos.min(), 1e-4)), np.log10(pos.max()), 80)
    ax.hist(pos, bins=bins, color="C2", alpha=0.85)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(np.median(pg_cost), color="C3", ls="--", lw=1.5, label=f"median={np.median(pg_cost):.3f}")
    ax.axvline(pg_cost.mean(), color="C1", ls="--", lw=1.5, label=f"mean={pg_cost.mean():.3f}")
    ax.set_xlabel("per-Gaussian matched cost  huber(‖pred−gt_σ‖²)  [log]")
    ax.set_ylabel("# Gaussians [log]")
    ax.set_title("Per-Gaussian loss-term distribution (heavy tail)\n" + title, fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_dist_per_gaussian.png"), dpi=130); plt.close(fig)

    # 3) loss vs t (per-sample scatter + binned mean)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(per_sample_t, per_sample_loss, s=8, alpha=0.35, color="C0", label="per object")
    tb = np.linspace(0, 1, 11)
    idx = np.digitize(per_sample_t, tb) - 1
    bm = [per_sample_loss[idx == k].mean() if (idx == k).any() else np.nan for k in range(10)]
    ax.plot(0.5 * (tb[:-1] + tb[1:]), bm, "s-", color="C3", lw=2, label="binned mean")
    ax.set_xlabel("t_value  (0=noise, 1=clean)"); ax.set_ylabel("per-sample recon loss")
    ax.set_title("Recon loss vs sampled t\n" + title, fontsize=9); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_vs_t.png"), dpi=130); plt.close(fig)

    # 4) per-Gaussian cost distribution by t-band
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    cmap = plt.get_cmap("viridis")
    bins = np.logspace(np.log10(max(pg_cost[pg_cost > 0].min(), 1e-4)), np.log10(pg_cost.max()), 70)
    for bi, (lo, hi) in enumerate(bands):
        m = (pg_t >= lo) & (pg_t < hi) & (pg_cost > 0)
        if m.sum() < 50:
            continue
        ax.hist(pg_cost[m], bins=bins, histtype="step", lw=1.8, density=True,
                color=cmap(bi / (len(bands) - 1)), label=f"t∈[{lo:.1f},{hi:.1f}) (med={np.median(pg_cost[m]):.3f})")
    ax.set_xscale("log"); ax.set_xlabel("per-Gaussian matched cost [log]"); ax.set_ylabel("density")
    ax.set_title("Per-Gaussian cost distribution by noise band\n" + title, fontsize=9)
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_dist_by_tband.png"), dpi=130); plt.close(fig)

    print(f"\n[done] wrote figures + dist.npz + metadata.json to:\n  {out_dir}")


if __name__ == "__main__":
    main()
