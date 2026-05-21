#!/usr/bin/env python
"""Atlas discontinuity probe — does the sphere->plane projection's local
non-smoothness predict where x0-prediction MSE is high?

Hypothesis: a single fixed `sphere2plane.npy` permutation cannot be locally
smooth for all objects, so plane-adjacent pixels can hold dissimilar Gaussians.
The conv patch-embed (8x8) and RoPE assume local coherence, so the model should
struggle most where the plane layout is most discontinuous.

For each plane pixel we compute three per-pixel maps over a small object set,
all in the normalized training space the model actually sees (DC-only 14ch):

  E_t[i,j]  error      = mean_n mean_c (x0 - pred_t)^2            (per t, needs model)
  D[i,j]    discontinuity = mean_n mean_{nb} mean_c (x0 - x0_nb)^2  (4-neighbor, x0 only)
  V[i,j]    variance   = mean_c Var_n( x0[c,i,j] )                (across-object difficulty)

CONFOUND: high-variance pixels are both harder (high E) and more discontinuous
(high D) by construction, so a raw corr(E, D) can be spurious. The headline is
the PARTIAL correlation r(E, D | V): does discontinuity predict error *beyond*
what intrinsic per-pixel variance explains? If it stays strongly positive, the
projection's local non-smoothness is independently implicated.

Expected trend if the mechanism is real: partial r(E,D|V) should RISE with t
(toward clean), because x_t = t*x0 + (1-t)*eps reveals more of x0's layout as
t->1; at low t the noise masks the layout. (Caveat: t>=0.8 is also where render
loss acts, per render_loss_noise_cutoff — read that end with that in mind.)

Mirrors jit/mse_atlas_probe.py for data/model loading. Foreground, single GPU,
small batch — safe to run alongside training.
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


# ── small helpers (kept standalone; mirrors jit/mse_atlas_probe.py) ────────
def find_latest_checkpoint(search_root: str) -> str | None:
    cands = [
        c for c in glob.glob(os.path.join(search_root, "**", "*.pt"), recursive=True)
        if Path(c).stem.isdigit()
    ]
    return max(cands, key=os.path.getmtime) if cands else None


def parse_t_values(raw: str) -> list[float]:
    vals = [float(v) for v in raw.split(",") if v.strip() != ""]
    if not vals:
        raise ValueError(f"--t_values produced an empty list: {raw!r}")
    for v in vals:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"t_value {v} out of range [0, 1]")
    return vals


def neighbor_discontinuity(x: torch.Tensor, torus: bool) -> torch.Tensor:
    """Per-object 4-neighbor channel-mean squared difference.

    x: [B, C, H, W] -> [B, H, W]. Bounded edges average over available
    neighbors; torus=True wraps left/right and top/bottom.
    """
    B, C, H, W = x.shape
    acc = torch.zeros(B, H, W, device=x.device, dtype=x.dtype)
    cnt = torch.zeros(H, W, device=x.device, dtype=x.dtype)
    # horizontal
    dh = ((x[:, :, :, 1:] - x[:, :, :, :-1]) ** 2).mean(1)  # [B,H,W-1]
    acc[:, :, 1:] += dh; acc[:, :, :-1] += dh
    cnt[:, 1:] += 1; cnt[:, :-1] += 1
    # vertical
    dv = ((x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2).mean(1)  # [B,H-1,W]
    acc[:, 1:, :] += dv; acc[:, :-1, :] += dv
    cnt[1:, :] += 1; cnt[:-1, :] += 1
    if torus:
        dwrap = ((x[:, :, :, 0] - x[:, :, :, -1]) ** 2).mean(1)  # [B,H]
        acc[:, :, 0] += dwrap; acc[:, :, -1] += dwrap
        cnt[:, 0] += 1; cnt[:, -1] += 1
        dwrap_v = ((x[:, :, 0, :] - x[:, :, -1, :]) ** 2).mean(1)  # [B,W]
        acc[:, 0, :] += dwrap_v; acc[:, -1, :] += dwrap_v
        cnt[0, :] += 1; cnt[-1, :] += 1
    return acc / cnt


# ── correlation stats over flattened pixel maps ────────────────────────────
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / (denom + 1e-12))


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def _partial(r_ab: float, r_ac: float, r_bc: float) -> float:
    denom = np.sqrt(max(1e-12, (1.0 - r_ac ** 2) * (1.0 - r_bc ** 2)))
    return float((r_ab - r_ac * r_bc) / denom)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, default="JiT-B/8", choices=list(JiT_3DGS_models.keys()))
    p.add_argument("--bottleneck", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--ckpt_search_root", type=str, default="output")
    p.add_argument("--weights", type=str, default="ema", choices=("ema", "model"))
    p.add_argument("--noise_schedule", type=str, default="squaredcos_cap_v2")
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
    p.add_argument("--t_values", type=str, default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=("bf16", "fp16", "none"))
    p.add_argument("--torus", action=argparse.BooleanOptionalAction, default=False,
                   help="Treat the plane as periodic (wrap edges) when measuring discontinuity.")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--cmap", type=str, default="magma")
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    t_values = parse_t_values(args.t_values)

    ckpt_path = args.resume or find_latest_checkpoint(args.ckpt_search_root)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint under {args.ckpt_search_root!r}; pass --resume.")
    print(f"[ckpt] {ckpt_path}  (weights={args.weights})")

    out_dir = args.output_dir or os.path.join(
        REPO_ROOT, "jit", "atlas_discontinuity_out",
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out]  {out_dir}")

    if args.sh_degree0_only:
        feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
        in_channels = len(DC_ONLY_FEATURE_INDICES)
    else:
        feature_indices = None
        in_channels = FULL_3DGS_FEATURE_DIM

    base = Standard3DGenDataset(
        obj_list=[args.obj_list], gs_path=args.gs_path, caption_path=None,
        mean_file=args.mean_file, std_file=args.std_file,
        sphere2plane_path=args.sphere2plane_path, exclude_keys_file=args.exclude_keys_file,
        rank_transform_file=args.rank_transform_file,
        clip_thresholds_file=args.clip_thresholds_file, text_embed_path=args.text_embed_path,
    )
    text_dim = int(base.text_pooled.shape[1])
    dataset = Text3DGenDataset(base, feature_indices=feature_indices,
                               return_full_for_render=False,
                               preload_to_cpu=False, lazy_cache_to_cpu=False)
    loader_gen = torch.Generator(); loader_gen.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        drop_last=False, generator=loader_gen)
    print(f"[data] dataset={len(dataset)}, probing {args.num_samples} @ batch {args.batch_size}, "
          f"in_channels={in_channels}, torus={args.torus}")

    model = JiT_3DGS_models[args.model](
        input_size=128, in_channels=in_channels, text_dim=text_dim,
        class_dropout_prob=0.0, learn_sigma=False,
        gradient_checkpointing=False, bottleneck=args.bottleneck,
    )
    if args.null_text_token_path and os.path.exists(args.null_text_token_path):
        null_np = load_null_text_token(args.null_text_token_path)
        model.load_null_embeddings(torch.from_numpy(null_np.astype(np.float32)))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt[args.weights], strict=False)
    if missing or unexpected:
        print(f"[load] non-strict: {len(missing)} missing, {len(unexpected)} unexpected")
    step = int(ckpt.get("step", ckpt.get("opt_step", -1)))
    print(f"[load] checkpoint step={step}")
    model.to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)

    diffusion = create_diffusion(timestep_respacing="", noise_schedule=args.noise_schedule,
                                 learn_sigma=False, predict_xstart=True)
    T = diffusion.num_timesteps
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.mixed_precision]

    # ── Accumulators (float64 on CPU) ─────────────────────────────────────
    H = W = 128
    sumsq = {t: torch.zeros(H, W, dtype=torch.float64) for t in t_values}  # E_t numerator
    sumD = torch.zeros(H, W, dtype=torch.float64)                          # discontinuity
    S1 = torch.zeros(in_channels, H, W, dtype=torch.float64)               # sum x0  (for V)
    S2 = torch.zeros(in_channels, H, W, dtype=torch.float64)               # sum x0^2
    n_seen = 0

    for batch_idx, batch in enumerate(loader):
        if n_seen >= args.num_samples:
            break
        x, y_pooled, _ = batch
        remaining = args.num_samples - n_seen
        if x.shape[0] > remaining:
            x = x[:remaining]; y_pooled = y_pooled[:remaining]
        bsz = x.shape[0]
        x = x.to(device, non_blocking=True).float()
        y_pooled = y_pooled.to(device, non_blocking=True).float()

        # Data-only maps (independent of t): discontinuity + variance moments.
        disc = neighbor_discontinuity(x, args.torus)          # [B,H,W]
        sumD += disc.sum(0).double().cpu()
        S1 += x.sum(0).double().cpu()
        S2 += (x ** 2).sum(0).double().cpu()

        # Fixed noise reused across all t for this batch.
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
            chmean = ((x - pred.float()) ** 2).mean(dim=1)    # [B,H,W]
            sumsq[t] += chmean.sum(0).double().cpu()

        n_seen += bsz
        print(f"  [{n_seen}/{args.num_samples}] batch {batch_idx} done")

    if n_seen == 0:
        raise RuntimeError("No samples processed.")

    # ── Finalize maps ─────────────────────────────────────────────────────
    E = {t: (sumsq[t] / n_seen).numpy() for t in t_values}                 # [H,W] per t
    D = (sumD / n_seen).numpy()                                            # [H,W]
    mean_x = S1 / n_seen
    var_c = (S2 / n_seen) - mean_x ** 2                                    # [C,H,W]
    V = var_c.clamp_min(0).mean(0).numpy()                                 # [H,W]

    Dflat, Vflat = D.reshape(-1), V.reshape(-1)
    rD = _rank(Dflat); rV = _rank(Vflat)
    r_DV = _pearson(Dflat, Vflat)
    sr_DV = _pearson(rD, rV)

    rows = []
    print("\n[corr] per-pixel correlations (n={} pixels, N={} objects)".format(H * W, n_seen))
    print("  t      r(E,D)  r(E,V)  r(D,V) | partial r(E,D|V) | sp r(E,D)  partial_sp(E,D|V)")
    for t in t_values:
        Eflat = E[t].reshape(-1)
        rE = _rank(Eflat)
        r_ED = _pearson(Eflat, Dflat)
        r_EV = _pearson(Eflat, Vflat)
        pr_EDV = _partial(r_ED, r_EV, r_DV)
        sr_ED = _pearson(rE, rD)
        sr_EV = _pearson(rE, rV)
        spr_EDV = _partial(sr_ED, sr_EV, sr_DV)
        rows.append(dict(t=t, r_ED=r_ED, r_EV=r_EV, r_DV=r_DV, partial_EDV=pr_EDV,
                         spearman_ED=sr_ED, spearman_partial_EDV=spr_EDV))
        print(f"  {t:<5}  {r_ED:6.3f}  {r_EV:6.3f}  {r_DV:6.3f} |     {pr_EDV:6.3f}     |  "
              f"{sr_ED:6.3f}      {spr_EDV:6.3f}")

    # ── Save raw arrays + metrics ─────────────────────────────────────────
    np.savez(os.path.join(out_dir, "maps.npz"),
             t_values=np.asarray(t_values), E=np.stack([E[t] for t in t_values], 0),
             D=D, V=V)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"checkpoint": ckpt_path, "checkpoint_step": step, "weights": args.weights,
                   "num_samples": n_seen, "torus": bool(args.torus), "r_DV": r_DV,
                   "rows": rows}, f, indent=2)

    # ── Figure 1: D and V maps ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    for ax, m, title in ((axes[0], D, "discontinuity D"), (axes[1], V, "across-object variance V")):
        im = ax.imshow(m, cmap=args.cmap)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Atlas layout maps  |  N={n_seen}  (r(D,V)={r_DV:.2f})", fontsize=11)
    fig.savefig(os.path.join(out_dir, "maps_D_V.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: D vs E_t hexbin per t ───────────────────────────────────
    n = len(t_values)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.4), squeeze=False)
    for j, t in enumerate(t_values):
        ax = axes[0, j]
        hb = ax.hexbin(Dflat, E[t].reshape(-1), gridsize=40, bins="log", cmap="viridis",
                       xscale="log", yscale="log", mincnt=1)
        row = rows[j]
        ax.set_title(f"t={t}\nr={row['r_ED']:.2f}  partial={row['partial_EDV']:.2f}", fontsize=9)
        ax.set_xlabel("discontinuity D")
        if j == 0:
            ax.set_ylabel("error E")
    fig.colorbar(hb, ax=axes[0, :].tolist(), fraction=0.02, pad=0.02, label="log count")
    fig.suptitle("Per-pixel error vs layout discontinuity (log-log)", fontsize=11)
    fig.savefig(os.path.join(out_dir, "scatter_E_vs_D.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: correlation summary vs t ────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(t_values, [r["r_ED"] for r in rows], "o-", label="r(E,D) raw")
    ax.plot(t_values, [r["r_EV"] for r in rows], "s--", label="r(E,V)")
    ax.plot(t_values, [r["partial_EDV"] for r in rows], "D-", color="crimson",
            label="partial r(E,D|V)  ← headline")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("t_value  (0 = noise, 1 = clean)")
    ax.set_ylabel("correlation")
    ax.set_title(f"Does projection discontinuity predict error?  step {step}, N={n_seen}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "corr_vs_t.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] wrote maps.npz + metrics.json + 3 figures to:\n  {out_dir}")


if __name__ == "__main__":
    main()
