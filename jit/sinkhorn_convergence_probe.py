#!/usr/bin/env python
"""Sinkhorn convergence probe — how many log-Sinkhorn iterations does the training
loss actually need, as a function of ``sinkhorn_eps`` and the noise level t?

WHY. ``_chamfer_recon_loss(sinkhorn_patch)`` runs a FIXED ``sinkhorn_iters`` (50) log-
domain Sinkhorn and DETACHES the plan, so the gradient is a soft-weighted MSE toward
``M·(Π @ gt)`` (the plan-blended target). Two things therefore matter for "is 50 enough":
  1. the plan must be ~doubly-stochastic (else it stops enforcing the bijection/coverage),
  2. the quantities training actually consumes — the loss ⟨Π, sq⟩ and the blended target
     Π @ gt — must have stopped moving.
Convergence rate scales ~O(1/eps) and depends on cost sharpness, which changes with t
(noisy → degenerate costs, fast; clean → sharp costs, slow) and over training. So the
right test sweeps iters × eps on REAL costs from a checkpoint.

WHAT IT REPORTS, per (eps, t, iters), vs an ``--ref_iters`` reference plan:
  * resid       — mean |Π.sum(dim=-1) − 1/M|, the row-marginal violation (the recursion
                  ends on the column update so the COLUMN marginal is exact; the row one
                  carries the error). →0 as the plan becomes doubly-stochastic. This is the
                  exact quantity logged online during training (see _chamfer_recon_loss).
  * loss        — ⟨Π, sq⟩ (mean over patches & batch), the detached training term.
  * loss_drift  — |loss − loss_ref| / loss_ref.
  * blend_drift — ‖M·(Π@gt) − M·(Π_ref@gt)‖_F / ‖M·(Π_ref@gt)‖_F: relative change in WHAT
                  THE GRADIENT PULLS TOWARD. Because the plan is detached, this is the most
                  decision-relevant convergence target — a plan can still show marginal
                  residual while the blended target it induces has already settled.
The reference's OWN resid is printed so you can see whether ``--ref_iters`` itself converged
(it may not, for very small eps).

For speed under a live training run the recursion is run ONCE to ref_iters per (eps, t),
snapshotting the plan at the grid iters — bit-for-bit identical to calling the training
kernel ``_sinkhorn_log`` with each iters value (verified by an assert), but ~Σgrid cheaper.

Single-GPU example (defaults match jit_sinkhorn_tune20260525_202939 / jit_train_gsplat.yaml):

    python jit/sinkhorn_convergence_probe.py \
        --resume output/jit_sinkhorn_tune20260525_202939/0003000.pt --weights ema \
        --patch_size 8 --eps_grid 0.05,0.02,0.01 --t_values 0.1,0.5,0.9 \
        --iters_grid 10,25,50,75,100,150,200,300,400 --ref_iters 2000 \
        --num_samples 16 --batch_size 16
"""
from __future__ import annotations

import argparse
import glob
import json
import math
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
    cands = [
        c for c in glob.glob(os.path.join(search_root, "**", "*.pt"), recursive=True)
        if Path(c).stem.isdigit()
    ]
    return max(cands, key=os.path.getmtime) if cands else None


def parse_floats(raw: str) -> list[float]:
    vals = [float(v) for v in raw.split(",") if v.strip() != ""]
    if not vals:
        raise ValueError(f"empty list from {raw!r}")
    return vals


def parse_ints(raw: str) -> list[int]:
    vals = [int(v) for v in raw.split(",") if v.strip() != ""]
    if not vals:
        raise ValueError(f"empty list from {raw!r}")
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


def sinkhorn_snapshots(C: torch.Tensor, eps: float, snap_iters: list[int], ref_iters: int):
    """Run log-Sinkhorn ONCE to ``ref_iters``, returning {iters: plan} for each iters in
    ``snap_iters ∪ {ref_iters}``. Mirrors ``_sinkhorn_log`` exactly (zero init, f-then-g
    per iteration, plan = exp(logK + f + g)), so snaps[k] == _sinkhorn_log(C, eps, k)."""
    M = C.shape[-1]
    logK = -C / eps
    logm = -math.log(M)
    f = torch.zeros(C.shape[:-1], device=C.device, dtype=C.dtype)
    g = torch.zeros_like(f)
    want = set(int(i) for i in snap_iters) | {int(ref_iters)}
    max_it = max(want)
    snaps: dict[int, torch.Tensor] = {}
    for it in range(1, max_it + 1):
        f = logm - torch.logsumexp(logK + g.unsqueeze(-2), dim=-1)
        g = logm - torch.logsumexp(logK + f.unsqueeze(-1), dim=-2)
        if it in want:
            snaps[it] = torch.exp(logK + f.unsqueeze(-1) + g.unsqueeze(-2))
    return snaps


def plan_metrics(plan: torch.Tensor, sq: torch.Tensor, yp: torch.Tensor, M: int):
    """resid (row-marginal violation), loss ⟨Π,sq⟩, blended target M·(Π@gt)."""
    resid = (plan.sum(dim=-1) - (1.0 / M)).abs().mean()
    loss = (plan * sq).sum(dim=(-1, -2)).mean(dim=1).mean()      # mean over patches & batch
    blend = M * torch.matmul(plan, yp)                           # (B, nP, M, C)
    return float(resid), float(loss), blend


def hard_metrics(plan: torch.Tensor, sq: torch.Tensor, M: int):
    """Hard-rounded assignment σ(i)=argmax_j Π[i,j] — the candidate hard-variant loss.
    Returns σ, the hard loss mean_i ‖pred_i−tgt_σ(i)‖² (= mean_i sq[i,σ(i)]), and the
    collision fraction (preds whose argmax column is shared with another pred in the same
    patch — i.e. how far the rounded assignment is from a true bijection)."""
    sigma = plan.argmax(dim=-1)                                       # (B, nP, M)
    hard_sq = sq.gather(-1, sigma.unsqueeze(-1)).squeeze(-1)          # (B, nP, M) = sq[i,σ(i)]
    hard_loss = float(hard_sq.mean())
    col_counts = torch.zeros_like(hard_sq)                            # (B, nP, M) over columns j
    col_counts.scatter_add_(-1, sigma, torch.ones_like(hard_sq))
    extra = (col_counts - 1.0).clamp_min(0).sum(dim=-1)              # duplicate picks per patch
    collision = float((extra / M).mean())
    return sigma, hard_loss, collision


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
    # Data (defaults mirror jit_sinkhorn_tune20260525_202939)
    p.add_argument("--obj_list", type=str, default="data/gaussianverse/all_obj_list_filtered.json")
    p.add_argument("--gs_path", type=str, default="data/gaussianverse")
    p.add_argument("--mean_file", type=str, default="data/stats/all_mean_postfix.pt")
    p.add_argument("--std_file", type=str, default="data/stats/all_std_postfix.pt")
    p.add_argument("--rank_transform_file", type=str, default="data/stats/rank_quantiles_8ch_clipped.pt")
    p.add_argument("--clip_thresholds_file", type=str, default="data/stats/clip_thresholds_opacity_scales.pt")
    p.add_argument("--text_embed_path", type=str, default="object_classification/text_tokens")
    p.add_argument("--null_text_token_path", type=str, default="object_classification/null_text_token.npz")
    p.add_argument("--sphere2plane_path", type=str, default="data/gaussianverse/sphere2plane.npy")
    p.add_argument("--exclude_keys_file", type=str, default="data/outlier_keys_8sigma.json")
    p.add_argument("--sh_degree0_only", action=argparse.BooleanOptionalAction, default=True)
    # Sweep knobs
    p.add_argument("--patch_size", type=int, default=8, help="P (chamfer_patch_size); M=P² cells/patch.")
    p.add_argument("--eps_grid", type=str, default="0.05,0.02,0.01")
    p.add_argument("--iters_grid", type=str, default="10,25,50,75,100,150,200,300,400")
    p.add_argument("--ref_iters", type=int, default=2000, help="Converged-reference iteration count.")
    p.add_argument("--t_values", type=str, default="0.1,0.5,0.9", help="FM: 0=noise, 1=clean.")
    # Probe params (kept small to coexist with a live training run)
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=("bf16", "fp16", "none"))
    p.add_argument("--output_dir", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    eps_grid = parse_floats(args.eps_grid)
    iters_grid = sorted(set(parse_ints(args.iters_grid)))
    t_values = parse_floats(args.t_values)
    P = int(args.patch_size)
    M = P * P
    REF = int(args.ref_iters)
    if REF <= max(iters_grid):
        raise ValueError(f"--ref_iters ({REF}) must exceed max(iters_grid)={max(iters_grid)}")

    # ── Resolve checkpoint / output ───────────────────────────────────────
    ckpt_path = args.resume or find_latest_checkpoint(args.ckpt_search_root)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint given and none under {args.ckpt_search_root!r}.")
    out_dir = args.output_dir or os.path.join(
        REPO_ROOT, "jit", "sinkhorn_convergence_probe_out",
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[ckpt] {ckpt_path}  (weights={args.weights})")
    print(f"[out]  {out_dir}")
    print(f"[grid] eps={eps_grid}  iters={iters_grid}  ref={REF}  t={t_values}  P={P} (M={M})")

    # ── Feature selection (must match training) ───────────────────────────
    if args.sh_degree0_only:
        feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
        in_channels = len(DC_ONLY_FEATURE_INDICES)
    else:
        feature_indices = None
        in_channels = FULL_3DGS_FEATURE_DIM

    # ── Dataset / loader ──────────────────────────────────────────────────
    base = Standard3DGenDataset(
        obj_list=[args.obj_list], gs_path=args.gs_path, caption_path=None,
        mean_file=args.mean_file, std_file=args.std_file,
        sphere2plane_path=args.sphere2plane_path, exclude_keys_file=args.exclude_keys_file,
        rank_transform_file=args.rank_transform_file, clip_thresholds_file=args.clip_thresholds_file,
        text_embed_path=args.text_embed_path,
    )
    text_dim = int(base.text_pooled.shape[1])
    dataset = Text3DGenDataset(base, feature_indices=feature_indices,
                               return_full_for_render=False, preload_to_cpu=False, lazy_cache_to_cpu=False)
    loader_gen = torch.Generator(); loader_gen.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        drop_last=False, generator=loader_gen)
    print(f"[data] dataset={len(dataset)} samples, text_dim={text_dim}; probing {args.num_samples}")

    # ── Model + checkpoint ────────────────────────────────────────────────
    model = JiT_3DGS_models[args.model](
        input_size=128, in_channels=in_channels, text_dim=text_dim, class_dropout_prob=0.0,
        learn_sigma=False, gradient_checkpointing=False, bottleneck=args.bottleneck)
    if args.null_text_token_path and os.path.exists(args.null_text_token_path):
        null_np = load_null_text_token(args.null_text_token_path)
        model.load_null_embeddings(torch.from_numpy(null_np.astype(np.float32)))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt[args.weights]
    missing, unexpected = model.load_state_dict(state, strict=False)
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

    # ── Gather one batch of real atlases ──────────────────────────────────
    xs, ys = [], []
    n = 0
    for batch in loader:
        x, y_pooled, _ = batch
        take = min(args.num_samples - n, x.shape[0])
        xs.append(x[:take]); ys.append(y_pooled[:take]); n += take
        if n >= args.num_samples:
            break
    x = torch.cat(xs, 0).to(device).float()
    y_pooled = torch.cat(ys, 0).to(device).float()
    bsz = x.shape[0]
    noise = torch.randn(x.shape, generator=torch.Generator(device=device).manual_seed(args.seed),
                        device=device, dtype=x.dtype)
    yp_gt = to_patches(x, P)  # canonical GT patches (B, nP, M, C); fixed across t
    print(f"[batch] {bsz} samples, patches/sample={yp_gt.shape[1]}")

    # one-time fidelity check: snapshot == exact kernel
    with torch.no_grad():
        _sqs = (yp_gt.pow(2).sum(-1).unsqueeze(-1) + yp_gt.pow(2).sum(-1).unsqueeze(-2)
                - 2 * (yp_gt @ yp_gt.transpose(-1, -2))).clamp_min(0)[:1, :2]
        k = max(iters_grid)
        snap_k = sinkhorn_snapshots(_sqs, eps_grid[0], [k], REF)[k]
        ref_k = _sinkhorn_log(_sqs, eps_grid[0], k)
        dev = float((snap_k - ref_k).abs().max())
        assert dev < 1e-5, f"snapshot != _sinkhorn_log (max dev {dev:.2e})"
        print(f"[check] snapshot matches _sinkhorn_log kernel (max dev {dev:.1e})")

    # ── Sweep ─────────────────────────────────────────────────────────────
    # results[(eps, t)] = dict(iters-> (resid, loss, loss_drift, blend_drift)), plus ref_resid/ref_loss
    results: dict = {}
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
        xp = to_patches(pred.float(), P)                    # (B, nP, M, C)
        aa = xp.pow(2).sum(-1); bb = yp_gt.pow(2).sum(-1)
        ab = xp @ yp_gt.transpose(-1, -2)
        sq = (aa.unsqueeze(-1) + bb.unsqueeze(-2) - 2 * ab).clamp_min(0)  # (B, nP, M, M)

        for eps in eps_grid:
            with torch.no_grad():
                snaps = sinkhorn_snapshots(sq, eps, iters_grid, REF)
                ref_resid, ref_loss, ref_blend = plan_metrics(snaps[REF], sq, yp_gt, M)
                ref_blend_norm = float(ref_blend.norm())
                ref_sigma, ref_hard_loss, ref_coll = hard_metrics(snaps[REF], sq, M)
                per_iter = {}
                for it in iters_grid:
                    resid, loss, blend = plan_metrics(snaps[it], sq, yp_gt, M)
                    loss_drift = abs(loss - ref_loss) / (abs(ref_loss) + 1e-12)
                    blend_drift = float((blend - ref_blend).norm()) / (ref_blend_norm + 1e-12)
                    sigma, hard_loss, coll = hard_metrics(snaps[it], sq, M)
                    agree = float((sigma == ref_sigma).float().mean())   # argmax matches converged
                    hard_drift = abs(hard_loss - ref_hard_loss) / (abs(ref_hard_loss) + 1e-12)
                    per_iter[it] = (resid, loss, loss_drift, blend_drift,
                                    agree, hard_loss, hard_drift, coll)
                del snaps
            results[(eps, float(t))] = {"per_iter": per_iter, "ref_resid": ref_resid,
                                        "ref_loss": ref_loss, "ref_hard_loss": ref_hard_loss,
                                        "ref_coll": ref_coll}
            print(f"[done] eps={eps} t={t}: ref_loss={ref_loss:.4f} ref_hard_loss={ref_hard_loss:.4f} "
                  f"ref_coll={ref_coll*100:.2f}% ref_resid={ref_resid:.2e}")

    # ── Console report ────────────────────────────────────────────────────
    print("\n" + "=" * 104)
    print(f"SINKHORN CONVERGENCE  |  {args.model} {args.weights} step {step}  |  N={bsz} P={P} (M={M})  ref_iters={REF}")
    print("  SOFT path:  blend_dr = ‖M·Π@gt − ref‖/‖ref‖ (what the soft gradient pulls toward).")
    print("  HARD path:  argmax_agr = frac of σ(i)=argmax_j Π[i,j] matching the converged σ; ")
    print("              hard_dr = |mean_i sq[i,σ(i)] − ref|/ref (the hard-variant LOSS); coll% = argmax non-bijection.")
    print("  KEY Q: does the hard path (argmax_agr→1, hard_dr→0) converge at FEWER iters than the soft path (blend_dr→0)?")
    print("=" * 104)
    for eps in eps_grid:
        for t in t_values:
            r = results[(eps, float(t))]
            print(f"\n eps={eps:<6} t={t:<4} (FM: 0=noise,1=clean)   "
                  f"ref_loss={r['ref_loss']:.4f}  ref_hard_loss={r['ref_hard_loss']:.4f}  "
                  f"ref_coll={r['ref_coll']*100:.2f}%  ref_resid={r['ref_resid']:.2e}"
                  + ("   <-- ref UNDER-CONVERGED" if r['ref_resid'] > 1e-3 else ""))
            print(f"   {'iters':>6} {'resid':>10} {'blend_dr':>9} | {'argmax_agr':>11} "
                  f"{'hard_loss':>10} {'hard_dr':>9} {'coll%':>7}")
            for it in iters_grid:
                resid, loss, ld, bd, agree, hloss, hdr, coll = r["per_iter"][it]
                flag = "  <-50" if it == 50 else ""
                print(f"   {it:>6} {resid:>10.2e} {bd:>9.2e} | {agree:>11.4f} "
                      f"{hloss:>10.4f} {hdr:>9.2e} {coll*100:>6.2f}%{flag}")

    # ── Save + plots ──────────────────────────────────────────────────────
    def _cube(idx):
        return np.asarray([[[results[(e, t)]["per_iter"][it][idx] for it in iters_grid]
                            for t in t_values] for e in eps_grid])
    np.savez(os.path.join(out_dir, "convergence.npz"),
             eps_grid=np.asarray(eps_grid), iters_grid=np.asarray(iters_grid),
             t_values=np.asarray(t_values), ref_iters=REF,
             resid=_cube(0), loss=_cube(1), loss_drift=_cube(2), blend_drift=_cube(3),
             argmax_agree=_cube(4), hard_loss=_cube(5), hard_drift=_cube(6), collision=_cube(7))
    meta = {"checkpoint": ckpt_path, "checkpoint_step": step, "weights": args.weights,
            "model": args.model, "patch_size": P, "M": M, "eps_grid": eps_grid,
            "iters_grid": iters_grid, "ref_iters": REF, "t_values": t_values,
            "num_samples": bsz, "in_channels": in_channels}
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # one figure per metric vs iters, line per (eps,t). The two HARD-path plots
    # (argmax_disagree, hard_drift) overlaid against the SOFT blend_drift answer "fewer iters?".
    plot_specs = [
        ("resid",            lambda v: max(v[0], 1e-6),       "row-marginal residual",        "log"),
        ("blend_drift_SOFT", lambda v: max(v[3], 1e-6),       "‖blend − ref‖/‖ref‖ (soft)",   "log"),
        ("argmax_disagree_HARD", lambda v: max(1.0 - v[4], 1e-6), "1 − argmax agreement (hard)", "log"),
        ("hard_loss_drift_HARD", lambda v: max(v[6], 1e-6),   "|hard_loss − ref|/ref (hard)", "log"),
        ("collision_rate",   lambda v: v[7],                  "argmax collision fraction",    "linear"),
    ]
    for metric, fn, ylab, scale in plot_specs:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for eps in eps_grid:
            for t in t_values:
                ys_ = [fn(results[(eps, float(t))]["per_iter"][it]) for it in iters_grid]
                ax.plot(iters_grid, ys_, marker="o", ms=3, label=f"eps={eps}, t={t}")
        ax.axvline(50, color="k", ls=":", alpha=0.5, label="current iters=50")
        if scale == "log":
            ax.axhline(1e-3, color="grey", ls="--", alpha=0.4)
        ax.set_yscale(scale); ax.set_xlabel("sinkhorn_iters"); ax.set_ylabel(ylab)
        ax.set_title(f"{metric} vs iters  ({args.model} {args.weights} step {step}, P={P})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.savefig(os.path.join(out_dir, f"{metric}_vs_iters.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"\n[done] wrote tables + convergence.npz + 3 plots to:\n  {out_dir}")


if __name__ == "__main__":
    main()
