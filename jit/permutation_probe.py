#!/usr/bin/env python
"""Permutation probe — how much does the model rely on the canonical atlas order?

The 128x128 input grid is a sphere->plane (OT-sorted) arrangement of the 16,384
Gaussians: the canonical permutation places spatially/semantically coherent
splats next to each other, and the JiT transformer sees them through 8x8 patch
embeddings + (learned) positional structure. This probe scrambles that order and
asks whether the x0-prediction still works.

For each sample we draw a *fresh* random permutation (per granularity) and feed
the permuted-noised atlas through the model. Because we permute the SAME noise,
``x_t_perm = perm(x_t)`` exactly — the only thing that changes is the spatial
arrangement, so any change in the prediction is purely the model's
non-equivariance.

Three MSEs are reported per flow-matching ``t_value`` (all in normalized training
space — the actual training objective, no rendering):

    canonical        = MSE( model(x_t),        x0 )            # reference
    perm_vs_perm     = MSE( model(perm(x_t)),  perm(x0) )      # task under scramble
    perm_vs_canonical= MSE( model(perm(x_t)),  x0 )            # sanity (should be high)

If the model were perfectly permutation-equivariant, ``perm_vs_perm`` would equal
``canonical`` and ``perm_vs_canonical`` would be large. The gap
``perm_vs_perm - canonical`` is the reliance-on-canonical-order signal.

Two granularities are run (``--granularities full,patch``):
    full   — shuffle all 16,384 splats freely (destroys all OT coherence).
    patch  — shuffle the 16x16 grid of 8x8 patches, keeping each patch's
             internal order (scrambles global layout, keeps local structure).

OUTPUT-permutation metrics (``--output_perm``, on by default) measure the dual
question — *how much does the model permute its OWN output?* On the canonical
(un-permuted) prediction, each pred cell is matched to the GT Gaussian it best
represents (feature-NN), and we report per ``t``:
    perm_gain   = L_index − L_chamfer   (~0 => keeps canonical order; large => permutes)
    mean_disp   = atlas-grid cells the matched Gaussian travelled
    in_patch@P  = fraction of matches staying in the same PxP patch (spatial scale
                  of the permutation -> informs chamfer_patch_size)
    exact%      = fraction whose match is its own cell
    collision%  = pred cells mapping to a duplicated GT (mode collapse)
These reuse the canonical forward pass; ``--match_space {feature,xyz}`` picks the
matching space, ``--patch_fracs`` the P values for in_patch.

Convention (flow matching, NOT DDPM — see CLAUDE.md):
    x_t = t_value * x_0 + (1 - t_value) * eps;  t=0 -> noise, t=1 -> clean.
    t_discrete = round(t_value * (T - 1)) is the integer step fed to t_embedder.

Single-GPU example (paths usually come from jit/permutation_probe.sh + .env):

    python jit/permutation_probe.py \
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
from jit.diffusion.gaussian_diffusion import _keops_available, _keops_argmin  # noqa: E402


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


def parse_granularities(raw: str) -> list[str]:
    vals = [g.strip() for g in raw.split(",") if g.strip() != ""]
    for g in vals:
        if g not in ("full", "patch"):
            raise ValueError(f"unknown granularity {g!r}; choose from full,patch")
    if not vals:
        raise ValueError("--granularities produced an empty list")
    return vals


# ---------------------------------------------------------------------------
# Permutation index construction (per-sample, fresh).
#
# All permutations are expressed as a per-sample index ``idx`` of shape
# (B, H*W) with the gather convention ``out[b, :, p] = x_flat[b, :, idx[b, p]]``.
# A patch permutation is just a block-structured special case of a free
# permutation, so the same gather applies to both.
# ---------------------------------------------------------------------------

def _make_full_perm_idx(bsz: int, hw: int, gen: torch.Generator, device) -> torch.Tensor:
    return torch.stack(
        [torch.randperm(hw, generator=gen, device=device) for _ in range(bsz)], dim=0
    )


def _patch_to_pos(H: int, W: int, P: int, device) -> torch.Tensor:
    """(npatch, P*P) of flat HW positions belonging to each row-major patch."""
    if H % P or W % P:
        raise ValueError(f"patch_size {P} does not divide grid {H}x{W}")
    nph, npw = H // P, W // P
    base = torch.arange(H * W, device=device).reshape(H, W)
    # (nph, P, npw, P) -> (nph, npw, P, P) -> (nph*npw, P*P)
    return base.reshape(nph, P, npw, P).permute(0, 2, 1, 3).reshape(nph * npw, P * P)


def _make_patch_perm_idx(bsz: int, H: int, W: int, P: int, gen: torch.Generator, device) -> torch.Tensor:
    pp = _patch_to_pos(H, W, P, device)        # (npatch, ppp); dst positions
    npatch = pp.shape[0]
    flat_dst = pp.reshape(-1)                   # covers 0..HW-1 exactly once
    idx = torch.empty(bsz, H * W, dtype=torch.long, device=device)
    for b in range(bsz):
        perm_p = torch.randperm(npatch, generator=gen, device=device)
        src = pp[perm_p]                        # output patch i sources from patch perm_p[i]
        idx[b].scatter_(0, flat_dst, src.reshape(-1))
    return idx


def _apply_perm(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x: (B,C,H,W); idx: (B,H*W) long -> permuted (B,C,H,W)."""
    B, C, H, W = x.shape
    x_flat = x.reshape(B, C, H * W)
    index = idx.unsqueeze(1).expand(B, C, H * W)
    return torch.gather(x_flat, 2, index).reshape(B, C, H, W)


def _per_sample_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean over (C,H,W) of (a-b)^2 -> (B,)."""
    return ((a.float() - b.float()) ** 2).mean(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
# Output-permutation metrics — "how much does the MODEL permute its output?"
#
# Distinct from the input-scramble experiment above. Given a CANONICAL (un-
# permuted) input x_t built from a real x0, the model predicts ``pred``. Because
# Chamfer training never constrains the output ordering, ``pred`` may place the
# right Gaussian in a different atlas cell than canonical. We measure that:
#   * match each pred cell to the GT Gaussian it best represents (feature-NN);
#   * displacement = atlas-grid distance from the pred cell to that GT cell's
#     home position; in-patch fraction (matched stays in same P×P patch) gives
#     the SPATIAL SCALE of the permutation -> informs chamfer_patch_size.
#   * permutation_gain = L_index - L_chamfer = how much reordering helps
#     (~0 => model preserves canonical order; large => heavy permutation).
#   * collision_rate = fraction of pred cells whose NN GT is a duplicate (mode
#     collapse: the model emitting the same Gaussian many times).
# Feature-NN is a proxy (not a bijection); collision_rate quantifies its slack.
# ---------------------------------------------------------------------------

def _nn_argmin(q: torch.Tensor, r: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Index of nearest r-row (squared L2) per q-row. q,r:(B,N,Ds) -> (B,N) long.

    KeOps when available (no NxN materialization); else a chunked-cdist fallback
    that bounds memory to (B, chunk, N)."""
    if _keops_available():
        return _keops_argmin(q, r)
    B, N, _ = q.shape
    out = torch.empty(B, N, dtype=torch.long, device=q.device)
    for s in range(0, N, chunk):
        d = torch.cdist(q[:, s:s + chunk], r)          # (B, c, N)
        out[:, s:s + chunk] = d.argmin(dim=-1)
    return out


def _output_perm_metrics(pred, x0, W, patch_sizes, match_space, chamfer_patch_size=4,
                         exact_assignment=False, exact_patch_subset=32):
    """Per-sample output-permutation metrics for one (pred, x0) batch.

    pred / x0: (B, C, H, W). Returns a dict of per-sample tensors (B,) plus an
    ``in_patch`` dict {P: (B,)}. ``match_space`` selects the matching feature
    space ('feature' = all C channels, 'xyz' = first 3)."""
    B, C, H, _ = pred.shape
    N = H * W
    pf = pred.reshape(B, C, N).transpose(1, 2).float().contiguous()  # (B, N, C)
    xf = x0.reshape(B, C, N).transpose(1, 2).float().contiguous()
    if match_space == "xyz":
        q, r = pf[..., :3].contiguous(), xf[..., :3].contiguous()
    else:
        q, r = pf, xf
    jstar = _nn_argmin(q, r)                                          # (B, N) GT idx per pred cell
    matched = torch.gather(xf, 1, jstar.unsqueeze(-1).expand(B, N, C))
    l_chamfer = ((pf - matched) ** 2).mean(dim=(1, 2))               # (B,) over N and C

    i_idx = torch.arange(N, device=pred.device)
    ri, ci = i_idx // W, i_idx % W                                    # (N,)
    rj, cj = jstar // W, jstar % W                                    # (B, N)
    disp = torch.sqrt(((rj - ri) ** 2 + (cj - ci) ** 2).float())     # (B, N) atlas-grid distance
    mean_disp = disp.mean(dim=1)                                      # (B,)
    exact_frac = (jstar == i_idx).float().mean(dim=1)                 # (B,) matched cell == own cell
    in_patch = {
        P: ((ri // P == rj // P) & (ci // P == cj // P)).float().mean(dim=1)  # (B,)
        for P in patch_sizes
    }
    collision = torch.tensor(
        [1.0 - torch.unique(jstar[b]).numel() / N for b in range(B)],
        device=pred.device,
    )
    # Within-patch collision — FAITHFUL to the chamfer_patch loss, which matches each
    # pred point to its nearest target WITHIN its own PxP patch (M=P*P candidates), not
    # globally. collision_patch[b] = mean over patches of (1 - distinct_targets_hit / M):
    # 0 => bijection (no collision), ->1 => mode collapse (pred points pile onto few
    # targets, leaving the rest uncovered & gradient-free).
    Pc = int(chamfer_patch_size)
    H = pred.shape[2]
    if Pc > 0 and (H % Pc == 0) and (W % Pc == 0):
        nH, nW = H // Pc, W // Pc
        M = Pc * Pc
        feat = pred if match_space != "xyz" else pred[:, :3]
        tgt = x0 if match_space != "xyz" else x0[:, :3]
        Cm = feat.shape[1]

        def _patches(t):  # (B,Cm,H,W) -> (B, nP, M, Cm); grouping == chamfer_patch loss
            return (t.float()
                    .reshape(B, Cm, nH, Pc, nW, Pc)
                    .permute(0, 2, 4, 1, 3, 5)
                    .reshape(B, nH * nW, Cm, M)
                    .transpose(-1, -2)
                    .contiguous())

        xpp, ypp = _patches(feat), _patches(tgt)              # (B, nP, M, Cm)
        sqp = (xpp.pow(2).sum(-1).unsqueeze(-1)
               + ypp.pow(2).sum(-1).unsqueeze(-2)
               - 2 * (xpp @ ypp.transpose(-1, -2))).clamp_min(0)   # (B, nP, M, M)
        jstar_p = sqp.argmin(dim=-1)                          # (B, nP, M) nearest tgt in-patch
        hitf = torch.zeros(B, nH * nW, M, device=pred.device)
        hitf.scatter_(-1, jstar_p, torch.ones_like(jstar_p, dtype=hitf.dtype))
        distinct = (hitf > 0).sum(-1).float()                 # (B, nP) distinct targets hit
        collision_patch = (1.0 - distinct / M).mean(dim=1)    # (B,)
        if exact_assignment:
            # Exact optimal assignment (Hungarian) per patch on the SAME within-patch cost,
            # subsampling patches to bound CPU cost. emd_patch[b] = mean over the patch subset
            # of (optimal matched sq-distance summed / M) = the EMD "ceiling" the Sinkhorn loss
            # targets (an exact within-patch bijection has 0 collision by construction). This is
            # >= the NN forward L_chamfer (bijection constraint costs >= greedy NN).
            from scipy.optimize import linear_sum_assignment
            nP_tot = sqp.shape[1]
            k = min(int(exact_patch_subset), nP_tot)
            pidx = torch.randperm(nP_tot, device=sqp.device)[:k]
            sub = sqp[:, pidx].detach().float().cpu().numpy()    # (B, k, M, M)
            cost = np.empty((B, k), dtype=np.float64)
            for b in range(B):
                for p in range(k):
                    ri, ci = linear_sum_assignment(sub[b, p])
                    cost[b, p] = sub[b, p][ri, ci].sum() / M     # mean matched sq-dist / patch
            emd_patch = torch.tensor(cost.mean(axis=1), device=pred.device, dtype=torch.float32)
        else:
            emd_patch = torch.full((B,), float("nan"), device=pred.device)
    else:
        collision_patch = torch.full((B,), float("nan"), device=pred.device)
        emd_patch = torch.full((B,), float("nan"), device=pred.device)
    return dict(l_chamfer=l_chamfer, mean_disp=mean_disp, exact_frac=exact_frac,
                in_patch=in_patch, collision=collision, collision_patch=collision_patch,
                emd_patch=emd_patch)


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
    p.add_argument("--granularities", type=str, default="full,patch",
                   help="Comma-separated subset of {full,patch}.")
    p.add_argument("--patch_size", type=int, default=None,
                   help="Patch side for 'patch' granularity. Default: parsed from --model "
                        "(e.g. JiT-B/8 -> 8).")
    # Output-permutation metrics ("how much does the MODEL permute its output?")
    p.add_argument("--output_perm", action=argparse.BooleanOptionalAction, default=True,
                   help="Also measure how much the model permutes its OWN output vs the "
                        "canonical GT (feature-NN displacement, in-patch fraction, "
                        "permutation gain L_index-L_chamfer, collision rate). Computed on the "
                        "canonical-input prediction, so it reuses the same forward pass.")
    p.add_argument("--match_space", type=str, default="feature", choices=("feature", "xyz"),
                   help="Feature space for matching pred->GT: 'feature' = all C channels "
                        "(most identifying), 'xyz' = first 3 (physical proximity).")
    p.add_argument("--chamfer_patch_size", type=int, default=4,
                   help="Within-patch size for the FAITHFUL collision metric matching "
                        "recon_loss=chamfer_patch (each pred point matched to nearest target "
                        "inside its own PxP patch). Should equal the run's chamfer_patch_size.")
    p.add_argument("--exact_assignment", action=argparse.BooleanOptionalAction, default=False,
                   help="Also compute the EXACT optimal-assignment (Hungarian, scipy) within-patch "
                        "cost = the EMD 'ceiling' that recon_loss=sinkhorn_patch targets (and >= the "
                        "NN L_chamfer). CPU; subsampled to --exact_patch_subset patches/sample.")
    p.add_argument("--exact_patch_subset", type=int, default=32,
                   help="Patches/sample for --exact_assignment (bounds CPU cost of the per-patch "
                        "Hungarian solves; the mean over the subset is an unbiased EMD estimate).")
    p.add_argument("--patch_fracs", type=str, default="4,8,16,32",
                   help="Comma-separated patch sides for the in-patch fraction (the spatial "
                        "scale of permutation). e.g. in_patch@8 => fraction of matches staying "
                        "within the model's 8x8 patch (informs chamfer_patch_size).")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8, help="Keep low to coexist with training.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0,
                   help="Seeds the sample subset, the fixed noise draw, and the permutations.")
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=("bf16", "fp16", "none"),
                   help="Autocast for the model forward (bf16 matches training).")
    p.add_argument("--max_vram_gb", type=float, default=10.0,
                   help="Hard cap on this process's CUDA allocator (GiB). Lets the probe "
                        "coexist with a training job: exceeding it OOMs the probe (caught "
                        "here) instead of stealing the trainer's memory. 0 disables the cap.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Default: jit/permutation_probe_out/run_<timestamp>.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    # Hard-cap allocator so a co-resident training job is never starved by the probe.
    if device.type == "cuda" and args.max_vram_gb > 0:
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        total_gb = torch.cuda.get_device_properties(dev_idx).total_memory / 1024**3
        frac = min(1.0, args.max_vram_gb / total_gb)
        torch.cuda.set_per_process_memory_fraction(frac, dev_idx)
        print(f"[vram] capping probe allocator at {args.max_vram_gb:.1f} GiB "
              f"({frac*100:.1f}% of {total_gb:.0f} GiB total)")
    t_values = parse_t_values(args.t_values)
    granularities = parse_granularities(args.granularities)
    patch_size = args.patch_size if args.patch_size is not None else int(args.model.split("/")[-1])
    patch_fracs = [int(p) for p in args.patch_fracs.split(",") if p.strip() != ""]

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
        REPO_ROOT, "jit", "permutation_probe_out",
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
    print(f"[perm] granularities={granularities}  patch_size={patch_size}  fresh-per-sample")

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
    if args.null_text_token_path and os.path.exists(args.null_text_token_path):
        null_np = load_null_text_token(args.null_text_token_path)
        model.load_null_embeddings(torch.from_numpy(null_np.astype(np.float32)))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt[args.weights]

    # Strip torch.compile ('_orig_mod.') and DDP ('module.') prefixes so a checkpoint
    # saved from a compiled/distributed model loads into this bare eval model. Without
    # this, every key mismatches, strict=False loads NOTHING, and the probe silently
    # runs on random weights (canonical MSE constant across t is the tell).
    def _strip(k):
        changed = True
        while changed:
            changed = False
            for pre in ("_orig_mod.", "module."):
                if k.startswith(pre):
                    k = k[len(pre):]
                    changed = True
        return k

    state = {_strip(k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] non-strict: {len(missing)} missing, {len(unexpected)} unexpected")
        if len(missing) > 5 or len(unexpected) > 5:
            print(f"       missing(head)={list(missing)[:5]}")
            print(f"       unexpected(head)={list(unexpected)[:5]}")
            raise SystemExit(
                "[load] ABORT: large state_dict mismatch after prefix-stripping — the probe "
                "would run on random weights. Check --model/--bottleneck/--sh_degree0_only "
                "match this checkpoint."
            )
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

    def forward(x_t, t_discrete, y):
        with torch.no_grad():
            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    return model(x_t, t_discrete, y).float()
            return model(x_t, t_discrete, y).float()

    # ── Accumulators (float64 sums of per-sample scalar MSE) ───────────────
    H = W = 128
    canon_sum = {t: 0.0 for t in t_values}
    pvp_sum = {g: {t: 0.0 for t in t_values} for g in granularities}   # vs permuted GT
    pvc_sum = {g: {t: 0.0 for t in t_values} for g in granularities}   # vs canonical GT
    # Output-permutation accumulators (sums of per-sample scalars).
    op_lcham = {t: 0.0 for t in t_values}
    op_disp = {t: 0.0 for t in t_values}
    op_exact = {t: 0.0 for t in t_values}
    op_coll = {t: 0.0 for t in t_values}
    op_coll_patch = {t: 0.0 for t in t_values}
    op_emd = {t: 0.0 for t in t_values}
    op_inpatch = {t: {P: 0.0 for P in patch_fracs} for t in t_values}
    n_seen = 0

    for batch_idx, batch in enumerate(loader):
        if n_seen >= args.num_samples:
            break
        x, y_pooled, hash_keys = batch
        remaining = args.num_samples - n_seen
        if x.shape[0] > remaining:
            x = x[:remaining]
            y_pooled = y_pooled[:remaining]
        bsz = x.shape[0]

        x = x.to(device, non_blocking=True).float()           # x0: canonical, normalized
        y_pooled = y_pooled.to(device, non_blocking=True).float()

        # Fixed noise for this batch, reused across every t (deterministic per batch).
        noise_gen = torch.Generator(device=device)
        noise_gen.manual_seed(args.seed * 1_000_003 + batch_idx)
        noise = torch.randn(x.shape, generator=noise_gen, device=device, dtype=x.dtype)

        # Fresh per-sample permutation indices for this batch (deterministic per batch).
        perm_gen = torch.Generator(device=device)
        perm_gen.manual_seed(args.seed * 7_919 + batch_idx)
        idx_by_gran = {}
        if "full" in granularities:
            idx_by_gran["full"] = _make_full_perm_idx(bsz, H * W, perm_gen, device)
        if "patch" in granularities:
            idx_by_gran["patch"] = _make_patch_perm_idx(bsz, H, W, patch_size, perm_gen, device)

        # Sanity (first batch only): every index row is a valid bijection.
        if batch_idx == 0:
            ref = torch.arange(H * W, device=device)
            for g, idx in idx_by_gran.items():
                ok = bool((idx.sort(dim=1).values == ref).all())
                assert ok, f"{g} permutation index is not a valid permutation"

        for t in t_values:
            t_value = torch.full((bsz,), float(t), device=device)
            t_discrete = torch.clamp((t_value * (T - 1)).round().long(), 0, T - 1)
            x_t = diffusion.flow_matching_q_sample(x, t_value, noise=noise)

            pred = forward(x_t, t_discrete, y_pooled)
            canon_sum[t] += float(_per_sample_mse(x, pred).sum().cpu())

            # Output-permutation metrics on the canonical-input prediction.
            if args.output_perm:
                m = _output_perm_metrics(pred, x, W, patch_fracs, args.match_space,
                                         chamfer_patch_size=args.chamfer_patch_size,
                                         exact_assignment=args.exact_assignment,
                                         exact_patch_subset=args.exact_patch_subset)
                op_lcham[t] += float(m["l_chamfer"].sum().cpu())
                op_disp[t] += float(m["mean_disp"].sum().cpu())
                op_exact[t] += float(m["exact_frac"].sum().cpu())
                op_coll[t] += float(m["collision"].sum().cpu())
                op_coll_patch[t] += float(m["collision_patch"].sum().cpu())
                if args.exact_assignment:
                    op_emd[t] += float(m["emd_patch"].sum().cpu())
                for P in patch_fracs:
                    op_inpatch[t][P] += float(m["in_patch"][P].sum().cpu())

            for g, idx in idx_by_gran.items():
                x_t_perm = _apply_perm(x_t, idx)     # == perm(x_t): same noise, scrambled order
                x0_perm = _apply_perm(x, idx)        # permuted GT
                pred_perm = forward(x_t_perm, t_discrete, y_pooled)
                pvp_sum[g][t] += float(_per_sample_mse(x0_perm, pred_perm).sum().cpu())
                pvc_sum[g][t] += float(_per_sample_mse(x, pred_perm).sum().cpu())

        n_seen += bsz
        print(f"  [{n_seen}/{args.num_samples}] batch {batch_idx} done")

    if n_seen == 0:
        raise RuntimeError("No samples were processed.")

    canon = {t: canon_sum[t] / n_seen for t in t_values}
    pvp = {g: {t: pvp_sum[g][t] / n_seen for t in t_values} for g in granularities}
    pvc = {g: {t: pvc_sum[g][t] / n_seen for t in t_values} for g in granularities}

    # ── Print table (input-scramble experiment) ──────────────────────────────
    print(f"\n[result] N={n_seen}  (FM: t=0 noise → t=1 clean)")
    header = f"  {'t':>5}  {'canonical':>11}"
    for g in granularities:
        header += f"  {g+'_pvP':>11}  {g+'_pvC':>11}  {g+'_gap':>9}"
    print(header)
    for t in t_values:
        row = f"  {t:>5}  {canon[t]:>11.6f}"
        for g in granularities:
            gap = pvp[g][t] - canon[t]
            row += f"  {pvp[g][t]:>11.6f}  {pvc[g][t]:>11.6f}  {gap:>9.5f}"
        print(row)
    print("  (pvP=vs permuted GT, pvC=vs canonical GT, gap=pvP-canonical = reliance-on-order)")

    # ── Output-permutation metrics (how much the model permutes its OWN output) ─
    op = {}
    if args.output_perm:
        op = {
            "l_chamfer": {t: op_lcham[t] / n_seen for t in t_values},
            "perm_gain": {t: canon[t] - op_lcham[t] / n_seen for t in t_values},
            "mean_disp": {t: op_disp[t] / n_seen for t in t_values},
            "exact_frac": {t: op_exact[t] / n_seen for t in t_values},
            "collision": {t: op_coll[t] / n_seen for t in t_values},
            "collision_patch": {t: op_coll_patch[t] / n_seen for t in t_values},
            "emd_patch": {t: op_emd[t] / n_seen for t in t_values},
            "in_patch": {P: {t: op_inpatch[t][P] / n_seen for t in t_values} for P in patch_fracs},
        }
        print(f"\n[output-perm] match_space={args.match_space}  "
              f"(L_index=canonical above; perm_gain=L_index-L_chamfer)")
        hdr = (f"  {'t':>5}  {'L_chamfer':>10}  {'perm_gain':>10}  {'mean_disp':>9}  "
               f"{'exact%':>7}  {'collide%':>8}  {f'P{args.chamfer_patch_size}coll%':>8}")
        if args.exact_assignment:
            hdr += f"  {'EMDopt':>9}"
        hdr += "".join(f"  inP@{P:<3}" for P in patch_fracs)
        print(hdr)
        for t in t_values:
            row = (f"  {t:>5}  {op['l_chamfer'][t]:>10.6f}  {op['perm_gain'][t]:>10.6f}  "
                   f"{op['mean_disp'][t]:>9.3f}  {100*op['exact_frac'][t]:>6.2f}%  "
                   f"{100*op['collision'][t]:>7.2f}%  {100*op['collision_patch'][t]:>7.2f}%")
            if args.exact_assignment:
                row += f"  {op['emd_patch'][t]:>9.6f}"
            row += "".join(f"  {100*op['in_patch'][P][t]:>5.1f}%" for P in patch_fracs)
            print(row)
        print("  (perm_gain~0 => keeps canonical order; mean_disp in atlas cells; "
              "inP@P = matched GT stays in same PxP patch; collide% = global dup matches; "
              f"P{args.chamfer_patch_size}coll% = WITHIN-{args.chamfer_patch_size}x{args.chamfer_patch_size}-patch "
              "collision = faithful to chamfer_patch loss = mode-collapse / gradient-sparsity"
              + ("; EMDopt = exact-assignment mean matched sq-dist = the sinkhorn_patch target "
                 "(>= L_chamfer; bijection => 0 collision)" if args.exact_assignment else "") + ")")

    # ── Save metadata JSON ───────────────────────────────────────────────────
    meta = {
        "checkpoint": ckpt_path,
        "checkpoint_step": step,
        "weights": args.weights,
        "model": args.model,
        "in_channels": in_channels,
        "sh_degree0_only": bool(args.sh_degree0_only),
        "t_values": t_values,
        "granularities": granularities,
        "patch_size": patch_size,
        "num_samples": n_seen,
        "batch_size": args.batch_size,
        "mixed_precision": args.mixed_precision,
        "seed": args.seed,
        "noise": "same noise permuted (x_t_perm = perm(x_t))",
        "perm_scope": "fresh per sample",
        "canonical_mse": {str(t): canon[t] for t in t_values},
        "perm_vs_perm_mse": {g: {str(t): pvp[g][t] for t in t_values} for g in granularities},
        "perm_vs_canonical_mse": {g: {str(t): pvc[g][t] for t in t_values} for g in granularities},
        "reliance_gap": {
            g: {str(t): pvp[g][t] - canon[t] for t in t_values} for g in granularities
        },
        "note": "MSE over (C,H,W) of (x0 - pred)^2 in normalized training space, "
                "averaged over samples. canonical=reference; perm_vs_perm=task under "
                "scramble; perm_vs_canonical=sanity (high). gap=perm_vs_perm-canonical "
                "is the model's reliance on the canonical atlas order.",
    }
    if args.output_perm:
        meta["output_perm"] = {
            "match_space": args.match_space,
            "patch_fracs": patch_fracs,
            "l_chamfer": {str(t): op["l_chamfer"][t] for t in t_values},
            "perm_gain": {str(t): op["perm_gain"][t] for t in t_values},
            "mean_disp_cells": {str(t): op["mean_disp"][t] for t in t_values},
            "exact_cell_frac": {str(t): op["exact_frac"][t] for t in t_values},
            "collision_rate": {str(t): op["collision"][t] for t in t_values},
            "collision_rate_inpatch": {str(t): op["collision_patch"][t] for t in t_values},
            "chamfer_patch_size": args.chamfer_patch_size,
            "exact_assignment": bool(args.exact_assignment),
            "emd_inpatch_optimal": ({str(t): op["emd_patch"][t] for t in t_values}
                                    if args.exact_assignment else None),
            "in_patch_frac": {str(P): {str(t): op["in_patch"][P][t] for t in t_values}
                              for P in patch_fracs},
            "note": "Output-permutation of the canonical-input prediction. Match each pred "
                    "cell to its nearest GT Gaussian (in match_space); L_chamfer=mean min "
                    "feature dist^2; perm_gain=L_index(canonical)-L_chamfer (~0 => keeps "
                    "canonical order); mean_disp=atlas-grid cells moved; in_patch_frac[P]= "
                    "fraction whose match stays in the same PxP patch; collision_rate= "
                    "fraction of pred cells mapping to a duplicated GT (mode collapse).",
        }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Figure: MSE vs t ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(t_values, [canon[t] for t in t_values], "o-", color="black",
            linewidth=2.2, label="canonical (reference)")
    color_cycle = {"full": "tab:red", "patch": "tab:blue"}
    for g in granularities:
        c = color_cycle.get(g, None)
        ax.plot(t_values, [pvp[g][t] for t in t_values], "s-", color=c,
                label=f"{g}: perm vs permuted GT")
        ax.plot(t_values, [pvc[g][t] for t in t_values], "x--", color=c, alpha=0.7,
                label=f"{g}: perm vs canonical GT")
    ax.set_xlabel("t_value  (0 = noise, 1 = clean)")
    ax.set_ylabel("mean MSE (normalized training space)")
    ax.set_title(f"Permutation reliance  ({args.model} {args.weights} step {step}, N={n_seen})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, "permutation_mse_vs_t.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Figure: output-permutation (perm gain + spatial locality vs t) ────────
    if args.output_perm:
        fig2, (axg, axl) = plt.subplots(1, 2, figsize=(13, 5))
        axg.plot(t_values, [canon[t] for t in t_values], "o-", color="black",
                 label="L_index (canonical)")
        axg.plot(t_values, [op["l_chamfer"][t] for t in t_values], "s-", color="tab:green",
                 label="L_chamfer (best match)")
        axg.plot(t_values, [op["perm_gain"][t] for t in t_values], "^--", color="tab:purple",
                 label="perm_gain = L_index − L_chamfer")
        axg.set_xlabel("t_value  (0 = noise, 1 = clean)")
        axg.set_ylabel("mean (normalized training space)")
        axg.set_title("Permutation gain")
        axg.grid(True, alpha=0.3)
        axg.legend(fontsize=8)
        for P in patch_fracs:
            axl.plot(t_values, [100 * op["in_patch"][P][t] for t in t_values], "o-",
                     label=f"in-patch @ {P}")
        axl.plot(t_values, [100 * op["exact_frac"][t] for t in t_values], "x--", color="black",
                 alpha=0.7, label="exact cell")
        axl.plot(t_values, [100 * op["collision"][t] for t in t_values], "s:", color="tab:red",
                 alpha=0.7, label="collision rate")
        axl.set_xlabel("t_value  (0 = noise, 1 = clean)")
        axl.set_ylabel("percent")
        axl.set_ylim(0, 100)
        axl.set_title(f"Permutation locality (match_space={args.match_space})")
        axl.grid(True, alpha=0.3)
        axl.legend(fontsize=8)
        fig2.suptitle(f"Output permutation  ({args.model} {args.weights} step {step}, N={n_seen})")
        fig2.savefig(os.path.join(out_dir, "output_permutation_vs_t.png"), dpi=130,
                     bbox_inches="tight")
        plt.close(fig2)

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        print(f"[vram] peak allocated={peak:.2f} GiB, peak reserved={reserved:.2f} GiB "
              f"(batch_size={args.batch_size})")

    figs = "permutation_mse_vs_t.png" + (" + output_permutation_vs_t.png" if args.output_perm else "")
    print(f"\n[done] wrote metadata.json + {figs} to:\n  {out_dir}")


if __name__ == "__main__":
    main()
