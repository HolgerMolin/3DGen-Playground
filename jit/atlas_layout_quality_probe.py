#!/usr/bin/env python
"""Atlas layout-quality probe — is the sphere->plane projection's local
roughness an *artifact* of the layout, or *intrinsic* to the geometry?

This is the attribution test for the discontinuity finding (jit/
atlas_discontinuity_probe.py): plane-layout discontinuity D predicts x0-pred
error, but D conflates (a) projection artifact (OT permutation puts dissimilar
Gaussians adjacent) vs (b) intrinsic content roughness (any 2D layout carries
it). This script separates them with NO model — purely data geometry.

For each Gaussian we measure local feature roughness under three neighbor graphs,
using the SAME channel-mean squared feature distance each time (only the
neighbor set changes):

  D_3D     = distance to its k true 3D-nearest-neighbors (by xyz)   — best achievable
  D_plane  = distance to its 4 plane-grid neighbors (current layout)
  D_random = distance to random Gaussians  (= 2 * per-channel var)  — no layout

Calibration:  D_3D <= D_plane <= D_random.
  smoothness_recovered = (D_random - D_plane) / (D_random - D_3D)
    1.0  -> layout is as smooth as ordering by true 3D adjacency (projection fine)
    0.0  -> layout no better than random (projection scrambles everything)
  ratio  = D_plane / D_3D : how much extra local roughness the layout injects.
  overlap = fraction of each splat's 4 plane-neighbors that are also among its
            k true 3D-nearest-neighbors (how well the projection preserves locality).

PRIMARY metric uses NON-XYZ channels (opacity/color/scale/rotation): selecting
3D neighbors by xyz would trivially deflate D_3D on the xyz dims, so excluding
them is the honest comparison for whether appearance/shape is scrambled. The
all-14-channel version is reported too.

Data-only: no checkpoint, no diffusion, no GPU model. Mirrors the other probes
for dataset loading. Foreground, single GPU, safe alongside training.
"""
from __future__ import annotations

import argparse
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

from dataloaders.standard_3dgen_loader import Standard3DGenDataset  # noqa: E402
from dataloaders.text_3dgen_loader import (  # noqa: E402
    Text3DGenDataset,
    DC_ONLY_FEATURE_INDICES,
    FULL_3DGS_FEATURE_DIM,
)


def plane_neighbor_disc(x: torch.Tensor) -> torch.Tensor:
    """Per-pixel 4-neighbor channel-mean squared diff. x:[C,H,W]->[H,W] (bounded)."""
    C, H, W = x.shape
    acc = torch.zeros(H, W, device=x.device, dtype=x.dtype)
    cnt = torch.zeros(H, W, device=x.device, dtype=x.dtype)
    dh = ((x[:, :, 1:] - x[:, :, :-1]) ** 2).mean(0)
    acc[:, 1:] += dh; acc[:, :-1] += dh; cnt[:, 1:] += 1; cnt[:, :-1] += 1
    dv = ((x[:, 1:, :] - x[:, :-1, :]) ** 2).mean(0)
    acc[1:, :] += dv; acc[:-1, :] += dv; cnt[1:, :] += 1; cnt[:-1, :] += 1
    return acc / cnt


def plane_neighbor_indices(H: int, W: int) -> np.ndarray:
    """[N,4] flat indices of up/down/left/right plane neighbors; -1 if off-grid."""
    N = H * W
    g = np.arange(N).reshape(H, W)
    nb = np.full((H, W, 4), -1, dtype=np.int64)
    nb[1:, :, 0] = g[:-1, :]
    nb[:-1, :, 1] = g[1:, :]
    nb[:, 1:, 2] = g[:, :-1]
    nb[:, :-1, 3] = g[:, 1:]
    return nb.reshape(N, 4)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obj_list", type=str, required=True)
    p.add_argument("--gs_path", type=str, required=True)
    p.add_argument("--mean_file", type=str, required=True, help="Needed to de-normalize xyz for 3D-NN.")
    p.add_argument("--std_file", type=str, required=True)
    p.add_argument("--rank_transform_file", type=str, default=None)
    p.add_argument("--clip_thresholds_file", type=str, default=None)
    p.add_argument("--text_embed_path", type=str, required=True)
    p.add_argument("--sphere2plane_path", type=str, required=True)
    p.add_argument("--exclude_keys_file", type=str, default=None)
    p.add_argument("--sh_degree0_only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--k", type=int, default=4, help="3D nearest-neighbors (match the 4 plane neighbors).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--cmap", type=str, default="magma")
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    out_dir = args.output_dir or os.path.join(
        REPO_ROOT, "jit", "atlas_layout_quality_out",
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")

    if args.sh_degree0_only:
        feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
        in_channels = len(DC_ONLY_FEATURE_INDICES)
    else:
        feature_indices = None
        in_channels = FULL_3DGS_FEATURE_DIM

    # xyz are channels 0,1,2 in both the full and DC-selected layout.
    xyz_ch = [0, 1, 2]
    nonxyz_ch = list(range(3, in_channels))

    # Per-channel mean/std to de-normalize xyz back to metric 3D space.
    mean_full = torch.load(args.mean_file, weights_only=True).float()
    std_full = torch.load(args.std_file, weights_only=True).float()
    xyz_mean = mean_full[[0, 1, 2]].to(device).view(3, 1, 1)
    xyz_std = std_full[[0, 1, 2]].to(device).view(3, 1, 1)

    base = Standard3DGenDataset(
        obj_list=[args.obj_list], gs_path=args.gs_path, caption_path=None,
        mean_file=args.mean_file, std_file=args.std_file,
        sphere2plane_path=args.sphere2plane_path, exclude_keys_file=args.exclude_keys_file,
        rank_transform_file=args.rank_transform_file,
        clip_thresholds_file=args.clip_thresholds_file, text_embed_path=args.text_embed_path,
    )
    dataset = Text3DGenDataset(base, feature_indices=feature_indices,
                               return_full_for_render=False,
                               preload_to_cpu=False, lazy_cache_to_cpu=False)
    loader_gen = torch.Generator(); loader_gen.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        drop_last=False, generator=loader_gen)
    print(f"[data] dataset={len(dataset)}, probing {args.num_samples} @ batch {args.batch_size}, "
          f"in_channels={in_channels}, k={args.k}")

    H = W = 128
    N = H * W
    nb_flat = torch.from_numpy(plane_neighbor_indices(H, W)).to(device)  # [N,4]
    nb_valid = nb_flat >= 0

    # Accumulators (sums over objects).
    def zmap():
        return torch.zeros(H, W, dtype=torch.float64)
    acc = {
        "Dplane_map": zmap(), "D3d_map": zmap(),               # non-xyz per-pixel maps
        "Dplane_nx": 0.0, "D3d_nx": 0.0, "Drand_nx": 0.0,
        "Dplane_all": 0.0, "D3d_all": 0.0, "Drand_all": 0.0,
        "overlap": 0.0,
    }
    n_seen = 0

    for batch in loader:
        if n_seen >= args.num_samples:
            break
        x, _, _ = batch
        remaining = args.num_samples - n_seen
        if x.shape[0] > remaining:
            x = x[:remaining]
        x = x.to(device, non_blocking=True).float()  # [B,C,H,W]
        B = x.shape[0]

        for b in range(B):
            xb = x[b]                                  # [C,H,W]
            feat = xb.reshape(in_channels, N).t()      # [N,C]
            feat_nx = feat[:, nonxyz_ch]               # [N, C-3]
            xyz = (xb[:3] * xyz_std + xyz_mean).reshape(3, N).t()  # [N,3] metric space

            # ---- plane-neighbor discontinuity (per-pixel, non-xyz + all) ----
            dpl_nx_map = plane_neighbor_disc(xb[nonxyz_ch])   # [H,W]
            dpl_all_map = plane_neighbor_disc(xb)             # [H,W]
            acc["Dplane_map"] += dpl_nx_map.double().cpu()
            acc["Dplane_nx"] += float(dpl_nx_map.mean().cpu())
            acc["Dplane_all"] += float(dpl_all_map.mean().cpu())

            # ---- 3D k-NN by xyz ----
            dist = torch.cdist(xyz, xyz)               # [N,N]
            knn = dist.topk(args.k + 1, largest=False).indices[:, 1:]  # [N,k] drop self
            nb_nx = feat_nx[knn]                        # [N,k,C-3]
            d3d_nx = ((feat_nx[:, None, :] - nb_nx) ** 2).mean(-1).mean(-1)  # [N]
            nb_all = feat[knn]
            d3d_all = ((feat[:, None, :] - nb_all) ** 2).mean(-1).mean(-1)
            acc["D3d_map"] += d3d_nx.reshape(H, W).double().cpu()
            acc["D3d_nx"] += float(d3d_nx.mean().cpu())
            acc["D3d_all"] += float(d3d_all.mean().cpu())

            # ---- random-pair baseline (analytic: E[(a-b)^2]=2*Var) ----
            acc["Drand_nx"] += float(2.0 * feat_nx.var(0, unbiased=False).mean().cpu())
            acc["Drand_all"] += float(2.0 * feat.var(0, unbiased=False).mean().cpu())

            # ---- plane/3D adjacency overlap ----
            match = (nb_flat[:, :, None] == knn[:, None, :]).any(-1) & nb_valid  # [N,4]
            acc["overlap"] += float(match.sum().cpu()) / float(nb_valid.sum().cpu())

            del dist
        n_seen += B
        print(f"  [{n_seen}/{args.num_samples}] done")

    if n_seen == 0:
        raise RuntimeError("No samples processed.")

    # ---- finalize ----
    def avg(key):
        return acc[key] / n_seen
    Dplane_map = (acc["Dplane_map"] / n_seen).numpy()
    D3d_map = (acc["D3d_map"] / n_seen).numpy()
    ratio_map = Dplane_map / np.clip(D3d_map, 1e-9, None)

    nx = {"D_3D": avg("D3d_nx"), "D_plane": avg("Dplane_nx"), "D_random": avg("Drand_nx")}
    nx["ratio_plane_over_3D"] = nx["D_plane"] / max(1e-12, nx["D_3D"])
    nx["smoothness_recovered"] = (nx["D_random"] - nx["D_plane"]) / max(1e-12, nx["D_random"] - nx["D_3D"])
    al = {"D_3D": avg("D3d_all"), "D_plane": avg("Dplane_all"), "D_random": avg("Drand_all")}
    al["ratio_plane_over_3D"] = al["D_plane"] / max(1e-12, al["D_3D"])
    al["smoothness_recovered"] = (al["D_random"] - al["D_plane"]) / max(1e-12, al["D_random"] - al["D_3D"])
    overlap = avg("overlap")

    print("\n[result] local roughness under three neighbor graphs (N={} objects, k={}):".format(n_seen, args.k))
    for tag, m in (("non-xyz (PRIMARY)", nx), ("all-14ch", al)):
        print(f"  {tag}")
        print(f"    D_3D={m['D_3D']:.4f}  D_plane={m['D_plane']:.4f}  D_random={m['D_random']:.4f}")
        print(f"    ratio D_plane/D_3D = {m['ratio_plane_over_3D']:.2f}x   "
              f"smoothness_recovered = {m['smoothness_recovered']*100:.1f}%")
    print(f"  plane/3D adjacency overlap (of 4 plane-nbrs, frac among 3D-{args.k}NN): {overlap*100:.1f}%")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"num_samples": n_seen, "k": args.k, "non_xyz": nx, "all_14ch": al,
                   "adjacency_overlap": overlap}, f, indent=2)
    np.savez(os.path.join(out_dir, "maps.npz"),
             Dplane_map=Dplane_map, D3d_map=D3d_map, ratio_map=ratio_map)

    # ---- Figure 1: calibration scale (non-xyz) ----
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    vals = [nx["D_3D"], nx["D_plane"], nx["D_random"]]
    labels = ["D_3D\n(best)", "D_plane\n(layout)", "D_random\n(none)"]
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]
    ax.scatter(vals, [0, 0, 0], s=120, c=colors, zorder=3)
    ax.plot([nx["D_3D"], nx["D_random"]], [0, 0], "k-", lw=1, zorder=1)
    for v, l, c in zip(vals, labels, colors):
        ax.annotate(l, (v, 0), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color=c)
    ax.set_yticks([])
    ax.set_xlabel("local feature roughness (non-xyz)")
    ax.set_title(f"Layout quality: smoothness_recovered = {nx['smoothness_recovered']*100:.0f}%  |  "
                 f"D_plane/D_3D = {nx['ratio_plane_over_3D']:.2f}x  |  overlap = {overlap*100:.0f}%",
                 fontsize=10)
    fig.savefig(os.path.join(out_dir, "calibration_scale.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: per-pixel maps ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, m, title in ((axes[0], Dplane_map, "D_plane (layout)"),
                         (axes[1], D3d_map, "D_3D (true neighbors)"),
                         (axes[2], ratio_map, "ratio D_plane / D_3D")):
        im = ax.imshow(m, cmap=args.cmap)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Atlas layout roughness, non-xyz  |  N={n_seen}", fontsize=11)
    fig.savefig(os.path.join(out_dir, "maps_layout_quality.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] wrote metrics.json + maps.npz + 2 figures to:\n  {out_dir}")


if __name__ == "__main__":
    main()
