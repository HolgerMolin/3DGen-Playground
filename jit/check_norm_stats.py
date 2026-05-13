"""Compare normalization stats variants on real dataloader samples.

Pulls a handful of raw (un-normalized) samples through Standard3DGenDataset,
applies each candidate (mean, std) pair to the same raw tensor, and reports
per-channel and aggregate stats of the result. Used to check whether the
new c11 stats produce well-conditioned normalized values vs. the originals.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.standard_3dgen_loader import Standard3DGenDataset


def _load_stats(path: str) -> np.ndarray:
    t = torch.load(path, map_location="cpu", weights_only=True)
    arr = t.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 3 and arr.shape[1:] == (1, 1):
        arr = arr[:, 0, 0]
    if arr.ndim != 1:
        raise ValueError(f"unexpected stats shape {arr.shape} for {path}")
    return arr


def _apply(pc_chw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # pc_chw: (C, H, W) float
    return (pc_chw - mean[:, None, None]) / (std[:, None, None] + 1e-8)


def _summarize(name: str, values: np.ndarray) -> None:
    flat = values.reshape(values.shape[0], -1)
    per_ch_mean = flat.mean(axis=1)
    per_ch_std = flat.std(axis=1)
    print(f"  {name}")
    print(
        f"    overall    mean={values.mean():+.4f}  std={values.std():.4f}  "
        f"min={values.min():+.3f}  max={values.max():+.3f}"
    )
    print(
        f"    per-chan   |mean| avg={np.abs(per_ch_mean).mean():.4f}  max={np.abs(per_ch_mean).max():.4f}  "
        f"std avg={per_ch_std.mean():.4f}  min={per_ch_std.min():.4f}  max={per_ch_std.max():.4f}"
    )
    p = np.percentile(values, [0.1, 1, 50, 99, 99.9])
    print(f"    pctiles    0.1%={p[0]:+.3f}  1%={p[1]:+.3f}  50%={p[2]:+.3f}  99%={p[3]:+.3f}  99.9%={p[4]:+.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_list", required=True, nargs="+")
    parser.add_argument("--gs_path", required=True)
    parser.add_argument("--sphere2plane_path", required=True)
    parser.add_argument(
        "--stats",
        action="append",
        required=True,
        help="label=mean.pt,std.pt — repeat for each variant to compare",
    )
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    variants = []
    for spec in args.stats:
        label, paths = spec.split("=", 1)
        mean_p, std_p = paths.split(",")
        variants.append((label.strip(), _load_stats(mean_p), _load_stats(std_p)))

    print("Stats shapes/ranges:")
    for label, m, s in variants:
        print(
            f"  {label:14s}  C={m.shape[0]}  mean[min,max]=[{m.min():+.3f},{m.max():+.3f}]  "
            f"std[min,max]=[{s.min():.4f},{s.max():.3f}]"
        )

    ds = Standard3DGenDataset(
        obj_list=args.obj_list,
        gs_path=args.gs_path,
        mean_file=None,
        std_file=None,
        sphere2plane_path=args.sphere2plane_path,
    )
    print(f"Dataset size: {len(ds)}")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)

    raws = []
    for i in indices:
        s = ds[int(i)]
        pc = s["point_cloud"]
        if isinstance(pc, torch.Tensor):
            pc = pc.detach().cpu().numpy()
        pc = pc.astype(np.float32)
        raws.append(pc)
        print(f"  sample idx={int(i):>6d}  shape={pc.shape}  dtype={pc.dtype}  "
              f"min={pc.min():+.3f}  max={pc.max():+.3f}")
    raw_batch = np.stack(raws, axis=0)  # (N, C, H, W)

    print("\nRaw (un-normalized) batch:")
    flat = raw_batch.transpose(1, 0, 2, 3).reshape(raw_batch.shape[1], -1)
    print(
        f"  per-chan mean range [{flat.mean(1).min():+.3f}, {flat.mean(1).max():+.3f}]  "
        f"std range [{flat.std(1).min():.4f}, {flat.std(1).max():.3f}]"
    )

    for label, mean, std in variants:
        normed = np.stack([_apply(pc, mean, std) for pc in raw_batch], axis=0)
        # treat batch as one C-major tensor for per-channel stats
        per_ch = normed.transpose(1, 0, 2, 3).reshape(normed.shape[1], -1)
        print(f"\n[{label}]")
        _summarize("normalized", per_ch.reshape(per_ch.shape[0], -1))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
