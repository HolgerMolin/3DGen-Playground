"""
k-means clustering of UMAP-reduced image embeddings.

Loads the cached UMAP coords produced by `cluster_image_embeddings.py`
(15-d projection of L2-normalised DINOv2 mean embeddings) and runs
sklearn KMeans with the requested k. Writes:

  image_kmeans_labels__k{K}.npz   — labels (N,) int32, keys (N,) object
  image_kmeans_summary__k{K}.txt  — size distribution per cluster

Usage
-----
python object_classification/kmeans_image_clusters.py --k 120
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_UMAP = (_SCRIPT_DIR /
                 "umap_image__neighbors30__mindist0.0__metriccosine__dim15.npz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--umap", default=str(_DEFAULT_UMAP),
                   help="Path to UMAP cache (.npz with umap_coords, keys)")
    p.add_argument("--output-dir", default=str(_SCRIPT_DIR))
    p.add_argument("--k", type=int, default=120, help="Number of clusters")
    p.add_argument("--n-init", type=int, default=10,
                   help="KMeans n_init (best of N restarts).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = Path(args.umap)
    if not cache.exists():
        raise FileNotFoundError(
            f"UMAP cache not found: {cache}\n"
            "Run cluster_image_embeddings.py first to generate it.")

    print(f"Loading UMAP coords from {cache}")
    data = np.load(cache, allow_pickle=True)
    coords = data["umap_coords"].astype(np.float32)
    keys = data["keys"].tolist()
    print(f"  coords {coords.shape}, {len(keys):,} keys")

    print(f"\nRunning KMeans k={args.k}, n_init={args.n_init}, seed={args.seed} ...")
    t0 = time.time()
    km = KMeans(n_clusters=args.k, n_init=args.n_init,
                random_state=args.seed, verbose=0)
    labels = km.fit_predict(coords).astype(np.int32)
    dt = time.time() - t0
    print(f"KMeans done in {dt:.1f}s. inertia={km.inertia_:.2e}")

    # Cluster size distribution
    sizes = np.bincount(labels, minlength=args.k).astype(np.int64)
    print(f"\nCluster size distribution (k={args.k}):")
    print(f"  count    : {len(sizes)}")
    print(f"  total    : {sizes.sum():,} (should match N={len(labels):,})")
    print(f"  min      : {sizes.min():,}")
    print(f"  max      : {sizes.max():,}")
    print(f"  median   : {int(np.median(sizes)):,}")
    print(f"  mean     : {sizes.mean():.1f}")
    print(f"  std      : {sizes.std():.1f}")
    for pct in (10, 25, 75, 90):
        print(f"  p{pct:>2}      : {int(np.percentile(sizes, pct)):,}")

    # Save labels
    labels_path = out_dir / f"image_kmeans_labels__k{args.k}.npz"
    np.savez(labels_path,
             labels=labels,
             keys=np.array(keys, dtype=object),
             centroids=km.cluster_centers_.astype(np.float32))
    print(f"\nSaved labels -> {labels_path}")

    # Per-cluster summary file (sizes + centroid norm in UMAP space)
    summary_path = out_dir / f"image_kmeans_summary__k{args.k}.txt"
    order = np.argsort(-sizes)
    centroids = km.cluster_centers_
    lines = [
        f"KMeans k={args.k} on UMAP coords {coords.shape}",
        f"Total points: {len(labels):,}",
        f"Sizes: min={sizes.min()} max={sizes.max()} "
        f"median={int(np.median(sizes))} mean={sizes.mean():.1f} "
        f"std={sizes.std():.1f}",
        "",
        f"  {'rank':>4}  {'cluster':>7}  {'size':>7}  {'cent_norm':>10}",
        "  " + "-" * 38,
    ]
    for rank, c in enumerate(order):
        lines.append(
            f"  {rank:>4}  {int(c):>7}  {int(sizes[c]):>7}  "
            f"{float(np.linalg.norm(centroids[c])):>10.4f}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
