"""
UMAP + HDBSCAN sweep over image embeddings (DINOv2, mean of 4 views).

Pipeline
--------
1. Load `image_embeddings.npz` (keys + 768-d mean DINOv2 vectors).
2. L2-renormalize each vector (the per-view embeddings were L2-normalized
   before being averaged, so the mean is no longer unit-norm).
3. UMAP-reduce to a clustering-friendly dimensionality (default 15-d) and
   cache the result.
4. Optionally grid-search HDBSCAN over the cached UMAP coordinates and
   report cluster size distributions.

The UMAP cache is written next to the input embeddings, named
`umap_image__neighbors{N}__mindist{D}__metric{M}__dim{K}.npz`. Re-running
with the same UMAP params reuses the cache.

Usage
-----
# 1) Compute (or load cached) UMAP only:
python object_classification/cluster_image_embeddings.py

# 2) Compute UMAP and run an HDBSCAN sweep, target ~80-120 clusters:
python object_classification/cluster_image_embeddings.py \\
    --sweep --target-min 80 --target-max 120 \\
    --output-csv object_classification/hdbscan_image_grid.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import hdbscan
import numpy as np
import umap


_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_EMBEDDINGS = _SCRIPT_DIR / "image_embeddings.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", default=str(_DEFAULT_EMBEDDINGS))
    parser.add_argument("--output-dir", default=str(_SCRIPT_DIR))
    # UMAP
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--umap-metric", default="cosine")
    parser.add_argument("--umap-dim", type=int, default=15)
    parser.add_argument("--seed", type=int, default=-1,
                        help="UMAP random_state. Negative = unseeded (parallel). "
                             "Setting a seed forces n_jobs=1 (much slower).")
    # Sweep
    parser.add_argument("--sweep", action="store_true",
                        help="After UMAP, grid-search HDBSCAN params.")
    parser.add_argument("--min-cluster-sizes", type=int, nargs="+",
                        default=[300, 500, 800, 1200, 1800, 2500])
    parser.add_argument("--min-samples", type=int, nargs="+",
                        default=[1, 5, 25, 100])
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--methods", nargs="+", default=["eom"],
                        choices=["eom", "leaf"])
    parser.add_argument("--target-min", type=int, default=80)
    parser.add_argument("--target-max", type=int, default=120)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def load_embeddings(path: str) -> tuple[np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=True)
    emb = data["embeddings"].astype(np.float32)
    keys = data["keys"].tolist()
    print(f"Loaded {len(keys):,} embeddings  shape={emb.shape}  from {path}")
    return emb, keys


def l2_normalize(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    out = (emb / norms).astype(np.float32)
    print(f"L2-renormalized. mean norm before={norms.mean():.4f} "
          f"min={norms.min():.4f} max={norms.max():.4f}")
    return out


def umap_cache_path(out_dir: Path, n_neighbors: int, min_dist: float,
                    metric: str, dim: int) -> Path:
    return out_dir / (
        f"umap_image__neighbors{n_neighbors}__mindist{min_dist}"
        f"__metric{metric}__dim{dim}.npz"
    )


def run_umap(emb: np.ndarray, n_neighbors: int, min_dist: float,
             metric: str, dim: int, seed: int | None) -> np.ndarray:
    print(f"\nRunning UMAP n_neighbors={n_neighbors} min_dist={min_dist} "
          f"metric={metric} n_components={dim} seed={seed} on {emb.shape} ...")
    t0 = time.time()
    kwargs = dict(
        n_components=dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        verbose=True,
        low_memory=True,
    )
    # Setting random_state forces n_jobs=1 inside UMAP, which is far too slow
    # for ~250k samples. Only set it if seed >= 0 was explicitly requested.
    if seed is not None and seed >= 0:
        kwargs["random_state"] = seed
    reducer = umap.UMAP(**kwargs)
    coords = reducer.fit_transform(emb).astype(np.float32)
    print(f"UMAP done in {time.time()-t0:.1f}s. coords {coords.shape}")
    return coords


def cluster_coherence(emb_unit: np.ndarray, labels: np.ndarray) -> dict:
    """Per-cluster mean cosine similarity to centroid, in the original
    (L2-normalised) 768-d space. Returns aggregate stats across clusters.

    `emb_unit` must already be L2-normalised. We compute the centroid as
    the mean of unit vectors, normalise it, then cosine-sim = dot product.
    """
    unique = [lbl for lbl in np.unique(labels) if lbl != -1]
    if not unique:
        return {"coh_median": 0.0, "coh_min": 0.0, "coh_p10": 0.0}
    means = []
    for lbl in unique:
        idx = np.where(labels == lbl)[0]
        v = emb_unit[idx]
        c = v.mean(axis=0)
        c /= max(np.linalg.norm(c), 1e-12)
        means.append(float((v @ c).mean()))
    arr = np.array(means)
    return {
        "coh_median": float(np.median(arr)),
        "coh_min": float(arr.min()),
        "coh_p10": float(np.percentile(arr, 10)),
    }


def evaluate_hdbscan(coords: np.ndarray, mcs: int, ms: int, eps: float,
                     method: str, emb_unit: np.ndarray | None = None) -> dict:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        cluster_selection_epsilon=eps,
        cluster_selection_method=method,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(coords)
    unique = set(labels)
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_noise = int((labels == -1).sum())
    n_total = len(labels)
    sizes = np.array([int((labels == lbl).sum()) for lbl in unique if lbl != -1])
    if sizes.size == 0:
        sizes = np.array([0])
    out = {
        "min_cluster_size": mcs,
        "min_samples": ms,
        "epsilon": eps,
        "method": method,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": 100.0 * n_noise / n_total,
        "size_min": int(sizes.min()),
        "size_median": float(np.median(sizes)),
        "size_max": int(sizes.max()),
        "size_mean": float(sizes.mean()),
        "size_p10": float(np.percentile(sizes, 10)),
        "size_p90": float(np.percentile(sizes, 90)),
        "labels": labels,
    }
    if emb_unit is not None and n_clusters > 0:
        out.update(cluster_coherence(emb_unit, labels))
    return out


def run_sweep(coords: np.ndarray, mcs_list: list[int], ms_list: list[int],
              eps_list: list[float], method_list: list[str],
              target_min: int, target_max: int,
              emb_unit: np.ndarray | None = None) -> list[dict]:
    combos = list(itertools.product(mcs_list, ms_list, eps_list, method_list))
    print(f"\nSweep: {len(combos)} configs "
          f"(target {target_min}-{target_max} clusters)")
    results = []
    for i, (mcs, ms, eps, m) in enumerate(combos, 1):
        t0 = time.time()
        r = evaluate_hdbscan(coords, mcs, ms, eps, m, emb_unit=emb_unit)
        dt = time.time() - t0
        in_target = target_min <= r["n_clusters"] <= target_max
        tag = "HIT " if in_target else "    "
        coh = r.get("coh_median", float("nan"))
        print(f"  [{i:>3}/{len(combos)}] {tag} mcs={mcs:>5} ms={ms:>4} "
              f"eps={eps:>4.2f} {m:<4} -> {r['n_clusters']:>4} clusters, "
              f"noise={r['noise_pct']:>5.1f}%, "
              f"sizes [{r['size_min']:>5},{int(r['size_median']):>5},{r['size_max']:>6}], "
              f"coh_med={coh:.3f}  {dt:.1f}s")
        r["in_target"] = in_target
        # Don't keep `labels` in the results list — too memory-heavy
        r.pop("labels", None)
        results.append(r)
    return results


def print_table(results: list[dict], target_min: int, target_max: int) -> None:
    hits = [r for r in results if r["in_target"]]
    # Rank by coherence (higher = tighter clusters), tiebreak by lower noise.
    hits.sort(key=lambda r: (-r.get("coh_median", 0.0), r["noise_pct"]))
    print(f"\n=== {len(hits)} configs in [{target_min},{target_max}] clusters "
          f"(ranked by per-cluster cosine coherence in 768-d space) ===")
    if not hits:
        print("(none — widen the target range or grid)")
        return
    header = (f"  {'mcs':>5} {'ms':>4} {'eps':>5} {'meth':<4} "
              f"{'k':>4} {'noise%':>7} {'min':>5} {'p10':>5} {'med':>5} "
              f"{'p90':>6} {'max':>6} {'coh_med':>7} {'coh_p10':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in hits:
        print(f"  {r['min_cluster_size']:>5} {r['min_samples']:>4} "
              f"{r['epsilon']:>5.2f} {r['method']:<4} "
              f"{r['n_clusters']:>4} {r['noise_pct']:>6.1f}% "
              f"{r['size_min']:>5} {int(r['size_p10']):>5} "
              f"{int(r['size_median']):>5} {int(r['size_p90']):>6} "
              f"{r['size_max']:>6} {r.get('coh_median', 0):>7.3f} "
              f"{r.get('coh_p10', 0):>7.3f}")


def save_csv(path: Path, results: list[dict]) -> None:
    fields = ["min_cluster_size", "min_samples", "epsilon", "method",
              "n_clusters", "n_noise", "noise_pct",
              "size_min", "size_p10", "size_median", "size_p90", "size_max",
              "size_mean", "coh_median", "coh_p10", "coh_min", "in_target"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"Saved sweep results -> {path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = umap_cache_path(out_dir, args.umap_neighbors, args.umap_min_dist,
                            args.umap_metric, args.umap_dim)
    if cache.exists():
        print(f"Loading cached UMAP coords from {cache}")
        coords = np.load(cache)["umap_coords"].astype(np.float32)
        keys = np.load(cache, allow_pickle=True)["keys"].tolist()
        print(f"  shape {coords.shape}, {len(keys):,} keys")
    else:
        emb, keys = load_embeddings(args.embeddings)
        emb = l2_normalize(emb)
        coords = run_umap(emb, args.umap_neighbors, args.umap_min_dist,
                          args.umap_metric, args.umap_dim, args.seed)
        np.savez(cache, umap_coords=coords, keys=np.array(keys, dtype=object))
        print(f"Cached UMAP coords -> {cache}")

    if args.sweep:
        # Reload + L2-normalise the original 768-d embeddings so we can
        # compute cosine coherence in the input space (UMAP space distances
        # aren't isometric to the input).
        emb_raw, _ = load_embeddings(args.embeddings)
        emb_unit = l2_normalize(emb_raw)
        del emb_raw
        results = run_sweep(coords, args.min_cluster_sizes, args.min_samples,
                            args.epsilons, args.methods,
                            args.target_min, args.target_max,
                            emb_unit=emb_unit)
        print_table(results, args.target_min, args.target_max)
        if args.output_csv:
            save_csv(Path(args.output_csv), results)


if __name__ == "__main__":
    main()
