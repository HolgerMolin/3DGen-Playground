"""
UMAP + HDBSCAN clustering of caption embeddings.

Loads caption_embeddings.npz produced by encode_captions.py, reduces the
384-dim vectors to 2-D with UMAP, clusters them with HDBSCAN, then saves:
  - cluster_labels.npz  — per-sample cluster assignments
  - cluster_plot.png    — 2-D scatter coloured by cluster
  - cluster_summary.txt — top captions / object keys per cluster

UMAP is the slow step. Its output is automatically cached in the output
directory as:

  umap_coords__neighbors{N}__mindist{D}__metric{M}.npz

On subsequent runs with the same UMAP settings the cached file is loaded
instead of recomputing. Delete or rename the file to force a fresh run.

Usage examples
--------------
# Default paths (reads from same folder as this script):
python object_classification/cluster_captions.py
# python object_classification/cluster_captions.py --hdbscan-min-cluster-size 100 --umap-neighbors 100 --hdbscan-min-samples 1
# Custom settings:
python object_classification/cluster_captions.py \
    --embeddings object_classification/caption_embeddings.npz \
    --output-dir object_classification \
    --umap-neighbors 30 \
    --umap-min-dist 0.05 \
    --hdbscan-min-cluster-size 20 \
    --hdbscan-min-samples 5
"""

import argparse
import textwrap
from collections import defaultdict
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_EMBEDDINGS = _SCRIPT_DIR / "caption_embeddings.npz"
_DEFAULT_OUTPUT_DIR = _SCRIPT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UMAP + HDBSCAN on caption embeddings."
    )
    parser.add_argument(
        "--embeddings",
        default=str(_DEFAULT_EMBEDDINGS),
        help=f"Path to .npz produced by encode_captions.py (default: {_DEFAULT_EMBEDDINGS})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help=f"Directory for output files (default: {_DEFAULT_OUTPUT_DIR})",
    )
    # UMAP
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors — controls local vs global structure (default: 30)",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.05,
        help="UMAP min_dist — lower = tighter clusters in projection (default: 0.05)",
    )
    parser.add_argument(
        "--umap-metric",
        default="cosine",
        help="UMAP distance metric (default: cosine)",
    )
    # HDBSCAN
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=15,
        help="HDBSCAN min_cluster_size (default: 15)",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples; defaults to min_cluster_size if omitted",
    )
    parser.add_argument(
        "--hdbscan-metric",
        default="euclidean",
        help="HDBSCAN metric applied to UMAP-reduced coords (default: euclidean)",
    )
    # Summary
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of example captions to show per cluster in the summary (default: 5)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the scatter plot (useful in headless environments)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embeddings(path: str) -> tuple[np.ndarray, list[str], list[str]]:
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    keys = data["keys"].tolist()
    captions = data["captions"].tolist()
    print(f"Loaded {len(keys):,} embeddings  shape={embeddings.shape}  from {path}")
    return embeddings, keys, captions


def umap_cache_path(output_dir: Path, n_neighbors: int, min_dist: float, metric: str) -> Path:
    """Return the path for the UMAP coordinate cache file for the given params."""
    return output_dir / f"umap_coords__neighbors{n_neighbors}__mindist{min_dist}__metric{metric}.npz"


def load_umap_cache(path: Path) -> np.ndarray:
    data = np.load(path)
    coords = data["umap_coords"]
    print(f"Loaded cached UMAP coords  shape={coords.shape}  from {path}")
    return coords


def save_umap_cache(path: Path, coords: np.ndarray) -> None:
    np.savez(path, umap_coords=coords)
    print(f"Saved UMAP cache  → {path}")


def run_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
) -> np.ndarray:
    print(
        f"\nRunning UMAP  (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}) …"
    )
    reducer = umap.UMAP(
        n_components=15,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        verbose=True,
    )
    coords = reducer.fit_transform(embeddings)
    print(f"  UMAP done. Output shape: {coords.shape}")
    return coords.astype(np.float32)


def run_hdbscan(
    coords: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
) -> np.ndarray:
    ms = min_samples if min_samples is not None else min_cluster_size
    print(
        f"\nRunning HDBSCAN  (min_cluster_size={min_cluster_size}, min_samples={ms}, metric={metric}) …"
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=ms,
        metric=metric,
        prediction_data=True,
        cluster_selection_epsilon=0.1,
    )
    labels = clusterer.fit_predict(coords)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"  HDBSCAN done.")
    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise points   : {n_noise:,}  ({100 * n_noise / len(labels):.1f}%)")
    return labels


def save_labels(output_dir: Path, coords: np.ndarray, labels: np.ndarray, keys: list[str]) -> None:
    out = output_dir / "cluster_labels.npz"
    np.savez(
        out,
        labels=labels,
        umap_coords=coords,
        keys=np.array(keys, dtype=object),
    )
    print(f"\nSaved labels → {out}")


def save_plot(
    output_dir: Path,
    coords: np.ndarray,
    labels: np.ndarray,
) -> None:
    out = output_dir / "cluster_plot.png"

    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "hsv")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Noise points first (grey, behind)
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            coords[noise_mask, 0],
            coords[noise_mask, 1],
            c="lightgrey",
            s=3,
            alpha=0.4,
            linewidths=0,
            label="noise",
            zorder=1,
        )

    # Clustered points
    cluster_labels = [l for l in unique_labels if l != -1]
    for i, label in enumerate(cluster_labels):
        mask = labels == label
        color = cmap(i / max(len(cluster_labels) - 1, 1))
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color],
            s=5,
            alpha=0.7,
            linewidths=0,
            label=f"cluster {label}",
            zorder=2,
        )

    ax.set_title(
        f"UMAP + HDBSCAN  —  {n_clusters} clusters  "
        f"({int(noise_mask.sum()):,} noise points)",
        fontsize=14,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.axis("equal")

    # Legend: only show if few enough clusters
    if n_clusters <= 30:
        ax.legend(
            markerscale=2,
            fontsize=7,
            loc="upper right",
            ncol=max(1, n_clusters // 15),
        )

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved plot    → {out}")


def save_summary(
    output_dir: Path,
    labels: np.ndarray,
    keys: list[str],
    captions: list[str],
    top_n: int,
) -> None:
    out = output_dir / "cluster_summary.txt"

    # Group indices by cluster
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_to_indices[int(label)].append(i)

    unique_labels = sorted(cluster_to_indices.keys())
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    lines: list[str] = [
        "=" * 80,
        f"HDBSCAN Cluster Summary — {n_clusters} clusters",
        "=" * 80,
        "",
    ]

    for label in unique_labels:
        indices = cluster_to_indices[label]
        tag = "NOISE" if label == -1 else f"Cluster {label:>3d}"
        lines.append(f"{'─' * 60}")
        lines.append(f"{tag}  ({len(indices):,} objects)")
        lines.append("")

        sample_indices = indices[:top_n]
        for idx in sample_indices:
            wrapped = textwrap.fill(captions[idx], width=72, subsequent_indent="    ")
            lines.append(f"  [{keys[idx]}]")
            lines.append(f"    {wrapped}")
            lines.append("")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved summary → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, keys, captions = load_embeddings(args.embeddings)

    cache_path = umap_cache_path(
        output_dir,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
    )
    if cache_path.exists():
        coords = load_umap_cache(cache_path)
    else:
        coords = run_umap(
            embeddings,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
        )
        save_umap_cache(cache_path, coords)

    labels = run_hdbscan(
        coords,
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        metric=args.hdbscan_metric,
    )

    save_labels(output_dir, coords, labels, keys)

    if not args.no_plot:
        save_plot(output_dir, coords, labels)

    save_summary(output_dir, labels, keys, captions, top_n=args.top_n)

    print("\nDone.")
    print(f"  cluster_labels.npz — load with: np.load(..., allow_pickle=True)")
    print(f"  Arrays: labels (N,), umap_coords (N, 2), keys (N,)")


if __name__ == "__main__":
    main()
