"""
Grid-search HDBSCAN hyperparameters over pre-computed UMAP coordinates.

Loads a UMAP cache file (produced by cluster_captions.py) and exhaustively
tries every combination of HDBSCAN settings.  Each result is scored by the
coefficient of variation (CV = std / mean) of cluster sizes — lower is more
uniform.  Configurations that exceed --max-clusters are discarded.

The ranked table is printed to stdout; use --output-csv to also save every
evaluated configuration.

Usage examples
--------------
# Auto-detect the UMAP cache in the default output dir:
python object_classification/tune_hdbscan.py

# Point at a specific cache file:
python object_classification/tune_hdbscan.py \
    --umap object_classification/umap_coords__neighbors30__mindist0.05__metriccosine.npz

# Wider grid, save all results:
python object_classification/tune_hdbscan.py \
    --min-cluster-sizes 50 100 200 500 \
    --min-samples 1 5 10 \
    --epsilons 0.0 0.05 0.1 0.2 \
    --methods eom leaf \
    --max-clusters 80 \
    --output-csv object_classification/hdbscan_grid.csv \
    --top 30
"""

import argparse
import csv
import itertools
from pathlib import Path

import hdbscan
import numpy as np


_SCRIPT_DIR = Path(__file__).resolve().parent

# Default grid values
_DEFAULT_MIN_CLUSTER_SIZES = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
_DEFAULT_MIN_SAMPLES = [1, 2, 3, 4, 5, 10, 20]       # None (= min_cluster_size) added implicitly
_DEFAULT_EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
_DEFAULT_METHODS = ["eom", "leaf"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search HDBSCAN params over cached UMAP coords."
    )
    parser.add_argument(
        "--umap",
        default=None,
        help=(
            "Path to umap_coords__*.npz cache file.  "
            "If omitted the script auto-detects one in --umap-dir."
        ),
    )
    parser.add_argument(
        "--umap-dir",
        default=str(_SCRIPT_DIR),
        help=f"Directory to search for a UMAP cache file (default: {_SCRIPT_DIR})",
    )
    # Grid axes
    parser.add_argument(
        "--min-cluster-sizes",
        type=int,
        nargs="+",
        default=_DEFAULT_MIN_CLUSTER_SIZES,
        metavar="N",
        help=f"min_cluster_size values to try (default: {_DEFAULT_MIN_CLUSTER_SIZES})",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        nargs="+",
        default=_DEFAULT_MIN_SAMPLES,
        metavar="N",
        help=(
            f"min_samples values to try (default: {_DEFAULT_MIN_SAMPLES}).  "
            "The value 0 is treated as 'use min_cluster_size' (i.e. the HDBSCAN default)."
        ),
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=_DEFAULT_EPSILONS,
        metavar="E",
        help=f"cluster_selection_epsilon values to try (default: {_DEFAULT_EPSILONS})",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=_DEFAULT_METHODS,
        choices=["eom", "leaf"],
        metavar="M",
        help=f"cluster_selection_method values to try (default: {_DEFAULT_METHODS})",
    )
    # Filtering / output
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=100,
        help="Hard upper bound on cluster count; configs above this are discarded (default: 100)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top-ranked configs to print (default: 20)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        metavar="PATH",
        help="Save all evaluated configurations to a CSV file",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# UMAP cache loading
# ---------------------------------------------------------------------------

def find_umap_cache(umap_dir: Path) -> Path:
    matches = sorted(umap_dir.glob("umap_coords__*.npz"))
    if not matches:
        raise FileNotFoundError(
            f"No umap_coords__*.npz file found in {umap_dir}.\n"
            "Run cluster_captions.py first to generate the UMAP cache."
        )
    if len(matches) > 1:
        print(f"Multiple UMAP cache files found; using the most recent:")
        for p in matches:
            print(f"  {p.name}")
        matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"Using UMAP cache: {matches[0]}")
    return matches[0]


def load_umap_coords(path: Path) -> np.ndarray:
    data = np.load(path)
    coords = data["umap_coords"].astype(np.float32)
    print(f"  UMAP coords shape: {coords.shape}")
    return coords


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def coefficient_of_variation(sizes: list) -> float:
    """CV = std / mean of cluster sizes (0 = perfectly uniform)."""
    if len(sizes) < 2:
        return 0.0
    arr = np.array(sizes, dtype=np.float64)
    mean = arr.mean()
    if mean == 0:
        return 0.0
    return float(arr.std() / mean)


def gini(sizes: list) -> float:
    """Gini coefficient of cluster sizes (0 = perfectly uniform)."""
    if len(sizes) < 2:
        return 0.0
    arr = np.sort(np.array(sizes, dtype=np.float64))
    n = len(arr)
    cumsum = np.cumsum(arr)
    return float((2 * np.dot(np.arange(1, n + 1), arr) - (n + 1) * cumsum[-1]) / (n * cumsum[-1]))


def evaluate(
    coords: np.ndarray,
    min_cluster_size: int,
    min_samples_raw: int,
    epsilon: float,
    method: str,
) -> dict:
    """Run HDBSCAN with given params and return a result dict."""
    ms = min_samples_raw if min_samples_raw != 0 else min_cluster_size

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=ms,
        metric="euclidean",
        cluster_selection_epsilon=epsilon,
        cluster_selection_method=method,
    )
    labels = clusterer.fit_predict(coords)

    unique = set(labels)
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_noise = int((labels == -1).sum())
    n_total = len(labels)

    sizes = [int((labels == lbl).sum()) for lbl in unique if lbl != -1]
    sizes_arr = np.array(sizes) if sizes else np.array([0])

    cv = coefficient_of_variation(sizes)
    gi = gini(sizes)

    return {
        "min_cluster_size": min_cluster_size,
        "min_samples": ms,
        "min_samples_raw": min_samples_raw,
        "epsilon": epsilon,
        "method": method,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": 100.0 * n_noise / n_total,
        "cv": cv,
        "gini": gi,
        "size_min": int(sizes_arr.min()) if sizes else 0,
        "size_median": float(np.median(sizes_arr)) if sizes else 0,
        "size_max": int(sizes_arr.max()) if sizes else 0,
        "size_mean": float(sizes_arr.mean()) if sizes else 0,
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid(
    coords: np.ndarray,
    min_cluster_sizes: list,
    min_samples_list: list,
    epsilons: list,
    methods: list,
    max_clusters: int,
) -> tuple:
    """Return (all_results, passing_results) sorted by CV ascending."""
    combos = list(itertools.product(min_cluster_sizes, min_samples_list, epsilons, methods))
    total = len(combos)
    print(f"\nGrid size: {total} combinations")
    print(f"Hard filter: n_clusters <= {max_clusters}\n")

    all_results = []
    for i, (mcs, ms_raw, eps, method) in enumerate(combos, 1):
        result = evaluate(coords, mcs, ms_raw, eps, method)
        all_results.append(result)

        status = "OK  " if result["n_clusters"] <= max_clusters else "SKIP"
        print(
            f"  [{i:>{len(str(total))}}/{total}] {status} "
            f"mcs={mcs:>5}  ms={result['min_samples']:>5}  "
            f"eps={eps:.2f}  method={method:<4}  "
            f"→ {result['n_clusters']:>3} clusters  "
            f"noise={result['noise_pct']:>5.1f}%  "
            f"CV={result['cv']:.4f}"
        )

    passing = [r for r in all_results if r["n_clusters"] <= max_clusters]
    passing.sort(key=lambda r: r["cv"])
    all_results.sort(key=lambda r: (r["n_clusters"] > max_clusters, r["cv"]))

    return all_results, passing


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(results: list, top: int, max_clusters: int) -> None:
    if not results:
        print(f"\nNo configurations found with n_clusters <= {max_clusters}.")
        return

    shown = results[:top]
    header = (
        f"  {'Rank':>4}  {'mcs':>6}  {'ms':>5}  {'eps':>5}  {'method':<4}  "
        f"{'clusters':>8}  {'noise%':>7}  {'CV':>7}  {'Gini':>6}  "
        f"{'min':>6}  {'median':>7}  {'max':>6}"
    )
    sep = "  " + "─" * (len(header) - 2)

    print(f"\nTop {min(top, len(results))} configurations (of {len(results)} passing, sorted by CV):\n")
    print(header)
    print(sep)

    for rank, r in enumerate(shown, 1):
        ms_label = str(r["min_samples"]) if r["min_samples_raw"] != 0 else f"{r['min_samples']}*"
        print(
            f"  {rank:>4}  {r['min_cluster_size']:>6}  {ms_label:>5}  "
            f"{r['epsilon']:>5.2f}  {r['method']:<4}  "
            f"{r['n_clusters']:>8}  {r['noise_pct']:>6.1f}%  "
            f"{r['cv']:>7.4f}  {r['gini']:>6.4f}  "
            f"{r['size_min']:>6}  {r['size_median']:>7.0f}  {r['size_max']:>6}"
        )

    if len(results) > top:
        print(f"  ... ({len(results) - top} more; use --top to show more or --output-csv to save all)")

    print("\n  * ms column: value marked with * uses min_cluster_size as min_samples (HDBSCAN default)\n")


def save_csv(path: Path, results: list) -> None:
    if not results:
        return
    fields = [
        "rank", "min_cluster_size", "min_samples", "epsilon", "method",
        "n_clusters", "n_noise", "noise_pct", "cv", "gini",
        "size_min", "size_median", "size_max", "size_mean",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rank, r in enumerate(results, 1):
            writer.writerow({"rank": rank, **r})
    print(f"Saved full grid results → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.umap:
        umap_path = Path(args.umap)
        if not umap_path.exists():
            raise FileNotFoundError(f"UMAP cache file not found: {umap_path}")
    else:
        umap_path = find_umap_cache(Path(args.umap_dir))

    coords = load_umap_coords(umap_path)

    # 0 in the CLI list means "use min_cluster_size" (HDBSCAN default behaviour)
    min_samples_list = [0] + args.min_samples

    all_results, passing = run_grid(
        coords,
        min_cluster_sizes=args.min_cluster_sizes,
        min_samples_list=min_samples_list,
        epsilons=args.epsilons,
        methods=args.methods,
        max_clusters=args.max_clusters,
    )

    print_table(passing, top=args.top, max_clusters=args.max_clusters)

    if args.output_csv:
        save_csv(Path(args.output_csv), all_results)

    print(
        f"Summary: {len(passing)}/{len(all_results)} configurations pass "
        f"(n_clusters <= {args.max_clusters})"
    )


if __name__ == "__main__":
    main()
