"""
Parse cluster_summary.txt and report the distribution of samples per label.

Reads the cluster counts written by cluster_captions.py and prints:
  - A ranked table (largest → smallest)
  - Percentile / descriptive statistics
  - An ASCII histogram of cluster sizes (noise excluded from histogram)

Usage
-----
python object_classification/cluster_distribution.py
python object_classification/cluster_distribution.py \
    --summary object_classification/cluster_summary.txt \
    --bins 20
"""

import argparse
import re
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_SUMMARY = _SCRIPT_DIR / "cluster_summary.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report sample-count distribution from cluster_summary.txt."
    )
    parser.add_argument(
        "--summary",
        default=str(_DEFAULT_SUMMARY),
        help=f"Path to cluster_summary.txt (default: {_DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=15,
        help="Number of bins for the ASCII histogram (default: 15)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of largest clusters to show in the ranked table (default: 20)",
    )
    parser.add_argument(
        "--bottom",
        type=int,
        default=10,
        help="Number of smallest clusters to show in the ranked table (default: 10)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Matches lines like:
#   NOISE  (10,658 objects)
#   Cluster   0  (697 objects)
_SECTION_RE = re.compile(
    r"^(?:(?P<noise>NOISE)|Cluster\s+(?P<label>-?\d+))\s+\((?P<count>[\d,]+)\s+objects?\)"
)


def parse_summary(path: Path) -> "tuple[dict, int]":
    """Return (counts_by_label, total_clusters).

    counts_by_label maps label → count.  Noise is stored under the key 'noise'.
    """
    counts: dict[int | str, int] = {}
    total_clusters = 0

    header_re = re.compile(r"HDBSCAN Cluster Summary\s+[—–-]+\s+(\d+)\s+clusters?")

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            m_header = header_re.search(line)
            if m_header:
                total_clusters = int(m_header.group(1))
                continue

            m = _SECTION_RE.match(line)
            if not m:
                continue

            count = int(m.group("count").replace(",", ""))
            if m.group("noise"):
                counts["noise"] = count
            else:
                counts[int(m.group("label"))] = count

    return counts, total_clusters


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[int], p: float) -> float:
    """Linear-interpolation percentile (0 ≤ p ≤ 100)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    idx = (p / 100) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _ascii_histogram(values: list[int], bins: int, bar_width: int = 40) -> list[str]:
    if not values:
        return ["  (no data)"]

    lo, hi = min(values), max(values)
    if lo == hi:
        return [f"  All values are {lo}"]

    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        bucket = min(int((v - lo) / step), bins - 1)
        counts[bucket] += 1

    max_count = max(counts)
    lines = []
    for i, c in enumerate(counts):
        bin_lo = int(lo + i * step)
        bin_hi = int(lo + (i + 1) * step)
        bar = "█" * int(bar_width * c / max_count) if max_count else ""
        lines.append(f"  {bin_lo:>7,}–{bin_hi:<7,} │ {bar:<{bar_width}} {c:,}")

    return lines


def report(counts: dict, total_clusters: int, bins: int, top: int, bottom: int) -> None:
    noise_count = counts.get("noise", 0)
    cluster_counts = {k: v for k, v in counts.items() if k != "noise"}
    total_samples = sum(counts.values())

    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    sizes = sorted(cluster_counts.values())

    print("=" * 70)
    print(f"  Cluster distribution report")
    print("=" * 70)
    print(f"  Clusters (excluding noise) : {total_clusters:,}")
    print(f"  Noise points               : {noise_count:,}  ({100 * noise_count / max(total_samples, 1):.1f}% of all samples)")
    print(f"  Total samples              : {total_samples:,}")
    print()

    # ── Descriptive statistics ────────────────────────────────────────────
    if sizes:
        mean = sum(sizes) / len(sizes)
        variance = sum((x - mean) ** 2 for x in sizes) / len(sizes)
        std = variance ** 0.5

        print("  Cluster-size statistics  (noise excluded)")
        print(f"  {'Min':<10}: {sizes[0]:,}")
        print(f"  {'P5':<10}: {_percentile(sizes, 5):,.1f}")
        print(f"  {'P25':<10}: {_percentile(sizes, 25):,.1f}")
        print(f"  {'Median':<10}: {_percentile(sizes, 50):,.1f}")
        print(f"  {'P75':<10}: {_percentile(sizes, 75):,.1f}")
        print(f"  {'P95':<10}: {_percentile(sizes, 95):,.1f}")
        print(f"  {'Max':<10}: {sizes[-1]:,}")
        print(f"  {'Mean':<10}: {mean:,.1f}")
        print(f"  {'Std dev':<10}: {std:,.1f}")
        print()

    # ── Ranked table ─────────────────────────────────────────────────────
    print(f"  Top {top} largest clusters")
    print(f"  {'Label':>8}  {'Count':>8}  {'% of total':>10}  {'% of clustered':>14}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*14}")
    clustered_total = sum(cluster_counts.values())
    for label, count in sorted_clusters[:top]:
        pct_total = 100 * count / max(total_samples, 1)
        pct_clust = 100 * count / max(clustered_total, 1)
        print(f"  {label:>8}  {count:>8,}  {pct_total:>9.2f}%  {pct_clust:>13.2f}%")

    if len(sorted_clusters) > top + bottom:
        print(f"  {'...':>8}")

    if bottom and len(sorted_clusters) > top:
        tail = sorted_clusters[-bottom:]
        if len(sorted_clusters) <= top:
            pass  # already printed above
        else:
            print(f"\n  Bottom {bottom} smallest clusters")
            print(f"  {'Label':>8}  {'Count':>8}  {'% of total':>10}  {'% of clustered':>14}")
            print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*14}")
            for label, count in tail:
                pct_total = 100 * count / max(total_samples, 1)
                pct_clust = 100 * count / max(clustered_total, 1)
                print(f"  {label:>8}  {count:>8,}  {pct_total:>9.2f}%  {pct_clust:>13.2f}%")
    print()

    # ── ASCII histogram ───────────────────────────────────────────────────
    print(f"  Cluster-size histogram  ({bins} bins, noise excluded)")
    print()
    for line in _ascii_histogram(sizes, bins):
        print(line)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    counts, total_clusters = parse_summary(summary_path)

    if not counts:
        print("No cluster sections found in the summary file.")
        return

    report(counts, total_clusters, bins=args.bins, top=args.top, bottom=args.bottom)


if __name__ == "__main__":
    main()

