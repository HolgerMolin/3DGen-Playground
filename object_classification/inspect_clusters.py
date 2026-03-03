"""
Interactive cluster inspector.

Lets you browse HDBSCAN clusters produced by cluster_captions.py — view
random caption samples, compare clusters side-by-side, and save highlighted
UMAP plots for any cluster.

Usage
-----
# Interactive mode (default):
python object_classification/inspect_clusters.py

# Jump straight to a cluster:
python object_classification/inspect_clusters.py --cluster 7

# Non-interactive: just print N samples from a cluster and exit:
python object_classification/inspect_clusters.py --cluster 7 --samples 20 --no-interactive

Commands available in interactive mode
---------------------------------------
  <number>          — show random samples from that cluster
  <a>,<b>,...       — show samples from multiple clusters side-by-side (e.g. 3,7,12)
  n                 — next batch of samples from the last viewed cluster(s)
  p <cluster>       — save a highlighted UMAP plot for that cluster
  l                 — list all clusters with sizes
  s <query>         — search captions for a keyword across all clusters
  q / exit          — quit
"""

import argparse
import random
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_LABELS = _SCRIPT_DIR / "cluster_labels.npz"
_DEFAULT_EMBEDDINGS = _SCRIPT_DIR / "caption_embeddings.npz"
_SAMPLE_WIDTH = 88


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(labels_path: str, embeddings_path: str):
    lab_data = np.load(labels_path, allow_pickle=True)
    labels = lab_data["labels"]
    coords = lab_data["umap_coords"]
    label_keys = lab_data["keys"].tolist()

    emb_data = np.load(embeddings_path, allow_pickle=True)
    emb_keys = emb_data["keys"].tolist()
    captions = emb_data["captions"].tolist()

    # Build key -> caption lookup
    key_to_caption = dict(zip(emb_keys, captions))

    # Build cluster -> list of (key, caption) pairs
    clusters: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for label, key in zip(labels.tolist(), label_keys):
        caption = key_to_caption.get(key, "[no caption]")
        clusters[int(label)].append((key, caption))

    return labels, coords, clusters


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def fmt_caption(key: str, caption: str, cluster_label: int | None = None) -> str:
    prefix = f"[{cluster_label}] " if cluster_label is not None else ""
    header = f"  {prefix}{key}"
    wrapped = textwrap.fill(caption, width=_SAMPLE_WIDTH, initial_indent="    ", subsequent_indent="    ")
    return f"{header}\n{wrapped}"


def print_samples(
    clusters: dict[int, list[tuple[str, str]]],
    cluster_ids: list[int],
    n: int,
    offset: int = 0,
    side_by_side: bool = False,
) -> int:
    """
    Print `n` samples from each cluster_id starting at `offset`.
    Returns the new offset (for paging).
    """
    show_label = len(cluster_ids) > 1

    if side_by_side and len(cluster_ids) > 1:
        # Interleave one sample from each cluster at a time
        items_per_cluster = [
            clusters[cid][offset : offset + n] for cid in cluster_ids
        ]
        max_items = max(len(x) for x in items_per_cluster)
        for i in range(max_items):
            for cid, items in zip(cluster_ids, items_per_cluster):
                if i < len(items):
                    key, cap = items[i]
                    print(fmt_caption(key, cap, cluster_label=cid if show_label else None))
            print()
    else:
        for cid in cluster_ids:
            items = clusters[cid]
            if not items:
                print(f"  [Cluster {cid}] — empty")
                continue
            batch = items[offset : offset + n]
            if not batch:
                print(f"  [Cluster {cid}] — no more samples beyond offset {offset}")
                continue
            if show_label:
                tag = "NOISE" if cid == -1 else f"Cluster {cid}"
                print(f"\n{'─'*60}")
                print(f"  {tag}  ({len(items):,} objects)  — showing {offset+1}–{offset+len(batch)}")
                print(f"{'─'*60}")
            for key, cap in batch:
                print(fmt_caption(key, cap, cluster_label=cid if show_label else None))
                print()

    return offset + n


def list_clusters(clusters: dict[int, list]) -> None:
    noise_count = len(clusters.get(-1, []))
    cluster_ids = sorted(k for k in clusters if k != -1)
    total = sum(len(v) for v in clusters.values())

    print(f"\n{'─'*60}")
    print(f"  {'ID':>4}  {'Size':>7}  {'%':>6}")
    print(f"{'─'*60}")
    for cid in cluster_ids:
        size = len(clusters[cid])
        print(f"  {cid:>4}  {size:>7,}  {100*size/total:>5.1f}%")
    print(f"{'─'*60}")
    print(f"  {'noise':>4}  {noise_count:>7,}  {100*noise_count/total:>5.1f}%")
    print(f"  {'TOTAL':>4}  {total:>7,}")
    print(f"{'─'*60}\n")


def search_captions(clusters: dict[int, list[tuple[str, str]]], query: str, max_hits: int = 20) -> None:
    query_lower = query.lower()
    hits: list[tuple[int, str, str]] = []
    for cid, items in clusters.items():
        for key, cap in items:
            if query_lower in cap.lower():
                hits.append((cid, key, cap))
                if len(hits) >= max_hits:
                    break
        if len(hits) >= max_hits:
            break

    if not hits:
        print(f"  No captions containing '{query}' found.")
        return

    print(f"\n  Found {len(hits)} match(es) for '{query}':\n")
    for cid, key, cap in hits:
        tag = "noise" if cid == -1 else str(cid)
        print(fmt_caption(key, cap, cluster_label=cid))
        print()


def save_highlight_plot(
    labels: np.ndarray,
    coords: np.ndarray,
    highlight_id: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))

    # Background: all other points
    bg_mask = labels != highlight_id
    ax.scatter(
        coords[bg_mask, 0], coords[bg_mask, 1],
        c="lightgrey", s=3, alpha=0.3, linewidths=0, zorder=1,
    )

    # Foreground: highlighted cluster
    hi_mask = labels == highlight_id
    ax.scatter(
        coords[hi_mask, 0], coords[hi_mask, 1],
        c="crimson", s=8, alpha=0.8, linewidths=0, zorder=2,
        label=f"Cluster {highlight_id}  (n={hi_mask.sum():,})",
    )

    tag = "NOISE" if highlight_id == -1 else f"Cluster {highlight_id}"
    ax.set_title(f"UMAP — {tag} highlighted  ({hi_mask.sum():,} / {len(labels):,} points)", fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.axis("equal")
    ax.legend(markerscale=2, fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively inspect HDBSCAN clusters of caption embeddings."
    )
    parser.add_argument(
        "--labels",
        default=str(_DEFAULT_LABELS),
        help=f"cluster_labels.npz from cluster_captions.py (default: {_DEFAULT_LABELS})",
    )
    parser.add_argument(
        "--embeddings",
        default=str(_DEFAULT_EMBEDDINGS),
        help=f"caption_embeddings.npz from encode_captions.py (default: {_DEFAULT_EMBEDDINGS})",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        default=None,
        help="Jump straight to this cluster ID on launch.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples to show per page (default: 10).",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Print samples and exit without entering the REPL.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

HELP_TEXT = """
Commands:
  <number>          show random samples from cluster <number>  (e.g.  7)
  <a>,<b>,...       compare clusters side-by-side             (e.g.  3,7,12)
  n                 next page of samples from last cluster(s)
  p <cluster>       save a highlighted UMAP plot for <cluster>
  l                 list all clusters with sizes
  s <query>         search captions for keyword across all clusters
  h / help          show this help
  q / exit          quit
"""


def repl(
    labels: np.ndarray,
    coords: np.ndarray,
    clusters: dict[int, list[tuple[str, str]]],
    samples_per_page: int,
    start_cluster: int | None,
) -> None:
    last_ids: list[int] = []
    offset = 0

    print(f"\n  Loaded {len(labels):,} objects in "
          f"{len([k for k in clusters if k != -1])} clusters "
          f"(+ {len(clusters.get(-1, []))} noise).")
    print("  Type 'l' to list clusters, 'h' for help, 'q' to quit.\n")

    # Auto-jump if --cluster was given
    if start_cluster is not None:
        if start_cluster not in clusters:
            print(f"  Cluster {start_cluster} not found. Type 'l' to list available clusters.")
        else:
            last_ids = [start_cluster]
            items = clusters[start_cluster]
            random.shuffle(items)
            offset = print_samples(clusters, last_ids, samples_per_page)

    while True:
        try:
            raw = input("cluster> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("q", "quit", "exit"):
            break

        if cmd in ("h", "help"):
            print(HELP_TEXT)
            continue

        if cmd == "l":
            list_clusters(clusters)
            continue

        if cmd == "n":
            if not last_ids:
                print("  No cluster selected yet.")
            else:
                offset = print_samples(clusters, last_ids, samples_per_page, offset=offset,
                                       side_by_side=len(last_ids) > 1)
            continue

        if cmd.startswith("s "):
            query = raw[2:].strip()
            if query:
                search_captions(clusters, query)
            continue

        if cmd.startswith("p "):
            try:
                cid = int(raw.split()[1])
            except (IndexError, ValueError):
                print("  Usage: p <cluster_id>")
                continue
            tag = "noise" if cid == -1 else str(cid)
            out = _SCRIPT_DIR / f"cluster_{tag}_highlight.png"
            save_highlight_plot(labels, coords, cid, out)
            continue

        # Cluster number(s): "7" or "3,7,12"
        try:
            ids = [int(x.strip()) for x in raw.split(",")]
        except ValueError:
            print(f"  Unknown command: '{raw}'. Type 'h' for help.")
            continue

        missing = [i for i in ids if i not in clusters]
        if missing:
            print(f"  Cluster(s) not found: {missing}. Type 'l' to list available clusters.")
            continue

        # Shuffle for variety each time a new cluster is selected
        if ids != last_ids:
            for cid in ids:
                random.shuffle(clusters[cid])
            offset = 0

        last_ids = ids
        offset = print_samples(
            clusters, last_ids, samples_per_page,
            offset=offset, side_by_side=len(ids) > 1,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"Loading cluster labels from {args.labels} …")
    labels, coords, clusters = load_data(args.labels, args.embeddings)

    if args.no_interactive:
        if args.cluster is None:
            sys.exit("[ERROR] --no-interactive requires --cluster <id>")
        if args.cluster not in clusters:
            sys.exit(f"[ERROR] Cluster {args.cluster} not found.")
        print_samples(clusters, [args.cluster], args.samples)
        return

    repl(labels, coords, clusters, args.samples, args.cluster)


if __name__ == "__main__":
    main()
