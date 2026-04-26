"""
Refine clusters on caption embeddings via two selectable strategies, both of
which finish with a similarity-based merge down to --target-count clusters.

Strategies (--strategy)
-----------------------
hdbscan-seed  (default)
  Seed from an existing cluster_labels.npz and:
    1. Reassign noise points (-1) to their nearest cluster centroid.
    2. Recursively split any cluster above --max-size with k-means.
    3. Merge smallest → most-similar neighbor until <= --target-count.
  Preserves rare morphological signatures discovered by HDBSCAN, but leaf
  clusters can be loose when the original cluster was diffuse.

fresh-kmeans
  Ignore the initial labels entirely and:
    1. Run spherical k-means on L2-normalised embeddings with K = --over-cluster.
    2. Merge smallest → most-similar neighbor until <= --target-count.
  Every leaf is as tight as k-means allows, so the final clusters are the
  morphologically-tightest groupings text embeddings can express. Preferred
  when cluster cohesion / morphological coherence matters more than
  preserving the HDBSCAN structure.

The default target of 100 clusters on 254k objects yields an ideal cluster
size of ≈2,500; --max-size defaults to 4,000 (1.5× target).

Usage
-----
# hdbscan-seed strategy (default):
python object_classification/refine_clusters.py

# fresh-kmeans: over-cluster at K=200 then merge down to 100.
python object_classification/refine_clusters.py \\
    --strategy fresh-kmeans \\
    --output-dir object_classification/clusters_refined_kmeans \\
    --over-cluster 200 \\
    --target-count 100
"""

import argparse
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_EMBEDDINGS = _SCRIPT_DIR / "caption_embeddings_gte_qwen2.npz"
_DEFAULT_LABELS = _SCRIPT_DIR / "clusters_full_250k" / "cluster_labels.npz"
_DEFAULT_OUTPUT = _SCRIPT_DIR / "clusters_refined"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine clusters via recursive splitting + similarity merging."
    )
    parser.add_argument("--embeddings", default=str(_DEFAULT_EMBEDDINGS))
    parser.add_argument("--initial-labels", default=str(_DEFAULT_LABELS))
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT))
    parser.add_argument(
        "--filter-list", default=None,
        help="Optional object-list JSON (e.g. aesthetic_list.json mapping "
             "hash_key → 'chunk/filename.tar.gz'). Embeddings/captions are "
             "subset to these stems before clustering.",
    )
    parser.add_argument(
        "--strategy", choices=["hdbscan-seed", "fresh-kmeans"], default="hdbscan-seed",
        help="hdbscan-seed: refine an existing HDBSCAN labelling. "
             "fresh-kmeans: over-cluster with k-means then merge (default: hdbscan-seed)",
    )
    parser.add_argument(
        "--over-cluster", type=int, default=None,
        help="[fresh-kmeans] Number of leaf k-means clusters before merging. "
             "Defaults to 2 × --target-count.",
    )
    parser.add_argument(
        "--target-count", type=int, default=100,
        help="Final cluster count target (merging stops at or below this; default: 100)",
    )
    parser.add_argument(
        "--max-size", type=int, default=4000,
        help="Any cluster above this size is split; merges are blocked above it "
             "unless no smaller merge is feasible (default: 4000)",
    )
    parser.add_argument(
        "--min-merge-sim", type=float, default=0.55,
        help="Minimum centroid cosine similarity for a merge to proceed. Below "
             "this the merge phase stops even if count > target (default: 0.55)",
    )
    parser.add_argument(
        "--tightness-threshold", type=float, default=0.875,
        help="After merging, any cluster whose mean cos-to-centroid is below "
             "this threshold is split into 2 to improve morphological coherence. "
             "Set to 0 to disable (default: 0.875).",
    )
    parser.add_argument(
        "--max-total", type=int, default=120,
        help="Hard upper bound on the final cluster count. If tightness-splitting "
             "would exceed this, only the loosest clusters are split (default: 120).",
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="Number of example captions per cluster in the summary (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_filter_stems(filter_list_path: str) -> set[str]:
    """Load an object-list JSON (hash_key → 'chunk/filename.tar.gz') and return
    the set of chunk/filename stems. Mirrors encode_captions.load_filter_keys."""
    with open(filter_list_path, "r", encoding="utf-8") as f:
        obj_list = json.load(f)
    return {v.removesuffix(".tar.gz") for v in obj_list.values()}


def load_inputs(embeddings_path: str, labels_path: str | None,
                filter_list_path: str | None = None):
    emb_data = np.load(embeddings_path, allow_pickle=True)
    embeddings = emb_data["embeddings"].astype(np.float32)
    emb_keys = emb_data["keys"].tolist()
    captions = emb_data["captions"].tolist()

    if filter_list_path is not None:
        filter_set = load_filter_stems(filter_list_path)
        mask = np.array([k in filter_set for k in emb_keys], dtype=bool)
        kept = int(mask.sum())
        if kept == 0:
            raise SystemExit(f"[ERROR] No embeddings matched filter list {filter_list_path}")
        print(f"Applied filter list {filter_list_path}: "
              f"{len(emb_keys):,} → {kept:,} embeddings")
        embeddings = embeddings[mask]
        emb_keys = [k for k, m in zip(emb_keys, mask) if m]
        captions = [c for c, m in zip(captions, mask) if m]

    if labels_path is None:
        return embeddings, None, emb_keys, captions, None

    print(f"Loading labels     from {labels_path}")
    lab_data = np.load(labels_path, allow_pickle=True)
    lab_labels = lab_data["labels"].astype(np.int64)
    lab_keys = lab_data["keys"].tolist()
    umap_coords = lab_data["umap_coords"] if "umap_coords" in lab_data.files else None

    if emb_keys == lab_keys:
        labels = lab_labels
    else:
        # Align by key; drop points missing from the label file (noise).
        key_to_label = dict(zip(lab_keys, lab_labels))
        labels = np.array([key_to_label.get(k, -1) for k in emb_keys], dtype=np.int64)
        if umap_coords is not None:
            key_to_coord = dict(zip(lab_keys, umap_coords))
            zero = np.zeros(umap_coords.shape[1], dtype=umap_coords.dtype)
            umap_coords = np.array([key_to_coord.get(k, zero) for k in emb_keys])

    return embeddings, labels, emb_keys, captions, umap_coords


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    uniq, cnt = np.unique(labels, return_counts=True)
    return {int(l): int(c) for l, c in zip(uniq, cnt)}


def report_sizes(tag: str, labels: np.ndarray) -> None:
    sizes = cluster_sizes(labels)
    noise = sizes.pop(-1, 0)
    vals = np.array(list(sizes.values()), dtype=np.float64)
    if len(vals) == 0:
        print(f"  [{tag}] no clusters")
        return
    cv = vals.std() / vals.mean() if vals.mean() else 0.0
    print(
        f"  [{tag}] {len(vals):>3} clusters  "
        f"min={int(vals.min()):>5,}  "
        f"median={int(np.median(vals)):>5,}  "
        f"max={int(vals.max()):>6,}  "
        f"mean={vals.mean():>6.0f}  "
        f"std={vals.std():>6.0f}  "
        f"CV={cv:.3f}  "
        f"noise={noise:,}"
    )


def centroids_of(emb_norm: np.ndarray, labels: np.ndarray, label_list: list[int]) -> np.ndarray:
    D = emb_norm.shape[1]
    out = np.zeros((len(label_list), D), dtype=np.float32)
    for i, lbl in enumerate(label_list):
        mask = labels == lbl
        out[i] = emb_norm[mask].mean(axis=0)
    return l2_normalize(out)


# ---------------------------------------------------------------------------
# Phase 1 — noise reassignment
# ---------------------------------------------------------------------------

def reassign_noise(emb_norm: np.ndarray, labels: np.ndarray) -> np.ndarray:
    noise_mask = labels == -1
    n_noise = int(noise_mask.sum())
    if n_noise == 0:
        return labels

    cluster_ids = sorted(set(int(l) for l in labels) - {-1})
    cents = centroids_of(emb_norm, labels, cluster_ids)

    # Chunked matmul to avoid a (71k, 1536) x (51, 1536) blowup peaking RAM.
    print(f"  reassigning {n_noise:,} noise points across {len(cluster_ids)} clusters...")
    idx_noise = np.where(noise_mask)[0]
    best = np.empty(n_noise, dtype=np.int64)
    chunk = 8192
    for s in range(0, n_noise, chunk):
        e = min(s + chunk, n_noise)
        sims = emb_norm[idx_noise[s:e]] @ cents.T  # (c, K)
        best[s:e] = sims.argmax(axis=1)

    out = labels.copy()
    out[idx_noise] = np.array(cluster_ids)[best]
    return out


# ---------------------------------------------------------------------------
# Phase 2 — recursive splitting
# ---------------------------------------------------------------------------

def split_one(emb_sub: np.ndarray, k: int, seed: int) -> np.ndarray:
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=4096,
        n_init=3,
        max_iter=200,
    )
    return km.fit_predict(emb_sub)


def cluster_tightness(emb_norm: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Return {label: mean cos similarity to centroid} for each non-noise cluster."""
    out: dict[int, float] = {}
    for lbl in sorted(set(int(l) for l in labels)):
        if lbl == -1:
            continue
        pts = emb_norm[labels == lbl]
        cent = pts.mean(axis=0)
        cent = cent / max(float(np.linalg.norm(cent)), 1e-12)
        out[lbl] = float((pts @ cent).mean())
    return out


def tightness_split(
    emb_norm: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    max_total: int,
    seed: int,
) -> np.ndarray:
    """Split each loose cluster (tightness < threshold) into 2 k-means pieces.

    Respects --max-total by splitting only the loosest clusters first, up to
    the budget of (max_total - current_count) additional clusters.
    """
    labels = labels.copy()
    tights = cluster_tightness(emb_norm, labels)
    loose = sorted(
        [(lbl, t) for lbl, t in tights.items() if t < threshold],
        key=lambda x: x[1],  # loosest first
    )
    if not loose:
        print(f"  all {len(tights)} clusters already above tightness {threshold:.3f}")
        return labels

    current_count = len(tights)
    budget = max_total - current_count
    to_split = loose[:budget] if budget > 0 else []
    skipped = len(loose) - len(to_split)
    print(f"  {len(loose)} cluster(s) below tightness {threshold:.3f}; "
          f"splitting {len(to_split)} (budget {budget}), skipping {skipped}")

    next_id = int(labels.max()) + 1
    for lbl, t in to_split:
        mask = labels == lbl
        pts = emb_norm[mask]
        sub = split_one(pts, 2, seed)

        # Report post-split tightness.
        sub_tights = []
        for s in range(2):
            sub_pts = pts[sub == s]
            c = sub_pts.mean(axis=0); c /= max(float(np.linalg.norm(c)), 1e-12)
            sub_tights.append((int((sub == s).sum()), float((sub_pts @ c).mean())))
        print(f"    cluster {lbl:>3} (n={mask.sum():,}, t={t:.3f}) → "
              f"[n={sub_tights[0][0]:,} t={sub_tights[0][1]:.3f}] + "
              f"[n={sub_tights[1][0]:,} t={sub_tights[1][1]:.3f}]")

        idx = np.where(mask)[0]
        labels[idx[sub == 1]] = next_id
        next_id += 1
    return labels


def fresh_kmeans_start(emb_norm: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Run k-means on the L2-normalised embeddings and return per-point labels.

    Running on unit-normalised vectors with Euclidean k-means is equivalent to
    spherical k-means on the original directions (up to the squared-distance /
    cosine correspondence), so centroid-cosine behaviour in downstream merging
    stays consistent with the split phase.
    """
    print(f"  running k-means on {emb_norm.shape[0]:,} points, K={k}...")
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=8192,
        n_init=5,
        max_iter=300,
        reassignment_ratio=0.02,
    )
    labels = km.fit_predict(emb_norm).astype(np.int64)
    return labels


def recursive_split(
    emb_norm: np.ndarray,
    labels: np.ndarray,
    max_size: int,
    target_size: int,
    seed: int,
) -> np.ndarray:
    labels = labels.copy()
    next_id = int(labels.max()) + 1

    it = 0
    while True:
        it += 1
        sizes = cluster_sizes(labels)
        sizes.pop(-1, None)
        oversized = sorted(
            [(lbl, sz) for lbl, sz in sizes.items() if sz > max_size],
            key=lambda x: -x[1],
        )
        if not oversized:
            break

        print(f"  iter {it}: {len(oversized)} oversized; "
              f"largest cluster {oversized[0][0]} has {oversized[0][1]:,} objects")

        for lbl, sz in oversized:
            mask = labels == lbl
            k = max(2, round(sz / target_size))
            sub_labels = split_one(emb_norm[mask], k, seed)

            # First sub-label keeps the original id; rest get new ids.
            new_ids = [lbl] + list(range(next_id, next_id + k - 1))
            next_id += k - 1
            idx = np.where(mask)[0]
            for j, nid in enumerate(new_ids):
                labels[idx[sub_labels == j]] = nid

    return labels


# ---------------------------------------------------------------------------
# Phase 3 — merge smallest into nearest neighbor
# ---------------------------------------------------------------------------

def merge_to_target(
    emb_norm: np.ndarray,
    labels: np.ndarray,
    target_count: int,
    min_sim: float,
    max_size: int,
) -> np.ndarray:
    """Iteratively merge the smallest cluster into its most-similar neighbor.

    A merge is skipped if it would push the destination above max_size AND a
    smaller-destination alternative exists. If no similar-enough neighbor is
    available for the smallest cluster we fall back to subsequent clusters,
    and finally stop when nothing passes min_sim.
    """
    labels = labels.copy()
    step = 0

    while True:
        sizes = cluster_sizes(labels)
        sizes.pop(-1, None)
        if len(sizes) <= target_count:
            break

        ids = sorted(sizes.keys())
        sz = np.array([sizes[i] for i in ids], dtype=np.int64)
        cents = centroids_of(emb_norm, labels, ids)
        sim = cents @ cents.T
        np.fill_diagonal(sim, -np.inf)

        # Try clusters in ascending size order. For each candidate, find the
        # most similar neighbor that (a) passes min_sim, and (b) preferably
        # won't exceed max_size after merging.
        order = np.argsort(sz)
        merged = False
        for i in order:
            neigh_order = np.argsort(-sim[i])  # descending similarity
            best_j = None
            best_j_any = None
            for j in neigh_order:
                if sim[i, j] < min_sim:
                    break
                if best_j_any is None:
                    best_j_any = int(j)
                if sz[i] + sz[j] <= max_size:
                    best_j = int(j)
                    break
            # If nothing under cap, accept the any-sim best (only once per pass
            # when i is the absolute smallest — this is the "relax cap" case).
            if best_j is None and i == order[0]:
                best_j = best_j_any
            if best_j is None:
                continue

            src, dst = ids[i], ids[best_j]
            s = sim[i, best_j]
            labels[labels == src] = dst
            step += 1
            if step <= 10 or step % 10 == 0:
                print(f"  step {step:>3}: merge {src}({sz[i]:,}) → {dst}({sz[best_j]:,})  "
                      f"sim={s:.3f}  (now {len(sizes) - 1} clusters)")
            merged = True
            break

        if not merged:
            print(f"  stop: no pair passes min_sim={min_sim}; "
                  f"{len(sizes)} clusters remain (target {target_count})")
            break

    return labels


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def renumber(labels: np.ndarray) -> np.ndarray:
    """Renumber labels to contiguous 0..K-1 by descending cluster size. Noise stays -1."""
    sizes = cluster_sizes(labels)
    noise = sizes.pop(-1, 0)
    # Sort clusters by size descending so cluster 0 is the biggest.
    ordered = [lbl for lbl, _ in sorted(sizes.items(), key=lambda x: -x[1])]
    remap = {old: new for new, old in enumerate(ordered)}
    if noise:
        remap[-1] = -1
    return np.array([remap[int(l)] for l in labels], dtype=np.int64)


def save_labels(output_dir: Path, labels: np.ndarray, keys: list[str],
                umap_coords: np.ndarray | None) -> None:
    out_path = output_dir / "cluster_labels.npz"
    payload = {"labels": labels, "keys": np.array(keys, dtype=object)}
    if umap_coords is not None:
        payload["umap_coords"] = umap_coords
    np.savez(out_path, **payload)
    print(f"Saved labels → {out_path}")


def save_summary(output_dir: Path, labels: np.ndarray, keys: list[str],
                 captions: list[str], top_n: int) -> None:
    out_path = output_dir / "cluster_summary.txt"
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        cluster_to_indices[int(lbl)].append(i)
    unique = sorted(cluster_to_indices.keys())
    n_clusters = len(unique) - (1 if -1 in unique else 0)

    lines: list[str] = [
        "=" * 80,
        f"Refined Cluster Summary — {n_clusters} clusters",
        "=" * 80,
        "",
    ]
    for lbl in unique:
        indices = cluster_to_indices[lbl]
        tag = "NOISE" if lbl == -1 else f"Cluster {lbl:>3d}"
        lines.append("─" * 60)
        lines.append(f"{tag}  ({len(indices):,} objects)")
        lines.append("")
        for idx in indices[:top_n]:
            wrapped = textwrap.fill(captions[idx], width=72, subsequent_indent="    ")
            lines.append(f"  [{keys[idx]}]")
            lines.append(f"    {wrapped}")
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved summary → {out_path}")


def save_plot(output_dir: Path, labels: np.ndarray, umap_coords: np.ndarray) -> None:
    if umap_coords is None or umap_coords.shape[1] < 2:
        return
    coords_2d = umap_coords[:, :2]
    unique = sorted(set(int(l) for l in labels))
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "hsv")

    fig, ax = plt.subplots(figsize=(12, 10))
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(coords_2d[noise_mask, 0], coords_2d[noise_mask, 1],
                   c="lightgrey", s=3, alpha=0.4, linewidths=0, zorder=1)

    cluster_labels = [l for l in unique if l != -1]
    for i, lbl in enumerate(cluster_labels):
        mask = labels == lbl
        color = cmap(i / max(len(cluster_labels) - 1, 1))
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   c=[color], s=4, alpha=0.7, linewidths=0, zorder=2)

    ax.set_title(f"Refined clusters — {n_clusters} clusters", fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.axis("equal")
    plt.tight_layout()
    out_path = output_dir / "cluster_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot    → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from {args.embeddings}")
    # Always attempt to load UMAP coords from the initial-labels file so that
    # fresh-kmeans still gets a plot. The HDBSCAN labels themselves are only
    # used when --strategy hdbscan-seed.
    embeddings, init_labels, keys, captions, umap_coords = load_inputs(
        args.embeddings, args.initial_labels, args.filter_list
    )
    if args.strategy == "fresh-kmeans":
        init_labels = None
    print(f"  {len(embeddings):,} embeddings, dim={embeddings.shape[1]}")

    emb_norm = l2_normalize(embeddings)
    target_size = max(1, len(embeddings) // args.target_count)
    print(f"\nStrategy: {args.strategy}  target={args.target_count} clusters  "
          f"ideal size={target_size:,}  max_size={args.max_size:,}  "
          f"min_merge_sim={args.min_merge_sim}")

    if args.strategy == "hdbscan-seed":
        labels = init_labels
        report_sizes("initial", labels)
        print("\n=== Phase 1 — reassign noise ===")
        labels = reassign_noise(emb_norm, labels)
        report_sizes("post-noise", labels)

        print("\n=== Phase 2 — recursive split ===")
        labels = recursive_split(emb_norm, labels, args.max_size, target_size, args.seed)
        report_sizes("post-split", labels)
    else:
        k_over = args.over_cluster or (2 * args.target_count)
        print(f"\n=== Phase 1/2 — fresh k-means over-clustering at K={k_over} ===")
        labels = fresh_kmeans_start(emb_norm, k_over, args.seed)
        report_sizes("post-kmeans", labels)

    print("\n=== Phase 3 — merge ===")
    labels = merge_to_target(
        emb_norm, labels, args.target_count, args.min_merge_sim, args.max_size
    )
    report_sizes("post-merge", labels)

    if args.tightness_threshold > 0:
        print(f"\n=== Phase 4 — tightness split (threshold={args.tightness_threshold}, "
              f"max_total={args.max_total}) ===")
        labels = tightness_split(
            emb_norm, labels, args.tightness_threshold, args.max_total, args.seed
        )
        report_sizes("post-tight", labels)

    labels = renumber(labels)
    report_sizes("final", labels)

    save_labels(output_dir, labels, keys, umap_coords)
    save_summary(output_dir, labels, keys, captions, args.top_n)
    if not args.no_plot:
        save_plot(output_dir, labels, umap_coords)

    print("\nDone.")


if __name__ == "__main__":
    main()
