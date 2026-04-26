"""Apply manual-audit merges and splits to clusters_refined_aesthetic.

Merges and splits were chosen by eyeballing cluster_summary.txt for
morphological coherence. All IDs below refer to the *display* labels in
clusters_refined_aesthetic/cluster_labels.npz (0..90, sorted by size desc).
"""

from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from refine_clusters import (
    cluster_sizes,
    cluster_tightness,
    l2_normalize,
    renumber,
    report_sizes,
    save_labels,
    save_plot,
    save_summary,
)


_SCRIPT_DIR = Path(__file__).resolve().parent
_EMB = _SCRIPT_DIR / "caption_embeddings_gte_qwen2.npz"
_LAB = _SCRIPT_DIR / "clusters_refined_aesthetic" / "cluster_labels.npz"
_OUT = _SCRIPT_DIR / "clusters_refined_aesthetic_audited"

# (keep, drop) — the `drop` cluster's points move into `keep`.
_MERGES: list[tuple[int, int]] = [
    (8, 42),   # rustic wooden houses/cabins
    (9, 13),   # standing men (3D models + stylized)
    (54, 69),  # rectangular boxes/crates
    (63, 64),  # long rifles (tactical + wooden)
    (52, 68),  # small plush animals (bunnies + teddy bears)
]
_SPLITS: list[int] = [33, 45, 53, 88, 67]
_SEED = 42


def bisect(emb_norm: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
    """k-means(2) on the subset selected by mask; returns 0/1 labels."""
    pts = emb_norm[mask]
    km = MiniBatchKMeans(
        n_clusters=2, random_state=seed, batch_size=4096, n_init=5, max_iter=300
    )
    return km.fit_predict(pts)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)

    emb_data = np.load(_EMB, allow_pickle=True)
    lab_data = np.load(_LAB, allow_pickle=True)
    emb_keys = emb_data["keys"].tolist()
    captions = emb_data["captions"].tolist()
    embeddings = emb_data["embeddings"].astype(np.float32)
    emb_norm = l2_normalize(embeddings)

    keys = lab_data["keys"].tolist()
    labels = lab_data["labels"].astype(np.int64).copy()
    umap_coords = lab_data["umap_coords"] if "umap_coords" in lab_data.files else None

    # Project embeddings onto the label ordering.
    ek = {k: i for i, k in enumerate(emb_keys)}
    idx = np.array([ek[k] for k in keys])
    emb_norm_lab = emb_norm[idx]
    captions_lab = [captions[ek[k]] for k in keys]

    report_sizes("before", labels)
    t_before = cluster_tightness(emb_norm_lab, labels)

    # --- Merges ---------------------------------------------------------
    print("\n=== Merges ===")
    for keep, drop in _MERGES:
        n_keep = int((labels == keep).sum())
        n_drop = int((labels == drop).sum())
        labels[labels == drop] = keep
        print(f"  merge {drop}({n_drop:,}) → {keep}({n_keep:,})  → {n_keep + n_drop:,}")
    report_sizes("post-merge", labels)

    # --- Splits ---------------------------------------------------------
    print("\n=== Splits ===")
    next_id = int(labels.max()) + 1
    for lbl in _SPLITS:
        mask = labels == lbl
        n = int(mask.sum())
        if n == 0:
            print(f"  skip split {lbl}: empty")
            continue
        sub = bisect(emb_norm_lab, mask, _SEED)
        pts = emb_norm_lab[mask]
        reports = []
        for s in range(2):
            sp = pts[sub == s]
            c = sp.mean(0)
            c /= max(float(np.linalg.norm(c)), 1e-12)
            reports.append((int((sub == s).sum()), float((sp @ c).mean())))
        idxs = np.where(mask)[0]
        labels[idxs[sub == 1]] = next_id
        print(f"  split {lbl:>3} (n={n:,}, t={t_before.get(lbl, 0):.3f}) → "
              f"[n={reports[0][0]:,} t={reports[0][1]:.3f}] + "
              f"[n={reports[1][0]:,} t={reports[1][1]:.3f}]  (new id {next_id})")
        next_id += 1
    report_sizes("post-split", labels)

    # --- Renumber and report --------------------------------------------
    labels = renumber(labels)
    report_sizes("final", labels)

    t_after = cluster_tightness(emb_norm_lab, labels)
    ts = np.array(list(t_after.values()))
    loose_before = sum(1 for v in t_before.values() if v < 0.875)
    loose_after = int((ts < 0.875).sum())
    print(f"\nTightness: median {np.median(ts):.3f}  P10 {np.percentile(ts, 10):.3f}  "
          f"min {ts.min():.3f}  loose(<0.875) {loose_after}/{len(ts)} "
          f"(was {loose_before})")

    # --- Save -----------------------------------------------------------
    save_labels(_OUT, labels, keys, umap_coords)
    save_summary(_OUT, labels, keys, captions_lab, top_n=5)
    save_plot(_OUT, labels, umap_coords)
    print("\nDone.")


if __name__ == "__main__":
    main()
