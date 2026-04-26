# Class-embedding workflow

End-to-end pipeline that turns object captions into discrete class labels via
sentence-embedding + clustering. All scripts live in this directory and are
run from the repo root. All generated files (`*.npz`, plots, summaries,
`clusters_*/` dirs, `object_to_class*.json`) are gitignored — regenerate
them on each machine.

## Prerequisites

- `.env` configured at the repo root (see `../README.md` and `../CLAUDE.md`).
  Relevant vars: `CAPTIONS_PATH`, `TDTOPIA_CAPTION_PATH`, `ALL_OBJ_JSON`.
- `downloaded/captions.json` produced by `../data/preprocess_captions.py`.
- Python deps from `../requirements.txt` plus `umap-learn`, `hdbscan`,
  `sentence-transformers`, `scikit-learn`, `matplotlib`. The Qwen2 encoder
  needs `trust_remote_code=True` and a GPU.
- Optional `--filter-list` JSON (e.g. `aesthetic_list.json`) to restrict
  encoding/refinement to a subset.

## Pipeline

### 1. Encode captions → embeddings

`encode_captions.py` runs a sentence-transformer over each caption and saves
`{embeddings, keys, captions}` as a `.npz`.

```bash
# Recommended for class labels: gte-Qwen2-1.5B-instruct (1536-d, GPU).
python object_classification/encode_captions.py --model gte-qwen2
```

Output: `caption_embeddings_gte_qwen2.npz` (or `caption_embeddings.npz` for
the bge-small fallback).

The Qwen2 instruction prompt is tuned for *morphological* similarity rather
than taxonomic/functional grouping — see the `_MODELS["gte-qwen2"]["prompt"]`
in `encode_captions.py`.

### 2. UMAP + HDBSCAN baseline (optional)

`cluster_captions.py` reduces the embeddings to 2D with UMAP and runs
HDBSCAN. The UMAP projection is cached as
`umap_coords__neighbors{N}__mindist{D}__metric{M}.npz` and reused on
subsequent runs (delete to recompute). Output: `cluster_labels.npz`,
`cluster_plot.png`, `cluster_summary.txt`.

```bash
python object_classification/cluster_captions.py
```

### 3. Tune HDBSCAN against the cached UMAP (optional)

`tune_hdbscan.py` grid-searches HDBSCAN over the cached UMAP coordinates
(it does **not** rerun UMAP). Configs are ranked by coefficient of variation
of cluster sizes; configs above `--max-clusters` (default 100) are dropped.

```bash
python object_classification/tune_hdbscan.py \
    --output-csv object_classification/hdbscan_grid.csv
```

### 4. Refine into the final class set (the production path)

`refine_clusters.py --strategy fresh-kmeans` is what produced the kept label
sets. It over-clusters with spherical k-means, merges nearest neighbours by
centroid cosine similarity, then optionally splits any remaining loose
clusters to enforce a tightness floor.

```bash
# Full set (~250k objects) → ~120 classes
python object_classification/refine_clusters.py \
    --strategy fresh-kmeans \
    --over-cluster 400 \
    --target-count 100 \
    --max-total 120 \
    --output-dir object_classification/clusters_refined_kmeans

# Aesthetic subset (~82k objects) → ~91 classes
python object_classification/refine_clusters.py \
    --strategy fresh-kmeans \
    --filter-list downloaded/aesthetic_chunk/filtered_aesthetic_list.json \
    --over-cluster 300 \
    --target-count 80 \
    --max-total 100 \
    --output-dir object_classification/clusters_refined_aesthetic
```

Each output dir contains `cluster_labels.npz` (`labels` + `keys`),
`cluster_plot.png`, `cluster_summary.txt`.

### 5. Manual audit (aesthetic subset only)

`apply_audit_edits.py` applies a hardcoded list of merges and 2-way
k-means splits chosen by reading `clusters_refined_aesthetic/cluster_summary.txt`.
The hardcoded IDs are valid only for that exact run — re-audit after
regenerating step 4.

```bash
python object_classification/apply_audit_edits.py
# → clusters_refined_aesthetic_audited/
```

## Inspection helpers

- `inspect_clusters.py` — interactive browser; sample captions, search,
  save highlighted UMAP plots.
- `cluster_distribution.py` — parses `cluster_summary.txt` and prints a
  ranked table + ASCII histogram of cluster sizes.

## Conventions

- **Keys** in every `.npz` are chunk/filename stems (e.g. `1876/9374307`),
  matching `captions.json` and the WebDataset sample names. Hash-UUIDs from
  `aesthetic_list.json` are converted by `load_filter_keys` /
  `load_filter_stems`.
- Bias toward **morphological coherence over semantic similarity** when
  tuning, even at the cost of CV. Target budget ≈100–120 classes.
