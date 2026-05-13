# Class-conditioning diagnostics

Three offline tools for inspecting how class-aware a JiT checkpoint is, plus an
orchestrator that ties them together. Read-only — none of these touch training
state.

## TL;DR

```bash
./jit/run_diagnostics.sh output/<run>/<step>.pt
```

Outputs land at `output/diagnostics/<run_id>/{probe,weights,diversity}/` where
`run_id = <ckpt_parent_dir>_<ckpt_stem>`.

Skip the slow render/DINOv2 step:

```bash
./jit/run_diagnostics.sh output/<run>/<step>.pt --skip-diversity
```

The DINOv2 feature cache lives at `output/diagnostics/_cache/dinov2_features/`
and is shared across all checkpoint runs (it depends only on the class map and
the sampling seed/n_per_class, not on the model weights). First diversity run
takes ~3 min; later runs that reuse the cache take ~10 s.

## Pipeline stages

### 1. Probe — `jit/probe_conditioning.py`

Measures `cfg_signal` (`||pred(y=k) − pred(null)||_RMS`) and `class_signal`
(`||pred(y=a) − pred(y=b)||_RMS`) over a configurable set of classes/pairs at
several t values, on both EMA and live weights. The training loop runs a small
in-loop version of this probe at every checkpoint via
`_measure_conditioning_signal` in `jit/train_gsplat.py`; this is the offline
extended version with a per-class JSON dump.

Per-class JSONs land at `probe/cond_probe_t{NNN}_per_class.json` for each t
value (e.g. `cond_probe_t050_per_class.json` for t=0.5).

Direct invocation:

```bash
python jit/probe_conditioning.py \
    --checkpoint output/<run>/<step>.pt \
    --t_values 0.3 0.5 0.7 \
    --per_class_dump_dir output/diagnostics/<run_id>/probe
```

Interpret:
- `cfg_signal` mean ≪ 0.05 across classes → conditioning has collapsed; CFG at
  sampling time is roughly a no-op.
- `class_signal` ≫ `cfg_signal` → all class embeddings cluster together
  off-axis from null. The model distinguishes classes from each other more than
  it distinguishes any class from "no class".
- Heavy upper tail (max ≫ mean) at t=0.5 → a few classes have learned strong
  conditioning; CFG will help for them. The strongest single-class number is the
  most useful upper bound on what the model can do.

### 2. Weight inspection — `jit/inspect_weights.py`

Loads the EMA (or live) weights and reports:
1. **LabelEmbedder geometry** — per-class L2 norm, pairwise cosine matrix,
   distance to null, PCA scatter. The story is in the cosine matrix: if every
   class has cosine > 0.8 with every other class AND with null, the embedder
   has collapsed and no downstream modulation can save it.
2. **Per-block AdaLN gate magnitudes** — forwards class embeddings through each
   block's `adaLN_modulation`, extracts the gate slices (`gate_msa`,
   `gate_mlp`), and reports mean/std/fraction-near-zero. AdaLN-Zero starts at
   exact zero, so any block with mean |gate| > ~0.01 has learned to use class
   info. Persistently zero blocks → the class signal is being silently dropped
   at that depth.
3. **AdaLN projection Frobenius norms** — same diagnostic as (2) but on the
   weights themselves rather than activations.
4. **Cross-reference** with the probe per-class JSON: weak vs strong class
   buckets, per-class final-block mean |gate|, and the Pearson correlation
   between gate magnitude and cfg_signal. If r ≈ 0 (as it was in the 04-23
   run), the AdaLN side is doing uniform-magnitude work; the variation is
   embedder-side.

Direct invocation:

```bash
python jit/inspect_weights.py \
    --checkpoint output/<run>/<step>.pt \
    --cfg_per_class_t 0.5
```

Default `--out_dir` is `output/diagnostics/<run_id>/weights/`. Default
`--cfg_per_class_json` is the t=0.5 dump in the matching probe directory.

### 3. Diversity — `jit/diagnose_class_diversity.py`

For each class:
1. Sample N=24 members (seed=0 by default).
2. Render each at 4 fixed cameras with gsplat (224×224).
3. Embed renders with DINOv2-S/14, average across views per object.
4. Class diversity = mean pairwise (1 − cosine) over the 24 averaged features.

Then correlate with the probe's per-class cfg_signal at the chosen t value
(default 0.5), partition the weak-20 classes by diversity quartile, and write a
visual sanity grid for 3 weak + 3 strong classes.

The DINOv2 features live in `output/diagnostics/_cache/dinov2_features/`.
Cache filename includes `n_per_class` and `seed` (e.g.
`dinov2_n24_seed0.npz`), so you can change those without silently reusing a
stale cache.

Direct invocation:

```bash
python jit/diagnose_class_diversity.py \
    --checkpoint output/<run>/<step>.pt
```

Defaults derive from `--checkpoint`: `--out_dir`,
`--probe_json` (uses the t=0.5 dump in `probe/`), and `--cache_dir` (shared
under `_cache/`).

Interpret:
- Strong correlation (positive Pearson > 0.3) → diverse clusters drive weak
  cfg_signal. Re-clustering candidates: weak-20 classes in the
  `incoherent_high_div` bucket.
- Null correlation (CIs straddle zero, as in the 04-23 run) → diversity is not
  the bottleneck. Look upstream at the embedder/AdaLN training dynamics. Strong
  classes with above-Q3 diversity are the existence proof: diverse clusters
  *can* be learned when the embedder cooperates.
- Sanity grid PNG lets you eyeball whether the metric matches human-visible
  coherence.

## Filing convention

```
output/diagnostics/
    _cache/
        dinov2_features/
            dinov2_n<N>_seed<S>.npz           # shared across checkpoint reruns
            dinov2_n<N>_seed<S>_keys.json
    <run_id>/
        SUMMARY.md                            # written by orchestrator
        probe/
            cond_probe_t030_per_class.json
            cond_probe_t050_per_class.json
            cond_probe_t070_per_class.json
            probe_log.txt
        weights/
            cosine_similarity_heatmap.png
            distance_to_null_sorted.png
            embedding_pca_2d.png
            gate_magnitude_per_block.png
            adaln_proj_norms.png
            final_block_gate_distribution.png
            final_block_gate_vs_cfg.png
            inspection_bundle.json
            weak_vs_strong_classes.json
            ... raw .npy and .json siblings ...
            inspect_log.txt
        diversity/
            per_class_diversity.json
            correlation_summary.json
            diversity_vs_cfg_scatter.png
            weak_partition.json
            visual_sanity_grid.png
            diagnosis_bundle.json
            diversity_log.txt
```

The `<run_id>` convention is `<ckpt_parent_dir>_<ckpt_stem>`. Computed in
`jit/_diag_paths.py` so all four CLI tools agree.

## Historical artifacts (predate this layout)

Earlier runs landed at one-off date-stamped directories. Kept as-is for
reference; new runs use the layout above.

- `output/2026-04-23_weight_inspection/` — first weight inspection on 0260000.
- `output/2026-04-23_class_diversity/` — first diversity diagnosis on 0260000.
- `runs/2026-04-23_cond_probe_0260000_ema_t030.json` — first per-class probe
  dump, t=0.3.
- `runs/2026-04-23_cond_probe_0260000_ema_t050.json` — same, t=0.5.

Each has its own `SUMMARY.md` recording the conclusions at the time.

## When to run

- After every meaningful checkpoint of a class-conditional run, especially if
  in-loop probe values look anomalous.
- After changing the LabelEmbedder, AdaLN init, dropout schedule, or label set.
- When debugging weak conditional sample quality before reaching for a CFG
  scale knob.

`--skip-diversity` is fine if you only want the probe + weight inspection
(takes ~30 s on an L4). Diversity adds ~3 min on the first run for a new
class_map / sampling seed; subsequent runs reuse the cache.
