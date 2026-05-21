# Atlas Layout & Per-Splat Error — Investigation Findings

**Date:** 2026-05-21
**Model:** JiT-B/8, DC-only 14-channel features (`--sh_degree0_only`)
**Checkpoints:** `output/jit_JiT-B_8_adaln_pooled_20260521_014137/{0020000,0080000}.pt` (EMA weights)
**Probe set:** 64 real objects, batch 8, fixed seed (deterministic subset + fixed noise).

---

## TL;DR

1. Per-splat x₀-prediction MSE is **spatially structured and position-locked** on the atlas — the structure survives averaging over 64 different objects, so it is a property of *plane position*, not of any one object.
2. That structure is strongly predicted by **how locally discontinuous the sphere→plane layout is**: partial correlation `r(error, discontinuity | variance)` reaches **0.78** in the high-noise regime — the regime that holds most of the training mass.
3. The render loss confounds the clean end: it applies only at `t_value ≥ 0.8`, so post-~40k checkpoints have their `t≥0.8` error structure overwritten. Pre-render-loss checkpoints (≤20k) are the clean control.
4. An attribution test (data-only, no model) shows the roughness is **largely a removable projection artifact, not intrinsic geometry**: the current `sphere2plane` layout recovers only **~7%** of the achievable local smoothness and preserves true 3D adjacency at **~chance level (0.1%)**. Ordering by true 3D proximity would be **~3× smoother**.

**Conclusion:** the fixed `sphere2plane.npy` atlas layout is a poor substrate for training. The model spends capacity fighting layout-induced high-frequency that a 3D-aware ordering could mostly eliminate.

---

## Background & conventions

- **Atlas = splats.** The model input is a `[B, C, 128, 128]` grid; the 128×128 plane is a sphere→plane (OT-sorted) arrangement of the 16,384 Gaussians via `sphere2plane.npy`. Each pixel is one Gaussian, so per-spatial-location MSE *is* per-splat MSE.
- **Flow matching (not DDPM).** `x_t = t·x₀ + (1−t)·ε`, with **t=0 → pure noise, t=1 → clean data**. `t_discrete = round(t·(T−1))`, `T=1000`. Timesteps sampled logit-normal with `P_mean=-0.5, P_std=1.5` (median t ≈ σ(−0.5) ≈ **0.38**, i.e. most training samples are moderate-to-high noise).
- **MSE is in normalized training space** — exactly the tensor the dataloader hands the trainer (clip → rank-transform → per-channel standardize), i.e. the real training objective. No rendering involved.
- **Render-loss schedule** (relevant throughout): `enable_render_loss_after: 20000`; `render_weight_schedule: [20000 → 0,0,0] → [40000 → 8,4,4]`; **`render_loss_noise_cutoff: 0.8`** (render loss applies only to samples with `t_value ≥ 0.8`). So the 20k checkpoint is effectively pre-render-loss; 80k is full strength.

---

## Tooling

Three standalone probes, each with a `.sh` launcher that sources `.env` and mirrors `train_gsplat.sh` path resolution (same mean/std + clip-thresholds + rank-transform pairing + sphere2plane + DC-only selection, so the data matches a live training run exactly). All run foreground on a single GPU and are light enough to run alongside training.

| Script | What it measures | Needs model? |
|---|---|---|
| `jit/mse_atlas_probe.py` | Per-splat x₀-pred MSE heatmaps across t | yes (fwd only) |
| `jit/atlas_discontinuity_probe.py` | Does layout discontinuity predict error? (partial corr) | yes (fwd only) |
| `jit/atlas_layout_quality_probe.py` | Is the roughness a projection artifact or intrinsic? | **no** (data only) |

Run examples (checkpoint auto-detects newest under `output/` if `RESUME` unset):

```bash
RESUME=output/jit_JiT-B_8_adaln_pooled_20260521_014137/0080000.pt ./jit/mse_atlas_probe.sh
RESUME=output/jit_JiT-B_8_adaln_pooled_20260521_014137/0020000.pt ./jit/atlas_discontinuity_probe.sh
./jit/atlas_layout_quality_probe.sh            # data-only, no checkpoint
```

Common overrides: `NUM_SAMPLES`, `BATCH_SIZE`, `WEIGHTS={ema,model}`, `T_VALUES=0.75,0.8,0.85,0.9`, `K=8` (3D neighbors).

---

## Finding 1 — Per-splat MSE is spatially structured, and the clean end regresses with render loss

`jit/mse_atlas_probe.py`. Mean per-splat MSE (channel-mean of `(x₀−pred)²`, averaged over 64 objects):

| t (FM) | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| **20k** | 0.721 | 0.568 | 0.305 | 0.107 | **0.022** |
| **80k** | 0.723 | 0.575 | 0.316 | 0.114 | **0.231** |

**Spatial structure** (per-panel-normalized heatmaps):
- High noise (t=0.1–0.5): smooth error concentrated in fixed **horizontal bands** of the atlas, darker center.
- Near clean (t=0.9, 80k): a sharp regular **16×16 grid of 8×8 cells** — the JiT **patch-embed seams**.

**The clean-end regression.** t ≤ 0.7 is essentially identical between checkpoints, but **t=0.9 jumps 0.022 → 0.231 (~10×)**. This is exactly the `render_loss_noise_cutoff=0.8` region: between 20k and 80k the objective there changed from pure x₀-MSE to photometric + LPIPS + alpha. The model is **trading x₀-reconstruction MSE for render quality at the clean end** — intended behavior, now quantified, and it shows up spatially as the patch-seam grid.

Figures: `jit/mse_atlas_probe_out/run_20260521_201845/` (80k), `run_20260521_203202/` (20k) — `mse_atlas_heatmaps.png`, `mse_vs_t.png`, `mse_atlas_per_sample.png`.

> The two-row heatmap figure shows the **same** maps under two color normalizations: top row = **shared scale** (compare magnitude across t), bottom row = **per-panel scale** (reveal spatial structure within each t, even where the absolute MSE is tiny).

---

## Finding 2 — Layout discontinuity predicts error (controlling for variance)

`jit/atlas_discontinuity_probe.py`. Per plane pixel, over the 64 objects, in normalized space:

- **E_t** — error: `mean_n mean_c (x₀ − pred_t)²` (per t).
- **D** — discontinuity: `mean_n mean_{4-nbr} mean_c (x₀ − x₀_nbr)²` (data only, checkpoint-independent).
- **V** — variance: `mean_c Var_n(x₀)` (across-object difficulty; data only).

**Confound handled:** high-variance pixels are both harder (high E) and more discontinuous (high D), so a raw `corr(E,D)` is partly spurious. The headline is the **partial correlation `r(E, D | V)`** — does discontinuity predict error *beyond* what intrinsic per-pixel variance explains. (`r(D,V)=0.567`, identical across checkpoints — confirms D, V are data-only.)

Partial `r(E, D | V)` (Pearson):

| t | **20k (pre-render-loss)** | **80k (render @ t≥0.8)** |
|---|---|---|
| 0.1 | 0.780 | 0.779 |
| 0.3 | 0.636 | 0.645 |
| 0.5 | 0.479 | 0.479 |
| 0.7 | 0.376 | 0.363 |
| 0.9 | **0.292** | **0.035** |

**Reading:**
1. **Discontinuity independently predicts error** — up to **0.78** at high noise. Partialling out V only drops the raw 0.85 → 0.78, so D is not just a variance proxy.
2. **The relationship is strongest at high noise and decays toward clean.** Mechanism: from near-pure noise the model predicts ≈ the smooth conditional mean, so it necessarily misses high-frequency content — and layout discontinuity *is* high-frequency content. Near clean it can copy local structure from its input, so layout smoothness matters less. Since training mass concentrates at moderate-to-high noise (median t≈0.38), **the layout roughness bites hardest in the most-trained regime.**
3. **Render loss confounds the clean end.** The two checkpoints are identical for t ≤ 0.7 and diverge only at t=0.9 — the cutoff. Render loss collapses the clean-end coupling **0.29 → 0.03** by overwriting the geometry-driven error structure with its own patch-seam pattern. *Use ≤20k checkpoints for clean geometry diagnostics.*

Figures: `jit/atlas_discontinuity_out/run_20260521_220135/` (80k), `run_20260521_220400/` (20k) — `corr_vs_t.png`, `scatter_E_vs_D.png`, `maps_D_V.png`.

---

## Finding 3 — The roughness is a projection artifact, not intrinsic (attribution test)

`jit/atlas_layout_quality_probe.py` (data only — no model, no checkpoint). For each Gaussian, local feature roughness under three neighbor graphs, **same channel-mean squared feature distance, only the neighbor set changes**:

- **D_3D** — distance to its `k=4` true 3D-nearest-neighbors (by de-normalized xyz) → best achievable.
- **D_plane** — distance to its 4 plane-grid neighbors (current layout).
- **D_random** — distance to random Gaussians (`= 2·var`) → no layout.

Calibration: `smoothness_recovered = (D_random − D_plane) / (D_random − D_3D)` (1 = layout as smooth as 3D-NN ordering; 0 = no better than random).

**Primary = non-xyz channels** (opacity / color / scale / rotation). Excluding xyz is the honest comparison, since picking 3D neighbors *by* xyz would trivially deflate D_3D on the xyz dims.

| metric (non-xyz) | value |
|---|---|
| D_3D (best achievable) | **0.493** |
| D_plane (current layout) | **1.473** |
| D_random (no layout) | **1.546** |
| **smoothness_recovered** | **6.9%** |
| ratio D_plane / D_3D | **2.99×** |
| plane↔3D adjacency overlap (of 4 plane-nbrs, frac among 3D-4NN) | **~0.1% (≈ chance)** |

All-14ch (incl. xyz) is even starker: D_3D=0.387, D_plane=1.436, D_random=1.642 → ratio **3.71×**, recovered **16.4%**.

**Reading:** `D_plane` (1.47) sits right next to `D_random` (1.55), far from `D_3D` (0.49). **The current sphere→plane layout is almost as rough as a random shuffle** — it recovers only ~7% of achievable local smoothness, and the splats it places adjacent are statistically near-independent of true 3D adjacency (0.1% overlap ≈ the ~0.02%/neighbor expected by chance). A 3D-proximity ordering would make local neighborhoods ~3× smoother. **So the roughness that predicts error (Finding 2) is largely a removable projection artifact.**

Figures: `jit/atlas_layout_quality_out/run_20260521_220904/` — `calibration_scale.png`, `maps_layout_quality.png`.

---

## Why it happens

`sphere2plane` optimizes a **sphere/direction parameterization**, which for these objects does **not** correspond to local feature smoothness. The model's conv patch-embed (8×8) and RoPE both assume **plane locality**, so every scrambled neighbor costs prediction error — concentrated, as Finding 2 shows, exactly where the layout is roughest and exactly in the noise regime the model is trained on most.

---

## Caveats

- **D_3D is a per-object kNN lower bound.** A single fixed permutation must serve all ~250K objects, so it cannot reach 0.49; the realistic floor for a *fixed* layout lies between 0.49 and 1.47. But landing at 1.47 (≈ random) shows the current map captures almost none of the cross-object-consistent 3D structure a fixed layout could exploit.
- **A 2D grid cannot realize a kNN graph exactly**, so a practical 3D-aware layout won't hit D_3D either — but the gap is large enough that even partial recovery should help.
- **Render loss overwrites clean-end (t≥0.8) structure.** Any diagnostic touching the clean end on post-~40k checkpoints sees render-loss structure, not underlying geometry.
- **Per-pixel V is estimated from 64 objects** (noisy but adequate as a control). Increase `--num_samples` to tighten.

---

## Recommended next steps

1. **Prototype a 3D-aware fixed layout** — rebuild the `sphere2plane.npy` analog by OT / space-filling-curve over **3D centroids** (or directly minimize plane-neighbor feature distance) instead of the sphere parameterization.
2. **Validate without retraining:** these probes give a model-free before/after metric.
   - `atlas_layout_quality_probe.py` → expect `smoothness_recovered` ↑, `D_plane/D_3D` ↓, overlap ↑.
   - `atlas_discontinuity_probe.py` on a checkpoint → expect `r(E, D | V)` ↓ (less layout-driven error).
3. **Only then** spend a training run, and compare per-splat MSE maps (`mse_atlas_probe.py`) old vs new layout at matched steps (use ≤20k or disable render loss for a clean comparison).
4. Optional deepening: per-channel-group breakdown (geometry vs color) of where the layout hurts most; finer t-grid around the 0.8 render cutoff to map the transition.

---

## Appendix — exact numbers & locations

**Checkpoint:** `output/jit_JiT-B_8_adaln_pooled_20260521_014137/` (JiT-B/8, EMA, steps 20k/40k/60k/80k @ 2.0 GB each).

**Discontinuity probe, full correlation table (20k):**

| t | r(E,D) | r(E,V) | r(D,V) | partial r(E,D\|V) | Spearman r(E,D) | Spearman partial |
|---|---|---|---|---|---|---|
| 0.1 | 0.849 | 0.757 | 0.567 | 0.780 | 0.827 | 0.742 |
| 0.3 | 0.761 | 0.606 | 0.567 | 0.636 | 0.720 | 0.584 |
| 0.5 | 0.583 | 0.388 | 0.567 | 0.479 | 0.521 | 0.408 |
| 0.7 | 0.448 | 0.262 | 0.567 | 0.376 | 0.380 | 0.302 |
| 0.9 | 0.366 | 0.233 | 0.567 | 0.292 | 0.328 | 0.251 |

**Discontinuity probe, full correlation table (80k):**

| t | r(E,D) | r(E,V) | r(D,V) | partial r(E,D\|V) | Spearman r(E,D) | Spearman partial |
|---|---|---|---|---|---|---|
| 0.1 | 0.848 | 0.763 | 0.567 | 0.779 | 0.826 | 0.742 |
| 0.3 | 0.769 | 0.626 | 0.567 | 0.645 | 0.730 | 0.592 |
| 0.5 | 0.592 | 0.410 | 0.567 | 0.479 | 0.530 | 0.408 |
| 0.7 | 0.444 | 0.277 | 0.567 | 0.363 | 0.381 | 0.293 |
| 0.9 | 0.054 | 0.045 | 0.567 | 0.035 | 0.295 | 0.204 |

**Output directories (raw `.npz` arrays + `metrics.json`/`metadata.json` + figures):**
- MSE: `jit/mse_atlas_probe_out/run_20260521_201845` (80k), `run_20260521_203202` (20k)
- Discontinuity: `jit/atlas_discontinuity_out/run_20260521_220135` (80k), `run_20260521_220400` (20k)
- Layout quality: `jit/atlas_layout_quality_out/run_20260521_220904`
