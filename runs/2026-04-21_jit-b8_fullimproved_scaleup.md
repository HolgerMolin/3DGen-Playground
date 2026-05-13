# Run: 2026-04-21 — JiT-B/8 scale-up to full corpus (254 k) + 120 improved-label classes + render_size 256

Source logs (in order; same `RESULTS_DIR=output/jit_JiT-B_8_full_improved/`):
1. `train_20260421_173602_1105566.err` — first attempt, obj list = `all_obj_list.json` (unfiltered), SIGTERM at step ~12 200 after `FileNotFoundError` in the loader (→ motivated the neighbor-index retry patch in `dataloaders/class_3dgen_loader.py`).
2. `train_20260421_190327_1112572.err` — restart from step 0 on `all_obj_list_filtered.json`, reached step ~44 800 before SIGTERM at 23:05.
3. `train_20260421_230912_1128704.err` — abortive relaunch (step 200 then killed).
4. `train_20260421_231132_1129228.err` — **current run, resumed from `0040000.pt`**; still in progress at time of writing — accelerate PID alive, latest log line Step 151 000 Epoch 18/70.

Latest checkpoint on disk: `0140000.pt` (ckpt_every=20 000; next write due at step 160 000).

Immediate successor to `2026-04-20_jit-b8_aesthetic_labelstd.md`. This is the run where the previously-successful recipe (label_embed_init_std=0.1, render-weight ramp, lr_min=1e-5, grad_accum=1) was held constant and the **data + conditioning axes were scaled**:

| axis | prior (labelstd run) | this run |
|---|---|---|
| Obj list | `aesthetic_list.json` (81 826 entries) | `all_obj_list_filtered.json` (254 338 entries, 3.1×) |
| Normalization | `data/stats/aesthetic_{mean,std}.pt` | `data/stats/all_{mean,std}.pt` (recomputed on full corpus) |
| Class map | `object_labels/object_to_class.json` (87 classes) | `object_labels/improved_labels/object_to_class.json` (**120 classes**, appearance-focused per commit 8e1a4f8) |
| Class imbalance | min=14, max=10 048 (~700:1) | min=682, max=4 315 (~6:1) — much flatter tail |
| Channel weights | mean 1.00, min 0.682, max 1.822 | mean 1.00, min 0.644, max 1.735 (recomputed; same shape) |
| `train_render_size` | 128 px | **256 px** (4× pixel count) |
| `epochs` | 120 | 70 (auto `max_opt_steps` = **556 360**, 1.81× prior) |

Four scale axes moved in lock-step. The failure mode below is therefore not uniquely attributable, but the *direction* is clear.

## Outcome (one line)

At ~151 k / 556 k opt steps (~27 % of cosine), Epoch 18/70, after ~22.5 h wall-clock across the restart lineage. MSE plateaued at ~0.45–0.46 — essentially identical to the 5-run lineage floor, so no regression from the scale-up. **But `cfg_signal` has collapsed to ~0.05 (prior run at step 40 k: 0.127; prior run at step 140 k: 0.152) and is not growing** — 3–4× worse class discrimination despite retaining `label_embed_init_std=0.1`. First run in the lineage where CFG is mathematically trivial again. This dominates all other findings.

## Hardware & duration (so far)

| | |
|---|---|
| GPUs | 2× L4 (yusu-l4.c.gcp-langmotion.internal) |
| Mixed precision | bf16 |
| Wall-clock (so far) | 22 h 31 m (lineage span 2026-04-21 17:36 → 2026-04-22 16:07; ~17 h in current PID) |
| Steps reached | 151 000 / 556 360 (cosine target) |
| Epochs reached | 18 / 70 |
| Throughput | ~3.2 step/s pre render-loss; **~1.65 step/s with render-loss** (prior run: 2.66 step/s) — render_size 128→256 costs ~38 % throughput |
| ETA to cosine completion | ~68 h remaining at 1.65 step/s (→ ~2026-04-25 ~12:00) |

## Model

| | |
|---|---|
| Variant | `JiT-B/8` (unchanged) |
| Params | 130.7 M |
| Spatial fold factor | 1 (disabled) |
| Input | 128×128 atlas, 14 channels (`sh_degree0_only`) |
| Classes | **120** (+1 null if `class_dropout_prob > 0`; `improved_labels`) |
| Class dropout | 0.1 |
| `label_embed_init_std` | 0.1 (retained from labelstd run) |
| Aux classifier | off |

## Diffusion / training objective

| | |
|---|---|
| Mode | DDPM container, flow-matching x0-prediction loss (predict_xstart=true) — same as labelstd run |
| Noise schedule | `squaredcos_cap_v2`, 1000 timesteps |
| Timestep sampling | logit-normal `t ~ sigmoid(N(0.0, 1.0))` |
| Channel loss weights | ENABLED (length-14, mean=1.0000, **min=0.644, max=1.735** — recomputed on full corpus, very similar shape to aesthetic weights) |
| Render loss | `enable_render_loss_after=50 000`; `render_weight_schedule` linear 50k→80k from `(0,0,0)` to `(rl1=0.300, α=0.150, lpips=0.020)`; `render_loss_noise_cutoff=0.4`; **`train_render_size=256` (↑ from 128)** |
| Render size | 256 px, 1 cam/sample, ref poses `artifacts/ref_camera.tar.gz` (52 cams) — masking observed: typical 3–8 of 16 samples masked per step by `t < 0.4` cutoff (no change from prior) |

## Optimizer & schedule

| | |
|---|---|
| Optimizer | AdamW, wd=0 |
| Peak LR | 2.0e-4 |
| LR schedule | cosine; warmup=500 opt steps; `lr_min=1e-5` |
| `lr_cosine_total_steps` | **0 → auto = 556 360** (1.81× labelstd run — this is the scale-up in training budget) |
| Effective batch | per-GPU 16 × 2 GPUs × grad_accum 1 = 32 |
| Grad accumulation | 1 |
| Grad clip | 4.0 (config) → **5.0 from overrides.yaml** |
| EMA decay | 0.995 |
| Gradient checkpointing | on |
| Per-loss grad-norm probe | **still not emitted** (no `GN[...]` lines in log, same gap as labelstd run). `grad_norm_log_every_n_prints: 10` present in yaml — probe plumbing looks broken, not gated |

## Data

| | |
|---|---|
| Dataset | GaussianVerse **full filtered** split — `all_obj_list_filtered.json` |
| Total entries / valid | 254 338 / 254 338 (filtered list has 0 skipped; the first attempt on unfiltered `all_obj_list.json` died with `FileNotFoundError`, now handled by 16-retry fallback in `class_3dgen_loader.py:__getitem__`) |
| Sampling | inverse-frequency `WeightedRandomSampler`, per-rank seed; class counts min=682, max=4 315, mean=2 119.5 — **much flatter tail than the aesthetic split** (was 14:10 048, ~700:1). Per-class sample count more than doubled (~2 120 vs ~940), so class starvation cannot explain the cond-signal drop. |
| Normalization | `data/stats/all_{mean,std}.pt` (recomputed on full corpus) |
| Sphere2plane | `gaussianverse/sphere2plane.npy` (16 384 points) |

## Loss trajectory (running MSE, 200-step window)

Steps 0–44 800 come from the fresh launch (1112572.err); steps 40 000+ come from the resume-from-0040000 (1129228.err), so there is a brief re-entry at step 40k (re-resumed the same checkpoint; no meaningful discontinuity).

| Step | Epoch | MSE | LR | Notes |
|---:|---:|---:|---:|---|
|     200 |  0 | 0.792 | 8.0e-5 | warmup (1112572) |
|   1 000 |  0 | 0.525 | 2.0e-4 | post-warmup peak LR |
|   5 000 |  0 | 0.471 | 2.0e-4 | |
|  10 000 |  1 | 0.463 | 2.0e-4 | MSE at labelstd-run's step-10k floor |
|  20 000 |  2 | 0.457 | 2.0e-4 | `cfg_signal=0.047`, `class_signal=0.049` (contrast labelstd@20k: 0.135 / 0.082) |
|  40 000 |  5 | 0.454 | 1.98e-4 | `cfg_signal=0.055`, `class_signal=0.056`; resume boundary |
|  50 200 |  6 | 0.457 | 1.96e-4 | **render loss engaged** (ramp start); raw render_l1=0.038, α=0.069, lpips=0.196 |
|  60 000 |  7 | 0.454 | 1.95e-4 | `cfg_signal=0.051`, `class_signal=0.056` |
|  70 000 |  8 | 0.471 | 1.93e-4 | ramp mid (rl1≈0.20); `cfg_signal=0.046`, `class_signal=0.047` |
|  80 000 | 10 | 0.468 | 1.91e-4 | ramp complete (rl1=0.300, α=0.150, lpips=0.020) |
| 100 000 | 12 | 0.454 | 1.85e-4 | `cfg_signal=0.059`, `class_signal=0.043` |
| 120 000 | 15 | 0.469 | 1.79e-4 | `cfg_signal=0.041`, `class_signal=0.050` |
| 140 000 | 17 | 0.458 | 1.72e-4 | last `[cond]` print |
| 150 200 | 18 | 0.452 | 1.68e-4 | latest — still drifting, LR only 16 % decayed |

MSE floor ≈ 0.455 — indistinguishable from the 5-run lineage's ~0.45–0.47 band. **Scaling the dataset by 3× did not move the floor**, consistent with the lineage finding that the floor is set by the `P_mean=0.0` logit-normal prior shape, not by data.

### MSE bucketed by `t_value` (clean fraction; t=1 ≡ clean)

Late-window averages (steps 140k–151k):
| t-bucket | mean MSE | typical n/batch |
|---|---:|---:|
| [0.00–0.25] (very noisy) | ~0.58 | 420–470 |
| [0.25–0.50]              | ~0.50 | 1 100–1 200 |
| [0.50–0.75]              | ~0.40 | 1 120–1 220 |
| [0.75–1.00] (near clean) | ~0.36 | 400–470 |

Spread ~1.61×. **N=6 now — same ~1.6× spread** across {aesthetic vs full, 87 vs 120 classes, ga8 vs ga1, 128 vs 256 render, labelstd 0.02 vs 0.1, epochs 60 vs 120 vs 70}. The t-spread is robustly a property of `P_mean=0.0`. `P_mean_schedule` is still the one untried lever that could move it.

### Render losses (post step 50 000)

Raw (unweighted), comparing to labelstd@comparable opt-steps-since-engagement:

| Step | Render L1 | Alpha L1 | Render LPIPS | Δ steps since engage | labelstd equiv. (L1 / α / LPIPS) |
|---:|---:|---:|---:|---:|---|
|  50 200 | 0.0382 | 0.0690 | 0.1960 | +200 | 0.049 / 0.085 / 0.282 |
|  60 000 | 0.0347 | 0.0565 | 0.1942 | +10 000 | 0.043 / 0.068 / 0.275 |
|  80 000 | 0.0312 | 0.0467 | 0.1896 | +30 000 | 0.035 / 0.050 / 0.240 |
| 100 000 | 0.0272 | 0.0399 | 0.1754 | +50 000 | 0.033 / 0.045 / 0.229 |
| 140 000 | 0.0273 | 0.0381 | 0.1736 | +90 000 | ≈0.030 / 0.040 / 0.218 (interp) |
| 150 200 | 0.0258 | 0.0366 | 0.1666 | +100 200 | — |

All three metrics are **lower at matched engagement-step** than the labelstd run: L1 is 0.026 vs labelstd's 0.028 at step 180k+ of rendering. This is surprising because:
- The render target is 256 px (2× linear resolution, 4× pixel count) — L1/LPIPS on finer targets is usually numerically larger, not smaller.
- The scene content is broader (full corpus) and should be objectively harder.
- `render_loss_num_cam=1` unchanged.

The most likely explanation is the recomputed normalization stats (`all_*.pt`): per-channel stds are slightly different from `aesthetic_*.pt`, which changes what the pixel head is reconstructing into. Net effect: the 256-px pixel head is clearly getting usable gradient — **render loss is the one place this run is strictly better** than labelstd at matched progress.

### Per-loss grad-norm probe

`grep 'GN\['` returns 0 hits in the current log, same gap as labelstd. `grad_norm_log_every_n_prints: 10` is in the YAML. Probe appears to be plumbing-broken, not disabled — worth a one-line debug pass before the next run.

## Conditioning signal probe (`[cond]` lines at checkpoints)

| Step | `cfg_signal` | `class_signal` | `pred_rms` | labelstd@same step | Δ cfg |
|---:|---:|---:|---:|---:|---:|
|  20 000 | 0.0474 | 0.0492 | 0.586 | 0.135 / 0.082 | **−65 %** |
|  40 000 | 0.0553 | 0.0564 | 0.627 | 0.127 / 0.077 | **−57 %** |
|  60 000 | 0.0514 | 0.0558 | 0.615 | 0.153 / 0.086 | **−66 %** |
|  80 000 | 0.0458 | 0.0467 | 0.650 | 0.140 / 0.098 | **−67 %** |
| 100 000 | 0.0592 | 0.0429 | 0.622 | 0.144 / 0.101 | **−59 %** |
| 140 000 | 0.0410 | 0.0500 | 0.626 | 0.152 / 0.100 | **−73 %** |

Flat at 0.04–0.06 across 120k steps with **no upward trend**, whereas the labelstd run rose monotonically from 0.135 (20k) to 0.219 (220k). `class_signal / pred_rms ≈ 0.08` here vs 0.22 at end of labelstd — perturbing the class label moves the model output by 8 % of activation norm, barely above noise. This is the first `cfg_signal < 0.1` reading in the lineage since ga8 (`label_embed_init_std=0.02` in the small-data regime).

Candidate mechanisms for the collapse (can't be separated without an ablation):
1. **Label semantics changed** — `improved_labels` (commit 8e1a4f8) optimizes for *appearance similarity* ("A small, spherical brownish rock with a rough and uneven surface…"). If two clusters now differ only in color/texture but produce overlapping 3DGS atlases, the trunk has no signal to discriminate them — class conditioning degrades to near-null. Cluster descriptions in `object_labels/improved_labels/cluster_summary.txt` support this: cluster 0 is "brownish rocks", others likely similarly texture-driven.
2. **More classes, more ambiguity** — 120 vs 87, with far flatter counts (6:1 vs 700:1). A WeightedRandomSampler over flat counts removes the huge common-class-dominance signal that was helping the aesthetic runs.
3. **More diverse data** — 3× objects, including non-aesthetic geometry. Same 130 M params now modeling a broader manifold; class-conditional contribution diluted.
4. **label_embed_init_std=0.1 may interact badly with 120 labels** — 40 % more embedding norm to absorb early gradient; not obviously bad, but unverified.

The labelstd run had `cfg_signal=0.135` at step 20 000 on the aesthetic split. This run, with the only *architectural* lever (init_std=0.1) preserved, has `cfg_signal=0.047` at the same step — a 2.87× regression. The regression is located in the data/label axes, not the model.

## Notable interventions (or lack thereof)

- **No mid-run overrides changes.** `overrides.yaml` applied every 1000 steps at stable `lr_scale=1.0, max_grad_norm=5.0, P_mean=0.0`.
- **Dataloader hardening landed mid-lineage**: `class_3dgen_loader.py:__getitem__` now wraps `_getitem_raw` with a 16-attempt neighbor-fallback on `FileNotFoundError`. This was driven by the first launch failing on stale entries in `all_obj_list.json`; the current run uses the already-filtered list so the retry never fires (0 warnings in log), but is correct insurance.
- **`train_gsplat.sh` now honors `RESULTS_DIR` env** (line 87), so this run's `RESULTS_DIR=output/jit_JiT-B_8_full_improved` cleanly separates from `output/jit_JiT-B/8_results_gsplat`. Good practice for scale-up runs; keep.
- `train_render_size=256` is a bundled change with the data scale-up. Should be isolated in a future ablation.

## Takeaways vs prior runs

1. **Scaling data + labels caused a cond-signal regression, not a gain.** The single most informative number in this run is `cfg_signal=0.04–0.06` where labelstd had 0.14–0.22. Everything the lineage has been optimizing toward (useful classifier-free guidance) has been partly undone. The MSE floor is stable, so this isn't capacity — it's *discriminability* of the labels.
2. **MSE floor is dataset-size-invariant to the third decimal.** 82 k aesthetic → 254 k full, same ~0.455 floor at comparable step. The lineage's t-spread-invariance finding now extends to data scale. `P_mean=0.0` remains the shape-setter.
3. **Render loss scales cleanly to 256 px** and gives lower absolute L1/α/LPIPS than the 128-px labelstd run at matched engagement-step — the one unqualified win. Throughput cost is 38 % (3.2 → 1.65 step/s with render on), which with the rest of the run not making progress is a net negative for *this* run. In isolation, the pixel-head improvement looks real.
4. **`improved_labels` may be net-negative for diffusion conditioning** even if it looks better as a clustering. Appearance-based clusters ("brownish rocks", "grey rocks", "white rocks") may not partition the 3DGS manifold cleanly. A label that the model cannot discriminate from its input distribution is a label it will ignore.
5. **Per-loss grad-norm probe is still not emitting.** Two runs in a row without it. This is the second time it has bitten an analysis — fix before the next launch.
6. **Training-budget scaling (556 k steps, 4+ days wall-clock) is already a large commitment** and has delivered zero measurable gain on the single metric we've been trying to move. Continuing to 556 k will *probably* incrementally improve render loss, *probably* leave MSE flat, and *likely* keep `cfg_signal` below 0.1. That is not a good use of the remaining ~68 h.

## Is scaling the correct next move?

**No — not yet, and not naïvely.** Three arguments against scaling further right now:

1. **MSE floor is not capacity-bound.** Five runs at JiT-B/8 and now a sixth have all plateaued at 0.45–0.47. Moving to JiT-L/8 (~4×) or JiT-XL/8 (~9×) is the canonical next step, and the labelstd writeup's §6 implies it's been on the table. But the signal that says "we need more capacity" — a sharp MSE drop late in training, MSE continuing to fall as LR anneals, per-t buckets closing — is **absent**. The current bottleneck looks like a training-objective / data-labeling shape issue, which a bigger model will reproduce at greater cost.

2. **Conditioning just regressed.** The one metric the lineage has been chasing (`cfg_signal > 0.2` to enable meaningful CFG) has gone backwards on the scale-up. A larger model on the same data + labels will, at best, reach the same ceiling; at worst, it will dilute the class signal further by spending capacity on the broader manifold. This is the wrong moment to commit 4–9× more FLOPs.

3. **The scale-up bundle can't be attributed.** Data size, class count, class semantics, normalization, and render resolution all moved together. We have no idea which one broke cond_signal. Adding *model scale* as a 6th simultaneous change guarantees we learn nothing about any of them.

**What to do instead — in order:**

1. **Qualitative sample audit now.** 155 validation PNGs exist under `dit_validation/`. Visually inspect a sweep across classes at fixed step (e.g. steps 60 k, 100 k, 140 k): do different class indices yield visibly different objects? With `cfg_signal=0.05` the answer may be "no" — confirming numerically what the probe says. Do this *before* committing to more training.
2. **Isolate label semantics.** Launch a short (~20 k step) sanity run on the **aesthetic split (87 classes) but with `improved_labels`** reduced to that subset, keeping everything else from labelstd. If `cfg_signal` collapses there too, the label *semantics* are the issue. If it recovers, it's the data or class-count axis.
3. **Enable `aux_classifier: true, aux_classifier_weight: 0.01`** (the other half of `class_conditioning.md §5`) on the next real run. This forces the trunk to stay class-discriminative via CE even if the flow-matching MSE is dominated by non-class signal. This is the lever most likely to rescue `cfg_signal` regardless of which of (1)–(4) broke it.
4. **Fix the per-loss GN probe.** Two runs in a row without it. Grep for `GN\[` in `jit/train_gsplat.py`; I suspect the print guard was changed without updating the emission format. Takes 15 min.
5. **Only then consider model scale.** Once `cfg_signal` is back to ≥0.15 on the full-corpus recipe, the JiT-L/8 jump is justifiable — and the MSE floor comparison becomes meaningful evidence for/against capacity being the bottleneck. Doing it before is lighting compute on fire.

Minor: revert `train_render_size` to 128 for the diagnostic runs in (2)–(3) to get throughput back; the 256-px win is real but not worth the 38 % slowdown while we're debugging conditioning.

## Suggested next run

1. **Kill the current run at the next checkpoint (step 160 000).** Projected 68 h remaining will not resolve the cond-signal collapse; the last 12 h of sunk compute has told us what we need to know.
2. **Short diagnostic run A — label ablation**, starting from the labelstd recipe (aesthetic split, 87 classes, render_size 128, lr_min=1e-5, label_embed_init_std=0.1), but swap the class map to the improved-labels equivalent filtered to aesthetic objects. 20 k steps is enough: `cfg_signal` at 20 k in the labelstd run was 0.135, any value below 0.08 is a clear regression signal on the labels.
3. **Diagnostic run B — aux classifier rescue**, from the *current* full-corpus checkpoint (0140000.pt). Turn `aux_classifier: true, aux_classifier_weight: 0.01` on. If this pulls `cfg_signal` up within 10–20 k steps, it's a usable lever for the scale-up regime even without solving (2).
4. Fix the per-loss GN probe before either (2) or (3).
5. Hold model scale, render resolution, training-budget scale constant until (2) tells us whether to revert to `object_labels/object_to_class.json` or stay on `improved_labels`.

## Reproducibility pointers

- **Startup banner** (lines 1–38 of `train_20260421_231132_1129228.err`) is authoritative: JiT-B/8, bf16, num_processes=2, per-GPU batch=16, `max_opt_steps=556 360`, `lr_min=1e-05`, P_mean=0.0, P_std=1.0, channel weights `[0.705, 0.794, 0.845, 1.735, 1.099, 1.053, 1.045, 0.975, 1.048, 0.971, 1.735, 0.672, 0.679, 0.644]`, `render_weight_schedule=[(50 000, 0, 0, 0), (80 000, 0.3, 0.15, 0.02)]`, `label_embed_init_std=0.1`, `train_render_size=256`, `enable_render_loss_after=50 000`, `render_loss_noise_cutoff=0.4`.
- Config as-of write-up: `jit/configs/jit_train_gsplat.yaml` uncommitted changes — `epochs: 70`, `train_render_size: 256`, channel weights updated. `overrides.yaml` unchanged (`lr_scale=1.0, max_grad_norm=5.0, P_mean=0.0`).
- Class map: `object_labels/improved_labels/object_to_class.json` (120 classes, appearance-focused; see `cluster_summary.txt`).
- Dataset: `/home/tiangexiang/gen3d/gaussianverse/all_obj_list_filtered.json` (254 338 entries, filtered for resolvable `.ply` paths).
- Normalization: `data/stats/all_mean.pt`, `data/stats/all_std.pt`.
- Resume chain: Run 1 (1105566) → filter list → Run 2 (1112572) from scratch → step 40 000 ckpt → current Run 4 (1129228) resumed from `0040000.pt`.
- 155 validation PNGs under `output/jit_JiT-B_8_full_improved/dit_validation/` (`val_every=1000`, heun, 100 steps, cfg=1.5) — **priority task: visually inspect these before next launch.**
- Latest checkpoint: `0140000.pt` (2.09 GB, written 2026-04-22 14:22).
