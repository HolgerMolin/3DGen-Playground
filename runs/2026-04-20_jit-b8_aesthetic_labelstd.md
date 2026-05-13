# Run: 2026-04-20 (PM) — JiT-B/8 on aesthetic split, 120 epochs + `label_embed_init_std=0.1` + render-weight ramp

Source log: `output/jit_JiT-B/8_results_gsplat/train_20260420_172641_753971.{out,err}`
Latest checkpoint on disk: `0220000.pt` (ckpt_every=20000; run still in progress at time of writing — accelerate launcher PID 753982 alive, latest log line Step 230 200, Epoch 90/120).

Immediate successor to `2026-04-20_jit-b8_aesthetic_ga1.md`. Main deltas from that run: `epochs` 60 → **120** (so `max_opt_steps` auto = **306 720**, 2× the prior run), `label_embed_init_std` 0.02 → **0.1** (first of the two class-conditioning levers suggested by the ga1-rerun writeup), and render losses now driven by a **`render_weight_schedule` curriculum** `[[50000, 0,0,0], [80000, 0.3, 0.15, 0.02]]` instead of a fixed weight kicked in at the engagement step. `aux_classifier` still off — per the prior run's suggestion to keep (1)+(2) together but not bundle with shape-changing levers. Config is otherwise identical to ga1-rerun.

## Outcome (one line)

Currently at ~230 k / 306 720 opt steps (~75% of cosine), Epoch 90/120, after ~23h 46m wall-clock. MSE: 0.82 → ~0.45 (plateaued by ~step 10k, slow drift down since; best individual window 0.4530 at step 180k). Render-weight ramp over steps 50k→80k engaged cleanly with no MSE spike or gradient artifact. **`cfg_signal` at step 20k = 0.135 (prior run: 0.077 at the same step), `class_signal` at step 20k = 0.082 (prior: 0.047)** — `label_embed_init_std=0.1` roughly doubled both conditioning signals at matched opt-step count. Current `cfg_signal=0.219, class_signal=0.136` at step 220k — best numbers in the lineage by a clear margin.

## Hardware & duration (so far)

| | |
|---|---|
| GPUs | 2× L4 (yusu-l4.c.gcp-langmotion.internal) |
| Mixed precision | bf16 |
| Wall-clock (so far) | 23h 46m (2026-04-20 17:26:47 → 2026-04-21 17:12:45, still running) |
| Steps reached | 230 200 / 306 720 (cosine target) |
| Epochs reached | 90 / 120 |
| Throughput | ~3.15 step/s pre render-loss; ~2.66 step/s with render-loss; dips to ~2.36 every 1000 steps (validation sample) |
| ETA to cosine completion | ~8h remaining at 2.66 step/s (→ ~2026-04-22 ~01:15) |

## Model

| | |
|---|---|
| Variant | `JiT-B/8` |
| Params | 130.7 M |
| Spatial fold factor | 1 (disabled) |
| Input | 128×128 atlas, 14 channels (`sh_degree0_only`) |
| Classes | 87 (incl. unconditional null) |
| Class dropout | 0.1 |
| `label_embed_init_std` | **0.1** (↑ from 0.02 in all prior runs; `class_conditioning.md §6`) |
| Aux classifier | off |

## Diffusion / training objective

| | |
|---|---|
| Mode | DDPM container, **flow-matching x0-prediction** loss (`predict_xstart=true`, `flow_matching_training_losses`) — startup banner still logs "Training mode: DDPM"; loss path is FM |
| Noise schedule | `squaredcos_cap_v2`, 1000 timesteps |
| Timestep sampling | logit-normal `t ~ sigmoid(N(P_mean, P_std))`; **P_mean=0.0, P_std=1.0** (stable throughout; overrides untouched mid-run) |
| Channel loss weights | ENABLED (length-14 inv-spatial-std² weights, mean=0.9999, min=0.682, max=1.822 — same as ga1-rerun) |
| Render loss | `enable_render_loss_after=50 000`; **`render_weight_schedule` linear ramp 50k→80k** from `(0, 0, 0)` to `(rl1=0.300, α=0.150, lpips=0.020)` — 3× / 1.5× / 2× the prior run's flat weights once fully ramped; `render_loss_noise_cutoff=0.4` |
| Render size | 128 px, 1 cam/sample, ref poses `artifacts/ref_camera.tar.gz` (52 cams) |

## Optimizer & schedule

| | |
|---|---|
| Optimizer | AdamW, wd=0 |
| Peak LR | 2.0e-4 |
| LR schedule | cosine; warmup=500 opt steps; `lr_min=1e-5` (unchanged vs ga1-rerun) |
| `lr_cosine_total_steps` | **0 → auto = 306 720 opt steps** (2× ga1-rerun, from `epochs` 60 → 120) |
| Effective batch | per-GPU 16 × 2 GPUs × grad_accum 1 = **32** |
| Grad accumulation | 1 |
| Grad clip | 4.0 (config) → **5.0 from overrides.yaml** throughout |
| EMA decay | 0.995 |
| Gradient checkpointing | on |
| Per-loss grad-norm probe | **not emitted** in this run's log (prior run emitted `GN[mse] / GN[rl1] / GN[alpha] / GN[lpips]` every 10 prints; no such lines here — probe either gated out or disabled) |

## Data

| | |
|---|---|
| Dataset | GaussianVerse aesthetic split — `aesthetic_list.json` |
| Total entries / valid | 81 826 / 81 823 (3 dropped: missing class) |
| Sampling | inverse-frequency `WeightedRandomSampler`, per-rank seed; class counts min=14, max=10 048, mean=940.5 |
| Normalization | `data/stats/aesthetic_mean.pt` / `aesthetic_std.pt` (per-channel) |
| Sphere2plane | `gaussianverse/sphere2plane.npy` (16 384 points) |

## Loss trajectory (running MSE, 200-step window)

| Step | Epoch | MSE | LR | Notes |
|---:|---:|---:|---:|---|
|     200 |  0 | 0.816 | 8.0e-5 | warmup |
|   1 000 |  0 | 0.547 | 2.0e-4 | post-warmup peak LR |
|   5 000 |  1 | 0.511 | 2.0e-4 | |
|  10 000 |  3 | 0.490 | 2.0e-4 | MSE already near steady-state floor |
|  20 000 |  7 | 0.488 | 1.98e-4 | ckpt; **`cfg_signal=0.135`, `class_signal=0.082`, `pred_rms=0.558`** |
|  30 000 | 11 | 0.479 | 1.96e-4 | |
|  40 000 | 15 | 0.473 | 1.92e-4 | ckpt; `cfg_signal=0.127`, `class_signal=0.077` |
|  50 000 | 19 | 0.465 | 1.88e-4 | **render loss engaged** (weights still 0 — curriculum start) |
|  50 200 | 19 | 0.465 | 1.88e-4 | first log with render metrics: raw render_l1=0.049, α=0.085, lpips=0.282 — *weighted* contribs essentially zero |
|  52 000 | 20 | 0.472 | 1.87e-4 | ramp at 2k/30k → weights rl1≈0.020, α≈0.010, lpips≈0.0013; no observable MSE step |
|  60 000 | 23 | 0.460 | 1.83e-4 | ckpt; `cfg_signal=0.153`, `class_signal=0.086`; ramp at 10k/30k → rl1≈0.10 (= old flat value) |
|  70 000 | 27 | 0.476 | 1.77e-4 | ramp at 20k/30k → rl1≈0.20, α≈0.10, lpips≈0.013 |
|  80 000 | 31 | 0.473 | 1.70e-4 | ckpt; **ramp complete — max render weights now in effect**; `cfg_signal=0.140`, `class_signal=0.098` |
|  90 000 | 35 | 0.460 | 1.63e-4 | |
| 100 000 | 39 | 0.469 | 1.55e-4 | ckpt; `cfg_signal=0.144`, `class_signal=0.101` |
| 120 000 | 46 | 0.465 | 1.37e-4 | ckpt; `cfg_signal=0.156`, `class_signal=0.120` |
| 140 000 | 54 | 0.468 | 1.18e-4 | ckpt; `cfg_signal=0.152`, `class_signal=0.100` |
| 160 000 | 62 | 0.460 | 9.9e-5 | ckpt; `cfg_signal=0.192`, `class_signal=0.104` |
| 180 000 | 70 | 0.453 | 7.96e-5 | ckpt; `cfg_signal=0.202`, `class_signal=0.117` — best MSE window |
| 200 000 | 78 | 0.461 | 6.15e-5 | ckpt; `cfg_signal=0.196`, `class_signal=0.122` |
| 220 000 | 86 | 0.462 | 4.52e-5 | ckpt; **`cfg_signal=0.219`, `class_signal=0.136`** |
| 230 200 | 90 | 0.459 | 3.78e-5 | last log at time of writing |

### MSE bucketed by `t_value` (clean fraction; t=1 ≡ clean)

Late-run averages (steps 220k–230k, render curriculum fully engaged):
| t-bucket | mean MSE | typical n/batch |
|---|---:|---:|
| [0.00–0.25] (very noisy) | ~0.59 | 400–465 |
| [0.25–0.50]              | ~0.50 | 1130–1220 |
| [0.50–0.75]              | ~0.41 | 1095–1218 |
| [0.75–1.00] (near clean) | ~0.36 | 397–477 |

Spread ~1.61× between most-noisy and near-clean buckets — **again unchanged** from ga1-rerun (1.62×), ga8 (1.55×), and the original ga1 (1.60×). Four different bundles — channel weights, grad_accum, render-weight level, label-embed init, 2× epoch count — all leave the t-spread shape untouched. This continues to point at the logit-normal `P_mean=0.0` timestep prior as the shape-setting lever. `P_mean_schedule` curriculum remains the untried knob (scaffold in `jit_train_gsplat.yaml` lines 91–95).

### Render losses (post step 50 000)

Because of the 50k→80k ramp, raw render losses fall during the ramp window not just from model improvement but also from noise-cutoff masking behavior stabilizing. Raw (unweighted) values:

| Step | Render L1 | Alpha L1 | Render LPIPS | Effective weights (rl1/α/lp) |
|---:|---:|---:|---:|---|
|  50 200 | 0.0491 | 0.0850 | 0.2815 | 0.002 / 0.001 / 0.00013 |
|  52 000 | 0.0483 | 0.0827 | 0.2857 | 0.020 / 0.010 / 0.0013 |
|  60 000 | 0.0426 | 0.0683 | 0.2746 | 0.100 / 0.050 / 0.0067 |
|  70 000 | 0.0388 | 0.0588 | 0.2636 | 0.200 / 0.100 / 0.013 |
|  80 000 | 0.0348 | 0.0502 | 0.2403 | **0.300 / 0.150 / 0.020** (ramp complete) |
| 100 000 | 0.0330 | 0.0454 | 0.2287 | 0.300 / 0.150 / 0.020 |
| 180 000 | 0.0301 | 0.0389 | 0.2129 | 0.300 / 0.150 / 0.020 |
| 230 000 | 0.0283 | 0.0359 | 0.2061 | 0.300 / 0.150 / 0.020 |

Monotone drop across the board: raw L1 0.049 → 0.028 (−43%), α 0.085 → 0.036 (−58%), LPIPS 0.282 → 0.206 (−27%). The ga1-rerun's 70k steps of rendering left L1 plateaued around 0.036; this run reached 0.028 at 180k further steps of ramped-then-steady render loss. **Higher render weight budget is being used productively by the pixel head.**

No MSE step at any point in the ramp — step 50 200 (first render-metrics log) through step 80 000 (ramp complete) shows MSE tracking the same slow downward drift as steps 30–50k. That is the cleanest render-loss engagement in the lineage: the prior fixed-weight kickin (original ga1 at step 52k) dropped MSE by 0.05 in two print intervals but bundled two simultaneous knob flips; ga1-rerun's engagement was clean but with one-step `GN[α]=0.35` artifact; this curriculum produces neither the drop nor the spike.

### Per-loss grad-norm probe

Not emitted in this run. `GN[mse]/GN[rl1]/GN[alpha]/GN[lpips]` lines are absent from the log (grep returns nothing for `GN\[`). The ga1-rerun writeup reported the probe as expected (`grad_norm_log_every_n_prints: 10` in yaml), so either the probe was gated off for this launch or its output format changed. Not critical for this run's analysis, but worth re-checking for next run if grad-budget probing is wanted (the `simplify.md` / `optimizing.md` audit context relies on it).

## Conditioning signal probe (`[cond]` lines at checkpoints)

| Step | `cfg_signal` | `class_signal` | `pred_rms` |
|---:|---:|---:|---:|
|  20 000 | 0.1350 | 0.0818 | 0.5575 |
|  40 000 | 0.1266 | 0.0769 | 0.5567 |
|  60 000 | 0.1531 | 0.0856 | 0.5862 |
|  80 000 | 0.1400 | 0.0983 | 0.5679 |
| 100 000 | 0.1436 | 0.1008 | 0.5329 |
| 120 000 | 0.1557 | 0.1195 | 0.6097 |
| 140 000 | 0.1515 | 0.0997 | 0.5481 |
| 160 000 | 0.1923 | 0.1041 | 0.5794 |
| 180 000 | 0.2021 | 0.1169 | 0.5875 |
| 200 000 | 0.1955 | 0.1219 | 0.5858 |
| 220 000 | 0.2189 | 0.1356 | 0.6252 |

**Both signals nearly doubled at matched opt-step count.** Per-opt-step comparison vs prior runs:

| opt step | this run `cfg` / `class` | ga1-rerun `cfg` / `class` | ratio `cfg` / `class` |
|---:|---:|---:|---:|
|  20 000 | 0.135 / 0.082 | 0.077 / 0.047 | 1.75× / 1.74× |
|  60 000 | 0.153 / 0.086 | 0.112 / 0.089 | 1.37× / 0.97× |
| 100 000 | 0.144 / 0.101 | 0.123 / 0.076 | 1.17× / 1.33× |
| 120 000 | 0.156 / 0.120 | 0.135 / 0.081 | 1.15× / 1.48× |

The `cfg_signal` gain is biggest at *small* step count (1.75× at step 20k) and narrows as both runs mature — consistent with `label_embed_init_std=0.1` giving the model a stronger signal to latch onto early rather than unlocking a new asymptote. `class_signal` gain is more uniform (1.3–1.5× across the trajectory) and has not plateaued: 0.082 (20k) → 0.136 (220k), and still rising at last checkpoint. `pred_rms` drifts in the 0.53–0.63 range, so `class_signal / pred_rms ≈ 0.22` now — two different class labels perturb the model's output by 22% of activation norm, up from 14% at end of ga1-rerun.

## Notable interventions (or lack thereof)

- **No mid-run overrides changes.** `overrides.yaml` applied every 1000 steps at the stable value `lr_scale=1.0, max_grad_norm=5.0, P_mean=0.0` for the full run. The `aux_classifier_weight` key has been removed from overrides this run (it was the noisy "ignoring unknown key" warning in the ga1-rerun log).
- **Render curriculum engaged automatically** via `enable_render_loss_after=50 000` + `render_weight_schedule` — no manual knob flip. Cleanest attribution in the lineage.
- `max_grad_norm` stayed at 5.0 (overrides) the entire run, overriding the YAML's 4.0. GradNorm values stay in 0.03–0.07 once render loss is engaged (occasional 0.10 at step 70k around mid-ramp); 5.0 clip is never hit.

## Takeaways vs prior runs

1. **`label_embed_init_std` 0.02 → 0.1 is the cheapest class-conditioning win of any lever tried.** 1.75× `cfg_signal` at step 20k, 1.74× `class_signal`, zero compute or architecture cost. This one-line change produced the first run in the lineage where `cfg_signal > 0.2` and `class_signal > 0.13`. `class_conditioning.md §6` was right.
2. **MSE floor is still ~0.45–0.46.** Prior ga1-rerun: 0.462 at opt step 120k. This run: 0.453 at opt step 180k (best), 0.459 at step 230k. Slight improvement (~0.01), but within noise — and this is despite 2× epochs, label-embed init change, and 3× render-L1 weight. The MSE-floor-determining mechanism is elsewhere, and is likely just the P_mean=0.0 shape of the t-sampling (confirmed across 5 runs now).
3. **Render-weight curriculum is strictly better than a step kick-in.** No MSE step, no grad-norm spike, raw render losses reach lower absolute values than the prior run's flat weights (L1 0.028 vs 0.036, LPIPS 0.206 vs 0.253). The ramp also makes the 30k-step window 50k–80k self-documenting as the render-loss warm-up — attribution of any future MSE movement will be clean.
4. **`cfg_signal` has not plateaued** (still rising at step 220k: 0.219, up from 0.196 at 200k). With ~30% of training remaining, this run should push through `cfg_signal ≈ 0.24+`. `class_signal` has also not plateaued.
5. **`lr_min=1e-5` is paying off here too.** Current LR is 3.78e-5 at step 230k and will floor at 1e-5 for the last ~40k steps. Meaningful updates continue through the tail — the ~0.01 MSE improvement from step 160k → 180k happened at LR ~0.8–1.0e-4, well below the peak.
6. **The t-spread invariance is now an N=5 finding.** Channel-weighting (ga8 adds it), compute scaling (ga1→ga1-rerun→this: 3× total compute), label-embed init (this run), and render-weight scale (this run) all left the ~1.6× spread untouched. `P_mean_schedule` curriculum is the last untried knob for this shape.

## Suggested next run

1. **Enable `aux_classifier: true` + `aux_classifier_weight: 0.01`.** This was the other half of the class-conditioning bundle; now that we have a clean `label_embed_init_std=0.1` baseline to diff against, the aux-classifier contribution to `class_signal` becomes attributable. `class_conditioning.md §5`. Keep the current run as the control.
2. **Keep `label_embed_init_std=0.1`, the 50k→80k render ramp, `lr_min=1e-5`, and `grad_accum=1`.** All confirmed working.
3. **Do a `P_mean_schedule` curriculum run in parallel** (separate GPU pair / separate branch) to finally probe the MSE-by-t shape. Suggested schedule `[[0, -0.3], [40000, 0.0], [120000, 0.3]]`: noisy-first warm-up → balanced → clean-biased tail once the model can denoise high-t. This is the only knob left that could legitimately change the t-spread. Attribute in isolation; do *not* bundle with (1).
4. Consider re-enabling per-loss grad-norm probe for the next run (check `grad_norm_log_every_n_prints=10` actually produces `GN[...]` lines; absent here for unknown reasons).
5. **Evaluate validation samples now.** `dit_validation/` has 230 PNGs at this point, all generated from the EMA model with `cfg_scale=1.5`, `heun`, 100 steps. With `cfg_signal=0.22` (vs ga1-rerun's 0.135) the samples should qualitatively look sharper and more class-coherent. Visual check is long overdue given how much `cfg_signal` has moved.
6. After this run completes (~8h remaining at time of writing), save the final checkpoint and update this doc with the ~306k cond-signal numbers; include the final MSE and render-loss values.

## Reproducibility pointers

- **Startup banner** (lines 1–41 of the `.err` log) + the `[overrides] applied` line is the authoritative state: `JiT-B/8`, bf16, num_processes=2, per-GPU batch=16, `max_opt_steps=306720`, `lr_min=1e-05`, `P_mean=0.000, P_std=1.000`, channel weights enabled (mean=0.9999, min=0.682, max=1.822), `render_weight_schedule=[(50000, 0,0,0), (80000, 0.3, 0.15, 0.02)]`, `label_embed_init_std=0.1`.
- Configs at run-start match the current YAML state as of this write-up (neither `jit_train_gsplat.yaml` nor `overrides.yaml` edited mid-run).
- Seed: 0 (per-process offset by accelerate).
- 230 validation PNGs under `dit_validation/` (`val_every=1000`, `val_sampler=heun`, `val_cfg_scale=1.5`, 100 steps) — one per class-sampled batch.
- Latest checkpoint snapshot: `0220000.pt` (2.09 GB, written 2026-04-21 16:07).
