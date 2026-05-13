# Run: 2026-04-19 (PM) — JiT-B/8 on aesthetic split, grad_accum=8

Source log: `output/jit_JiT-B/8_results_gsplat/train_20260419_075229_429924.{out,err}`
Last checkpoint: `0153360.pt` (ckpt_every=20000; training ran to completion at the cosine target).

Immediate successor to `2026-04-19_jit-b8_aesthetic.md`; same date, same model, same data. Main deltas from that run: `gradient_accumulation_steps` 1 → **8**, `lr_cosine_total_steps` hardcoded-150k → **auto (19200)**, `channel_loss_weights` off → **on**, render loss engaged cleanly via `enable_render_loss_after=50000` instead of mid-run overrides flip.

## Outcome (one line)

Trained JiT-B/8 with flow-matching x0-prediction for 59 epochs / ~19.2k opt steps (153,360 micro-steps) over ~13h. Cosine fully completed. MSE: 1.03 → 0.49 (plateaued by ~opt step 5k, flat thereafter). Final MSE **higher** than prior run's 0.469 and `class_signal` lower (0.065 vs 0.074) despite 2× the epoch count — symptom of the 8× optimizer-step reduction.

## Hardware & duration

| | |
|---|---|
| GPUs | 2× L4 (yusu-l4.c.gcp-langmotion.internal) |
| Mixed precision | bf16 |
| Wall-clock | 12h 55m (07:52:29 → 20:48:05) |
| Micro-steps reached | 153 360 |
| Opt steps reached | **~19 170 / 19 200 (cosine target)** — fully completed |
| Epochs reached | 59 / 60 |
| Throughput | ~4.0 step/s pre render-loss; ~2.6 step/s with render-loss |

## Model

| | |
|---|---|
| Variant | `JiT-B/8` |
| Params | 130.7 M |
| Spatial fold factor | 1 (disabled) |
| Input | 128×128 atlas, 14 channels (`sh_degree0_only`) |
| Classes | 87 (incl. unconditional null) |
| Class dropout | 0.1 |

## Diffusion / training objective

| | |
|---|---|
| Mode | DDPM container, **flow-matching x0-prediction** loss (`predict_xstart=true`, `flow_matching_training_losses`) — note: startup log says "Training mode: DDPM" but the loss path is FM |
| Noise schedule | `squaredcos_cap_v2`, 1000 timesteps |
| Timestep sampling | logit-normal `t ~ sigmoid(N(P_mean, P_std))`; **P_mean=0.0 throughout**, P_std=1.0 |
| Channel loss weights | **ENABLED**: length-14 inv-spatial-std² weights, mean=0.9999, min=0.682, max=1.822 (values `[0.766, 0.799, 0.757, 1.822, 1.036, 0.979, 1.006, 0.950, 0.980, 1.023, 1.822, 0.689, 0.687, 0.682]`) |
| Render loss | `enable_render_loss_after=50000` (engaged cleanly at step 50000 with `render_l1=0.1, alpha_l1=0.1, lpips=0.01`); `noise_cutoff=0.4` |
| Render size | 128 px, 1 cam/sample, ref poses `artifacts/ref_camera.tar.gz` (52 cams) |

## Optimizer & schedule

| | |
|---|---|
| Optimizer | AdamW, wd=0 |
| Peak LR | 2.0e-4 |
| LR schedule | cosine; warmup=500 opt steps; `lr_min=0` |
| `lr_cosine_total_steps` | **0 → auto = 19 200 opt steps** (new auto path: `len(loader) × epochs / grad_accum` after `accelerate.prepare`) |
| Effective batch | per-GPU 16 × 2 GPUs × **grad_accum 8 = 256** |
| Grad accumulation | **8** (was 1 in prior run) |
| Grad clip | 4.0 (config) → **5.0 from overrides.yaml** throughout |
| EMA decay | 0.995 |
| Gradient checkpointing | on |

## Data

| | |
|---|---|
| Dataset | GaussianVerse aesthetic split — `aesthetic_list.json` |
| Total entries / valid | 81 826 / 81 823 (3 dropped: missing class) |
| Sampling | inverse-frequency `WeightedRandomSampler`, per-rank seed; class counts min=14, max=10 048, mean=940.5 |
| Normalization | `data/stats/aesthetic_mean.pt` / `aesthetic_std.pt` (per-channel) |
| Sphere2plane | `gaussianverse/sphere2plane.npy` (16 384 points) |

## Loss trajectory (running MSE, 200-step window)

Micro-step ≈ 8 × opt-step for this run (grad_accum=8).

| Micro-step | Opt step | Epoch | MSE | LR | Notes |
|---:|---:|---:|---:|---:|---|
|     200 |     25 |  0 | 1.031 | 1.0e-5 | warmup |
|   1 000 |    125 |  0 | 0.705 | 5.0e-5 | |
|   5 000 |    625 |  1 | 0.550 | 2.0e-4 | post-warmup peak LR |
|  10 000 |  1 250 |  3 | 0.522 | 1.99e-4 | |
|  15 000 |  1 875 |  5 | 0.509 | 1.97e-4 | |
|  20 000 |  2 500 |  7 | 0.517 | 1.94e-4 | ckpt; **cond `cfg_signal=0.015`, `class_signal=0.013`** |
|  30 000 |  3 750 | 11 | 0.508 | 1.85e-4 | |
|  40 000 |  5 000 | 15 | ~0.50 | 1.67e-4 | ckpt; `cfg_signal=0.038`, `class_signal=0.027` |
|  50 000 |  6 250 | 19 | 0.492 | 1.57e-4 | **render loss auto-engaged** (no override flip) |
|  60 000 |  7 500 | 23 | ~0.51 | 1.47e-4 | ckpt; `cfg_signal=0.040`, `class_signal=0.033` |
|  70 000 |  8 750 | 27 | 0.493 | 1.18e-4 | render L1≈0.043, α≈0.067, LPIPS≈0.283 |
|  80 000 | 10 000 | 31 | — | — | ckpt; `cfg_signal=0.064`, `class_signal=0.062` |
|  90 000 | 11 250 | 35 | 0.477 | 7.6e-5 | |
| 100 000 | 12 500 | 39 | 0.485 | 5.7e-5 | ckpt; `cfg_signal=0.065`, `class_signal=0.062` |
| 110 000 | 13 750 | 43 | 0.491 | 3.9e-5 | |
| 120 000 | 15 000 | 46 | 0.484 | 2.4e-5 | ckpt; (cond probe not captured — LR already small) |
| 130 000 | 16 250 | 50 | 0.491 | 1.2e-5 | |
| 140 000 | 17 500 | 54 | 0.486 | 3.9e-6 | ckpt; `cfg_signal=0.073`, `class_signal=0.065` |
| 150 000 | 18 750 | 58 | 0.493 | 2.5e-7 | cosine nearly exhausted |
| 153 360 | 19 170 | 59 | 0.489 | ≈0 | final — `Training complete` |

### MSE bucketed by `t_value` (clean fraction; t=1 ≡ clean)

Late-run averages (steps 140k–150k, render loss on):
| t-bucket | mean MSE | typical n/batch |
|---|---:|---:|
| [0.00–0.25] (very noisy) | ~0.61 | 420–470 |
| [0.25–0.50] | ~0.53 | 1130–1210 |
| [0.50–0.75] | ~0.44 | 1120–1210 |
| [0.75–1.00] (near clean) | ~0.39 | 400–480 |

Spread ~1.55× between most-noisy and near-clean buckets — **essentially identical to prior run's ~1.6×**, despite `channel_loss_weights` being enabled. Either the weights aren't the right lever for t-bucket spread (they rebalance *channels*, not timesteps), or the adjustment was too small to overcome the logit-normal sampling bias.

### Render losses (post step 50 000)

| | typical value |
|---|---:|
| Render L1 | 0.040 |
| Alpha L1 | 0.060 |
| Render LPIPS | 0.271 |

No visible MSE drop at step 50k when render loss engaged (unlike the prior run's ~0.05 drop at its manual flip at step 52k) — MSE had already plateaued at ~0.49 by step ~30k, so there was nothing left for render loss to drag down.

## Conditioning signal probe (`[cond]` lines at checkpoints)

| Micro-step | Opt step | `cfg_signal` | `class_signal` | `pred_rms` |
|---:|---:|---:|---:|---:|
|  20 000 |  2 500 | 0.0149 | 0.0126 | 0.4568 |
|  40 000 |  5 000 | 0.0377 | 0.0266 | 0.5835 |
|  60 000 |  7 500 | 0.0397 | 0.0327 | 0.5244 |
|  80 000 | 10 000 | 0.0642 | 0.0619 | 0.6685 |
| 100 000 | 12 500 | 0.0646 | 0.0624 | 0.6635 |
| 140 000 | 17 500 | 0.0727 | 0.0652 | 0.6461 |

Rising but under-budget. Per-opt-step progress is comparable to prior run (prior reached `cfg_signal=0.067` at opt step 60k; this run reaches 0.073 at opt step 17.5k — i.e. roughly on the same trajectory). The issue is the total opt-step budget: 19k is simply not enough for the conditioning head to separate to a usable `cfg_signal ≥ 0.1`. CFG still mathematically weak; `class_signal` 0.065 is *below* prior's 0.074.

## Notable interventions (or lack thereof)

- **No mid-run overrides changes.** `overrides.yaml` was stable at `render_loss_weight=0.1, alpha_mask_loss_weight=0.1, lpips_loss_weight=0.01, P_mean=0.0, max_grad_norm=5.0, lr_scale=1.0` for the full run (applied every 1000 steps, no value edits).
- Render loss engaged automatically at step 50000 via `enable_render_loss_after`, not via overrides flip. Clean attribution.
- `max_grad_norm` stayed at 5.0 (overrides) the entire run, overriding the YAML's 4.0.

## Takeaways vs prior run

1. **`gradient_accumulation_steps` 1 → 8 is the dominant regression.** Same 60-epoch budget, but 8× fewer optimizer updates (19k vs prior's 68k reached / 150k planned). Conditioning head is update-starved; per-opt-step trajectory is fine.
2. **Final MSE 0.489 > prior 0.469.** Bigger effective batch (256 vs 32) stabilizes gradients but cannot compensate for 8× fewer steps; combined with cosine fully decaying to ~0, the last 30% of training (~opt step 13k+, LR < 3e-5) contributed little.
3. **Render loss engagement was clean this time** — no confound in the loss curve, but also no dramatic drop, because MSE had already plateaued.
4. **Channel loss weighting did not visibly flatten the MSE/t spread** (still ~1.55×). Expected, since it's a channel-wise not timestep-wise reweighting. Need a separate lever (e.g. higher P_mean, or t-bucket reweighting) if flattening the t-curve is the goal.
5. **Cosine auto path (`lr_cosine_total_steps: 0`) worked correctly** — resolved to 19 200 opt steps, cosine fully completed. Good reproducibility improvement over prior's hardcoded 150k.
6. **Validation samples**: 74 images saved under `dit_validation/` (val_every=1000 in micro-steps → ~1 val/epoch). Not evaluated numerically here; worth a visual check — `cfg_signal=0.07` predicts gray-blob-adjacent output.

## Suggested next run

1. **Revert `gradient_accumulation_steps` to 1** (or at most 2). With 60 epochs this gives auto-cosine target ~150k opt steps — matches prior's plan, 8× more updates than this run.
2. **Raise `lr_min` from 0 to ~1e-5.** Keeps the tail productive instead of running dead.
3. Leave peak LR at 2e-4 — stable at both batch sizes.
4. If bundling changes, consider adding aux classification loss + `LabelEmbedder` init std 0.02→0.1 (see `class_conditioning.md §5–6`) to attack `cfg_signal` directly.
5. Do not bundle multiple changes in one run — this run's bundled delta (grad_accum + channel_weights + auto-cosine + render-engagement) makes the MSE regression unattributable.

## Reproducibility pointers

- **Startup banner** (lines 1–32 of the `.err` log) + the `[overrides] applied` line is the authoritative state: `JiT-B/8`, bf16, num_processes=2, per-GPU batch=16, `max_opt_steps=19200`, `P_mean=0.000, P_std=1.000`, channel weights enabled (mean=0.9999, min=0.682, max=1.822).
- Configs at run-start match the current YAML state as of this write-up (neither file was edited mid-run).
- Seed: 0 (per-process offset by accelerate).
- For comparison runs, capture `jit/configs/jit_train_gsplat.yaml` and `jit/configs/overrides.yaml` snapshots at run start (copy to `runs/<date>_*/configs/`).
