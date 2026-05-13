# Run: 2026-04-19 — JiT-B/8 on aesthetic split

Source log: `output/jit_JiT-B/8_results_gsplat/train_20260419_003649_344988.{out,err}`
Last checkpoint: `0060000.pt` (ckpt_every=20000; run terminated by SIGTERM at step 68600).

## Outcome (one line)

Trained JiT-B/8 with flow-matching x0-prediction for 26 epochs / 68.6k opt steps over ~7h. MSE went 0.83 → 0.47; render-loss never auto-engaged (override held weights at 0 for the first 22k steps after threshold), then was manually enabled at step 52k, dropping MSE another ~0.03 within a few hundred steps. Killed before cosine completion (~46% of scheduled steps).

## Hardware & duration

| | |
|---|---|
| GPUs | 2× L4 (yusu-l4.c.gcp-langmotion.internal) |
| Mixed precision | bf16 |
| Wall-clock | 6h 54m (00:36:54 → 07:30:42) |
| Steps reached | 68 600 / 150 000 (cosine target) |
| Epochs reached | 26 / 60 |
| Throughput | ~3.0 step/s pre render-loss; ~2.7 step/s with render-loss |

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
| Mode | DDPM container, **flow-matching x0-prediction** loss (`predict_xstart=true`, `flow_matching_training_losses`) |
| Noise schedule | `squaredcos_cap_v2`, 1000 timesteps |
| Timestep sampling | logit-normal `t ~ sigmoid(N(P_mean, P_std))`; **P_mean=−0.5 (start) → 0.0 (override at step 52k)**, P_std=1.0 |
| Channel loss weights | length-14 inverse-spatial-std, mean=1.000, range [0.682, 1.822] |
| Render loss | configured `enable_render_loss_after=30000`; *kept at weight 0 in overrides until manual flip at step ~52000*, then `render_l1=0.1, alpha_l1=0.1, lpips=0.0 → 0.01` (lpips added at step ~55k); `noise_cutoff=0.4` (skipped on noisy halves) |
| Render size | 128 px, 1 cam/sample, ref poses `artifacts/ref_camera.tar.gz` (52 cams) |

## Optimizer & schedule

| | |
|---|---|
| Optimizer | AdamW, wd=0 |
| Peak LR | 2.0e-4 |
| LR schedule | cosine; warmup=500 opt steps; `lr_min=0` |
| `lr_cosine_total_steps` | **150 000 (hardcoded — not the new auto path)** |
| Effective batch | per-GPU 16 × 2 GPUs × grad_accum 1 = **32** |
| Grad accumulation | 1 |
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

| Step | Epoch | MSE | LR | Notes |
|---:|---:|---:|---:|---|
| 200 | 0 | 0.830 | 8.0e-5 | warmup |
| 1 000 | 0 | 0.583 | 2.0e-4 | post-warmup peak LR |
| 5 000 | 1 | 0.548 | 2.0e-4 | |
| 10 000 | 3 | 0.535 | 1.98e-4 | |
| 20 000 | 7 | 0.535 | 1.92e-4 | ckpt; cond `class_signal=0.055`, `cfg_signal=0.058` |
| 30 000 | 11 | ~0.51 | 1.83e-4 | render loss eligible but overrides keep weights 0 |
| 40 000 | 15 | 0.519 | 1.67e-4 | ckpt; `class_signal=0.069`, `cfg_signal=0.060`, `pred_rms=0.572` |
| 50 000 | 19 | ~0.51 | 1.50e-4 | |
| **52 000** | **20** | 0.518 → **0.469** | 1.47e-4 | **render_loss flipped on (rl1=0.1, α=0.1); P_mean -0.5 → 0.0** |
| 60 000 | 23 | 0.474 | 1.33e-4 | ckpt; `class_signal=0.074`, `cfg_signal=0.067`, `pred_rms=0.600` |
| 68 600 | 26 | 0.469 | 1.14e-4 | last log before SIGTERM |

### MSE bucketed by `t_value` (clean fraction; t=1 ≡ clean)

Late-run averages (steps 60k–68k):
| t-bucket | mean MSE | typical n/batch |
|---|---:|---:|
| [0.00–0.25] (very noisy) | ~0.59 | 410–470 |
| [0.25–0.50] | ~0.52 | 1130–1210 |
| [0.50–0.75] | ~0.43 | 1140–1220 |
| [0.75–1.00] (near clean) | ~0.38 | 410–460 |

After P_mean shift to 0 at step 52k, t-distribution rebalanced toward clean (clean-half count ~1600 → ~1640; very-noisy count fell from ~870 → ~430). MSE drop was strongest on the now-undersampled noisy bucket.

### Render losses (post step 52k)

| | typical value |
|---|---:|
| Render L1 | 0.040 |
| Alpha L1  | 0.061 |
| Render LPIPS | 0.270 (only after enabled at ~55k) |

## Conditioning signal probe (`[cond]` lines at checkpoints)

| Step | `cfg_signal` | `class_signal` | `pred_rms` |
|---:|---:|---:|---:|
| 20 000 | 0.0575 | 0.0553 | 0.4998 |
| 40 000 | 0.0596 | 0.0687 | 0.5723 |
| 60 000 | 0.0673 | 0.0740 | 0.6001 |

`class_signal` rises monotonically — class conditioning is being learned. CFG margin (`cfg_signal`) is roughly tracking it, suggesting drop-prob 0.1 is leaving usable unconditional density.

## Notable mid-run interventions (not pre-planned)

- **Step ~52 000**: overrides edited to set `render_loss_weight=0.1, alpha_mask_loss_weight=0.1` and **`P_mean=-0.5 → 0.0`** simultaneously. MSE drop of ~0.05 within ~2 print intervals is *confounded* by both changes; cannot attribute to either alone.
- **Step ~55 000**: LPIPS weight bumped 0 → 0.01 in overrides.
- `max_grad_norm` stayed at 5.0 (overrides) the entire run, overriding the YAML's 4.0.

## Takeaways for future comparison

1. **Render loss never auto-engaged** even though `enable_render_loss_after=30000` had been crossed — overrides.yaml weight=0 dominates. For a controlled ablation, set the override in line with config or set `render_loss_weight: 0.1` in overrides from the start.
2. **Compound change at step 52k** confounds the most interesting drop in the loss curve. Future runs should change one knob at a time, or enable both from step 0.
3. **Cosine target was hardcoded to 150k** but the run used effective batch 32 with grad_accum 1 (≈2 560 opt steps/epoch → 60-epoch budget ≈ 153 k opt steps). Coincidentally well-sized; with the new auto path this becomes deterministic.
4. **MSE-by-t curve has ~1.6× spread** between most-noisy and near-clean buckets. P_mean tuning shifts where compute is spent without obviously equalizing the curve.
5. **Conditioning is learning slowly** (`class_signal` 0.055 → 0.074 over 40k steps). Worth probing whether higher class_dropout, longer training, or a stronger class embedding closes the gap with `pred_rms ≈ 0.6`.

## Reproducibility pointers

- Configs at the time of run were edited mid-run; YAML in repo no longer reflects start-of-run state. Authoritative state for this run = the `[overrides] applied` lines in the .err log + the startup banner (lines 1–32).
- Seed: 0 (per-process offset by accelerate).
- For comparison runs, capture `jit/configs/jit_train_gsplat.yaml` and `jit/configs/overrides.yaml` snapshots at run start (e.g. copy to `runs/<date>_*/configs/`).
