# Run: 2026-04-20 — JiT-B/8 on aesthetic split, grad_accum=1 (rerun)

Source log: `output/jit_JiT-B/8_results_gsplat/train_20260420_021602_559800.{out,err}`
Latest checkpoint on disk: `0120000.pt` (ckpt_every=20000; run still in progress at time of writing — PID 559811 alive, latest log line Step 123 000, Epoch 48/60).

Immediate successor to `2026-04-19_jit-b8_aesthetic_ga8.md`. Main deltas from that run: `gradient_accumulation_steps` 8 → **1**, `lr_min` 0 → **1e-5**, `lr_cosine_total_steps` auto = **153 360 opt steps** (8× more updates than the ga8 run), `enable_render_loss_after` 30 000 → **50 000** (but the ga8 run already used 50 000). Config is otherwise identical to ga8.

## Outcome (one line)

Currently at ~123 k / 153 360 opt steps (~80% of cosine), Epoch 48/60, after ~14h wall-clock. MSE: 0.82 → ~0.465 (plateaued by ~step 20k, slow drift down since). Render loss auto-engaged cleanly at step 50 000, produced a one-step GradNorm spike (0.30 from α grad) that normalized within a few hundred steps. **`cfg_signal` crossed the useful-CFG threshold (0.10) around step 40k and is at 0.135 at step 120k — the first run where classifier-free guidance is mathematically nontrivial.**

## Hardware & duration (so far)

| | |
|---|---|
| GPUs | 2× L4 (yusu-l4.c.gcp-langmotion.internal) |
| Mixed precision | bf16 |
| Wall-clock (so far) | 13h 45m (02:16:11 → 16:01:30, still running) |
| Steps reached | 123 000 / 153 360 (cosine target) |
| Epochs reached | 48 / 60 |
| Throughput | ~3.2 step/s pre render-loss; ~2.65 step/s with render-loss (drops to ~0.97 step/s on log-boundary steps due to per-loss grad-norm probes) |
| ETA to cosine completion | ~3h remaining at current 2.65 step/s |

## Model

| | |
|---|---|
| Variant | `JiT-B/8` |
| Params | 130.7 M |
| Spatial fold factor | 1 (disabled) |
| Input | 128×128 atlas, 14 channels (`sh_degree0_only`) |
| Classes | 87 (incl. unconditional null) |
| Class dropout | 0.1 |
| `label_embed_init_std` | 0.02 (unchanged; see `class_conditioning.md §6`) |
| Aux classifier | off |

## Diffusion / training objective

| | |
|---|---|
| Mode | DDPM container, **flow-matching x0-prediction** loss (`predict_xstart=true`, `flow_matching_training_losses`) — startup banner still logs "Training mode: DDPM"; loss path is FM |
| Noise schedule | `squaredcos_cap_v2`, 1000 timesteps |
| Timestep sampling | logit-normal `t ~ sigmoid(N(P_mean, P_std))`; **P_mean=0.0, P_std=1.0** (stable — overrides.yaml has not been edited mid-run) |
| Channel loss weights | ENABLED (length-14 inv-spatial-std² weights, mean=0.9999, min=0.682, max=1.822 — same as ga8 run) |
| Render loss | `enable_render_loss_after=50 000` (engaged cleanly at step 50 000 with `render_l1=0.1, alpha_l1=0.1, lpips=0.01` from overrides); `render_loss_noise_cutoff=0.4` |
| Render size | 128 px, 1 cam/sample, ref poses `artifacts/ref_camera.tar.gz` (52 cams) |

## Optimizer & schedule

| | |
|---|---|
| Optimizer | AdamW, wd=0 |
| Peak LR | 2.0e-4 |
| LR schedule | cosine; warmup=500 opt steps; **`lr_min=1e-5`** (was 0 in ga8) |
| `lr_cosine_total_steps` | **0 → auto = 153 360 opt steps** (8× the ga8 run, because `grad_accum` reverted from 8 to 1) |
| Effective batch | per-GPU 16 × 2 GPUs × **grad_accum 1 = 32** |
| Grad accumulation | **1** (was 8 in ga8) |
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

With `grad_accum=1`, opt-step == micro-step for this run.

| Step | Epoch | MSE | LR | Notes |
|---:|---:|---:|---:|---|
|     200 |  0 | 0.815 | 8.0e-5 | warmup |
|   1 000 |  0 | 0.547 | 2.0e-4 | post-warmup peak LR |
|   5 000 |  1 | 0.512 | 2.0e-4 | |
|  10 000 |  3 | ~0.50 | 1.98e-4 | |
|  20 000 |  7 | 0.494 | 1.92e-4 | ckpt; **`cfg_signal=0.077`, `class_signal=0.047`** |
|  30 000 | 11 | ~0.48 | 1.82e-4 | |
|  40 000 | 15 | 0.479 | 1.70e-4 | ckpt; **`cfg_signal=0.104` (crosses useful-CFG threshold), `class_signal=0.063`** |
|  50 000 | 19 | 0.471 | 1.55e-4 | **render loss auto-engaged** via `enable_render_loss_after` |
|  52 000 | 20 | 0.482 | 1.52e-4 | first render-loss step; GradNorm 0.30 (α grad 0.35 — single-step artifact) |
|  60 000 | 23 | 0.464 | 1.37e-4 | ckpt; `cfg_signal=0.112`, `class_signal=0.089` |
|  80 000 | 31 | 0.472 | 9.9e-5 | ckpt; `cfg_signal=0.117`, `class_signal=0.075` |
| 100 000 | 39 | 0.466 | 6.2e-5 | ckpt; `cfg_signal=0.123`, `class_signal=0.076` |
| 110 000 | 43 | 0.468 | 4.5e-5 | render L1≈0.037, α≈0.053, LPIPS≈0.254 |
| 120 000 | 46 | 0.462 | 3.2e-5 | ckpt; **`cfg_signal=0.135`, `class_signal=0.081`** |
| 123 000 | 48 | 0.466 | 2.8e-5 | last log at time of writing |

### MSE bucketed by `t_value` (clean fraction; t=1 ≡ clean)

Late-run averages (steps 110k–123k, render loss on):
| t-bucket | mean MSE | typical n/batch |
|---|---:|---:|
| [0.00–0.25] (very noisy) | ~0.60 | 400–470 |
| [0.25–0.50]              | ~0.51 | 1130–1210 |
| [0.50–0.75]              | ~0.41 | 1130–1210 |
| [0.75–1.00] (near clean) | ~0.37 | 400–470 |

Spread ~1.62× — **unchanged** from the ga8 run (~1.55×) and the original ga1 run (~1.6×). Channel loss weights and 8× more opt steps both left the MSE/t shape untouched. This continues to point at the logit-normal `P_mean=0.0` timestep prior as the shape-setting lever, not channel weighting.

### Render losses (post step 50 000)

| | typical value |
|---|---:|
| Render L1 | 0.036 |
| Alpha L1 | 0.052 |
| Render LPIPS | 0.253 |

Small monotone drift downward (L1 0.044 → 0.036 over 70k post-engagement steps; LPIPS 0.28 → 0.25). No visible MSE drop at step 50k — MSE had already plateaued at ~0.47 by step ~20k (same pattern as ga8). Render loss provides a ~8:1 gradient advantage to LPIPS vs MSE in absolute terms (`GN[mse]≈0.038`, `GN[lpips]≈0.016`, but lpips is weighted 0.01 — pre-weight LPIPS grad is ~1.6 vs MSE's 0.038, ~42×), though `render_loss_noise_cutoff=0.4` masks it on ~30–60% of samples per step (log line `[render-loss] step=... masked N/16 samples`).

### Per-loss grad norms at step 120 000

```
GradNorm: 0.0537
  GN[mse]: 0.0388   GN[rl1]: 0.0114   GN[alpha]: 0.0292   GN[lpips]: 0.0163
```

Post-weight contributions are balanced (all in 0.01–0.04 range). MSE still the single largest contributor, LPIPS held in check by the 0.01 weight. Compared to the prior XL/8 run (`jit_training_evaluation.md §3`, where `GN[lpips]` was 0.608 vs MSE 0.066 — LPIPS 9× MSE), this run's gradient budget is not LPIPS-dominated.

## Conditioning signal probe (`[cond]` lines at checkpoints)

| Step | `cfg_signal` | `class_signal` | `pred_rms` |
|---:|---:|---:|---:|
|  20 000 | 0.0773 | 0.0474 | 0.499 |
|  40 000 | 0.1039 | 0.0633 | 0.598 |
|  60 000 | 0.1123 | 0.0885 | 0.588 |
|  80 000 | 0.1173 | 0.0754 | 0.598 |
| 100 000 | 0.1229 | 0.0760 | 0.596 |
| 120 000 | 0.1349 | 0.0806 | 0.578 |

**First run in this lineage where `cfg_signal` exceeds 0.10** (the rule-of-thumb threshold for CFG being a nontrivial lift) — and it does so by step 40k. Trajectory:

- ga1 prior (68k steps, P_mean=-0.5 until 52k): `cfg=0.067` at step 60k
- ga8 prior (19.2k opt steps): `cfg=0.073` at step 17.5k
- **this run (123k opt steps so far):** `cfg=0.135` at step 120k

The ga8 run's per-opt-step trajectory was on the same slope as this run; ga8 simply ran out of opt-step budget. Reverting grad_accum to 1 was the correct call — `cfg_signal` now has room to keep rising and the absolute number is the best of the three.

`class_signal` stalled around 0.08 after step 60k — still fundamentally below `pred_rms ≈ 0.58`, i.e. two different class labels still only perturb output by ~14% of activation norm. CFG margin is rising faster than class discrimination, consistent with the model learning "class vs null" more than "class A vs class B". This is the lever `class_conditioning.md §4–5` (split adaLN + aux classifier) still targets.

## Notable interventions (or lack thereof)

- **No mid-run overrides changes.** `overrides.yaml` applied every 1000 steps at the stable value `lr_scale=1.0, max_grad_norm=5.0, render_loss_weight=0.1, alpha_mask_loss_weight=0.1, lpips_loss_weight=0.01, P_mean=0.0` for the full run. One benign warning: `[overrides] ignoring unknown key: aux_classifier_weight` — added to the overrides file but the trainer's override list doesn't accept it yet (and the run was started without `--aux_classifier` so the head isn't instantiated regardless).
- Render loss engaged automatically at step 50 000 via `enable_render_loss_after` — same mechanism that worked in ga8. Clean attribution.
- `max_grad_norm` stayed at 5.0 (overrides) the entire run, overriding the YAML's 4.0.

## Takeaways vs prior runs

1. **Reverting `gradient_accumulation_steps` 8 → 1 was the correct call.** Per-opt-step trajectory is slightly ahead of the ga1 prior (similar absolute numbers at each opt step) and far ahead of ga8 (which ran out of updates at opt step ~19k). This is the first run of the three where `cfg_signal` reaches a useful-CFG regime.
2. **MSE floor is ~0.46 regardless of effective batch / grad_accum / channel weights.** Prior ga1: 0.469 at opt step 68k. ga8: 0.489 at opt step 19k (update-starved). This run: 0.462 at opt step 120k. The delta across all three is within noise; MSE floor appears to be a function of the loss + P_mean shape, not of total compute.
3. **Render loss engagement is reliably clean** via `enable_render_loss_after=50 000` (both this run and ga8) — no more mid-run override flips. The step-52k GradNorm spike (one-step α grad of 0.35) is visible but self-correcting and does not need intervention.
4. **`lr_min=1e-5` (vs ga8's 0)** keeps the tail productive: the current LR is 2.8e-5 and will floor at 1e-5 rather than 0. With ~30k steps remaining, this should still produce useful updates through the end of cosine. Prior ga8 run's last 30% of training happened at LR < 3e-5 decaying to ~0; visibly dead.
5. **MSE-by-t curve shape is invariant** across all three runs (~1.55–1.62× spread). Not a channel-weight problem, not a compute problem. The next lever to try is a `P_mean` shift (or a `P_mean_schedule` curriculum — the scaffold is already in `jit_train_gsplat.yaml` lines 91–95).
6. **Class discrimination still the bottleneck for quality.** `cfg_signal` rising past 0.1 is necessary but not sufficient — `class_signal` is stuck at ~0.08 since step 60k. The `c = t + y` architectural bottleneck (`class_conditioning.md §4`) and the zero-init adaLN head are still in play. Next run should enable `aux_classifier` and bump `label_embed_init_std` to 0.1.

## Suggested next run

1. **Enable `aux_classifier: true` + `aux_classifier_weight: 0.01`** — cheapest lever for `class_signal`, no shape change. `class_conditioning.md §5`.
2. **Bump `label_embed_init_std` 0.02 → 0.1** — one-line, low-risk. `class_conditioning.md §6`.
3. Try a `P_mean_schedule` curriculum: `[[0, -0.3], [30000, 0.0], [80000, 0.2]]` to shift compute toward near-clean samples once the model is competent at high-noise denoising. Probes MSE-by-t spread.
4. Keep `grad_accum=1`, `lr_min=1e-5`, `enable_render_loss_after=50 000` — these are all working.
5. Do **not** bundle (1)+(2)+(3) into a single run. (1)+(2) together is fine (both target class conditioning); put (3) in its own run — it changes the MSE-by-t shape and mixing the two makes attribution impossible.
6. After this run completes (~3h remaining at time of writing), save the final checkpoint and update this doc with the 140k / 153 360 cond-signal numbers.

## Reproducibility pointers

- **Startup banner** (lines 1–32 of the `.err` log) + the `[overrides] applied` line is the authoritative state: `JiT-B/8`, bf16, num_processes=2, per-GPU batch=16, `max_opt_steps=153360`, `lr_min=1e-05`, `P_mean=0.000, P_std=1.000`, channel weights enabled (mean=0.9999, min=0.682, max=1.822).
- Configs at run-start match the current YAML state as of this write-up (neither file edited mid-run apart from the harmless `aux_classifier_weight` key in overrides).
- Seed: 0 (per-process offset by accelerate).
- Log-boundary throughput dips (2.6 → 0.97 step/s every 2000 steps) are from `_measure_per_loss_grad_norms` running N separate backward passes at `grad_norm_log_every_n_prints=10` intervals — expected, documented in `optimizing.md` "What was audited but not implemented".
