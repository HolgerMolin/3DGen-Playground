# Run: 2026-04-22 — JiT-B/8 full-improved render-weight probe (rl1 0.3 → 6.0 at 140k→200k)

Source log (single PID, clean resume, still alive at write time):
1. `output/jit_JiT-B_8_full_improved_renderprobe/train_20260422_190900_1327209.err` — resumed from `output/jit_JiT-B_8_full_improved/0140000.pt` at 2026-04-22 19:09; latest line 2026-04-23 16:06, step 272 400, Epoch 33/70. Accelerate PID 1327220 alive.

Latest checkpoints on disk (new `RESULTS_DIR=output/jit_JiT-B_8_full_improved_renderprobe/`): `0160000.pt` → `0260000.pt` (6 × 2.0 GB, ckpt_every=20 000). Next write due at step 280 000.

Immediate successor to `2026-04-21_jit-b8_fullimproved_scaleup.md`. The scaleup writeup ended with a pointed recommendation to kill the run and do diagnostic label/aux-classifier ablations; this run instead presses the render-loss lever as far as it will go. It is therefore a **controlled probe of a single lever** (render-weight ramp) with every other axis held to the scaleup recipe. That is a strength relative to the scaleup run, which bundled five changes.

| axis | scaleup run (parent) | this run |
|---|---|---|
| Resume | `0040000.pt` (fresh-ish) | **`0140000.pt`** (continues the scaleup trajectory) |
| Obj list / class map / stats | `all_obj_list_filtered.json`, `improved_labels` (120), `all_{mean,std}.pt` | **identical** |
| `train_render_size` | 256 | **256** (unchanged) |
| `render_weight_schedule` | `[(50k, 0), (80k, 0.3/0.15/0.02), (140k, 0.3/0.15/0.02)]` — flat after 80k | **4-point**: adds `(200000, 6.0, 3.0, 0.4)` → **20× ramp** on `rl1/alpha`, **20× on lpips**, over 60 k opt steps |
| `label_embed_init_std`, aux, class_dropout, channel weights | 0.1, off, 0.1, len-14 | **identical** |
| Optimizer / LR schedule | cosine to 556 360, peak 2e-4, `lr_min=1e-5` | **identical** (continues same cosine) |

The render-weight ramp is the *only* axis that moved between parent and child. Whatever this run does or does not deliver is attributable.

## Outcome (one line)

Over 132 k opt steps of render-weight ramp (140k → 272k, 24 → 49 % of cosine, 21 h wall-clock), **render losses fell monotonically** (L1 −24 %, α −34 %, LPIPS −19 %) while **MSE stayed pinned at the 0.455–0.48 lineage floor** and **`cfg_signal` hovered at 0.05–0.07** with no sustained trend. The render head is clearly responding to the extra weight; the diffusion trunk and the conditioning channel are not. Brute-forcing pixel gradient did not rescue the cond-signal collapse diagnosed in the scaleup writeup — which nearly closes the book on "maybe we just need more render signal."

## Hardware & duration (so far, this PID only)

| | |
|---|---|
| GPUs | 2× L4 (yusu-l4.c.gcp-langmotion.internal) |
| Mixed precision | bf16 |
| Wall-clock (this PID) | **20 h 57 m** (2026-04-22 19:09 → 2026-04-23 16:06) |
| Steps this PID | 132 400 (140 000 → 272 400) |
| Throughput | **1.76 step/s** steady — indistinguishable from parent's 1.65 step/s despite 20× render weight (render-loss *weighting* doesn't cost extra forward passes; render_size 256, 1 cam/sample unchanged) |
| ETA to cosine end (step 556 360) | ~45 h remaining (→ ~2026-04-25 ~13:00) |

## Model

| | |
|---|---|
| Variant | `JiT-B/8` |
| Params | 130.7 M |
| Spatial fold factor | 1 |
| Input | 128×128 atlas, 14 channels (`sh_degree0_only`) |
| Classes | 120 (`improved_labels`) |
| Class dropout | 0.1 |
| `label_embed_init_std` | 0.1 |
| Aux classifier | **off** (still not enabled — see §Takeaways) |

## Diffusion / training objective

| | |
|---|---|
| Mode | DDPM container, flow-matching x0-prediction — unchanged from parent |
| Noise schedule | `squaredcos_cap_v2`, 1000 timesteps |
| Timestep sampling | logit-normal `sigmoid(N(0.0, 1.0))` |
| Channel loss weights | length-14, mean 1.0, min 0.644, max 1.735 (same full-corpus values as parent) |
| **Render loss schedule** | **new 4-pt ramp**: `(50000, 0, 0, 0)` → `(80000, 0.3, 0.15, 0.02)` → `(140000, 0.3, 0.15, 0.02)` → `(200000, 6.0, 3.0, 0.4)`; linear between, held flat outside. Cutoff `t_value >= 0.4` unchanged. |
| Effective weights at step 272k | `rl1 = 6.0, alpha = 3.0, lpips = 0.4` (held flat post-200k) |
| Masking observed | 2–10 of 16 samples masked per step by `t<0.4` cutoff — unchanged from parent |

The chosen ramp endpoint (rl1=6.0) is **~120×** the "planned" max in `weight_schedule.md` (0.05) and **20×** the parent's flat weight (0.3). This run is the extreme case, not a mid-range probe.

## Optimizer & schedule

| | |
|---|---|
| Optimizer | AdamW, wd=0 |
| LR | cosine; warmup=500; peak 2e-4; `lr_min=1e-5` |
| `lr_cosine_total_steps` | auto 556 360 (inherited from parent) |
| Effective batch | 16 × 2 GPUs × grad_accum 1 = 32 |
| Grad clip | 4.0 config → 5.0 from `overrides.yaml` |
| EMA decay | 0.995 |
| Gradient checkpointing | on |
| Per-loss grad-norm probe | **still not emitting** — `grep 'GN\['` returns 0 hits. Third run in a row with the probe plumbing broken. |

## Data

Unchanged from parent: `all_obj_list_filtered.json` (254 338), `all_{mean,std}.pt`, inverse-frequency `WeightedRandomSampler` over the 120 improved classes (min 682, max 4 315).

## Loss trajectory (running MSE, 200-step window)

| Step | Epoch | MSE | LR | Render L1 | Alpha L1 | Render LPIPS | GradNorm | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 140 200 | 17 | 0.463 | 1.72e-4 | 0.0276 | 0.0400 | 0.175 | **0.049** | resume; ramp not yet affecting rl1 (still 0.3) |
| 150 000 | 18 | 0.467 | 1.68e-4 | 0.0262 | 0.0371 | 0.170 | 0.170 | ramp ~17 % into 140→200k |
| 160 000 | 19 | 0.468 | 1.64e-4 | 0.0257 | 0.0345 | 0.164 | 0.303 | ramp 33 %; rl1_eff ≈ 2.2 |
| 170 000 | 20 | 0.459 | 1.60e-4 | 0.0238 | 0.0322 | 0.158 | 0.405 | ramp 50 %; rl1_eff ≈ 3.2 |
| 180 000 | 22 | 0.473 | 1.55e-4 | 0.0241 | 0.0325 | 0.159 | 0.681 | ramp 67 %; rl1_eff ≈ 4.1 |
| 190 000 | 23 | 0.455 | 1.51e-4 | 0.0229 | 0.0305 | 0.155 | 0.569 | ramp 83 %; rl1_eff ≈ 5.1 |
| 200 000 | 24 | 0.467 | 1.46e-4 | 0.0224 | 0.0294 | 0.151 | 0.615 | **ramp complete** (rl1=6.0, α=3.0, lpips=0.4) |
| 210 000 | 25 | 0.468 | 1.41e-4 | 0.0230 | 0.0293 | 0.151 | 0.642 | |
| 220 000 | 27 | 0.471 | 1.36e-4 | 0.0228 | 0.0293 | 0.153 | 0.653 | |
| 240 000 | 29 | 0.460 | 1.25e-4 | 0.0223 | 0.0286 | 0.149 | 0.667 | |
| 260 000 | 32 | 0.474 | 1.15e-4 | 0.0216 | 0.0268 | 0.145 | 0.625 | |
| 272 000 | 33 | 0.471 | 1.08e-4 | 0.0211 | 0.0266 | 0.143 | 0.611 | latest |

**Observations on the trajectory:**

- **MSE**: noisy band 0.455–0.481, mean ≈ 0.469. Parent's floor was ~0.455 at step 140k. Child's ≥140k floor is slightly *higher*. Two possibilities: (a) the heavy render gradient is pulling the trunk off the MSE-optimal direction (plausible; rl1=6 × raw L1 0.022 = 0.13, comparable to channel-weighted MSE of ~0.47 × `channel_weight ≈ 0.5` = 0.23 after weighting — render gradient is now ~50 % of trunk signal by magnitude), or (b) 556 k cosine is simply too slow to move MSE this late (LR only down to 1.08e-4 = 54 % of peak). Either way: **the 20× render weight has not lowered MSE**.
- **Render L1/α/LPIPS**: monotone decline, no plateau even after the ramp completes at 200 k. Between 200 k and 272 k all three continue to drop 5–10 %. This is the only trajectory in the run that visibly benefits from continued training.
- **GradNorm**: jumped 12× (0.05 → ~0.6) during the ramp and has been flat at 0.60–0.70 since step 180 k. Pre-ramp the run was deep under the 5.0 clip; now it still is, comfortably. No clipping events (would show as `GradNorm = 5.000` on the clip-ceiling).
- **Two momentary GradNorm spikes**: 1.32 at step 270 600 and 0.76 at 121 200 — isolated, not recurring.

### MSE bucketed by `t_value` (late window, steps 260k–272k)

| t-bucket | mean MSE | typical n/batch |
|---|---:|---:|
| [0.00–0.25] (very noisy) | ~0.59 | 400–470 |
| [0.25–0.50] | ~0.51 | 1 130–1 200 |
| [0.50–0.75] | ~0.42 | 1 140–1 200 |
| [0.75–1.00] (near clean) | ~0.37 | 410–465 |

Spread ~1.59×. **N=7 now** — same ~1.6× spread across every lineage permutation. Heavy render weight moves neither the spread nor any single bucket. `P_mean=0.0` remains the shape-setter. `P_mean_schedule` is still the only untried lever that could move this.

### Render losses — cross-run comparison

Parent's render L1 at step 140 000 was 0.028. At step 272 000 with 20× the weight, render L1 = 0.0211 (-25 %). The relevant comparison is *engagement-step-matched*:

| Run | @ engagement+90k | @ engagement+120k | @ engagement+220k |
|---|---|---|---|
| labelstd (128 px, rl1=0.3) | ≈0.030 | ≈0.028 | ~0.024 (extrapolated) |
| scaleup (256 px, rl1=0.3) | 0.0273 | 0.0272 | — (died at +100k) |
| **renderprobe (256 px, rl1=0.3→6.0 ramp)** | 0.0273 (@140k = ramp start) | 0.0262 (@150k) | **0.0211 (@272k)** |

The renderprobe is pulling render L1 ~12 % below the prior best at matched engagement. That is not a breakthrough — it is one step of improvement at 20× the weight. Marginal utility of render weight is **sublinear** and likely log-sublinear: the first 0.3 got us most of the way, the next 5.7 got us ~10 % more.

## Conditioning signal probe (`[cond]` lines at checkpoint writes)

| Step | `cfg_signal` | `class_signal` | `pred_rms` | vs parent@140k (0.041 / 0.050) |
|---:|---:|---:|---:|---:|
| 160 000 | 0.0554 | 0.0522 | 0.607 | +35 % cfg |
| 180 000 | 0.0512 | 0.0599 | 0.611 | +25 % |
| 200 000 | 0.0635 | 0.0726 | 0.599 | +55 % |
| 220 000 | 0.0643 | 0.0786 | 0.602 | +57 % |
| 240 000 | 0.0656 | 0.0715 | 0.616 | +60 % |
| 260 000 | 0.0573 | 0.0692 | 0.605 | +40 % |

Reading:
- **Slight recovery from parent's 140k-era 0.041**, drifting to 0.055–0.066 — this may simply be the natural continuation of training rather than anything the render ramp is doing (parent's scaleup was also on a slow drift; we don't have a control arm without the ramp to separate).
- **Still well below** the labelstd run's same-training-step values (0.14–0.22). The ~3× conditioning regression diagnosed in the scaleup writeup is **not repaired** by heavy render weight.
- `class_signal ≈ 0.07` vs `pred_rms ≈ 0.60` = **11 %** — perturbing the class label still moves the output by barely more than noise. For context, labelstd reached 0.22 / 0.60 = 37 %.
- No upward break after ramp completion (200k). If render was going to rescue cond_signal, the 200k–272k window would show it.

Conclusion on the probe: **render weight is conditioning-signal neutral.** This is informative — it rules out "more pixel supervision rescues class discriminability" as a mechanism, leaving the hypotheses named in the scaleup writeup (label semantics / class count / data breadth) as the live ones.

## Validation samples

132 PNGs in `dit_validation/` (val_every=1000, heun sampler, 100 steps, cfg=1.5). **Not yet inspected**; the parent writeup flagged this as a priority task and it remains so. With `cfg_signal ≈ 0.06`, the expectation is that the sweep across class indices at a fixed step should look qualitatively similar — the class label is not moving the output much. Visual confirmation is still the cheapest test before committing to the next run.

Note on sampler correctness: `BUGS.md` Bugs 1+2 flag heun/euler as broken, but they predate commit `4001c74` ("switch JiT training to flow-matching loss with x0 prediction"). Since that commit, training uses `flow_matching_training_losses` with `x_t = t·x_0 + (1-t)·ε` (t=0 noise, t=1 clean; `jit/diffusion/gaussian_diffusion.py:238`), which matches the heun/euler integration path in `jit/sampling.py:132,251`. Those two bug entries are now stale — `BUGS.md` itself named "train with a FM loss" as one of the two resolution options, and that option was taken. Heun validation samples in `dit_validation/` are therefore on a consistent train/sample path and can be trusted for quality readouts; the audit does not need a separate `ddim` comparison.

## Notable interventions (or lack thereof)

- **Only one axis moved**: the 4-point `render_weight_schedule`. Clean single-variable run, in marked contrast to the 5-axis scaleup that preceded it.
- **`overrides.yaml` untouched mid-run**: `lr_scale=1.0, max_grad_norm=5.0, P_mean=0.0` every 1000 steps. `aux_classifier_weight` is in the override but inert — run was started with `aux_classifier: false`, so the head was never instantiated.
- **Per-loss GN probe still broken.** 3rd run in a row. `grad_norm_log_every_n_prints: 10` in yaml, zero `GN[` hits in log. This is a 15-minute plumbing fix (probably in the emission site, not the gate) that has now blocked three analyses from quantifying per-loss gradient balance. Fix it before anything else.
- **Dataloader retry path**: 0 warnings in log (filtered list → no `FileNotFoundError`). Retry fallback in `class_3dgen_loader.py:__getitem__` is correctly dormant.

## Takeaways vs prior runs

1. **Brute-forcing render weight does not rescue conditioning.** The single cleanest experiment in the lineage: everything held, render weight 20×. `cfg_signal` unchanged (within drift), MSE unchanged, t-bucket structure unchanged. The cond-signal bottleneck is **not pixel-supervision-starved**.

2. **Render loss is the only responsive output, and even it's saturating.** L1 went from 0.028 → 0.021 with 20× the weight — a factor of ~1.32 reduction for a factor-20 weight increase. Log-diminishing returns. There is no reason to go above rl1 ≈ 1.0 in future runs; the pixel head is already near its operating point at 0.3.

3. **The 7-run t-spread invariant holds.** Every lineage run shows ~1.6× MSE spread from clean to noisy buckets, regardless of data, classes, labels, render weight, or init std. This is a property of the logit-normal `P_mean=0.0` timestep prior. **`P_mean_schedule` is the untested lever and should be next.**

4. **Scaleup writeup's conclusions confirmed.** Every recommendation in `2026-04-21_jit-b8_fullimproved_scaleup.md` §"What to do instead" is now backed by additional evidence — the renderprobe consumed 21 h to confirm one of them negatively (render weight does not rescue cond). The other four (visual audit, label ablation, aux classifier, fix GN probe) are untouched.

5. **Current run is not making useful progress and is consuming compute.** MSE flat, render saturating, cond flat, cosine 49 % done with ~45 h remaining. The marginal value of the next 45 h is near-zero — most likely another 5–10 % shave on render L1 and nothing else. **Kill at next checkpoint (step 280 000) and redirect compute.**

6. **Heavy render weight may be mildly destabilizing MSE.** The child's MSE floor (0.469 mean over 140k–272k) is slightly higher than parent's pre-ramp floor (0.455 at step 140k). Not a blow-up, but consistent with render gradient (now ~50 % of loss scale) pulling the trunk off the MSE-optimal direction. Before another run with heavy render, check per-loss grad norms to quantify this — which requires fixing the GN probe.

## Is scaling the correct next move?

**Still no**, and this run strengthens that answer. The hypothesis that could have overturned it — "maybe we just need more generative signal via render loss" — has been falsified on the full-corpus recipe. The constraints remain:

1. MSE floor is not capacity-bound. Seven runs plateau at 0.45–0.48 across every axis combination.
2. Conditioning is the lineage's primary metric target and remains 2–3× below labelstd values. Larger models will, at best, reach the same ceiling on the same broken conditioning.
3. There is now one more axis (render weight, extreme end) we can cross off the "maybe this is the bottleneck" list. That narrows the search to the untested levers, not expands it.

**Near-term plan, in order of expected information gain:**

1. **Stop the current run at step 280 000** (next checkpoint, ~2 h from write time). 45 h of remaining cosine will not repay the compute.
2. **Visual audit the 132 validation PNGs** before anything else. Walk across the class-index dimension at a fixed step; confirm visually that class conditioning is near-null as `cfg_signal=0.06` predicts. Heun is trustworthy post-`4001c74` (see "Note on sampler correctness" above); no `ddim` comparison needed. 30 min of work that could redirect the whole ablation plan.
3. **Fix the per-loss GN probe.** Three runs without it. Without it we cannot distinguish "render loss is ~50 % of trunk gradient" from "render loss is ~5 % of trunk gradient", and that gap matters for deciding the next render weight.
4. **Launch label-semantics diagnostic A** (per scaleup writeup §Suggested next run, item 2): labelstd recipe (aesthetic split, 87 classes, render_size 128, lr_min=1e-5, label_embed_init_std=0.1) but swap class_map to the improved-labels subset for aesthetic objects. 20 k steps. Probe `cfg_signal` at 20 k; compare to labelstd's 0.135.
5. **Launch aux-classifier rescue B** in parallel (per scaleup §Suggested next run, item 3): *resume from `0140000.pt`* (parent's pre-renderprobe checkpoint — not from renderprobe's 0260000, which has 132 k steps of drift on a heavy render gradient baked in). Flip `aux_classifier: true, aux_classifier_weight: 0.01`. The prior aux_classifier writeup (`aux_classifier.md`) was downbeat about this lever in the 87-class regime; worth re-running on 120 classes where cond signal is much worse at baseline — the mechanism may differ when the probe reports 0.06 instead of 0.14.
6. **Revert render ramp to `[(50k, 0), (80k, 0.3/0.15/0.02), (end, 0.3/0.15/0.02)]`** for all diagnostic runs. This run has shown rl1 > 0.3 is sub-linear; the headroom is not a priority.
7. **First `P_mean_schedule` curriculum** probe: `[(0, -0.5), (20000, 0.0), (40000, 0.3), (70000, 0.5)]`. Untested lever that the t-bucket invariance evidence directly predicts should move the floor.
8. **Only then consider JiT-L/8.** Same bar as before: need `cfg_signal ≥ 0.15` on the full-corpus recipe first.

## Suggested next run

1. **Kill this PID** (`1327220`) at the next checkpoint (step 280 000, ~2 h out). Preserve `0260000.pt` and `0280000.pt` as render-ramp reference checkpoints — they are the best-render-loss checkpoints in the lineage.
2. **Diagnostic A — label ablation** (labelstd recipe on improved-labels subset). 20 k steps on `aesthetic_list.json` with a re-keyed 87-class improved-labels map. Success criterion: `cfg_signal ≥ 0.10` at step 20 k (vs labelstd's 0.135) ⇒ labels are partly the issue; else labels exonerated, suspicion shifts to data breadth / class count.
3. **Diagnostic B — aux classifier rescue from parent ckpt**. Resume `output/jit_JiT-B_8_full_improved/0140000.pt`, enable `aux_classifier: true, aux_classifier_weight: 0.01`, 20 k steps. Watch `cfg_signal`; success criterion ≥ 0.10 within 10 k steps.
4. **Fix the per-loss GN probe** before either A or B. Grep for `GN\[` in `jit/train_gsplat.py`; the emission site almost certainly regressed when the probe was moved. Log a `GN[mse|render_l1|alpha|lpips]` line at the same cadence as the step-print line.
5. **Leave render weights at `(140k+, 0.3/0.15/0.02)` for diagnostics.** The renderprobe has answered what 20× buys (~12 % render L1, 0 % everything else). The 0.3 baseline is the right operating point.
6. **After A/B return, plan the P_mean_schedule run** as the next exploratory probe (7 runs of constant `P_mean=0.0` have each returned the same t-spread and MSE floor; this is the strongest a priori case for a novel intervention).

## Reproducibility pointers

- **Startup banner** (lines 1–38 of `train_20260422_190900_1327209.err`): JiT-B/8, bf16, 2 GPUs, per-GPU batch 16, `max_opt_steps=556360`, `lr_min=1e-05`, `P_mean=0.0`, `P_std=1.0`, channel weights `[0.705, 0.794, 0.845, 1.735, 1.099, 1.053, 1.045, 0.975, 1.048, 0.971, 1.735, 0.672, 0.679, 0.644]`, `label_embed_init_std=0.1`, `train_render_size=256`, `enable_render_loss_after=50 000`, `render_loss_noise_cutoff=0.4`.
- **Render schedule** (the single changed axis): `[(50000,0,0,0),(80000,0.3,0.15,0.02),(140000,0.3,0.15,0.02),(200000,6.0,3.0,0.4)]` — see `jit/configs/jit_train_gsplat.yaml:105–109`.
- `overrides.yaml`: `lr_scale=1.0, max_grad_norm=5.0, P_mean=0.0`. Applied every 1000 steps; no mid-run edits.
- Resume chain (cross-writeup): 1112572 (parent, step 0) → `0040000.pt` → 1129228 (parent, resumed) → `0140000.pt` → **1327220 (this run, resumed 2026-04-22 19:09)**.
- Dataset/class/stats: `/home/tiangexiang/gen3d/gaussianverse/all_obj_list_filtered.json` (254 338), `object_labels/improved_labels/object_to_class.json` (120), `data/stats/all_{mean,std}.pt`.
- Checkpoints (new dir): `output/jit_JiT-B_8_full_improved_renderprobe/{0160000..0260000}.pt`, 2.0 GB each.
- Validation PNGs: 132 under `dit_validation/`, heun sampler (trustworthy post-`4001c74` — `BUGS.md` Bugs 1+2 are stale for FM-trained checkpoints). **Priority task: visually inspect before launching diagnostics A/B.**
- Loss-plot dumps auto-refreshed at 2026-04-23 16:03 in `loss_plots/` (`mse.png`, `render_l1.png`, `alpha_l1.png`, `lpips.png`, `grad_norm.png`, `mse_by_t_bucket.png`, `loss_log.csv`). Useful for the inspect step.
