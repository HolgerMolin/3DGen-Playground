# JiT sampler — OOD-collapse diagnosis

Working notes on why the JiT flow-matching sampler emits long thin spike artifacts (high-scale outlier Gaussians) even though the model itself can produce clean single-step `x0` predictions from heavily noised GT.

## TL;DR

The bug is **OOD model collapse on drifted iterates**: the sampler's intermediate `x_t` doesn't live on the `t·x0 + (1-t)·ε` manifold the trainer ever showed the model, and at certain drift levels the model's `x0_pred` itself goes spiky.

Sampler-side knobs that **don't help**: integrator (Heun ↔ Euler), truncate-and-snap, fewer steps, constant-strength Langevin noise injection.

Sampler-side knob that **does help**: **logit-normal-spaced inference timesteps** (matching the trainer's `sigmoid(N(0,1))` `t` density). Cuts scale-tail max by 19–33% across all channels with no model retraining.

The durable fix remains model-side (self-conditioning / iterate-aware training), but logit-normal inference is a free ~20–30% win that should ship as the default.

## Setup

- **Checkpoint used for the diagnostic runs**: `output/jit_B8_no_bottleneck_20260430_194526/0120000.pt` (JiT-B/8, EMA, sh_degree0_only, predict_xstart, 14-channel atlas).
- **Debug entry point**: `jit/debug_sampler_vs_singlestep.py`.
- For 20 fixed objects (seed 0), the script:
  1. **Partial sampler** — noise GT to `t_start ∈ {0.1, 0.3, 0.5}` and run the Heun sampler from `t_start → 1.0`. 5 snapshots per trajectory.
  2. **Single-step** — noise GT to `t ∈ {0.9, …, 0.1}` and run one model forward to read out raw `x0`.
- All forward passes wrapped in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. Production inference (`jit/infer_gsplat.py`) runs without autocast — flagged here for awareness, not a confound for the spike question (single-step uses bf16 too and stays clean).

## Symptom

Trajectory snapshots for `obj06` (and most other objects) show clean single-step renders at every probed `t`, but the partial-sampler trajectory grows long thin "spike" Gaussians especially in `ts0.1` runs. The spikes are scale-tail outliers — they get exponentiated by the renderer (`exp(raw_scale)` clamped at `raw_scale=8` ≈ 2980 in physical units), so a few rogue Gaussians dominate the rendered view.

CSV from the original baseline (Heun + integrate to 1.0) localizes the heavy tail to the scale channels:

| channel (max, physical units) | GT | partial ts0.1 | partial ts0.3 | partial ts0.5 | single t=0.5 | single t=0.1 |
|---|---|---|---|---|---|---|
| scale_x | 0.114 | 0.196 | **0.139** | 0.176 | 0.031 | 0.007 |
| scale_y | 0.067 | 0.131 | 0.218 | **0.270** | 0.046 | 0.003 |
| scale_z | 0.095 | 0.141 | 0.099 | 0.142 | 0.023 | 0.003 |

Partial-sampler outputs hit 1.5–4× the GT scale max; single-step never goes above GT.

The `v_per_group.png` plot also showed a pronounced `||v||_RMS` spike at `t≈0.95` for **every** channel group — the integrator is "still pulling hard" toward `x0_pred` right where `t_eps=0.05` clamps the velocity divisor.

## Hypotheses considered

1. **OOD trajectory drift** — sampler iterates leave the noise-perturbed-GT manifold and the model misbehaves there.
2. **`t_eps`-clamp dilution near `t=1`** — with `Δt=0.01` and `denom=t_eps=0.05`, each step in `[0.95, 1]` only does 20% of the work toward `x0_pred`, so any drift accumulated by `t=0.95` survives into the final image.
3. **Heun's corrector amplifies drift** — Heun evaluates the model at `t_next` on the Euler-predicted state, which is one step further into drifted territory.

Tests below probe each.

## Test 1 — iter vs `x0_pred` rendered side-by-side

**Setup**: in `_partial_heun_with_v_log`, after each iterate snapshot also compute and stash `x0_pred(sample, t)`. Render both and stack into the grid (iter row + x0_pred row per `t_start`).

**Output**: `output/debug_sampler_vs_singlestep_x0test/`.

**What we expected to see**:
- `x0_pred` clean at every iterate ⇒ bug is in the integrator (the model is fine).
- `x0_pred` itself goes spiky as the iterate drifts ⇒ model collapses on OOD input.

**What we actually saw** (mixed, but tilted toward OOD):
- For some objects (e.g. `obj00` vase) the `x0_pred` row is clean at every snapshot.
- For others (`obj13` dog, `obj17` shoe, `obj06` human) `x0_pred` is **clean at the warm-start** (just-noised-GT), **spiky at intermediate sampling snapshots** (`t≈0.31` for `ts0.1`, `t≈0.61` for `ts0.5`), and **clean again at late snapshots** (`t≈0.78–1.0`).

The pattern is consistent: the model is fine when fed properly noised GT, and fine when fed late-stage iterates that have settled, but **collapses on mid-trajectory iterates that have drifted off the training manifold**. That's OOD model behavior, not an integrator bug.

## Test 4 — Euler-only

**Setup**: skip Heun's corrector — `v_used = v_t` at every step. CLI: `--euler_only`.

**Output**: `output/debug_test4_euler/`.

**Result**: scale-tail max barely moves. Visually the same OOD `x0_pred` spikes at the same intermediate snapshots.

| channel (max) | GT | test 1 (Heun) | **test 4 (Euler)** | Δ |
|---|---|---|---|---|
| scale_x ts0.3 | 0.119 | 0.313 | **0.302** | −3% |
| scale_y ts0.3 | 0.113 | 0.196 | **0.174** | −11% |
| scale_z ts0.5 | 0.115 | 0.190 | **0.181** | −5% |

p99 across all scale channels is essentially identical. The end-of-trajectory `||v||_RMS` spike at `t≈0.95` is also still present.

**Verdict**: Heun's corrector contributes ~5–10% to the scale tail but is not the dominant source. Removing it doesn't fix the OOD collapse.

## Test 5 — truncate at `t=0.95`, snap to `x0_pred`

**Setup**: integrate over `[t_start, 0.95]` only (same `Δt` density), then emit `x0_pred(sample, 0.95)` as the final state. CLI: `--truncate_at 0.95`.

**Output**: `output/debug_test5_truncate/`.

**Result**: scale-tail max actually slightly *worse* in some channels.

| channel (max) | GT | test 1 | **test 5 (truncate+snap)** | Δ |
|---|---|---|---|---|
| scale_x ts0.3 | 0.119 | 0.313 | **0.368** | +18% |
| scale_y ts0.3 | 0.113 | 0.196 | **0.205** | +5% |
| scale_z ts0.5 | 0.115 | 0.190 | **0.201** | +6% |

The `v_per_group.png` plot confirms the `t≈0.95` velocity spike is gone (we never enter `[0.95, 1]`). The intermediate-`t` spiky `x0_pred` images are *unchanged* — visually identical to test 1.

**Verdict**: avoiding the unstable tail doesn't help, because the OOD problem is mid-trajectory, not in the tail. Snapping inherits the model's collapse behavior on whichever iterate it sees.

## Triangulation

Tests 1 + 4 + 5 land on the same answer: **the integrator is not the bottleneck**. Switching from Heun to Euler or stopping early both leave the spike tail essentially unchanged because every variant feeds the same model the same drifted iterate and gets back the same OOD `x0_pred`.

The model can produce clean `x0` from training-distribution `x_t = t·x0 + (1-t)·ε` for any `t`, including `t=0.1`. It collapses (high-scale outliers in `x0_pred`) when fed an iterate that has drifted off that manifold by mid-trajectory.

## Test 6 — fewer inference steps (`--num_inference_steps 20`)

**Setup**: same defaults as test 1 (Heun, integrate to `t=1`), but with 20 inference steps instead of 100. CLI: `--num_inference_steps 20`.

**Output**: `output/debug_test_20steps/`.

**Result**: scale-tail max **worse**, not better.

| channel (max) | GT | test 1 (100 steps) | **test 6 (20 steps)** | Δ vs test 1 |
|---|---|---|---|---|
| scale_x ts0.3 | 0.119 | 0.313 | **0.407** | +30% |
| scale_y ts0.5 | 0.113 | 0.212 | **0.221** | +4% |
| scale_z ts0.5 | 0.115 | 0.190 | **0.221** | +16% |
| scale_x ts0.1 | 0.119 | 0.211 | **0.225** | +7% |

p99 is roughly the same. Visually `obj13` and `obj17` show the same OOD `x0_pred` spikes at intermediate snapshots, possibly slightly more pronounced.

**Verdict**: drift does **not** compound geometrically over steps — it's roughly proportional to total integration distance. With 20 steps each `Δt≈0.05` is 5× bigger, so a single bad `x0_pred` contributes 5× more drift per step. The total over 19 steps ends up similar to (or worse than) 99 small Heun steps. This rules out "compounding-per-step" and confirms the OOD failure mode is fundamentally a **model-side** property of evaluating on iterates that don't lie on the training manifold — no integrator schedule fixes it.

## Test 7 — stochastic / Langevin per-step noise (`--noise_inject 0.1`)

**Setup**: same defaults as test 1 (Heun, 100 steps, integrate to `t=1`), but after each non-final step add `noise_inject · √|Δt| · ε` with `ε ∼ N(0, I)`. CLI: `--noise_inject 0.1`.

**Output**: `output/debug_test7_stochastic/`.

**Result**: scale-tail max **mixed** — some channels modestly better, others noticeably worse.

| channel (max) | GT | test 1 | **test 7 (stochastic)** | Δ |
|---|---|---|---|---|
| scale_x ts0.1 | 0.119 | 0.211 | **0.278** | +32% |
| scale_x ts0.3 | 0.119 | 0.313 | **0.328** | +5% |
| scale_y ts0.5 | 0.113 | 0.212 | **0.191** | −10% |
| scale_y ts0.3 | 0.113 | 0.196 | **0.277** | +41% |
| scale_z ts0.5 | 0.115 | 0.190 | **0.171** | −10% |

p99 also goes both directions. Visually `obj13` and `obj17` still show the same OOD `x0_pred` spikes at intermediate snapshots; some new texture from the injected noise is visible on iterates but isn't dragging things back to the manifold.

**Verdict**: at `noise_inject=0.1`, Langevin steps don't help. The injected noise sometimes randomly helps (when it pushes the iterate toward the manifold) but sometimes hurts (when it pushes further OOD). A more principled SDE schedule (e.g. EDM-style with t-dependent churn) might do better, but at this strength the result is a wash.

## Test 8 — logit-normal-spaced inference timesteps (`--timestep_schedule logit_normal`)

**Setup**: same defaults as test 1 (Heun, 100 steps, integrate to `t=1`), but inference timesteps are spaced according to `sigmoid(N(0,1))` density — dense near `t=0.5`, sparse near the endpoints, matching the training-time `t` distribution. CLI: `--timestep_schedule logit_normal`.

**Output**: `output/debug_test8_logitnormal/`.

**Result**: **consistent 19–33% reduction** in scale-tail max across all channels and all `t_start` values. In several cases the max is essentially at GT level.

| channel (max) | GT | test 1 | **test 8 (logit-normal)** | Δ |
|---|---|---|---|---|
| scale_x ts0.1 | 0.119 | 0.211 | **0.164** | −22% |
| scale_x ts0.3 | 0.119 | 0.313 | **0.244** | −22% |
| scale_x ts0.5 | 0.119 | 0.248 | **0.192** | −23% |
| scale_y ts0.1 | 0.113 | 0.205 | **0.139** | −32% |
| scale_y ts0.3 | 0.113 | 0.196 | **0.150** | −23% |
| scale_y ts0.5 | 0.113 | 0.212 | **0.152** | −28% |
| scale_z ts0.1 | 0.115 | 0.147 | **0.114** | −22% (≈ GT) |
| scale_z ts0.3 | 0.115 | 0.150 | **0.122** | −19% |
| scale_z ts0.5 | 0.115 | 0.190 | **0.127** | −33% |

p99 across scale channels also drops by ~10–15%. Mid-trajectory `x0_pred` snapshots still go spiky for the same problem objects (`obj13`, `obj17` etc.), so the model is still collapsing on drifted iterates — but the **trajectory drifts less**, so the final output is much cleaner.

**Why it works**: the model's velocity field is best-trained around `t≈0.5` (where logit-normal sampling concentrates training mass). Linear-spaced inference visits each `t` value with equal frequency, which means the integrator spends time in regions where the velocity field is poorly fit (the tails near `t=0` and `t=1`). Logit-normal-spaced inference spends more steps where the field is well-trained, so each step's approximation error is smaller and the iterate stays closer to the training manifold.

**Verdict**: the first sampler-side change that actually helps. ~20–30% reduction in scale tail with no model retraining.

## Where to go from here

After tests 1, 4, 5, 6, 7, and 8:

- **Test 8 (logit-normal timesteps) is a clear win** and should be the new default for inference. Code change is two lines once promoted into `jit/sampling.py:251`.
- **Test 7 (stochastic sampling)** at `noise_inject=0.1` doesn't help, but the strength was a guess; an EDM-style t-dependent churn schedule may still be worth one more experiment.
- **Self-conditioning / iterate augmentation in training** remains the most promising long-term fix. Combined with logit-normal inference, this is the most likely path to closing the gap to single-step quality.

Updated priority order:

1. **Logit-normal-spaced inference timesteps** — already validated by test 8, ship as the default.
2. **Self-conditioning / iterate augmentation in training** — addresses the OOD root cause, not just the symptoms.
3. **Stochastic / SDE-style sampling with a tuned schedule** — speculative; only worth it if (1) and (2) leave residual problems.

Already ruled out:

- Changing integrator (Heun ↔ Euler) — test 4: ~5–10% effect.
- Truncating before `t=1` and snapping to `x0_pred` — test 5: roughly neutral / slightly worse.
- Lowering `t_eps` — would only affect `[0.95, 1]`, where damage is already done.
- **Fewer inference steps** — test 6: actually *worse* (+30% on scale_x ts0.3 max).
- **Constant-strength Langevin noise injection** at `0.1` — test 7: mixed; some channels worse.

## Reproducing

```bash
# All runs use the same checkpoint and seed; only the variant differs.
CKPT=output/jit_B8_no_bottleneck_20260430_194526/0120000.pt
COMMON="--checkpoint $CKPT --num_objects 20 --seed 0"

source .3dgen/bin/activate
# Test 1: default (Heun, full integration to t=1.0).
python jit/debug_sampler_vs_singlestep.py $COMMON \
    --output_dir output/debug_sampler_vs_singlestep_x0test
# Test 4: Euler only.
python jit/debug_sampler_vs_singlestep.py $COMMON --euler_only \
    --output_dir output/debug_test4_euler
# Test 5: truncate at t=0.95 and snap to x0_pred.
python jit/debug_sampler_vs_singlestep.py $COMMON --truncate_at 0.95 \
    --output_dir output/debug_test5_truncate
# Test 6: same defaults, 20 inference steps instead of 100.
python jit/debug_sampler_vs_singlestep.py $COMMON --num_inference_steps 20 \
    --output_dir output/debug_test_20steps
# Test 7: stochastic / Langevin per-step noise injection.
python jit/debug_sampler_vs_singlestep.py $COMMON --noise_inject 0.1 \
    --output_dir output/debug_test7_stochastic
# Test 8: logit-normal-spaced inference timesteps (THE WIN).
python jit/debug_sampler_vs_singlestep.py $COMMON --timestep_schedule logit_normal \
    --output_dir output/debug_test8_logitnormal
```

Each run takes ~6 min on a partially-loaded L4. Outputs per run: `grids/obj??_*.png` (per-object 8×5 visualization), `histograms_{standardized,physical}.png`, `summary_stats.csv`, `v_per_group.png`.
