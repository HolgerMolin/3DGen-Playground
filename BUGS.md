# Known Algorithmic Bugs in the Inference Pipeline

Identified via code review of `jit/sampling.py`, `jit/train_gsplat.py`, and `jit/diffusion/gaussian_diffusion.py`.

---

## Bug 1: JiT ODE — Integration direction is reversed ✓ RESOLVED

**Affects:** `heun`, `euler` samplers
**Severity:** Critical — these samplers produce garbage output
**File:** `jit/sampling.py:250`
**Resolved by:** commit `4001c74` — training switched to `flow_matching_training_losses` with convention `t=0 → noise, t=1 → clean` (`jit/diffusion/gaussian_diffusion.py:238`). Under that convention, `linspace(0.0, 1.0, ...)` starting from a random-noise sample **is** the denoising direction. This entry described the bug under the old DDPM-trained convention; no code change in `jit/sampling.py` was needed once training was moved to the matching FM interpolation.

### What the code does

```python
sample = noise_scale * torch.randn(shape, ...)          # start: pure noise
timesteps = torch.linspace(0.0, 1.0, num_inference_steps + 1, ...)
for step_idx in range(max(0, num_inference_steps - 1)):
    sample = stepper(t_value=timesteps[step_idx], t_next=timesteps[step_idx + 1], ...)
```

`timesteps` runs `0.0 → 1.0`. In the JiT / Flow Matching convention the code adopts:
- `t = 0` → clean data
- `t = 1` → pure noise

The sampler initialises with pure noise and then integrates **toward more noise**. It is running the forward (noising) process, not the reverse (denoising) process.

### What it should do

Start at `t = 1` (pure noise) and integrate to `t = 0` (clean data):

```python
timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, ...)
```

The step direction and sign of the velocity are then naturally correct because `t_next < t_value` gives a negative `step`, which moves the sample in the denoising direction.

---

## Bug 2: JiT ODE — Velocity formula is inconsistent with the training forward process ✓ RESOLVED

**Affects:** `heun`, `euler` samplers
**Severity:** Critical — even if direction is fixed (Bug 1), velocity is wrong
**Files:** `jit/sampling.py:132-133`, `jit/train_gsplat.py:764`
**Resolved by:** commit `4001c74` — this entry listed two options ("train with a Flow Matching loss" or "replace the ODE sampler with DDIM/DDPM/DPM"); the first was taken. `flow_matching_q_sample` uses `x_t = t·x_0 + (1-t)·ε` (t=0 noise, t=1 clean), under which `dx/dt = x_0 - ε = (x_0 - x_t)/(1-t)`. That is exactly the velocity formula at `jit/sampling.py:132-134`. Training and inference are now on the same FM interpolation path; the sampler no longer needs to be edited.

### What the code does

The velocity at continuous time `t ∈ [0, 1]` is computed as:

```python
# sampling.py:131-133
x_cond = model(sample_input, t_batch, class_labels)   # model predicts x̂₀
denom = (1.0 - t_value).clamp_min(t_eps)
v_cond = (x_cond - sample) / denom                    # (x̂₀ - xₜ) / (1 - t)
```

This formula is derived from the **Flow Matching linear interpolation** forward process:

```
xₜ = (1 - t) · x₀  +  t · ε
```

Under that process, `dx_t/dt = ε - x₀`, and rearranging gives `ε - x₀ = (xₜ - x₀) / t`. The velocity toward x₀ (denoising direction) is `(x₀ - xₜ) / t`, i.e., the denominator should be `t` and the sign is opposite to what is written.

### What the model is actually trained on

```python
# train_gsplat.py:764
loss_dict = diffusion.training_losses(model, x, t, ...)
```

`training_losses` calls `q_sample`, which uses the **DDPM** forward process:

```
xₜ = √ᾱₜ · x₀  +  √(1 - ᾱₜ) · ε
```

This is a curved, schedule-dependent path — not the linear FM path. Two consequences:

1. **Denominator / sign are wrong for FM** — even if one argued FM conventions: the correct FM velocity from an x₀ prediction is `(xₜ - x̂₀) / t`, not `(x̂₀ - xₜ) / (1 - t)`.
2. **The ODE trajectory is out-of-distribution for the model** — the model was trained on DDPM-noised inputs `√ᾱₜ · x₀ + √(1-ᾱₜ) · ε`. During the ODE integration the running sample `xₜ` follows a linear FM path, which the model has never seen. Its predictions will be unreliable at every step.

### Resolution options

Either:
- **Train with a Flow Matching loss** (linear interpolation forward process) to match the FM-style ODE sampler; or
- **Replace the ODE sampler with DDIM/DDPM/DPM**, which correctly use the `alpha_bar` schedule that matches DDPM training.

---

## Bug 3: DDIM — `cfg_scale` is silently ignored (no CFG implementation) ✓ FIXED

**Affects:** `ddim` sampler  
**Severity:** High — the primary reason DDIM output underrepresents model quality  
**File:** `jit/sampling.py:520-532`, `jit/sampling.py:351-394`  
**Fixed in:** `jit/sampling.py` — `sample_with_ddim` now accepts `cfg_scale` and `cfg_interval`; `sample_model` forwards both.

### What the code does

`sample_model` accepts `cfg_scale` for all samplers but does not forward it to `sample_with_ddim`:

```python
# sampling.py:520-532
if sampler == "ddim":
    return sample_with_ddim(
        model=model,
        shape=shape,
        class_labels=class_labels,
        num_inference_steps=num_inference_steps,
        device=device,
        predict_xstart=predict_xstart,
        noise_schedule=noise_schedule,
        diffusion_steps=diffusion_steps,
        eta=ddim_eta,
        generator=generator,
        # cfg_scale is NOT passed
    )
```

Inside `sample_with_ddim`, the model is called with a single forward pass (conditional only):

```python
model_kwargs = {"y": class_labels}

def model_fn(x, t, y):
    return model(x.to(dtype=sample_dtype), t, y)

sample = diffusion.ddim_sample_loop(model_fn, shape, ..., model_kwargs=model_kwargs, ...)
```

There is no unconditional forward pass and no guidance interpolation. A user who passes `--cfg_scale 4.0 --sampler ddim` gets exactly the same output as `--cfg_scale 1.0`.

### Why this matters

The model is trained with class-label dropout (`class_dropout_prob`, default 0.1), which teaches it to handle null-class inputs. This enables classifier-free guidance at inference. CFG is the primary lever for improving class-conditional sample quality. Without it, DDIM produces effectively weakly-conditional samples regardless of the `cfg_scale` setting.

The `heun`/`euler` samplers do implement CFG correctly in `_jit_velocity_from_xstart`, but they are broken for other reasons (Bugs 1 and 2). DDIM is the only working sampler with no CFG.

### Fix

Add a CFG wrapper inside `sample_with_ddim`. The model's `y_embedder` uses index `num_classes` as the null token (matching the dropout logic in `LabelEmbedder`). A minimal CFG model function:

```python
null_labels = torch.full_like(class_labels, num_classes)

def model_fn_cfg(x, t, y):
    x_in = torch.cat([x, x], dim=0)
    t_in = torch.cat([t, t], dim=0)
    y_in = torch.cat([y, null_labels], dim=0)
    out = model(x_in.to(dtype=sample_dtype), t_in, y_in)
    out_cond, out_uncond = out.chunk(2, dim=0)
    return out_uncond + cfg_scale * (out_cond - out_uncond)
```

---

## Bug 4: DDIM — starts denoising from severely undertrained timesteps

**Affects:** `ddim`, and to a lesser extent `dpm`/`ddpm`  
**Severity:** Medium — degrades quality via compounding early-step errors  
**Files:** `jit/train_gsplat.py:217-220`, `jit/diffusion/respace.py:12-62`

### Training timestep distribution

```python
# train_gsplat.py:217-220
def _sample_jit_timesteps(batch_size, num_timesteps, device, p_mean, p_std):
    probs = torch.sigmoid(torch.randn(batch_size, device=device) * p_std + p_mean)
    return torch.clamp((probs * num_timesteps).long(), min=0, max=num_timesteps - 1)
```

With defaults `P_mean = -0.8`, `P_std = 0.8`:

- `sigmoid(-0.8) ≈ 0.31` → training concentrates around `t ≈ 310` out of 1000
- The distribution decays rapidly toward high `t`; `t > 600` is very rarely sampled

### What DDIM does at inference

DDIM starts at `t = 999` and steps down to `t = 0`. With 50 inference steps, the evenly-spaced schedule visits approximately `t ∈ {999, 979, 959, ..., 19}`. The first ~35 of those 50 steps are at `t > 310` — in the undertrained regime.

The DDIM update for step `t → t_prev` depends on `x̂₀(xₜ, t)`. Poor predictions at high `t` propagate through the DDIM update rule into `x_{t-1}`, which then corrupts `x̂₀` at the next step. Errors compound across the entire trajectory.

### Options

- **Clip the starting timestep**: Begin DDIM at a lower starting `t` (e.g., `t_start = 600`) to avoid the undertrained high-noise regime. This reduces diversity but may improve coherence.
- **Use a training schedule that covers high timesteps uniformly**: Increase `P_mean` or use uniform sampling to ensure the model is trained on the full `[0, 999]` range.
- **Use a non-uniform inference schedule** biased toward the training distribution (e.g., inverse-CDF sampling from the logit-normal to select inference timesteps).

---

## Summary Table

| Bug | Sampler(s) | Severity | Status | Short description |
|-----|-----------|----------|--------|-------------------|
| 1 | `heun`, `euler` | Critical | ✓ Resolved (`4001c74`) | ODE direction — resolved by training on the matching FM interpolation |
| 2 | `heun`, `euler` | Critical | ✓ Resolved (`4001c74`) | FM velocity vs. DDPM training — resolved by switching training to FM loss |
| 3 | `ddim` | High | ✓ Fixed | `cfg_scale` parameter silently ignored; no CFG implementation in DDIM |
| 4 | `ddim`, `dpm`, `ddpm` | Medium | Open | Training biased to `t ≈ 310`; DDIM starts from `t = 999` where model is undertrained |

Post commit `4001c74` the training objective is flow-matching with x₀ prediction on the linear interpolation `x_t = t·x_0 + (1-t)·ε` (t=0 noise, t=1 clean). The `heun` and `euler` samplers integrate the matching ODE on that path and are now consistent with training. `dpm` and `ddpm` still use the `alpha_bar` DDPM schedule via `DPMSolverMultistepScheduler` / `SpacedDiffusion` — these are **no longer consistent with training** after `4001c74` (training no longer produces the DDPM-noised inputs they assume). Until updated for FM, treat `dpm`/`ddpm` as off-distribution samplers for the current JiT checkpoints; prefer `heun`/`euler`. `ddim` has the same training/inference mismatch plus Bug 4.
