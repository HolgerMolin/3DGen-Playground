# JiT Training Evaluation — Weaknesses and Evidence

Evaluation of the JiT-XL/8 training run at `output/jit_JiT-XL/8_results_gsplat/` (checkpoint `0200000.pt`, training continued to ~207k steps at time of writing). Ranked roughly by impact on sample quality, with the evidence each claim rests on.

## 1. Training is not JiT — it's DDPM with JiT-style timestep weighting

`jit/train_gsplat.py:856` calls `diffusion.training_losses`, which at `jit/diffusion/gaussian_diffusion.py:237` noises with `x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε`. The JiT paper uses `x_t = (1-t)·x_0 + t·ε` and predicts velocity (or x₀ in the flow-matching sense). Only the logit-normal timestep prior was copied over.

Consequences:

- `val_sampler: heun` in `jit/configs/jit_train_gsplat.yaml:69` is a flow-matching ODE integrator and is numerically inconsistent with the training noising — that's why every `dit_validation/*.png` is a gray blob even though the base model can actually predict x₀.
- You lose the high-SNR end of the JiT schedule where flow matching is well-conditioned, and gain the numerical stiffness of `squaredcos_cap_v2` at `t→T` instead.
- The "predict_xstart + DDPM" target is valid, just not JiT — calling it JiT is mostly misleading.

## 2. Timestep sampling and inference start point disagree

With `P_mean=-0.3, P_std=0.8`, the histogram of discrete t actually seen during training is:

```
t in [   0,  100):   0.9%
t in [ 100,  200):   7.9%
t in [ 200,  300):  16.1%
t in [ 300,  400):  20.0%   ← mode
t in [ 400,  500):  19.8%
t in [ 500,  600):  16.5%
t in [ 600,  700):  11.3%
t in [ 700,  800):   5.8%
t in [ 800,  900):   1.6%
t in [ 900, 1000):   0.1%
```

But any sampler (DDIM/DPM/DDPM) starts from pure noise at `t = T-1 = 999`. The model effectively never saw that regime during training — the coarse-shape step is untrained. This is the #1 reason generated samples look like particle clouds instead of coherent objects.

## 3. Render loss gradient dominates MSE 8×, and only on low-noise samples

From the latest training log (step 206k):

```
GN[mse]: 0.066   GN[rl1]: 0.011   GN[alpha]: 0.024   GN[lpips]: 0.608
```

LPIPS alone is ~9× the MSE gradient. And `render_loss_noise_cutoff: 0.4` at `jit/configs/jit_train_gsplat.yaml:33` means render loss is masked out for any `t` with `ᾱ_t < 0.4` — i.e., only `t ≲ 500`. Combined with #2, the model's dominant gradient signal is "make the rendered image at low noise look nice." High-noise denoising gets ~1/9 of the learning signal.

## 4. MSE plateaus almost immediately; base model doesn't improve past ~15k steps

Sampled from the full run:

```
step    200:  MSE 0.9479   (warmup)
step  15000:  MSE 0.4577
step  30000:  MSE 0.4319   (render loss kicks in)
step 100000:  MSE 0.4230
step 200000:  MSE 0.4260
step 206800:  MSE 0.4240
```

Cross-checkpoint probe, base weights, `RMSE(x₀_pred, x) at t=0`:

```
step 20000  → 0.738
step 100000 → 0.724
step 180000 → 0.723
step 200000 → 0.719
```

The model fit 90% of what it ever fits in the first 15k steps and has been grinding LPIPS for 190k steps without improving x₀ prediction. Meanwhile the cosine schedule's `lr_cosine_total_steps: 0` auto-computes to ~58k optimizer steps; at step 206k (≈6.4k optimizer steps) LR has only decayed from 2e-4 to 1.95e-4 — essentially constant.

## 5. EMA is under-converged by design

`ema_decay: 0.9999` gives an effective window of `1/(1-d) = 10000` EMA updates, but EMA is only updated on optimizer sync (every `gradient_accumulation_steps=32`). So 200k training steps → 6.25k EMA updates → window is wider than the entire run. Empirically:

```
step    weights   RMSE(t=0)   std(pred)   cond-vs-null
 20k    ema       1.167       0.016       0.0000   (basically dead)
100k    ema       1.073       0.182       0.0009
180k    ema       0.950       0.417       0.0033
200k    ema       0.927       0.455       0.0038
200k    base      0.719       0.906       0.0120
```

EMA std rises linearly, not exponentially — it's simply not converged. For this optimizer-step budget, `ema_decay` should be around `0.999` (window ≈ 1000 updates), not `0.9999`. Inference should load base weights, not EMA (`jit/infer_gsplat.py` default is now `--no-use_ema`).

## 6. Class conditioning has collapsed

At step 200k, base weights, pure noise input, t=900:

- `‖pred(noise, y_A) - pred(noise, null)‖_RMS = 0.012`
- `‖pred(noise, y_A) - pred(noise, y_B)‖_RMS = 0.014`
- `pred std ≈ 0.9`

So conditioning perturbs output by ~1.3% of activation norm, and two different class labels look as different as one class vs. null. CFG on this checkpoint is mathematically a no-op. The embedding table in `jit/models.py:234` initialized at `std=0.02` and nothing is pulling it apart because most of the training signal (LPIPS) doesn't care about the class.

## 7. Class distribution is ~1000:1 skewed

Across 5000 random samples of the dataset: 86/87 classes observed; biggest class has 646 samples, smallest has 1. Class 33 alone is 13% of the data. Even if class conditioning weren't collapsed, tail classes would get almost no training — and `--overfit` probing won't notice.

## 8. Per-channel normalization is not unit-variance

Averaged per-channel spatial std across 200 samples of the DC-only 14-channel data:

```
[0.233, 0.188, 0.854, 0.248, 0.747, 0.770, 0.767, 0.852,
 0.819, 0.831, 0.621, 0.982, 0.995, 1.013]
```

Channels 0, 1, 3 have 3–5× smaller effective variance than channels 11–13. The logit-normal noise schedule and the MSE loss treat all channels equivalently, so channels with small variance contribute ~1/16th as much MSE as channels with large variance. Low-variance channels are effectively untrained relative to high-variance ones. This could be a difference between "global stats" and "per-channel stats" in the `.pt` mean/std files, or a genuine property of the 3DGS feature distribution that the normalization didn't correct.

---

## Suggested priority order if you fix these

1. **Decide what diffusion you're actually training.** If the goal is JiT, swap `training_losses` for flow-matching noising and keep `heun`; then the existing sampler works. If you want to keep DDPM, change `val_sampler` default to `ddim` (already done in inference) and accept you're doing logit-normal-weighted DDPM with predict_xstart.
2. **Shift the timestep prior rightward** (`P_mean: +0.5` to `+1.0`) or use uniform sampling for the first ~5k steps — the model must see `t→T` to generate anything from noise.
3. **Rebalance gradients**: either cap LPIPS weight until MSE has decayed further, or enable render loss across the full `t` range (drop `render_loss_noise_cutoff` or raise it after MSE converges). The current setup trains a "low-noise renderer" more than a "denoiser".
4. **Lower `ema_decay` to 0.999** for this optimizer-step budget; otherwise treat EMA as decorative.
5. **Address class imbalance** with class-balanced sampling, or drop low-frequency classes into an "other" bucket. Raising `class_dropout_prob` won't help until the embedding table has separated.
6. **Verify normalization stats**: per-channel spatial std varying 5× is worth a short script to confirm mean/std files were computed per-channel over the whole dataset, not accidentally per-object-then-averaged.
