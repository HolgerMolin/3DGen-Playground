# Training Performance Optimization Notes

**Branch:** `perf/training-optimizations`  
**Base branch:** `dev`  
**Status:** All changes implemented, not yet committed or merged.

---

## Context

This document records all performance optimizations made to the JiT diffusion training pipeline. It is written so that a future session with no prior context can understand, verify, and extend this work.

### What the training pipeline does (relevant to perf)

- **Model:** Transformer (JiT-{S,B,L,XL}/8) operating on 3DGS feature grids — 128×128 grid × 59 channels = 16,384 Gaussians per object.
- **Patch tokenization:** 8×8 patches → 256 sequence tokens for /8 variants.
- **Diffusion:** Standard DDPM with 1000 timesteps, MSE loss, eps or x0 prediction.
- **Mixed precision:** fp16 by default (`--mixed_precision fp16`).
- **Gradient accumulation:** 32 steps default (`--gradient_accumulation_steps 32`).
- **Render loss:** Optional photometric loss via gsplat; disabled by default, enabled after a warmup step count.
- **Key training files:**
  - `jit/train_gsplat.py` — main training loop
  - `jit/models.py` — JiT transformer, RoPE embeddings, TimestepEmbedder
  - `jit/diffusion/gaussian_diffusion.py` — DDPM forward/backward, `_extract_into_tensor`
  - `utils/gsplat_render_util.py` — render loss helpers, plane↔sphere permutation

---

## Optimizations implemented

### 1. `_extract_into_tensor`: GPU schedule cache + `expand()` instead of `+ zeros()`

**File:** `jit/diffusion/gaussian_diffusion.py`  
**Lines (post-edit):** module-level `_SCHEDULE_TENSOR_CACHE` dict (~line 19); `_extract_into_tensor` function (~line 874)

**What the old code did:**
```python
# Called 4× per training step (2 in q_sample, 2 in _predict_xstart_from_eps)
res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
while len(res.shape) < len(broadcast_shape):
    res = res[..., None]
return res + th.zeros(broadcast_shape, device=timesteps.device)
```

Two problems:
1. `th.from_numpy(arr).to(device=...)` copies the 4 KB schedule array CPU→GPU on every call.
2. `+ th.zeros(broadcast_shape)` allocates a **14.8 MB** tensor (shape `(4,59,128,128)` fp32) to force broadcasting, then adds — ~15 μs per call for the zeros alloc + memset + add kernel.

**What the new code does:**
```python
_SCHEDULE_TENSOR_CACHE: dict = {}   # module-level

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    key = (id(arr), str(timesteps.device))
    arr_t = _SCHEDULE_TENSOR_CACHE.get(key)
    if arr_t is None:
        arr_t = th.from_numpy(arr).float().to(device=timesteps.device)
        _SCHEDULE_TENSOR_CACHE[key] = arr_t
    res = arr_t[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
```

Cache key uses `id(arr)` because schedule arrays are instance attributes on `GaussianDiffusion` that live for the full training run — `id()` is stable.

`expand()` returns a non-contiguous view (zero-copy); all downstream uses are element-wise multiplications which broadcast correctly.

**Estimated savings:** ~76 μs/step (4 calls × ~19 μs saved per call).

---

### 2. RoPE dtype cache in `VisionRotaryEmbeddingFast`

**File:** `jit/models.py`  
**Class:** `VisionRotaryEmbeddingFast`  
**Lines:** `forward()` method (~line 51)

**What the old code did:**
```python
def forward(self, x):
    cos = self.freqs_cos.to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    sin = self.freqs_sin.to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin
```

In fp16 training, `freqs_cos`/`freqs_sin` are stored as float32 buffers. The `.to(dtype=x.dtype)` triggers a CUDA dtype-conversion kernel on a 64 KB tensor (shape `(256, 64)` for B/8) on **every** call. This is called `2 × depth` times per forward pass (q and k in every block): 24× for B/8, 56× for XL/8.

**What the new code does:**
```python
def forward(self, x):
    if not hasattr(self, '_rope_cache_dtype') or self._rope_cache_dtype != x.dtype:
        self._cos_cache = self.freqs_cos.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        self._sin_cache = self.freqs_sin.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        self._rope_cache_dtype = x.dtype
    return x * self._cos_cache + rotate_half(x) * self._sin_cache
```

`_cos_cache` / `_sin_cache` are plain Python attributes (not buffers), so they don't appear in `state_dict`. They are invalidated if dtype changes (e.g. during eval in a different precision). Device is already correct because registered buffers follow `model.to(device)`.

**Estimated savings:** ~120 μs/step (B/8: 24 calls × ~5 μs); ~280 μs/step (XL/8: 56 calls × ~5 μs).

---

### 3. `TimestepEmbedder`: pre-compute `freqs` as a registered buffer

**File:** `jit/models.py`  
**Class:** `TimestepEmbedder`  
**Lines:** `__init__` and `timestep_embedding` (~line 188)

**What the old code did:**
```python
@staticmethod
def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    ...
```

`torch.arange` + `torch.exp` on CPU + `.to(device)` ran on every forward pass.

**What the new code does:**
- `__init__` computes `freqs` once and calls `self.register_buffer("_freqs", freqs, persistent=False)`.
- `timestep_embedding` is now an instance method using `self._freqs` (already on device).
- `persistent=False` means the buffer moves with `model.to(device)` but is excluded from `state_dict` / checkpoints.
- Added `max_period=10000` parameter to `__init__` for forward compatibility.

**Checkpoint compatibility:** No state_dict keys added (persistent=False). Existing checkpoints load without modification.

**Estimated savings:** ~8 μs/step (small; only called once per forward).

---

### 4. `update_ema`: `zip` over parameters instead of dict lookup

**File:** `jit/train_gsplat.py`  
**Function:** `update_ema` (~line 432)

**Old:**
```python
ema_params = dict(ema_model.named_parameters())
model_params = dict(model.named_parameters())
for key in ema_params:
    ema_params[key].mul_(decay).add_(model_params[key].data, alpha=1 - decay)
```
Builds two Python dicts from named_parameters generators on every optimizer step.

**New:**
```python
for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
    ema_p.mul_(decay).add_(model_p.data, alpha=1 - decay)
```

Assumes parameter ordering is consistent (it always is for the same model class). The dict approach was defensive but unnecessary.

**Estimated savings:** ~3–5 μs/micro-step (amortized from ~80–150 μs per optimizer step / 32 grad_accum steps).

---

### 5. Defer `.item()` to log time (GPU accumulate without sync)

**File:** `jit/train_gsplat.py`  
**Lines:** initialization (~line 770), accumulation (~line 1018), log block (~line 1026), reset (~line 1057)

**What the old code did:**
```python
# Every micro-step:
log_loss += mse_loss.item()             # GPU→CPU sync
log_render_l1 += render_l1_loss.item() # GPU→CPU sync
log_render_alpha_l1 += render_alpha_l1_loss.item()
log_render_lpips += render_lpips_loss.item()
```
Four `.item()` calls = four GPU→CPU synchronization points per micro-step on every process.

**What the new code does:**
```python
# Init (before training loop):
log_loss = torch.zeros([], device=device)
log_render_l1 = torch.zeros([], device=device)
log_render_alpha_l1 = torch.zeros([], device=device)
log_render_lpips = torch.zeros([], device=device)

# Every micro-step (no sync):
log_loss += mse_loss.detach()
log_render_l1 += render_l1_loss.detach()
log_render_alpha_l1 += render_alpha_l1_loss.detach()
log_render_lpips += render_lpips_loss.detach()

# Log block only (single sync per log_every steps):
avg_loss = log_loss.item() / log_steps

# Reset:
log_loss.zero_()
```

Non-main processes accumulate and never call `.item()` — avoiding syncs entirely on worker ranks.

**Estimated savings:** ~15–20 μs/step (small for the typical setup where GPU is already idle at the `.item()` call site, but eliminates pipeline stalls in tighter training loops).

---

### 6. `norm_mean` / `norm_std` loaded directly to GPU

**File:** `jit/train_gsplat.py`  
**Lines:** norm stats loading (~line 705)

**Old:**
```python
norm_mean_full = torch.load(args.mean_file, weights_only=True).float().cpu()
norm_std_full = torch.load(args.std_file, weights_only=True).float().cpu()
```

**New:**
```python
norm_mean_full = torch.load(args.mean_file, weights_only=True).float().to(device)
norm_std_full = torch.load(args.std_file, weights_only=True).float().to(device)
```

`_denormalize_point_cloud` (called 2× per render step) does `.to(device=point_cloud.device, dtype=...)`. With the stats already on device, this is a no-op. Shape of stats: `(59,)` — tiny, no memory concern.

**Estimated savings:** ~10 μs/render-step. Only matters when render loss is active.

---

### 7. Inverse sphere-plane permutation cache

**File:** `utils/gsplat_render_util.py`  
**Function:** `_plane_to_point_cloud_batch` (~line 120)  
**Lines:** module-level `_SPHERE_TO_PLANE_INV_CACHE` dict (~line 24)

**What the old code did:**
```python
perm = plane_to_sphere.to(device=planes.device)
sphere_to_plane = torch.empty_like(perm)
sphere_to_plane[perm] = torch.arange(num_points, device=planes.device, dtype=perm.dtype)
return flat.index_select(1, sphere_to_plane)
```
Recomputed the inverse of the 16,384-element permutation on every call. Called 2–3× per render step.

**What the new code does:**
```python
_SPHERE_TO_PLANE_INV_CACHE: dict = {}   # module-level

cache_key = (plane_to_sphere.data_ptr(), str(planes.device))
sphere_to_plane = _SPHERE_TO_PLANE_INV_CACHE.get(cache_key)
if sphere_to_plane is None:
    perm = plane_to_sphere.to(device=planes.device)
    sphere_to_plane = torch.empty_like(perm)
    sphere_to_plane[perm] = torch.arange(num_points, device=planes.device, dtype=perm.dtype)
    _SPHERE_TO_PLANE_INV_CACHE[cache_key] = sphere_to_plane
return flat.index_select(1, sphere_to_plane)
```

`data_ptr()` is used (not `id()`) because it remains stable as long as the underlying tensor storage is alive, which it is — `plane_to_sphere` is loaded once from `sphere2plane.npy` and kept for the full run.

**Estimated savings:** ~30–45 μs/render-step. Only when render loss is active.

---

### 8. Pre-allocated zero sentinel for render losses

**File:** `jit/train_gsplat.py`  
**Lines:** initialization (~line 797), per-step assignment (~line 865)

**Old** (inside the training loop, every step):
```python
render_l1_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
render_alpha_l1_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
render_lpips_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
```

**New:**
```python
# Before training loop:
_zero_render_loss = torch.zeros([], dtype=torch.float32, device=device)

# Inside the loop:
render_l1_loss = _zero_render_loss
render_alpha_l1_loss = _zero_render_loss
render_lpips_loss = _zero_render_loss
```

When render loss IS computed, `_compute_render_loss_for_batch` returns new tensors and the variables are reassigned normally. The sentinel is only active for steps where render loss is skipped. All three variables point to the same tensor object — this is safe because none of them are modified in-place.

**Estimated savings:** ~6 μs/step (3 GPU allocations avoided). Applies every step when render loss is disabled.

---

### 9. `torch.compile` opt-in flag

**File:** `jit/train_gsplat.py`  
**Lines:** compile block (~line 640), `--compile` argument (~line 1249)

**Usage:** `--compile` or `--no-compile` (default: off)

**What it does:** Calls `torch.compile(model)` before `accelerator.prepare()`. This means:
- The compiled `OptimizedModule` is wrapped by DDP (correct order).
- EMA is `deepcopy`'d before compilation so it stays an eager model (used for checkpointing/inference only, no forward passes during training).
- `accelerator.unwrap_model(model)` returns the `OptimizedModule`, whose `state_dict()` / `load_state_dict()` proxy transparently to the original module. Resume from checkpoint works correctly.

**Compatibility notes:**
- Works with gradient checkpointing (`use_reentrant=False`, already used in the codebase).
- First training step is slow (~5–60 seconds depending on model size) while Triton kernels are compiled; subsequent steps are faster.
- If compilation fails (e.g. for an unsupported op), PyTorch falls back to eager automatically.
- Tested correct order: EMA deepcopy → compile → diffusion init → optimizer → `accelerator.prepare`.

**Estimated savings: 10–25% per step** (15–30% for forward pass, less for backward). Dominates all other optimizations combined.

---

## Estimated savings summary

Baseline assumptions: JiT-B/8, batch=4, fp16, A100 80GB, gradient_checkpointing=True, render loss disabled.

| Optimization | Savings/micro-step | % of ~10ms step (B/8) | % of ~50ms step (XL/8) |
|---|---|---|---|
| Schedule cache + expand | ~76 μs | 0.76% | 0.15% |
| RoPE dtype cache | ~120 μs | 1.2% | 0.24% |
| Defer `.item()` | ~18 μs | 0.18% | 0.04% |
| norm stats on GPU | ~10 μs (render only) | — | — |
| Inverse perm cache | ~35 μs (render only) | — | — |
| EMA zip, pre-alloc, embedder buffer | ~27 μs | 0.27% | 0.05% |
| **Subtotal (no render loss)** | **~241 μs** | **~2.4%** | **~0.5%** |
| **torch.compile** | **~1–2.5 ms (B/8)** | **~10–25%** | **~10–20%** |

Key caveat: if data loading is the bottleneck (e.g. training from raw PLY files without `--preload_to_cpu`), compute step time is longer and all percentages shrink. Profile with `torch.profiler` or compare `steps/sec` before/after to get real numbers.

---

## What was audited but not implemented

The following issues were identified but not acted on, with reasons:

| Issue | Location | Why skipped |
|---|---|---|
| `_measure_per_loss_grad_norms` does N separate backward passes | `train_gsplat.py:445` | Intentional design — needs full separate backward per loss; changing it would affect what's measured |
| `RMSNorm` float32 upcast via fused `F.rms_norm` | `models.py:63` | Would be a single-kernel speedup (~5–10%) but risk changing numerical behavior in bf16 |
| `_render_gsplat_batch` expand+contiguous | `gsplat_render_util.py:261` | `contiguous()` after `expand()` is required by gsplat; unavoidable |
| `Attention.forward` qkv permute+unbind | `models.py:101` | Already optimal; standard pattern |
| `SwiGLUFFN` chunk | `models.py:133` | Returns views, already zero-copy |
| Checkpoint saving blocking on main process | `train_gsplat.py:1072` | Async save would add complexity; ckpt_every=10000 means it's rare |

---

## Potential future optimizations (not implemented)

These are valid but were out of scope for this session:

1. **`torch.compile(mode='reduce-overhead')`** — More aggressive than `mode='default'`; reduces kernel launch overhead by capturing CUDA graphs. May break with dynamic shapes (variable batch sizes, gradient accumulation variability). Try if `mode='default'` works stably.

2. **`RMSNorm` with `F.rms_norm`** — PyTorch 2.4+ has a fused RMSNorm kernel. The current manual implementation does float32 upcast, pow, mean, rsqrt, scale, downcast — 6+ kernel launches per norm. `F.rms_norm` does it in one. Called `(2 × depth + 1)` times per forward. Change only if training dynamics are verified to be identical (test: same loss curve for 1000 steps with and without).

3. **Fused AdamW** — `torch.optim.AdamW` can use a fused CUDA kernel: `torch.optim.AdamW(params, fused=True)`. Requires CUDA and fp16/bf16 params. Reduces optimizer step time, especially for large models.
   ```python
   opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0, fused=True)
   ```

4. **Apply RoPE to q and k in a single batched call** — Currently `rope(q)` and `rope(k)` run sequentially through `VisionRotaryEmbeddingFast.forward`. Stacking them into a single tensor, applying RoPE once, then unbinding would halve the number of element-wise ops. Low priority.

5. **Pin preloaded CPU cache tensors** — In `class_3dgen_loader.py`, preloaded tensors in `/dev/shm` are not pinned. `pin_memory=True` on the DataLoader triggers on-the-fly pinning. Pre-pinning the shared cache tensors at build time would avoid this overhead.

6. **`DataLoader` `prefetch_factor` tuning** — Currently defaults to 2. For large objects (59×128×128 fp32 ≈ 3.7 MB each), prefetch=4 with persistent_workers=True could hide more I/O latency.

---

## How to verify the changes work

```bash
# Quick smoke test (single GPU, 50 steps)
python jit/train_gsplat.py \
    --model JiT-B/8 \
    --obj_list /path/to/obj_list.json \
    --gs_path /path/to/gs \
    --mean_file /path/to/mean.pt \
    --std_file /path/to/std.pt \
    --class_map object_labels/object_to_class.json \
    --sphere2plane_path data/sphere2plane.npy \
    --batch_size 2 \
    --gradient_accumulation_steps 1 \
    --epochs 1 \
    --log_every 10 \
    --ckpt_every 100 \
    --results_dir /tmp/test_run

# With torch.compile:
python jit/train_gsplat.py ... --compile

# Benchmark steps/sec before vs after (compare dev branch vs this branch):
git stash   # stash changes temporarily
# run 100 steps, note steps/sec
git stash pop
# run same 100 steps, compare
```

To check that `_extract_into_tensor` caching is working, add a temporary `print(len(_SCHEDULE_TENSOR_CACHE))` after a few steps — it should stabilize at 6–8 (number of distinct schedule arrays × number of GPU devices).

---

## Files changed on this branch

| File | What changed |
|---|---|
| `jit/diffusion/gaussian_diffusion.py` | `_SCHEDULE_TENSOR_CACHE` dict; `_extract_into_tensor` uses cache + `expand()` |
| `jit/models.py` | `VisionRotaryEmbeddingFast.forward` RoPE cache; `TimestepEmbedder` freqs buffer |
| `jit/train_gsplat.py` | `update_ema` zip; GPU log accumulators + deferred `.item()`; norm stats to device; `_zero_render_loss` pre-alloc; `torch.compile` + `--compile` flag |
| `utils/gsplat_render_util.py` | `_SPHERE_TO_PLANE_INV_CACHE`; `_plane_to_point_cloud_batch` uses cache |

Note: `jit/diffusion/gaussian_diffusion.py`, `jit/sampling.py`, `jit/train_gsplat.py`, `jit/train_gsplat.sh`, `requirements.txt`, and `.env.example` also have pre-existing uncommitted changes from the `dev` branch that came along with the branch creation. Those are not part of this optimization work.
