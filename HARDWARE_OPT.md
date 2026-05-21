Systems optimization:
- Instead of filtering by t, take top half for fixed input shape, combined with LPIPS compiled could win 5-10% throughput

---

# JiT A100 throughput audit (2026-05-14)

Constraints: render-loss path must remain bit-identical (no formula/cadence/cap changes).
Spare GPU memory: ~25% on A100-80GB.

Baseline measured from `output/jit_JiT-B_8_full_improved/train_20260514_033341_42247.err`:
JiT-B/8, bs=256, bf16, `torch.compile(mode="default")`, render loss active.
~3.08 steps/s steady-state, drops to ~2.04 steps/s at every `val_every=1000` boundary.

## Tier 1 — risk-free, training-math-identical

### 1. Fused AdamW
File: [jit/train_gsplat.py:1350](jit/train_gsplat.py#L1350)
Change: `torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0, fused=True)`
Verified: torch 2.11.0+cu128, A100 sm_80, `fused=True` supported.
Expected: 1–4% wall-clock (more on JiT-L/XL).

### 2. Remove three per-step CPU↔GPU syncs
Each one stalls the main stream every micro-batch and serializes against NCCL/prefetch.

- [jit/train_gsplat.py:1810](jit/train_gsplat.py#L1810): `if not torch.isfinite(mse_loss):`
  → either drop entirely (the dump is only needed at first occurrence) or gate behind `step % args.log_every == 0`.
- [jit/train_gsplat.py:1881](jit/train_gsplat.py#L1881): `n_kept = int(keep_mask.sum().item())`
  → keep the slice (it's the memory cap), defer the `.item()` into the `if step % log_every == 0` block. Use `x0_pred_render.shape[0]` checks instead of `n_kept` for the runtime branch.
- [jit/train_gsplat.py:2038-2039](jit/train_gsplat.py#L2038-L2039): `if torch.isfinite(grad_norm): log_grad_norm += grad_norm.item()`
  → accumulate as GPU tensor: `log_grad_norm += grad_norm.where(grad_norm.isfinite(), torch.zeros_like(grad_norm))`; `.item()` once per print.

Expected: 2–5% wall-clock combined.

### 3. Compile mode = max-autotune-no-cudagraphs
File: [jit/train_gsplat.py:1327](jit/train_gsplat.py#L1327)
Change: `model = torch.compile(model, mode="max-autotune-no-cudagraphs")`
Same numerical path as default; Triton picks better tile sizes. Safe under DDP.
Expected: 0–15%, A/B over a 1k-step run.

## Tier 1.5 — converts spare GPU memory into throughput

### 4. Compile mode = reduce-overhead (CUDA Graphs)
File: [jit/train_gsplat.py:1327](jit/train_gsplat.py#L1327)
Change: `model = torch.compile(model, mode="reduce-overhead")`
Captures the forward+backward as a CUDA Graph, removes per-kernel launch overhead.
JiT-B/8 has O(hundreds) of kernels/step; this is exactly where launch overhead matters.

Risk: CUDA Graphs need stable strides between iters. The existing channels-last Conv2d fix at
[jit/train_gsplat.py:1307-1320](jit/train_gsplat.py#L1307-L1320) plus `gradient_as_bucket_view=True`
already protect against the DDP stride drift. Order of operations is already correct:
compile before `accelerator.prepare`.

Validation procedure:
```bash
TORCH_LOGS=recompiles,graph_breaks ./jit/train_gsplat.sh JiT-B/8
```
If recompiles stop after ~step 10, ship it. If recompile loop, fall back to #3.

Optional follow-up (use more of the headroom):
```python
import torch._inductor.config as ic
ic.triton.cudagraphs = True
ic.max_autotune = True
```

Expected: 5–15%. Most likely single biggest win.

## Tier 2 — small effort, training-behavior-neutral

### 5. Fused EMA update with `torch._foreach_*`
File: [jit/train_gsplat.py:1018-1023](jit/train_gsplat.py#L1018-L1023)
Current loop launches ~2 kernels per parameter. JiT-L/XL has ~300+ param tensors.
Replacement:
```python
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_ps = list(ema_model.parameters())
    model_ps = [p.data for p in model.parameters()]
    torch._foreach_mul_(ema_ps, decay)
    torch._foreach_add_(ema_ps, model_ps, alpha=1 - decay)
```
Bit-identical. Expected: ~1% on B/8, more on L/XL.

### 6. Reduce validation cadence
File: [jit/configs/jit_train_gsplat.yaml](jit/configs/jit_train_gsplat.yaml) (`val_every: 1000` → `5000`).
Also remove or gate the every-1000-step `_measure_conditioning_signal` block marked TEMP at
[jit/train_gsplat.py:2186-2218](jit/train_gsplat.py#L2186-L2218) (keep it at checkpoint time only).
The 16-tile grid does 50-step heun sampling + gsplat render every 1000 steps; that's the
~2.04 sps dip in the baseline log.
Expected: 10–20% (it's ~20% of current wall time at val_every=1000).

### 7. Skip `_compute_null_repel_loss` when weight=0 and not a print step
File: [jit/train_gsplat.py:1963-1971](jit/train_gsplat.py#L1963-L1971)
Currently runs every step purely for the `cos(c, null)` diagnostic. Gate on
`step % args.log_every == 0 or args.null_repel_weight > 0`.
Expected: sub-1%.

## Test plan (suggested A/B procedure)

Use `--max_steps 1000` per arm so each run is ~5 min. Run on a single GPU first to remove
DDP-variance from the signal. Take steps/sec averaged over steps 200–1000 (skip warmup).

| Arm | Change |
|---|---|
| baseline | current main |
| t1 | + fused AdamW + sync removals |
| t1.3 | t1 + `max-autotune-no-cudagraphs` |
| t1.4 | t1 + `reduce-overhead` |
| t1.4+ic | t1.4 + inductor.max_autotune + inductor.cudagraphs |
| t2 | best-of-t1.* + foreach EMA + val_every=5000 + null_repel gate |

Tier 1 stacks linearly. Compile-mode (#3 vs #4) is mutually exclusive — pick the winner.

## Tier 3 — render-loss reformulations (objective preserved or near-preserved)

The render loss does not need to be bit-identical; reformulations that compute the
same (or close-to-same) objective more efficiently are in scope. "Do it less often"
or "subsample the batch" are NOT in scope — those wins are already understood.

### Disk & dataset baseline (used by A0/A sizing below)
- Dataset: **253,611 valid samples × 52 cameras = 13.2 M (obj, cam) pairs**
- Disk available on `/`: **665 GB** (with `~/3dgen_cache` already at 434 GB and `output/` at 139 GB)
- Per (obj, cam) render at 224 bf16: **~393 KB** (RGB+alpha)
- Full GT-render cache (52 cams, 224, bf16): **~5.0 TB** — does NOT fit
- 8 fixed cams, 128, bf16: **~260 GB** — fits comfortably

### A0. Cache the GT Gaussian-construction output (camera-independent, ~70 GB)
Cheapest win that preserves the objective bit-exactly.
The output of `_constrain_denormalized_point_cloud_for_render` (means/scales/opacity/quats/colors)
depends only on the object, not the camera. Per object: ~16384 × 14 floats × 2B ≈ **460 KB**.
**Total for full dataset: ~70 GB.** Trivially fits in `/dev/shm` (84 GB) or on `/`.

Add a `cached_gaussians` field to `Class3DGenDataset` (extends the existing `lazy_cache_to_cpu`
machinery). On every render-loss step, GT skips: `_plane_to_point_cloud_batch`,
`_denormalize_point_cloud`, `_apply_rank_inverse_for_render`, sigmoid/exp/quat-normalize.
Pred path is unchanged. Rasterization still runs both sides.

Eliminates ~5% of render-loss step time. Expected: **~2% wall-clock**, zero risk.

### A. Cache GT renders (deterministic in `(object, camera_index)`)
Bigger win, requires committing to a camera subset due to disk:

- **A1 lazy cache, 8 fixed cams × 128 × bf16 (~260 GB):** Pick 8 cameras at startup, restrict
  per-step sampling to that subset, memoize GT (RGB, alpha) per `(idx, cam_idx)` on first
  encounter via the same `lazy_cache_to_cpu` infra at
  [dataloaders/class_3dgen_loader.py:847-867](dataloaders/class_3dgen_loader.py#L847-L867).
  Combines naturally with reformulation **D** (render at 128). View distribution per object
  shrinks 52 → 8 — audit on loss curve.
- **A2 offline:** [data/download_renderings.sh](data/download_renderings.sh) pulls ~8.5 TB of
  pre-rendered 2D views; `Standard3DGenDataset._load_renderings` reads them. If poses align
  with `ref_camera_tar` (verify), the runtime GT rasterization disappears entirely. No new
  disk cost beyond the existing download.

Eliminates ~50% of render-loss compute (~20% of render-loss step time when you also count
the avoided pre-render pipeline). Expected: **~7% wall-clock** (A1), more if A2 lands.

### B. `torch.compile` the pre-render pipeline
File: [utils/gsplat_render_util.py:265-313](utils/gsplat_render_util.py#L265-L313).
`_constrain_denormalized_point_cloud_for_render` + `_apply_rank_inverse_for_render` +
`_normalize_quaternions_with_identity_fallback` is ~15 element-wise / gather /
searchsorted ops on `(B, 16384, 14)` tensors. Trivially fusible.

```python
_constrain_compiled = torch.compile(_constrain_denormalized_point_cloud_for_render, dynamic=False)
```
Same shape contract on every call. Runs on both GT and pred sides.
Expected: 1–3% of step time per render-loss step.

### C. GT / pred construction on separate CUDA streams
GT path is `no_grad`. Independent of pred until the final L1. Put GT on a side stream;
sync only before L1 compute. No math change.

### D. Render at 128×128, upsample for LPIPS only
File: [jit/configs/jit_train_gsplat.yaml:67](jit/configs/jit_train_gsplat.yaml#L67) `train_render_size: 224`.
128 matches the native 3DGS grid resolution. L1 and alpha-L1 evaluate at 128;
bilinear-upsample only for the LPIPS forward (which needs ≥ 224 input).
Rasterization cost is O(H·W) → ~3× cheaper. Loss surface shifts slightly toward
low-frequency content; defensible since LPIPS still carries the high-frequency term.

### E. Linearized render loss (research bet)
Render loss exists because rasterization is nonlinear in `(xyz, scale, rotation)`.
Linearize around GT once, training-time loss becomes a sparse matmul:

Offline, per `(object, cam_idx)`:
  1. `x0_gt.requires_grad_(True)`
  2. Render → backward `R.sum()`
  3. Store `J = ∂R/∂x0_gt` as a sparse tensor (per pixel depends on tens of Gaussians)

Online:
```
L_surrogate = |J @ (x0_pred - x0_gt)|_1 + alpha-equivalent + LPIPS-of-(J @ Δ)
```

First-order Taylor of the original objective around GT. Same gradient at
`pred == gt`; close to identical when `pred ≈ gt`. The regime that matters most for
render loss. Pilot validation: train 5–10k steps with the linearized variant vs. the
true renderer, compare downstream sample quality.

Trades runtime gsplat rasterization for offline Jacobian compute + cheap sparse matmul.
Render-loss steps approach the cost of regular MSE steps.

### F. Skip the rank-inverse on GT
For GT specifically, raw pre-rank-transform Gaussian parameters exist (they produced
the source `.ply`). Cache those in the CPU lazy cache instead of running the full
normalize → rank-inverse → sigmoid/exp pipeline at training time. Subsumed by (A) at
the render level but cheaper in storage if you don't want to cache full renders.

## Not pursuing (per user — known/obvious)

- Render-loss capping / temporal striding (#6 in original audit). "Do it less often" is
  already known. Reformulations that preserve the objective are in scope (Tier 3 above).
- Batch-size increase to consume spare memory. Affects effective LR / AdamW momentum stats.

## Expected combined ceiling

Tier 1 + best compile mode + Tier 2: **20–35% wall-clock** with no change to training math.
Dominant contributions: val cadence change + compile mode + sync removal.
