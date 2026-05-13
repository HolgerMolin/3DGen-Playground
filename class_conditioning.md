# Class conditioning — diagnosis and remaining fixes

Class conditioning in the JiT-XL/8 run at `output/jit_JiT-XL/8_results_gsplat/` is collapsed: at step 200k (previous run in `jit_training_evaluation.md`), `‖pred(x_t, y_A) − pred(x_t, null)‖_RMS ≈ 0.012` and `‖pred(x_t, y_A) − pred(x_t, y_B)‖_RMS ≈ 0.014` — i.e. swapping the class has about the same effect as dropping it, at ~1.3% of activation norm. CFG is mathematically a no-op on that checkpoint.

## Root causes (code-level)

- `c = t + y` addition at `jit/models.py:450` — class embedding competes with timestep embedding for modulation magnitude. Timestep is always informative; class signal gets swamped.
- adaLN-Zero init at `jit/models.py:410–417` — `adaLN_modulation[-1]` weight and bias start at zero, so class has literally zero effect on the output for the first few optimizer steps. Training must lift adaLN off zero before class can influence anything.
- `class_dropout_prob=0.2` (old default) — 20% of updates train the null-class path, rewarding a "ignore class" local minimum.
- `LabelEmbedder` init at std=0.02 (`jit/models.py:402`) — fragile signal, nothing big to grab onto.
- Class imbalance ~1000:1 across 87 classes (class 33 alone is ~13% of data).

## Done in this branch

- [x] **(1) Class-balanced sampler** — `_build_class_balanced_sampler` in `jit/train_gsplat.py`. `WeightedRandomSampler` with inverse-frequency weights, per-rank generator (`seed + rank`), `replacement=True`, `num_samples = len(dataset)`. Enabled via `--class_balanced_sampler` / `class_balanced_sampler: true` (YAML default on). Verified on a 1000:10:1 synthetic skew: output frequencies → 0.336 / 0.341 / 0.322.
- [x] **(2) Drop `class_dropout_prob` 0.2 → 0.1** — YAML. Argparse default was already 0.1; the YAML was overriding it upward.

## Remaining fixes (ranked by impact/cost)

### 3. Replace class IDs with text (caption) conditioning — biggest lever

The dataloader already plumbs captions (`dataloaders/fast_3dgen_loader.py:199,214`), and `.env` has `CAPTIONS_PATH`. A small frozen text encoder (SigLIP-base or CLIP ViT-L) gives a dense, semantically structured conditioning signal that eliminates the 87-bucket granularity problem and handles the long tail for free.

**Implementation sketch:**

- Swap `LabelEmbedder` (`jit/models.py:227`) for a `TextEmbedder` that:
  - Runs frozen text encoder once per caption (cache embeddings per object ID to avoid recomputing — captions don't change).
  - Projects `text_dim → hidden_size` via a small learned MLP.
  - Keeps CFG dropout by randomly replacing with a learned null-text embedding.
- Dataloader path: `Class3DGenDataset` already returns captions via the wrapped base dataset — plumb through training batches alongside labels (or replace labels entirely).
- Precompute caption embeddings in a one-time pass, store as `.pt` keyed by object ID. Load into RAM at dataset init.

**Expected impact:** biggest single quality lift for conditional generation. Text is strictly more information than `∈ [0, 87)`.

**Cost:** moderate — one new model head, one dataloader change, one preprocessing script. ~half a day.

### 4. Split `t` and `y` conditioning paths

Current `c = t + y` forces one adaLN head to carry both. Remove the competition:

**Option (a) — separate adaLN heads, sum post-modulation:**
```python
shift_t, scale_t, gate_t, ... = self.adaLN_t(t).chunk(6, -1)
shift_y, scale_y, gate_y, ... = self.adaLN_y(y).chunk(6, -1)
shift, scale, gate = shift_t + shift_y, scale_t + scale_y, gate_t + gate_y
```

**Option (b) — concat and double-width adaLN:** cleaner, one-line change:
```python
self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(2 * hidden, 6 * hidden))
# in forward: c = torch.cat([t, y], dim=-1)
```

Keep the zero-init on `adaLN_modulation[-1]`. Only the input width changes.

**Expected impact:** class gets its own parameter budget for modulation. Combined with (1)+(2), probably enough to get CFG working.

**Cost:** ~30 lines. Requires retraining from scratch (shape change).

### 5. Auxiliary classification loss

Add a small classification head on pooled tokens, cross-entropy against `y`. Forces class embeddings (and the trunk) to carry class-discriminative features independent of the denoising MSE.

```python
# In model.__init__
self.aux_classifier = nn.Linear(hidden_size, num_classes)

# In forward: after blocks, before final_layer
cls_logits = self.aux_classifier(x.mean(dim=1))  # pool tokens
# return (x_pred, cls_logits)

# In train step
aux_loss = F.cross_entropy(cls_logits, y_clean)  # use un-dropped labels
total_loss = mse_loss + 0.01 * aux_loss
```

Weight ~0.01 so it doesn't dominate. Important: use the un-dropped labels (the actual class), not the post-CFG-dropout version. Drops the aux loss for samples that were CFG-dropped.

**Expected impact:** guarantees embeddings carry usable class signal. Works even if (4) isn't done.

**Cost:** ~20 lines, no shape changes, can be enabled with a flag.

### 6. Raise embedding init std 0.02 → 0.1

One line at `jit/models.py:402`:
```python
nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.1)
```

Gives class signal a larger starting magnitude so it's not dwarfed by timestep during the first few thousand updates, when adaLN is still lifting off zero.

**Expected impact:** small on its own, useful combined with (4)/(5). Low risk.

### 7. Fold tail classes into "other"

Only do this if (1)+(3) don't produce separation on tail classes. Concretely:

- Count samples per class in `object_labels/object_to_class.json`.
- Pick a threshold (e.g. ≥500 samples per class → keep; below → remap to a shared "other" ID).
- Rewrite the class map to a new file; update `num_classes` implicitly via `max(v) + 1`.

**Expected impact:** cleaner per-class signal if many tail classes are too small to learn individually. Negative if done prematurely — destroys information.

**Cost:** one preprocessing script.

## Diagnostics to add before / during training

These are cheap and tell you whether any of the above is actually working. Without them, you're flying blind:

```python
# Every N checkpoints, on a fixed held-out batch x_t at a fixed t (say t=0.3):
with torch.no_grad():
    pred_cond_A = model(x_t, t, y=class_A)
    pred_cond_B = model(x_t, t, y=class_B)  # different class
    pred_null   = model(x_t, t, y=null)

    # (a) Conditional vs unconditional — measures whether CFG will do anything
    cfg_signal = (pred_cond_A - pred_null).square().mean().sqrt() / pred_cond_A.square().mean().sqrt()

    # (b) Class-specific — measures whether conditioning discriminates between classes
    class_signal = (pred_cond_A - pred_cond_B).square().mean().sqrt() / pred_cond_A.square().mean().sqrt()

    logger.info(f"[cond] cfg_signal={cfg_signal:.4f}  class_signal={class_signal:.4f}")
```

**Interpretation:**

- `cfg_signal < 0.01` → conditioning collapsed, CFG is a no-op.
- `cfg_signal > 0.1` and `class_signal > 0.05` → conditioning works, CFG will help.
- `cfg_signal > 0.1` but `class_signal ≈ 0` → model uses "any class vs null" but doesn't discriminate between classes. Still a problem.

## Order of operations for the next full run

1. Start with (1) + (2) (already in this branch) + (5) auxiliary classification loss (cheap, reversible).
2. Add the cond-signal diagnostic. Run to ~20k opt steps.
3. If `cfg_signal` is still < 0.05, add (4) split modulation and restart.
4. Separately, as a bigger experiment: (3) text conditioning. Do this in its own run — it's a different enough change that mixing it with the above makes attribution impossible.
5. Only add (7) tail-class folding after (3) if conditioning still fails on low-frequency classes.
