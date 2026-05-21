# Auxiliary classification loss — findings

Follow-up to `class_conditioning.md` section 5 ("Auxiliary classification loss"). The aux head was implemented and enabled live on a JiT-B/8 run resuming from step 120,000, with `aux_classifier_weight=0.01` via `overrides.yaml`. This doc records the ~1,200-step observation window (120,000 → 121,200) and what it does and does not tell us about training.

## Setup

- **Model:** JiT-B/8, resumed from checkpoint at step 120,000 (120k steps of prior class-conditional diffusion training with `c = t + y` additive conditioning).
- **Head:** `nn.Linear(hidden_size, num_classes)` over mean-pooled transformer tokens, before `final_layer`. Xavier-uniform init via the generic `_basic_init` in `jit/models.py:391`.
- **Loss:** Cross-entropy on un-dropped labels (`drop_mask` excluded so CFG-dropped rows don't train the head against the null slot). `jit/train_gsplat.py:1183–1197`.
- **Weight:** `aux_classifier_weight=0.01`, hot-loaded from `jit/configs/overrides.yaml`.
- **Other conditioning changes active:** `class_balanced_sampler=true`, `class_dropout_prob=0.1`. Render loss on with `render_w=0.1, alpha_w=0.1, lpips_w=0.01`. `P_mean=0.0`.

## Observations

Raw log from the first ~1,200 post-enable steps:

```
step    MSE     Render_L1  Alpha_L1  LPIPS   Aux     GradNorm
120200  0.4965  0.0438     0.0683    0.2791  10.7144 0.7708
120400  0.4762  0.0387     0.0562    0.2614   1.0129 0.3217
120600  0.4708  0.0384     0.0544    0.2580   0.2380 0.1933
120800  0.4733  0.0370     0.0531    0.2543   0.1059 0.1383
121000  0.4724  0.0369     0.0525    0.2544   0.0607 0.1016
121200  0.4597  0.0368     0.0518    0.2512   0.0455 0.0910
```

Summary:

- **Aux loss collapsed 99.6% in 1,000 steps** (10.71 → 0.046). Uniform random on 48 classes is `ln(48) ≈ 3.87`; the initial 10.7 reflects xavier-init on a linear head fed large-magnitude pooled features from a trunk that has already trained for 120k steps.
- **MSE fell ~7%** (0.4965 → 0.4597) over the same window.
- **Render losses fell 7–16%** (continuing the prior trajectory — no visible inflection at step 120,000).
- **GradNorm tracks aux** (0.77 → 0.09), confirming aux loss dominated the gradient scale early and decayed alongside it.

## Interpretation

### The aux loss improvement is the classifier fitting, not the trunk improving

Evidence:

1. **Linear probe convergence is fast and trunk-independent.** Cross-entropy on linearly separable features is convex in the head's weights. A fresh `nn.Linear` converges orders of magnitude regardless of trunk state.
2. **Gradient-share asymmetry.** The head receives 100% of its own gradient. The trunk receives aux gradients scaled by `aux_classifier_weight=0.01`, mixed against MSE (~0.47) and render losses — aux's share of the trunk update is ~1% and shrinking as aux collapses.
3. **Loss scales don't match.** Aux fell 99.6%; MSE fell 7%; render fell 7–16%. Reorganizing the trunk enough to solve classification would move MSE far more than was observed. Most of what moved was the head.
4. **The trunk was already class-aware at step 120,000.** Class labels enter via `c = t + y` at every block. 120k prior steps of class-conditional diffusion embed class identity into pooled tokens. The head just learned the readout of what was already there.

### Why this lever is probably weak for this run

1. **No new information for the trunk.** A diagnostic probe confirms class info already exists in the features. It doesn't inject new signal.
2. **Near-zero CE → near-zero gradient.** Once `p ≈ onehot`, `∂CE/∂trunk ≈ 0` on easy examples. All remaining signal comes from a few hard samples, which may or may not be the ones generation cares about.
3. **Discriminative ≠ generative.** Classification rewards features that separate classes. Generation needs features that reconstruct within-class variation. Pressuring the trunk to be more classifier-like trades generative capacity for discriminative capacity.
4. **Conditioning mechanism, not conditioning readout, is the real bottleneck.** The CFG-collapse analysis in `class_conditioning.md` points to `c = t + y` and zero-init adaLN as the root causes. An aux head on top of the trunk doesn't change how `y` flows into modulation — it just scores what comes out.

### When aux classification *would* be useful (and isn't, here)

- **Cold start:** warm-starting a class-conditional model. Not applicable — the run resumed from step 120,000.
- **Noisy/sparse labels:** forcing the model to learn despite weak supervision. Not applicable — the 48 labels from `object_labels/object_to_class.json` are clean, and the trunk already learned them.
- **Interpretability probe:** measuring *what* the trunk encodes without depending on generation quality. Legitimate, but that's a diagnostic, not a training improvement.

## What the weight is actually doing

At steady state `aux_loss ≈ 0.046`:

| weight | aux contribution | vs MSE (~0.47) | trunk gradient share |
|---|---|---|---|
| 0.01 (current) | 4.6e-4 | ~0.1% | negligible |
| 0.1 | 4.6e-3 | ~1% | ~1% |
| 1.0 | 4.6e-2 | ~10% | ~10% (but most ≈0 due to ∂CE vanishing) |

Bumping the weight amplifies gradients on the few hard-example rows, which may or may not align with reconstruction. Bumping to 1.0 directly is not recommended — initial aux was 10.7, and at weight=1.0 that would have swamped MSE (0.5) by ~20× at the moment of enabling. The weight is only reasonable once aux has already converged.

## Recommendation

**Turn aux off, or leave at 0.01 as a cheap monitor.** It is not a relevant lever for this run's quality problem. Redirect effort to mechanisms that actually change how class information flows through the network or how training signal is weighted:

1. **Split `t` and `y` conditioning paths** (`class_conditioning.md` §4). Gives class its own modulation budget rather than competing additively with timestep. Biggest lever in this family but requires a retrain.
2. **Noise-schedule / `P_mean` sweep.** Currently `P_mean=0.0` per overrides — worth trying `P_mean∈{-0.5, -0.25, 0.0, +0.25}` on short probes. The high-noise regime is under-trained per `jit_training_evaluation.md` §2.
3. **Render weight balance.** LPIPS at 0.25 is the slowest-moving component; the gradient balance between MSE and LPIPS is documented in `jit_training_evaluation.md` §3.
4. **Per-channel loss weighting.** If specific 3DGS feature channels (opacity, scale, rotation) are under-learned, the flat MSE hides it — `channel_loss_weights` already plumbed through `flow_matching_training_losses`.
5. **CFG scale at inference**, not training — if class separation is the user-visible symptom.

## Quick decision rule for future "should I try loss X" questions

Before adding another loss term, check:

- **Is the trunk already encoding what the loss measures?** If a linear probe converges in <1k steps, yes — the loss will become a readout, not a training signal. (Aux here: failed this check.)
- **Does the gradient survive once the loss converges?** Cross-entropy at near-perfect accuracy has vanishing gradient. (Aux here: failed this check.)
- **Does it change the *mechanism* or just add a *readout*?** Splitting adaLN changes mechanism. A pooled classifier on top is a readout. (Aux here: readout only.)

If all three answer "no," the loss is more likely to be a monitor than a lever.

## Files touched by the aux experiment

| File | What changed |
|---|---|
| `jit/models.py` | Added `aux_classifier` arg, `nn.Linear(hidden, num_classes)` head, stashes `_aux_logits` during forward at `jit/models.py:468–471` |
| `jit/train_gsplat.py` | Pre-samples `drop_mask` so aux ignores CFG-dropped rows; CE on un-dropped logits; adds `aux_classifier_weight` to `total_loss`; logs `Aux` in the per-step printout |
| `jit/configs/jit_train_gsplat.yaml` | `aux_classifier: true`, `aux_classifier_weight: 0.01`, `label_embed_init_std: 0.02` |
| `jit/configs/overrides.yaml` | `aux_classifier_weight` hot-reloadable |

## Status

- [x] Aux head implemented and validated (loss converges → trunk features are linearly class-separable).
- [ ] **Recommendation: disable or leave at 0.01 as a monitor.** Not a priority lever for this training run. Revisit only if a future experiment needs a discriminative probe.
