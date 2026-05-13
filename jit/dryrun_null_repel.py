"""Dry-run of one training step with the null-repel loss active.

Builds a real JiT model (JiT-S/8, the smallest variant — keeps memory small),
mirrors the training loop's compute:
  - flow-matching main loss
  - _compute_null_repel_loss on the LabelEmbedder's embedding_table
  - sum with configurable lambda, backward, inspect gradients.

Runs the step twice: once with lambda=0.0 (baseline, no behavioral change) and
once with lambda=0.01. For each run we print the two scalars the training
loop logs (`repel` and `cos(c,null)`) and verify gradient flow matches
expectations (null row grad == baseline null row grad when repel is enabled,
because stopgrad keeps the regularizer off the null row).

No dataset required — runs on synthetic tensors.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GS_ROOT = os.path.join(REPO_ROOT, "submodules", "gaussian-splatting")
if GS_ROOT not in sys.path:
    sys.path.insert(0, GS_ROOT)

from jit.diffusion import create_diffusion  # noqa: E402
from jit.models import JiT_3DGS_models  # noqa: E402
from jit.train_gsplat import _compute_null_repel_loss, _sample_jit_timesteps  # noqa: E402


def build_model(num_classes: int, in_channels: int):
    return JiT_3DGS_models["JiT-S/8"](
        input_size=128,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=0.1,
        learn_sigma=False,
        gradient_checkpointing=False,
        aux_classifier=False,
        label_embed_init_std=0.1,
    )


def run_step(model, diffusion, x, y, *, lam: float, margin: float, num_classes: int):
    """Mirror the training loop's per-step compute. Returns scalars + grad snapshot."""
    model.zero_grad()

    t_value, t = _sample_jit_timesteps(
        x.shape[0], diffusion.num_timesteps, x.device, p_mean=-0.5, p_std=1.0
    )
    noise = torch.randn_like(x)
    loss_dict = diffusion.flow_matching_training_losses(
        model, x, t_value, t, model_kwargs=dict(y=y), noise=noise,
    )
    mse_loss = loss_dict["loss"].mean()

    repel_loss, class_null_cos_mean = _compute_null_repel_loss(
        model.y_embedder.embedding_table.weight, num_classes, margin
    )

    total_loss = mse_loss + lam * repel_loss
    total_loss.backward()

    emb_grad = model.y_embedder.embedding_table.weight.grad.detach().clone()
    return dict(
        mse=mse_loss.item(),
        repel=repel_loss.item(),
        cos=class_null_cos_mean.item(),
        total=total_loss.item(),
        emb_grad=emb_grad,
    )


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dry-run] device={device}")

    num_classes = 120  # matches JiT-XL training config
    in_channels = 14   # DC-only features
    B = 2

    model = build_model(num_classes=num_classes, in_channels=in_channels).to(device)
    diffusion = create_diffusion(
        timestep_respacing="",
        noise_schedule="squaredcos_cap_v2",
        learn_sigma=False,
        predict_xstart=True,
    )

    # DiT zero-init sets final_layer.linear.weight = 0, which (a) blocks
    # gradient flow to the trunk and (b) keeps the embedding grad pinned at 0
    # on the very first step. A mid-training checkpoint doesn't have this —
    # so simulate that by breaking zero-init before the dry-run.
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model.final_layer.linear.weight)
        model.final_layer.linear.weight.mul_(0.05)  # small scale, realistic early-train
        torch.nn.init.normal_(
            model.final_layer.adaLN_modulation[-1].weight, std=0.02
        )

        # Fresh init puts class embeddings near-orthogonal to null (cos ~ 0),
        # below the 0.5 margin → regularizer dormant. Force the geometry into
        # the drifted state observed on the real checkpoint (cos(c,null) ≈ 0.9)
        # so the hinge is active and we can verify repel actually moves classes.
        emb = model.y_embedder.embedding_table.weight  # (num_classes+1, D)
        D = emb.shape[1]
        null = torch.randn(D, device=device)
        null = null / null.norm()
        perturb = 0.1 * torch.randn(num_classes, D, device=device)
        drifted_classes = null.unsqueeze(0) + perturb  # ~aligned with null
        emb[:num_classes] = drifted_classes
        emb[num_classes] = null

    # Synthetic batch shaped like a normalized DC-only atlas.
    x = torch.randn(B, in_channels, 128, 128, device=device)
    y = torch.randint(0, num_classes, (B,), device=device, dtype=torch.long)

    # --- Step 1: lambda=0.0 (baseline, no behavioral change) ---
    print("\n[lambda=0.0] baseline: repel computed but contributes zero-weighted term")
    torch.manual_seed(42)  # fix RNG so the two runs below are comparable
    r0 = run_step(model, diffusion, x, y, lam=0.0, margin=0.5, num_classes=num_classes)
    print(
        f"  mse={r0['mse']:.4f}  repel={r0['repel']:.6f}  "
        f"cos(c,null)={r0['cos']:+.4f}  total={r0['total']:.4f}"
    )
    print(
        f"  emb_grad  class_rows_max={r0['emb_grad'][:num_classes].abs().max().item():.2e}  "
        f"null_row_max={r0['emb_grad'][num_classes].abs().max().item():.2e}"
    )

    # --- Step 2: lambda=0.01 (regularizer engaged) ---
    print("\n[lambda=0.01] regularizer engaged")
    torch.manual_seed(42)  # same RNG -> same mse_loss, so any difference in total comes from repel
    r1 = run_step(model, diffusion, x, y, lam=0.01, margin=0.5, num_classes=num_classes)
    print(
        f"  mse={r1['mse']:.4f}  repel={r1['repel']:.6f}  "
        f"cos(c,null)={r1['cos']:+.4f}  total={r1['total']:.4f}"
    )
    print(
        f"  emb_grad  class_rows_max={r1['emb_grad'][:num_classes].abs().max().item():.2e}  "
        f"null_row_max={r1['emb_grad'][num_classes].abs().max().item():.2e}"
    )

    # --- Verify gradient-flow properties ---
    print("\n[verify]")

    # MSE should match between runs (same RNG seed, same forward).
    mse_diff = abs(r0["mse"] - r1["mse"])
    assert mse_diff < 1e-5, f"baseline vs repel MSE differ: {mse_diff}"
    print(f"  mse matches across lambda=0 vs 0.01 (diff={mse_diff:.2e}) -> main loss untouched")

    # The null row gradient should be *identical* between lambda=0 and lambda=0.01
    # (because the regularizer detaches null — the only grad on null is from the
    # main loss, which is identical across runs).
    null_diff = (
        r1["emb_grad"][num_classes] - r0["emb_grad"][num_classes]
    ).abs().max().item()
    assert null_diff < 1e-6, (
        f"null row grad changed after adding repel (should be stopgrad'd): max delta {null_diff}"
    )
    print(f"  null-row grad identical across lambda=0 vs 0.01 (max delta={null_diff:.2e}) -> stopgrad works")

    # Class rows should differ — repel adds gradient to classes.
    class_diff = (
        r1["emb_grad"][:num_classes] - r0["emb_grad"][:num_classes]
    ).abs().max().item()
    assert class_diff > 0.0, (
        f"class row grads did not change after adding repel (unexpected): max delta {class_diff}"
    )
    print(f"  class-row grad differs across lambda=0 vs 0.01 (max delta={class_diff:.2e}) -> repel reaches classes")

    # cos(c, null) diagnostic should be identical across runs (weights unchanged pre-backward).
    cos_diff = abs(r0["cos"] - r1["cos"])
    assert cos_diff < 1e-5, f"diagnostic cos mismatch: {cos_diff}"
    print(f"  cos(c,null) diagnostic identical (diff={cos_diff:.2e}) -> pure-function of weights")

    print("\nDry-run passed.  Training-loop log fields for this step:")
    print(f"  [lambda=0.00] Repel (not shown) | cos(c,null): {r0['cos']:+.4f}")
    print(f"  [lambda=0.01] Repel: {r1['repel']:.4f} | cos(c,null): {r1['cos']:+.4f}")


if __name__ == "__main__":
    main()
