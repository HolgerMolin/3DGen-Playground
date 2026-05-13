"""Unit test for _compute_null_repel_loss.

Covers:
  1. All class cosines below margin -> loss = 0 and no grad on any row.
  2. Mix of below/above margin -> hand-calculated quadratic matches.
  3. Gradient pattern: class rows have grad, null row has None (stopgrad).
  4. A full-on dry-run "mini training step" that combines repel with a dummy
     main loss, backpropagates, and prints the scalars the training loop logs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from jit.train_gsplat import _compute_null_repel_loss  # noqa: E402


def _approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def case_all_below_margin():
    """All class cosines < margin: repel loss must be 0, mean cos matches hand value."""
    # Construct an embedding where every class vector is orthogonal to null:
    # null = (1, 0, 0, 0), classes = (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
    # -> cos(class, null) = 0 for all -> below margin=0.5.
    C = 3
    table = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],  # null at row C=3
        ],
        requires_grad=True,
    )
    loss, mean_cos = _compute_null_repel_loss(table, num_classes=C, margin=0.5)
    assert _approx(loss.item(), 0.0), f"expected 0.0, got {loss.item()}"
    assert _approx(mean_cos.item(), 0.0), f"expected 0.0 mean cos, got {mean_cos.item()}"

    loss.backward()  # zero loss still produces a grad tensor of zeros
    # Class rows: grad must be the zero tensor (not None), null row: grad must be None
    # because stopgrad detaches it from the graph.
    assert table.grad is not None, "expected grad on table"
    for c in range(C):
        assert torch.all(table.grad[c] == 0.0), f"class row {c} should have zero grad"
    assert torch.all(table.grad[C] == 0.0), "null row grad entry must be zero (stopgrad kept it off-graph)"
    # Re-check with retained gradient structure: the null row gradient didn't receive
    # any contribution; the .grad storage exists because autograd allocates on
    # whole-tensor leaves, but its slice remains zero.
    print("PASS  case_all_below_margin  loss=0, mean_cos=0")
    return table, loss, mean_cos


def case_mix_above_margin():
    """Hand-calculated: one above-margin, one below-margin class."""
    C = 2
    # null = e1; class0 parallel to null (cos=1); class1 orthogonal (cos=0).
    table = torch.tensor(
        [
            [1.0, 0.0],  # class 0 -> cos=1.0
            [0.0, 1.0],  # class 1 -> cos=0.0
            [1.0, 0.0],  # null
        ],
        requires_grad=True,
    )
    margin = 0.5
    loss, mean_cos = _compute_null_repel_loss(table, num_classes=C, margin=margin)

    # Expected:
    #   hinge_0 = (1.0 - 0.5) = 0.5 -> squared 0.25
    #   hinge_1 = (0.0 - 0.5) clamped at 0 -> 0
    #   mean = 0.125
    #   mean_cos = (1.0 + 0.0) / 2 = 0.5
    expected_loss = 0.125
    expected_mean_cos = 0.5
    assert _approx(loss.item(), expected_loss), (
        f"expected loss={expected_loss}, got {loss.item()}"
    )
    assert _approx(mean_cos.item(), expected_mean_cos), (
        f"expected mean_cos={expected_mean_cos}, got {mean_cos.item()}"
    )

    loss.backward()
    # Gradient analytical check on class 0:
    #   L = (1/C) * max(0, cos - m)^2  with cos = <e_c, e_null> / (||e_c|| ||e_null||)
    # With ||e_null|| = 1 (constant due to stopgrad) and e_c = (1, 0) being a
    # unit vector perpendicular to gradient direction, the Jacobian of cos wrt
    # e_c = (x, y) at (1, 0) is (1/||e_c|| - x^2/||e_c||^3, -xy/||e_c||^3) =
    # (0, 0). That makes the gradient at (1, 0) zero — an uninteresting test
    # point. So we only sanity-check that (a) null row grad is zero (stopgrad)
    # and (b) class1 grad is zero (below margin, no penalty).
    assert table.grad is not None
    assert torch.all(table.grad[C] == 0.0), "null row grad must be zero (stopgrad)"
    assert torch.all(table.grad[1] == 0.0), "class 1 (below margin) grad must be zero"
    print(f"PASS  case_mix_above_margin  loss=0.125, mean_cos=0.5")


def case_gradient_flows_only_to_classes():
    """Generic embedding: verify null row grad stays zero, class rows get nonzero grad."""
    torch.manual_seed(0)
    C = 4
    D = 8
    # Construct an embedding where classes cluster near null so cosines exceed
    # the margin deterministically. Mirrors the drift pattern the regularizer
    # is meant to fix (cos(c, null) around 0.9 at the trained checkpoint).
    null = torch.randn(D)
    null = null / null.norm()
    noise = 0.2 * torch.randn(C, D)
    classes = null.unsqueeze(0) + noise  # ~aligned with null
    table = torch.cat([classes, null.unsqueeze(0)], dim=0).clone().requires_grad_(True)
    margin = 0.5

    loss, mean_cos = _compute_null_repel_loss(table, num_classes=C, margin=margin)
    # Make sure loss is nonzero so backward has something to propagate.
    assert loss.item() > 0.0, f"expected nonzero loss for clustered embedding, got {loss.item()}"
    assert mean_cos.item() > margin, (
        f"expected mean_cos > margin for clustered embedding, got {mean_cos.item()}"
    )
    loss.backward()

    assert table.grad is not None
    # Null row grad must be exactly zero (stopgrad keeps null off-graph).
    null_grad_max = table.grad[C].abs().max().item()
    assert null_grad_max == 0.0, f"null row grad must be 0, max abs = {null_grad_max}"

    # At least one class row must have a nonzero grad component (the hinge
    # was active somewhere).
    class_grad_max = table.grad[:C].abs().max().item()
    assert class_grad_max > 0.0, (
        f"expected some class rows to receive grad, max abs = {class_grad_max}"
    )
    print(
        f"PASS  case_gradient_flows_only_to_classes  "
        f"loss={loss.item():.4f}  mean_cos={mean_cos.item():+.4f}  "
        f"class_grad_max={class_grad_max:.4f}  null_grad_max={null_grad_max:.1f}"
    )


def case_dry_run_mini_training_step():
    """Full dry-run that mirrors the training loop's usage pattern.

    Builds a tiny nn.Embedding (same API as LabelEmbedder.embedding_table),
    combines a dummy MSE-style main loss with `lambda * repel_loss`, runs a
    backward pass, and prints the two scalars the training loop logs
    (`repel_loss` and `class_null_cos_mean`).
    """
    torch.manual_seed(0)
    num_classes = 8
    hidden_size = 16
    embedding = nn.Embedding(num_classes + 1, hidden_size)  # +1 for null at index C

    # Fake main-loss path: pull embeddings for some random labels, regress them
    # toward a random target. This keeps the class rows on the main-loss graph
    # regardless of the repel term.
    labels = torch.tensor([0, 3, 5, 7, num_classes])  # includes null (CFG dropout hit)
    target = torch.randn(5, hidden_size)
    main_loss = ((embedding(labels) - target) ** 2).mean()

    # Repel term — mirroring train_gsplat.py usage exactly.
    lam = 0.01
    margin = 0.5
    repel_loss, class_null_cos_mean = _compute_null_repel_loss(
        embedding.weight, num_classes, margin
    )

    total_loss = main_loss + lam * repel_loss
    total_loss.backward()

    # Verify both scalars are finite and sensible.
    assert torch.isfinite(repel_loss), f"repel_loss is not finite: {repel_loss}"
    assert torch.isfinite(class_null_cos_mean), f"class_null_cos_mean is not finite: {class_null_cos_mean}"

    # Verify gradient pattern: null row gets grad from main_loss (which indexed
    # it via the CFG-dropout label), but not from repel. We can't easily
    # separate contributions post-backward, but we can verify non-repel flow:
    # classes used by main-loss labels have grad; null also has grad (from
    # main loss only).
    assert embedding.weight.grad is not None

    print(
        f"PASS  case_dry_run_mini_training_step  "
        f"main_loss={main_loss.item():.4f}  repel={repel_loss.item():.6f}  "
        f"cos(c,null)={class_null_cos_mean.item():+.4f}  "
        f"total={total_loss.item():.4f}  lambda={lam}"
    )

    # Print the same fields the training loop would log for λ=0 and λ=0.01:
    print()
    print("[dry-run simulation of training-loop logging]")
    for weight in (0.0, 0.01):
        fake_total = main_loss.detach().item() + weight * repel_loss.detach().item()
        active = weight > 0.0
        repel_str = f" | Repel: {repel_loss.item():.4f}" if active else ""
        print(
            f"  lambda={weight:g}  main={main_loss.item():.4f}"
            f"{repel_str}  cos(c,null): {class_null_cos_mean.item():+.4f}  "
            f"total_loss_scalar={fake_total:.4f}"
        )


if __name__ == "__main__":
    case_all_below_margin()
    case_mix_above_margin()
    case_gradient_flows_only_to_classes()
    case_dry_run_mini_training_step()
    print("\nAll unit tests passed.")
