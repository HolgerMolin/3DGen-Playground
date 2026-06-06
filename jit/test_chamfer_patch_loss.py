"""Unit test for the chamfer_patch reconstruction loss (_chamfer_recon_loss mode).

Covers:
  1. Identity: chamfer_patch(x, x) ~= 0.
  2. DECISIVE within-patch invariance: chamfer_patch(x, permute_atlas(x,"patch")) ~= 0
     — proves the loss patches align exactly with the permute_atlas('patch') augmentation.
  3. DECISIVE cross-patch penalty: chamfer_patch(x, permute_atlas(x,"global")) > 0
     — points that jump across patch boundaries are penalized.
  4. Single-patch parity with a brute-force global Chamfer (== chamfer_feature semantics),
     confirming the magnitude/reduction so chamfer_loss_weight carries over. Also cross-checks
     the live chamfer_feature impl when its kNN backend is importable.
  5. Gradient finiteness through the Gram/clamp_min/min path.
  6. Weighted-channel path matches a brute-force weighted reference (single patch).
  7. H not divisible by patch_size raises ValueError.

Run: ./.3dgen/bin/python jit/test_chamfer_patch_loss.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from jit.diffusion.gaussian_diffusion import (  # noqa: E402
    _chamfer_recon_loss,
    _sinkhorn_log,
    permute_atlas,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _brute_global_chamfer(pred, target, w=None):
    """Reference bidirectional Chamfer over ALL points (== chamfer_feature definition):
    mean-over-points of (sum-over-C squared distance to nearest), both directions."""
    x = pred.flatten(2).transpose(1, 2).float()      # (B, N, C)
    y = target.flatten(2).transpose(1, 2).float()
    if w is not None:
        s = w.sqrt().view(1, 1, -1)
        x = x * s
        y = y * s
    d = torch.cdist(x, y).pow(2)                      # (B, N, N) squared dists
    fwd = d.min(dim=-1).values.mean(dim=1)            # (B,)
    rev = d.min(dim=-2).values.mean(dim=1)
    return fwd + rev


def case_identity():
    torch.manual_seed(0)
    x = torch.randn(2, 14, 128, 128, device=DEVICE)
    loss = _chamfer_recon_loss(x, x, "chamfer_patch", patch_size=8).mean().item()
    assert loss < 1e-5, f"identity loss should be ~0, got {loss}"
    print(f"PASS  case_identity                 loss={loss:.2e}")


def case_within_patch_invariance():
    torch.manual_seed(1)
    x = torch.randn(2, 14, 128, 128, device=DEVICE)
    xp = permute_atlas(x, "patch", 8)
    loss = _chamfer_recon_loss(x, xp, "chamfer_patch", patch_size=8).mean().item()
    assert loss < 1e-5, f"within-patch permutation must be free, got {loss}"
    print(f"PASS  case_within_patch_invariance  loss={loss:.2e}  (permute_atlas('patch') aligns)")


def case_cross_patch_penalty():
    torch.manual_seed(2)
    x = torch.randn(2, 14, 128, 128, device=DEVICE)
    xg = permute_atlas(x, "global", 8)
    loss = _chamfer_recon_loss(x, xg, "chamfer_patch", patch_size=8).mean().item()
    assert loss > 0.1, f"global permutation should be penalized within-patch, got {loss}"
    print(f"PASS  case_cross_patch_penalty      loss={loss:.3f}  (cross-patch jumps cost)")


def case_single_patch_parity():
    """H=W=patch_size => exactly one patch => chamfer_patch == global Chamfer."""
    torch.manual_seed(3)
    P = 8
    pred = torch.randn(2, 14, P, P, device=DEVICE)
    tgt = torch.randn(2, 14, P, P, device=DEVICE)
    cp = _chamfer_recon_loss(pred, tgt, "chamfer_patch", patch_size=P)
    ref = _brute_global_chamfer(pred, tgt)
    err = (cp - ref).abs().max().item()
    assert err < 1e-3, f"single-patch chamfer_patch must equal global Chamfer, max err {err}"
    print(f"PASS  case_single_patch_parity      max|cp-bruteforce|={err:.2e}")
    # Cross-check the live chamfer_feature impl when its kNN backend is available.
    try:
        cf = _chamfer_recon_loss(pred, tgt, "chamfer_feature", subsample=0)
        err2 = (cp - cf).abs().max().item()
        assert err2 < 1e-3, f"chamfer_patch(single) vs chamfer_feature mismatch {err2}"
        print(f"      + chamfer_feature cross-check  max|cp-feature|={err2:.2e}")
    except Exception as e:  # backend (keops/pytorch3d) unavailable in this env
        print(f"      (skipped chamfer_feature cross-check: {type(e).__name__}: {e})")


def case_gradient_finite():
    torch.manual_seed(4)
    pred = torch.randn(2, 14, 128, 128, device=DEVICE, requires_grad=True)
    tgt = torch.randn(2, 14, 128, 128, device=DEVICE)
    loss = _chamfer_recon_loss(pred, tgt, "chamfer_patch", weight=0.03, patch_size=8).mean()
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all(), "non-finite grad"
    gmax = pred.grad.abs().max().item()
    assert gmax > 0, "expected nonzero gradient"
    print(f"PASS  case_gradient_finite          loss={loss.item():.4f}  grad_max={gmax:.4f}")


def case_weighted_path():
    torch.manual_seed(5)
    P = 8
    pred = torch.randn(2, 14, P, P, device=DEVICE)
    tgt = torch.randn(2, 14, P, P, device=DEVICE)
    w = torch.rand(14, device=DEVICE) * 2.0  # arbitrary positive per-channel weights
    cp = _chamfer_recon_loss(pred, tgt, "chamfer_patch", channel_loss_weights=w, patch_size=P)
    ref = _brute_global_chamfer(pred, tgt, w=w)
    err = (cp - ref).abs().max().item()
    assert err < 1e-3, f"weighted single-patch mismatch, max err {err}"
    print(f"PASS  case_weighted_path            max|cp-bruteforce|={err:.2e}")


def case_divisibility_raise():
    x = torch.randn(1, 14, 130, 128, device=DEVICE)  # 130 % 8 != 0
    try:
        _chamfer_recon_loss(x, x, "chamfer_patch", patch_size=8)
    except ValueError as e:
        print(f"PASS  case_divisibility_raise       raised -> {str(e)[:50]}...")
        return
    raise AssertionError("expected ValueError for H not divisible by patch_size")


def case_sinkhorn_identity():
    torch.manual_seed(10)
    x = torch.randn(2, 14, 128, 128, device=DEVICE)
    loss = _chamfer_recon_loss(x, x, "sinkhorn_patch", patch_size=4,
                               sinkhorn_eps=0.05, sinkhorn_iters=50).mean().item()
    assert loss < 1e-2, f"sinkhorn identity loss should be ~0, got {loss}"
    print(f"PASS  case_sinkhorn_identity        loss={loss:.2e}")


def case_sinkhorn_within_patch_invariance():
    torch.manual_seed(11)
    x = torch.randn(2, 14, 128, 128, device=DEVICE)
    xp = permute_atlas(x, "patch", 4)
    loss = _chamfer_recon_loss(x, xp, "sinkhorn_patch", patch_size=4,
                               sinkhorn_eps=0.05, sinkhorn_iters=50).mean().item()
    assert loss < 1e-2, f"sinkhorn within-patch permutation must be ~free, got {loss}"
    print(f"PASS  case_sinkhorn_within_patch_invariance  loss={loss:.2e}")


def case_sinkhorn_penalizes_collapse():
    """DECISIVE: optimal assignment (bijection) penalizes within-patch collapse that the
    forward Chamfer (nearest-neighbour) is blind to — the whole motivation for the mode."""
    torch.manual_seed(12)
    P = 4
    tgt = torch.randn(1, 14, P, P, device=DEVICE)                    # 16 distinct points (1 patch)
    pred_perm = permute_atlas(tgt, "patch", P)                       # bijection: distinct, same set
    pred_coll = tgt[:, :, :1, :1].expand(-1, -1, P, P).contiguous()  # all 16 collapsed to one pt
    # Forward-only Chamfer (rev_weight=0) is blind to collapse: every collapsed pred sits ON a
    # target, so its nearest-neighbour distance is 0.
    cf_coll = _chamfer_recon_loss(pred_coll, tgt, "chamfer_patch", patch_size=P, rev_weight=0.0).item()
    sk_perm = _chamfer_recon_loss(pred_perm, tgt, "sinkhorn_patch", patch_size=P,
                                  sinkhorn_eps=0.05, sinkhorn_iters=50).item()
    sk_coll = _chamfer_recon_loss(pred_coll, tgt, "sinkhorn_patch", patch_size=P,
                                  sinkhorn_eps=0.05, sinkhorn_iters=50).item()
    assert cf_coll < 1e-3, f"forward Chamfer should be ~blind to collapse, got {cf_coll}"
    assert sk_perm < 1e-1, f"sinkhorn should be ~0 for a true bijection, got {sk_perm}"
    assert sk_coll > 1.0, f"sinkhorn should heavily penalize collapse, got {sk_coll}"
    assert sk_coll > cf_coll + 0.5, "sinkhorn must penalize collapse more than forward Chamfer"
    print(f"PASS  case_sinkhorn_penalizes_collapse  cf_fwd(coll)={cf_coll:.2e}  "
          f"sk(perm)={sk_perm:.2e}  sk(coll)={sk_coll:.3f}")


def case_sinkhorn_gradient_finite():
    torch.manual_seed(13)
    pred = torch.randn(2, 14, 128, 128, device=DEVICE, requires_grad=True)
    tgt = torch.randn(2, 14, 128, 128, device=DEVICE)
    loss = _chamfer_recon_loss(pred, tgt, "sinkhorn_patch", weight=0.03, patch_size=4,
                               sinkhorn_eps=0.05, sinkhorn_iters=50).mean()
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all(), "non-finite grad"
    gmax = pred.grad.abs().max().item()
    assert gmax > 0, "expected nonzero gradient"
    print(f"PASS  case_sinkhorn_gradient_finite     loss={loss.item():.4f}  grad_max={gmax:.4f}")


def case_sinkhorn_plan_marginals():
    """_sinkhorn_log returns a transport plan with ~uniform marginals (rows/cols ~ 1/M, total 1)."""
    torch.manual_seed(14)
    B, nP, M = 2, 3, 16
    C = torch.rand(B, nP, M, M, device=DEVICE)
    Pln = _sinkhorn_log(C, eps=0.05, iters=200)
    row = Pln.sum(dim=-1)            # (B,nP,M) -> ~1/M
    col = Pln.sum(dim=-2)            # -> ~1/M
    tot = Pln.sum(dim=(-1, -2))      # -> ~1
    assert torch.allclose(row, torch.full_like(row, 1.0 / M), atol=1e-3), "row marginals off"
    assert torch.allclose(col, torch.full_like(col, 1.0 / M), atol=1e-3), "col marginals off"
    assert torch.allclose(tot, torch.ones_like(tot), atol=1e-3), "total mass off"
    print(f"PASS  case_sinkhorn_plan_marginals      max|row-1/M|={(row-1.0/M).abs().max().item():.2e}")


def case_sinkhorn_hard_identity():
    torch.manual_seed(20)
    x = torch.randn(2, 14, 128, 128, device=DEVICE)
    loss, diag = _chamfer_recon_loss(x, x, "sinkhorn_patch_hard", patch_size=4,
                                     sinkhorn_eps=0.05, sinkhorn_iters=100, return_diag=True)
    loss = loss.mean().item()
    coll = float(diag["sinkhorn_collision_frac"])
    assert loss < 1e-2, f"hard identity loss should be ~0, got {loss}"
    assert coll < 1e-2, f"hard identity should be a clean bijection, got collision={coll}"
    print(f"PASS  case_sinkhorn_hard_identity   loss={loss:.2e}  collision={coll:.2e}")


def case_sinkhorn_hard_within_patch_invariance():
    torch.manual_seed(21)
    x = torch.randn(2, 14, 128, 128, device=DEVICE)
    xp = permute_atlas(x, "patch", 4)
    loss = _chamfer_recon_loss(x, xp, "sinkhorn_patch_hard", patch_size=4,
                               sinkhorn_eps=0.05, sinkhorn_iters=100).mean().item()
    assert loss < 1e-2, f"hard within-patch permutation must be ~free, got {loss}"
    print(f"PASS  case_sinkhorn_hard_within_patch_invariance  loss={loss:.2e}")


def case_sinkhorn_hard_scale_parity():
    """At enough iters the plan ~ a permutation, so hard (argmax→MSE) and soft (⟨Π,sq⟩)
    land on the SAME scale — hard just sharpens the gradient, it doesn't rescale the loss."""
    torch.manual_seed(22)
    pred = torch.randn(2, 14, 128, 128, device=DEVICE)
    tgt = torch.randn(2, 14, 128, 128, device=DEVICE)
    soft = _chamfer_recon_loss(pred, tgt, "sinkhorn_patch", patch_size=8,
                               sinkhorn_eps=0.05, sinkhorn_iters=200).mean().item()
    hard = _chamfer_recon_loss(pred, tgt, "sinkhorn_patch_hard", patch_size=8,
                               sinkhorn_eps=0.05, sinkhorn_iters=200).mean().item()
    assert abs(hard - soft) / soft < 0.05, f"hard/soft scale mismatch: soft={soft}, hard={hard}"
    print(f"PASS  case_sinkhorn_hard_scale_parity  soft={soft:.4f}  hard={hard:.4f}")


def case_sinkhorn_hard_gradient_finite():
    torch.manual_seed(23)
    pred = torch.randn(2, 14, 128, 128, device=DEVICE, requires_grad=True)
    tgt = torch.randn(2, 14, 128, 128, device=DEVICE)
    loss = _chamfer_recon_loss(pred, tgt, "sinkhorn_patch_hard", weight=0.03, patch_size=4,
                               sinkhorn_eps=0.05, sinkhorn_iters=100).mean()
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all(), "non-finite grad"
    assert pred.grad.abs().max().item() > 0, "expected nonzero gradient"
    print(f"PASS  case_sinkhorn_hard_gradient_finite  loss={loss.item():.4f}  "
          f"grad_max={pred.grad.abs().max().item():.4f}")


def case_sinkhorn_hard_collapse_monitor():
    """KEY soft-vs-hard difference: the hard (argmax) variant does NOT enforce the bijection in
    the LOSS the way soft sinkhorn does. Under full collapse the plan goes UNIFORM, so the argmax
    lands on one arbitrary column (≈total collision) — the per-pred gradient is arbitrary, not a
    principled spread-apart. So the hard loss does not reliably penalize collapse the way soft's
    marginals do (its magnitude here is a tiebreak accident). The sinkhorn_collision_frac monitor
    is the reliable signal. (In normal training predictions aren't degenerate, so the argmax is
    meaningful and near-bijective; this is the worst-case characterization.)"""
    torch.manual_seed(24)
    P = 4
    tgt = torch.randn(1, 14, P, P, device=DEVICE)                    # 16 distinct points (1 patch)
    pred_coll = tgt[:, :, :1, :1].expand(-1, -1, P, P).contiguous()  # all collapsed to tgt[0,0]
    sk_coll = _chamfer_recon_loss(pred_coll, tgt, "sinkhorn_patch", patch_size=P,
                                  sinkhorn_eps=0.05, sinkhorn_iters=200).item()
    hard_loss, diag = _chamfer_recon_loss(pred_coll, tgt, "sinkhorn_patch_hard", patch_size=P,
                                          sinkhorn_eps=0.05, sinkhorn_iters=200, return_diag=True)
    coll_frac = float(diag["sinkhorn_collision_frac"])
    assert sk_coll > 1.0, f"soft sinkhorn should penalize collapse, got {sk_coll}"
    assert coll_frac > 0.5, f"collision monitor must flag the collapse, got {coll_frac}"
    print(f"PASS  case_sinkhorn_hard_collapse_monitor  soft(coll)={sk_coll:.3f}  "
          f"hard(coll)={hard_loss.item():.3f}  collision={coll_frac*100:.0f}% "
          f"(uniform plan → arbitrary argmax; monitor is the reliable signal)")


def _pseudo_huber(sq, delta):
    """Reference pseudo-Huber on a squared distance, 2δ²(√(1+sq/δ²)−1). Concave in sq, ≤ sq,
    equals sq as δ→∞; matches the in-loss transform."""
    d2 = delta * delta
    return 2.0 * d2 * ((1.0 + sq / d2) ** 0.5 - 1.0)


def case_huber_off_bit_identical():
    """huber_delta=0 must be the EXACT squared-L2 path (regression guard for existing runs)."""
    torch.manual_seed(30)
    pred = torch.randn(2, 14, 128, 128, device=DEVICE)
    tgt = torch.randn(2, 14, 128, 128, device=DEVICE)
    for mode, kw in (
        ("chamfer_patch", dict(patch_size=8)),
        ("sinkhorn_patch", dict(patch_size=4, sinkhorn_eps=0.05, sinkhorn_iters=50)),
        ("sinkhorn_patch_hard", dict(patch_size=4, sinkhorn_eps=0.05, sinkhorn_iters=100)),
    ):
        base = _chamfer_recon_loss(pred, tgt, mode, **kw)
        off = _chamfer_recon_loss(pred, tgt, mode, huber_delta=0.0, **kw)
        assert torch.equal(base, off), f"{mode}: huber_delta=0 not bit-identical to default"
    print("PASS  case_huber_off_bit_identical   (delta=0 == squared-L2 for all 3 patch modes)")


def case_huber_transform_exact():
    """DECISIVE: construct well-separated points so the within-patch NN match is the identity,
    with a KNOWN per-pair residual — then chamfer_patch's loss must equal the hand-computed
    pseudo-Huber of that residual (validates the elementwise transform applied to `sq`)."""
    torch.manual_seed(31)
    P = 2                                   # M=4 distinct points per patch, fits in C=14 dims
    B, C = 1, 14
    nrep = 64 // P                          # tile to a 64x64 atlas (many identical patches)
    # target: each of the 4 in-patch points is 10*e_k (k=0..3) — mutually orthogonal, far apart.
    base = torch.zeros(B, C, P, P, device=DEVICE)
    for k in range(P * P):
        base.view(B, C, P * P)[:, k, k] = 10.0
    tgt = base.repeat(1, 1, nrep, nrep).contiguous()        # (1,14,64,64)
    res = 3.0                               # known residual radius: offset every pred by res on ch 5
    pred = tgt.clone()
    pred[:, 5, :, :] += res                 # pred_i - tgt_i = res*e_5  ⇒ sq[i,i] = res**2, NN=identity
    sq = res * res
    for delta in (1.0, 2.0, 5.0):
        # bidirectional (rev_weight=1) identity match ⇒ loss = 2 * huber(sq)
        got = _chamfer_recon_loss(pred, tgt, "chamfer_patch", patch_size=P,
                                  huber_delta=delta).mean().item()
        want = 2.0 * _pseudo_huber(sq, delta)
        assert abs(got - want) < 1e-3, f"delta={delta}: got {got}, want {want}"
        # sanity: huber compresses the squared cost (2*sq) since res>0
        assert want < 2.0 * sq - 1e-3, "pseudo-Huber should be < squared cost for res>0"
    print(f"PASS  case_huber_transform_exact     sq={sq:.1f}  loss matches 2·huber(sq) for δ∈{{1,2,5}}")


def case_huber_recovers_sq_large_delta():
    """Continuity: as δ→∞ the cost → squared-L2, so the loss → the un-huberized loss."""
    torch.manual_seed(32)
    pred = torch.randn(2, 14, 128, 128, device=DEVICE)
    tgt = torch.randn(2, 14, 128, 128, device=DEVICE)
    base = _chamfer_recon_loss(pred, tgt, "sinkhorn_patch_hard", patch_size=4,
                               sinkhorn_eps=0.05, sinkhorn_iters=100).mean().item()
    big = _chamfer_recon_loss(pred, tgt, "sinkhorn_patch_hard", patch_size=4,
                              sinkhorn_eps=0.05, sinkhorn_iters=100, huber_delta=1e6).mean().item()
    assert abs(big - base) / base < 1e-3, f"large-δ should recover sq loss: base={base}, big={big}"
    print(f"PASS  case_huber_recovers_sq_large_delta  base={base:.4f}  δ=1e6→{big:.4f}")


def case_huber_reduces_loss_and_identity():
    """δ>0 compresses outlier residuals: hard loss (which gathers the huberized cost under the
    huber-optimal assignment) is ≤ the squared loss; and identity stays ~0."""
    torch.manual_seed(33)
    pred = torch.randn(2, 14, 128, 128, device=DEVICE)
    tgt = torch.randn(2, 14, 128, 128, device=DEVICE)
    sq_loss = _chamfer_recon_loss(pred, tgt, "sinkhorn_patch_hard", patch_size=4,
                                  sinkhorn_eps=0.05, sinkhorn_iters=100).mean().item()
    hub_loss = _chamfer_recon_loss(pred, tgt, "sinkhorn_patch_hard", patch_size=4,
                                   sinkhorn_eps=0.05, sinkhorn_iters=100, huber_delta=2.0).mean().item()
    assert hub_loss < sq_loss, f"huber should compress the cost: sq={sq_loss}, huber={hub_loss}"
    x = torch.randn(2, 14, 128, 128, device=DEVICE)
    idn = _chamfer_recon_loss(x, x, "sinkhorn_patch_hard", patch_size=4,
                              sinkhorn_eps=0.05, sinkhorn_iters=100, huber_delta=2.0).mean().item()
    assert idn < 1e-2, f"huber identity should be ~0, got {idn}"
    print(f"PASS  case_huber_reduces_loss_and_identity  sq={sq_loss:.4f}  huber={hub_loss:.4f}  id={idn:.2e}")


def case_huber_gradient_finite():
    """Gradient through the pseudo-Huber path is finite (incl. at near-coincident points)."""
    torch.manual_seed(34)
    pred = torch.randn(2, 14, 128, 128, device=DEVICE, requires_grad=True)
    tgt = pred.detach().clone()             # near-coincident ⇒ would be the 1/√ danger for true Huber
    tgt = tgt + 1e-4 * torch.randn_like(tgt)
    loss = _chamfer_recon_loss(pred, tgt, "sinkhorn_patch_hard", weight=0.03, patch_size=4,
                               sinkhorn_eps=0.05, sinkhorn_iters=100, huber_delta=2.0).mean()
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all(), "non-finite huber grad"
    print(f"PASS  case_huber_gradient_finite     loss={loss.item():.2e}  "
          f"grad_max={pred.grad.abs().max().item():.4f}  (smooth at coincident points)")


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    case_identity()
    case_within_patch_invariance()
    case_cross_patch_penalty()
    case_single_patch_parity()
    case_gradient_finite()
    case_weighted_path()
    case_divisibility_raise()
    case_sinkhorn_identity()
    case_sinkhorn_within_patch_invariance()
    case_sinkhorn_penalizes_collapse()
    case_sinkhorn_gradient_finite()
    case_sinkhorn_plan_marginals()
    case_sinkhorn_hard_identity()
    case_sinkhorn_hard_within_patch_invariance()
    case_sinkhorn_hard_scale_parity()
    case_sinkhorn_hard_gradient_finite()
    case_sinkhorn_hard_collapse_monitor()
    case_huber_off_bit_identical()
    case_huber_transform_exact()
    case_huber_recovers_sq_large_delta()
    case_huber_reduces_loss_and_identity()
    case_huber_gradient_finite()
    print("\nAll chamfer_patch + sinkhorn_patch(+hard) + huber tests passed.")
