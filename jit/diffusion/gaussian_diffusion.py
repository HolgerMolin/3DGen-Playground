# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


import math
import os

import numpy as np
import torch as th
import enum

from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl

# Cache GPU copies of diffusion schedule arrays so _extract_into_tensor doesn't
# pay the from_numpy→to(device) transfer overhead on every training step.
# Keyed by (id(numpy_array), device_str); safe because schedule arrays are
# instance attributes that live for the full training run.
_SCHEDULE_TENSOR_CACHE: dict = {}


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# --- KeOps backend for the Chamfer kNN search (optional; ~2.4-2.8x faster) -----
# pytorch3d's kNN is correct but memory-bandwidth-bound. KeOps does the same
# nearest-neighbor search as an online reduction (no NxN materialization) and is
# substantially faster on this 16,384-point cloud. We use KeOps only for the
# (non-differentiable) argmin search and keep the differentiable distance in torch
# via gather — numerically identical to pytorch3d. Falls back to pytorch3d if
# pykeops or a CUDA toolchain is unavailable.
_KEOPS_STATE = {"checked": False, "ok": False}


def _keops_setup_cuda_env():
    """Point KeOps' runtime JIT at a CUDA toolchain (nvcc + libnvrtc) via env vars.

    KeOps compiles/links its kernels with g++/nvrtc at first use, which need the
    CUDA lib dir on LIBRARY_PATH/LD_LIBRARY_PATH. Idempotent; auto-detects a system
    CUDA install when CUDA_PATH/CUDA_HOME are unset. The launcher should also export
    these before process start so the compiled kernel loads reliably.
    """
    if os.environ.get("_KEOPS_CUDA_ENV_DONE"):
        return
    cuda = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if not cuda or not os.path.isdir(cuda):
        for cand in ("/usr/local/cuda", "/usr/local/cuda-12.8"):
            if os.path.isdir(cand):
                cuda = cand
                break
    if cuda and os.path.isdir(cuda):
        os.environ["CUDA_PATH"] = cuda
        os.environ.setdefault("CUDA_HOME", cuda)
        for var, sub in (("PATH", "bin"), ("LIBRARY_PATH", "lib64"), ("LD_LIBRARY_PATH", "lib64")):
            val = os.path.join(cuda, sub)
            cur = os.environ.get(var, "")
            if val not in cur.split(os.pathsep):
                os.environ[var] = val + (os.pathsep + cur if cur else "")
    os.environ["_KEOPS_CUDA_ENV_DONE"] = "1"


def _keops_available():
    """True if pykeops imports (and its CUDA env is set up); cached. Else pytorch3d."""
    if not _KEOPS_STATE["checked"]:
        _KEOPS_STATE["checked"] = True
        try:
            _keops_setup_cuda_env()
            from pykeops.torch import LazyTensor  # noqa: F401
            _KEOPS_STATE["ok"] = True
        except Exception:
            _KEOPS_STATE["ok"] = False
    return _KEOPS_STATE["ok"]


def _keops_argmin(x, y):
    """Index of nearest y-row (squared L2) per x-row. x:(B,N,Ds) y:(B,M,Ds) -> (B,N) long.

    Search only — argmin indices carry no gradient (as in any kNN). Inputs are
    detached so KeOps does no autograd bookkeeping; the differentiable distance is
    recomputed in torch by the caller on the gathered rows.
    """
    from pykeops.torch import LazyTensor
    x_i = LazyTensor(x.detach().unsqueeze(2).contiguous())  # (B,N,1,Ds)
    y_j = LazyTensor(y.detach().unsqueeze(1).contiguous())  # (B,1,M,Ds)
    return ((x_i - y_j) ** 2).sum(-1).argmin(dim=2)[..., 0].long()  # (B,N)


def _gather_rows(src, idx):
    """src:(B,M,C), idx:(B,N) -> (B,N,C): row idx[b,n] of src[b]."""
    return th.gather(src, 1, idx.unsqueeze(-1).expand(-1, -1, src.shape[-1]))


def _subsample_query_idx(B, N, M, device):
    """``(B, M)`` long: M distinct uniform query indices per batch element (no replacement).

    Used to draw a fresh random subset of *query* points for the Chamfer search each
    step. argsort-of-rand is the cheap vectorized way to sample without replacement per
    row; the cost is negligible next to the kNN search.
    """
    return th.rand(B, N, device=device).argsort(dim=1)[:, :M]


def _sinkhorn_log(C, eps, iters):
    """Entropic-OT transport plan via log-domain Sinkhorn (uniform marginals).

    ``C``: (B, nP, M, M) squared-distance cost. Returns the (B, nP, M, M) soft transport
    plan assigning M pred points to M target points within each patch, marginals a=b=1/M.
    Fixed ``iters`` + pure tensor ops keep it torch.compile-safe (no data-dependent control
    flow, no CPU sync). As ``eps`` -> 0 the plan -> a hard permutation (exact assignment);
    larger ``eps`` -> softer/blurrier plan (re-collapses toward the mean).
    """
    M = C.shape[-1]
    logK = -C / eps                                              # (B, nP, M, M)
    logm = -math.log(M)                                         # log uniform marginal
    f = th.zeros(C.shape[:-1], device=C.device, dtype=C.dtype)  # (B, nP, M)
    g = th.zeros_like(f)
    for _ in range(int(iters)):
        f = logm - th.logsumexp(logK + g.unsqueeze(-2), dim=-1)
        g = logm - th.logsumexp(logK + f.unsqueeze(-1), dim=-2)
    return th.exp(logK + f.unsqueeze(-1) + g.unsqueeze(-2))     # (B, nP, M, M) plan


# Lazily-compiled handle for _sinkhorn_log. The loop is memory-bound: eager, each iter
# materializes the full (B,nP,M,M) broadcast `logK + g.unsqueeze(...)` to HBM before the
# logsumexp reduction (200 such ~1GB round-trips/step at B=256). Inductor FUSES the broadcast
# into the reduction, keeping it in SRAM — measured ~8x on the Sinkhorn term (1463->184 ms,
# B=128/100 iters), numerically identical fp32 (same math, just fused). Compile-safe by design
# (fixed iters, no data-dependent control flow / CPU sync). Compiles once; shapes are static
# because training uses drop_last=True. Opt-in via compile_sinkhorn so probes/tests stay eager.
_sinkhorn_log_compiled = None


def _resolve_sinkhorn(compiled: bool):
    """Return the eager or (lazily) compiled _sinkhorn_log. Specializes on shape/eps/iters;
    fine since those are constant within a run (drop_last=True; iters launch-time fixed)."""
    global _sinkhorn_log_compiled
    if not compiled:
        return _sinkhorn_log
    if _sinkhorn_log_compiled is None:
        _sinkhorn_log_compiled = th.compile(_sinkhorn_log)
    return _sinkhorn_log_compiled


def _chamfer_recon_loss(pred, target, mode, channel_loss_weights=None, weight=1.0,
                        subsample=0, patch_size=8, rev_weight=1.0,
                        sinkhorn_eps=0.05, sinkhorn_iters=50, return_diag=False,
                        compile_sinkhorn=False, huber_delta=0.0, residual_mad=False):
    """Permutation-invariant reconstruction loss over the per-Gaussian point set.

    ``pred`` / ``target`` are atlas tensors ``(B, C, H, W)``. Chamfer/kNN matching is
    permutation-invariant, so the spatial grid is flattened straight to a point cloud
    ``(B, N=H*W, C)`` — the sphere/atlas ordering is irrelevant (the whole point of
    switching away from index-aligned MSE).

    Modes:
      * ``chamfer_feature``   — match Gaussians by their full C-dim feature vector;
        loss is the bidirectional Chamfer of those vectors.
      * ``chamfer_geometric`` — match Gaussians by xyz, then MSE on all channels of
        the matched pairs (both directions).
      * ``chamfer_patch``     — bidirectional Chamfer restricted to *within* each
        ``patch_size × patch_size`` patch (the model's Conv2d(stride=patch_size)
        patchify). A middle ground between global Chamfer (fully permutation-free) and
        index-MSE: bounds a Gaussian's correspondence to its own patch, restoring a
        coarse per-patch alignment. Pure-torch (no KeOps/pytorch3d), exact (no
        ``subsample``), and ~``N/patch_size²`` cheaper than the global kNN. Uses the
        IDENTICAL patch grouping as ``permute_atlas('patch')``, so the two compose:
        within-patch permutation leaves a patch's point set unchanged → loss invariant.
      * ``sinkhorn_patch``     — same per-patch grouping/cost as ``chamfer_patch`` but the
        match is the OPTIMAL ASSIGNMENT (entropic-OT EMD via log-Sinkhorn) instead of hard
        nearest-neighbour. With M pred == M target the optimum is a bijection, forcing
        coverage (within-patch collision → 0) and a dense gradient while staying within-patch
        permutation-invariant. ``sinkhorn_eps`` controls hard↔soft; ``rev_weight`` is a no-op
        (OT marginals are symmetric). Reuses ``sq``; the plan is detached so the gradient is
        the squared distance under the (soft-)optimal assignment.

    The kNN search (feature/geometric) runs on KeOps when available (~2.4-2.8x faster),
    else pytorch3d; both give identical results. Distances are in normalized model space.
    ``channel_loss_weights`` (if given) re-weights channels inside the distance:
    ``feature``/``patch`` scale each channel by ``sqrt(w_c)`` (affects match + distance);
    ``geometric`` applies ``w_c`` to the per-channel squared error.

    ``subsample`` (>0 and < N): cap the search at ``M = subsample`` *query* points per
    direction, drawn fresh and uniformly (without replacement) each call. The full
    target cloud is kept, so every query still finds its exact nearest neighbour; since
    the point reduction is a mean, the mean over M queries is an unbiased estimate of
    the mean over all N. Cost scales ~linearly in M (the search is O(M·N·C)), trading a
    little gradient variance for a large speedup. ``subsample=0`` (or ``>= N``) uses all
    points and is bit-for-bit the previous behaviour.

    ``rev_weight`` scales the backward (GT-as-query) term in all modes — the
    coverage/recall direction (every GT Gaussian must have a nearby prediction).
    ``rev_weight=1`` is the symmetric Chamfer; ``>1`` upweights coverage (fights
    mode-collapse, pushes the model to represent every GT Gaussian somewhere).

    ``huber_delta`` (>0) robustifies the within-patch ground cost (``chamfer_patch`` /
    ``sinkhorn_patch`` / ``sinkhorn_patch_hard``) with a pseudo-Huber on the pair radius
    ``r=‖x−y‖``: ``cost = 2δ²(√(1+r²/δ²)−1)`` — quadratic (= squared-L2) for inliers,
    linear (``2δr``) for outliers, so a few large-residual pairs stop dominating the
    assignment and the gradient. ``δ`` is the residual radius where quad bends to linear;
    the ``2δ²`` scaling fixes the inlier scale to the squared cost (so ``weight`` stays
    calibrated) and ``δ→∞`` recovers the exact squared cost. ``0`` (default) = off
    (bit-identical squared-L2). No-op for the ``feature``/``geometric`` modes.

    ``residual_mad`` (diagnostic, ``return_diag`` only): when set, report the robust scale
    of the matched-residual radius ``r=‖pred−tgt_σ(i)‖`` for the ``sinkhorn`` modes —
    ``median``, ``MAD = median(|r−median r|)``, ``p90`` and ``p99`` of ``r`` — measured on
    the RAW (pre-huber) cost so the numbers are the natural residual scale, in the same
    units as ``huber_delta``. Lets a ``huber_delta=0`` run inform a principled ``δ`` (e.g.
    ``δ ≈ median + k·MAD``). No-op for the ``feature``/``geometric``/``chamfer_patch`` modes.

    Returns a per-sample tensor ``(B,)`` (un-reduced over batch so downstream
    t-bucketing / logging still works), already scaled by ``weight``. When
    ``return_diag=True`` returns ``(loss, diag)`` instead, where ``diag`` is a dict of
    scalar tensors (``sinkhorn_marginal_resid`` for ``sinkhorn_patch`` — the Sinkhorn
    convergence monitor — and ``recon_residual_{median,mad,p90,p99}`` when
    ``residual_mad``); empty for the other modes.
    """
    def _ret(loss, diag=None):
        return (loss, diag or {}) if return_diag else loss

    # kNN kernels (both backends) require float32; cast keeps autocast (fp16/bf16)
    # forward passes from breaking them. The cast is differentiable so grads still
    # reach ``model_output`` through the torch-side distance.
    pred_pc = pred.flatten(2).transpose(1, 2).float().contiguous()      # (B, N, C)
    target_pc = target.flatten(2).transpose(1, 2).float().contiguous()  # (B, N, C)

    w = None
    if channel_loss_weights is not None:
        w = channel_loss_weights.to(device=pred_pc.device, dtype=pred_pc.dtype).clamp_min(0)

    use_keops = _keops_available()

    B, N = pred_pc.shape[0], pred_pc.shape[1]
    M = subsample if (subsample and 0 < subsample < N) else 0  # 0 => use all points

    if mode == "chamfer_feature":
        x, y = pred_pc, target_pc
        if w is not None:
            scale = w.sqrt().view(1, 1, -1)  # weighted Euclidean: Σ w_c (Δ_c)²
            x = x * scale
            y = y * scale
        # Query points for each direction: a fresh random subset of M when subsampling,
        # else the full cloud. The full target (y / x) is always searched, so each
        # query's nearest neighbour is exact.
        if M:
            xq = _gather_rows(x, _subsample_query_idx(B, N, M, x.device))  # (B,M,C)
            yq = _gather_rows(y, _subsample_query_idx(B, N, M, y.device))
        else:
            xq, yq = x, y
        if use_keops:
            # search full C-dim feature space; differentiable distance via gather.
            sq_f = (xq - _gather_rows(y, _keops_argmin(xq, y))) ** 2  # (B,M|N,C)
            sq_r = (yq - _gather_rows(x, _keops_argmin(yq, x))) ** 2
            # rev_weight scales the backward (GT-as-query) term — the coverage/recall
            # direction (every GT Gaussian must have a nearby prediction).
            loss_b = sq_f.sum(-1).mean(1) + rev_weight * sq_r.sum(-1).mean(1)
        else:
            from pytorch3d.loss import chamfer_distance
            # single_directional so we can feed the subsampled query set on each side.
            kw = dict(batch_reduction=None, point_reduction="mean", norm=2)
            loss_f, _ = chamfer_distance(xq, y, single_directional=True, **kw)  # (B,)
            loss_r, _ = chamfer_distance(yq, x, single_directional=True, **kw)
            loss_b = loss_f + rev_weight * loss_r
        return _ret(weight * loss_b)

    if mode == "chamfer_geometric":
        wv = w.view(1, 1, -1) if w is not None else None
        pred_xyz = pred_pc[..., :3].contiguous()
        gt_xyz = target_pc[..., :3].contiguous()
        # Subsample the query side (pred for forward, target for reverse); match against
        # the full opposite cloud by xyz, then squared error on all channels.
        if M:
            pred_q = _gather_rows(pred_pc, _subsample_query_idx(B, N, M, pred_pc.device))    # (B,M,C)
            gt_q = _gather_rows(target_pc, _subsample_query_idx(B, N, M, target_pc.device))  # (B,M,C)
            pred_q_xyz = pred_q[..., :3].contiguous()
            gt_q_xyz = gt_q[..., :3].contiguous()
        else:
            pred_q, gt_q = pred_pc, target_pc
            pred_q_xyz, gt_q_xyz = pred_xyz, gt_xyz
        if use_keops:
            gt_match = _gather_rows(target_pc, _keops_argmin(pred_q_xyz, gt_xyz))    # (B,M|N,C)
            pred_match = _gather_rows(pred_pc, _keops_argmin(gt_q_xyz, pred_xyz))    # (B,M|N,C)
        else:
            from pytorch3d.ops import knn_gather, knn_points
            gt_match = knn_gather(target_pc, knn_points(pred_q_xyz, gt_xyz, K=1).idx)[:, :, 0, :]
            pred_match = knn_gather(pred_pc, knn_points(gt_q_xyz, pred_xyz, K=1).idx)[:, :, 0, :]
        sq_fwd = (pred_q - gt_match) ** 2   # forward: pred query -> nearest target by xyz
        sq_rev = (gt_q - pred_match) ** 2   # reverse: target query -> nearest pred by xyz
        if wv is not None:
            sq_fwd = sq_fwd * wv
            sq_rev = sq_rev * wv
        return _ret(weight * 0.5 * (sq_fwd.mean(dim=(1, 2)) + rev_weight * sq_rev.mean(dim=(1, 2))))

    if mode in ("chamfer_patch", "sinkhorn_patch", "sinkhorn_patch_hard"):
        # Bidirectional Chamfer WITHIN each patch_size×patch_size patch. Works from the
        # original (B,C,H,W) atlas (not the flattened pred_pc/target_pc) so we can group
        # points by patch exactly as permute_atlas('patch') / the Conv2d patchify do.
        P = int(patch_size)
        B, C, H, W = pred.shape
        if H % P or W % P:
            raise ValueError(
                f"chamfer_patch needs H,W divisible by patch_size={P}; got {(H, W)}"
            )
        nH, nW = H // P, W // P
        M = P * P  # Gaussians per patch

        def to_patches(t):
            # (B,C,H,W) -> (B, nP, M, C); identical grouping to permute_atlas('patch').
            return (
                t.float()
                .reshape(B, C, nH, P, nW, P)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(B, nH * nW, C, M)
                .transpose(-1, -2)
                .contiguous()
            )

        xp = to_patches(pred)      # (B, nP, M, C)
        yp = to_patches(target)
        if w is not None:
            scale = w.sqrt().view(1, 1, 1, -1)  # weighted Euclidean, as in chamfer_feature
            xp = xp * scale
            yp = yp * scale
        # Squared in-patch distances via the Gram form ‖x‖²+‖y‖²−2x·y. Avoids the
        # (B,nP,M,M,C) broadcast (~15GB at B=256/C=14); materializes only (B,nP,M,M)
        # (~1.07GB at B=256/P=8). No sqrt -> gradient-safe at coincident points.
        aa = xp.pow(2).sum(-1)                       # (B, nP, M)
        bb = yp.pow(2).sum(-1)                       # (B, nP, M)
        ab = xp @ yp.transpose(-1, -2)              # (B, nP, M, M)
        sq = (aa.unsqueeze(-1) + bb.unsqueeze(-2) - 2 * ab).clamp_min(0)  # (B, nP, M, M)
        # Keep a handle on the RAW (pre-huber) squared distances so the residual-MAD
        # diagnostic always reports the natural residual radius r=√sq, in the same units
        # as `huber_delta`, regardless of whether the huber transform is applied below.
        # When huber_delta==0 (the intended diagnostic run) the transform is a no-op, so
        # sq_raw is exactly the matched cost. (Just a tensor reference — no copy; the huber
        # block rebinds `sq` to a new tensor, leaving sq_raw pointing at the original.)
        sq_raw = sq
        if huber_delta and huber_delta > 0.0:
            # Pseudo-Huber on the per-pair radius r=‖x−y‖ (r²=sq): robustifies the ground
            # cost so a few large-residual pairs (bad matches / high-noise outliers) stop
            # dominating both the OT assignment AND (via the hard gather / soft <P,sq>) the
            # gradient. Reusing the name `sq` makes every downstream consumer use the robust
            # cost. A single elementwise map of the (B,nP,M,M) sq tensor — keeps the Gram-form
            # memory trick (no (B,nP,M,M,C) broadcast) and stays outside the compiled Sinkhorn
            # loop (changing δ never recompiles). The 2δ² scaling fixes the INLIER scale to sq:
            #   small r → sq (= MSE cost, so chamfer_loss_weight stays calibrated)
            #   large r → 2δ·r (linear ⇒ bounded gradient δ/√sq)
            #   δ → ∞ recovers sq exactly. grad d/dsq = 1/√(1+sq/δ²) ∈ (0,1], smooth at r=0.
            # Use the rationalized form 2δ²(√(1+u)−1) = 2·sq/(√(1+u)+1), u=sq/δ², which is
            # algebraically identical but free of the √(1+u)−1 catastrophic cancellation that
            # underflows to 0 (instead of → sq) for large δ in fp32.
            d2 = huber_delta * huber_delta
            sq = 2.0 * sq / ((1.0 + sq / d2).sqrt() + 1.0)
        patch_diag = {}
        if mode == "chamfer_patch":
            fwd = sq.min(dim=-1).values              # pred  -> nearest in-patch target
            rev = sq.min(dim=-2).values              # target -> nearest in-patch pred
            # mean over points (patches × M) — sum-over-C already in sq, matches chamfer_feature scale.
            loss_b = fwd.mean(dim=(1, 2)) + rev_weight * rev.mean(dim=(1, 2))  # (B,)
        else:  # sinkhorn_patch (soft) / sinkhorn_patch_hard — optimal-assignment matching
            # Optimal assignment of the M pred points to the M target points within each patch,
            # approximated by entropic-OT log-Sinkhorn. The plan only decides the MATCHING —
            # the gradient flows through `sq`, never through the plan — so it is computed under
            # no_grad. Otherwise autograd records the iters-deep recursion for a backward that
            # never uses it (~1.1 GB/iter of dead graph at B=128 — the dominant training-memory
            # cost and a hard cap on sinkhorn_iters); under no_grad peak memory is O(1) in iters.
            # rev_weight is irrelevant (OT marginals already cover both directions).
            with th.no_grad():
                # compiled (fused logsumexp) when compile_sinkhorn — ~8x, numerically identical.
                P = _resolve_sinkhorn(compile_sinkhorn)(sq, sinkhorn_eps, sinkhorn_iters)  # (B,nP,M,M); no graph
                if mode == "sinkhorn_patch_hard":
                    sigma = P.argmax(dim=-1)                          # (B, nP, M) assigned target per pred
                if return_diag:
                    # Sinkhorn convergence monitor. The log-domain recursion ends on the column
                    # update, so the COLUMN marginal (sum over pred i, dim=-2) is exact (=1/M);
                    # the ROW marginal (sum over target j, dim=-1) carries the residual. mean
                    # |rowsum − 1/M| → 0 as the plan converges to doubly-stochastic, rising if
                    # sinkhorn_iters is too few for the eps / cost sharpness. ~free.
                    row = P.sum(dim=-1)                               # (B, nP, M)
                    patch_diag["sinkhorn_marginal_resid"] = (row - (1.0 / M)).abs().mean()
                    if mode == "sinkhorn_patch_hard":
                        # Collision fraction: preds whose argmax target is shared with another
                        # pred in the patch — i.e. how far the rounded assignment is from a true
                        # bijection. Floors at the eps non-permutation (≠0), not at full conv.
                        col_counts = th.zeros_like(row)              # (B, nP, M) counts over columns j
                        col_counts.scatter_add_(-1, sigma, th.ones_like(row))
                        extra = (col_counts - 1.0).clamp_min(0).sum(dim=-1)  # duplicate picks per patch
                        patch_diag["sinkhorn_collision_frac"] = (extra / M).mean()
                    if residual_mad:
                        # Robust scale of the matched-residual RADIUS r=‖pred−tgt_σ(i)‖ (r²=sq_raw
                        # at the assigned target) — the exact quantity huber_delta thresholds. Use
                        # the hard argmax assignment in BOTH sinkhorn modes (the soft plan has no
                        # single match) and gather from the RAW (pre-huber) cost so the number is
                        # the natural residual scale regardless of huber_delta. median/MAD/p90/p99
                        # over the whole micro-batch point set; the window mean is taken upstream.
                        sig = sigma if mode == "sinkhorn_patch_hard" else P.argmax(dim=-1)
                        r = sq_raw.gather(-1, sig.unsqueeze(-1)).squeeze(-1).clamp_min(0).sqrt().reshape(-1)
                        med = r.median()
                        patch_diag["recon_residual_median"] = med
                        patch_diag["recon_residual_mad"] = (r - med).abs().median()
                        qs = th.quantile(r, th.tensor([0.9, 0.99], device=r.device, dtype=r.dtype))
                        patch_diag["recon_residual_p90"] = qs[0]
                        patch_diag["recon_residual_p99"] = qs[1]
            if mode == "sinkhorn_patch_hard":
                # HARD assignment: MSE to the single argmax-matched target (DETR-style hard
                # matching). σ is detached, so the gradient is a crisp per-pred MSE toward
                # tgt_σ(i) — no soft blend, hence no mean-pull. Gradient flows through `sq`
                # gathered at the detached σ. mean_i sq[i,σ(i)] = the same scale as the soft
                # <P, sq> when the plan is a permutation.
                hard_sq = sq.gather(-1, sigma.unsqueeze(-1)).squeeze(-1)  # (B, nP, M) = sq[i,σ(i)]
                loss_b = hard_sq.mean(dim=(1, 2))                     # (B,)
            else:  # sinkhorn_patch (soft)
                # SOFT: <P, sq> is the plan-weighted (blended) matched distance; the induced
                # per-pred gradient pulls toward M·Σ_j P[i,j]·tgt_j (a blend → mean-pull when P
                # is blurry). Plan mass sums to 1 per patch → ~chamfer_patch scale.
                loss_b = (P * sq).sum(dim=(-1, -2)).mean(dim=1)      # (B,) gradient via `sq` only
        return _ret(weight * loss_b, patch_diag)

    raise ValueError(f"unknown recon_loss mode: {mode!r}")


def permute_atlas(x, mode="none", patch_size=8, generator=None):
    """Randomly permute the per-Gaussian ordering of an atlas ``(B, C, H, W)``.

    Augmentation for Chamfer training: the model is otherwise only ever shown the
    canonical sphere2plane ordering (every training input is a noised *canonical*
    atlas), so at inference — where a sampler feeds the model's own non-canonical
    output back in — the ordering is OOD and samples collapse. Permuting the input
    each step teaches order-robustness. Permutation is applied to the same Gaussians
    across all channels (xyz/opacity/SH of a point move together); ε is i.i.d. so it
    needs no matching permutation, and the Chamfer target is permutation-invariant so
    the loss is unchanged. ONLY valid with a Chamfer recon_loss — index-aligned MSE
    becomes unlearnable at high noise (the ordering signal is gone in ε).

    Modes:
      * ``"none"``   — identity.
      * ``"patch"``  — permute the ``patch_size²`` Gaussians *within* each patch,
        independently per patch/sample. Matches the model's Conv2d(stride=patch_size)
        patchify exactly (each patch is a contiguous P×P block), so the patch-level
        positional embedding stays meaningful — only the fine, within-patch ordering
        is randomized. Targets the hypothesis that the model permutes *locally*.
      * ``"global"`` — permute all ``H·W`` Gaussians across the whole atlas, per
        sample. Maximal augmentation; also makes the positional embedding
        uninformative (every patch becomes a random bag of points).

    A fresh permutation is drawn on every call (on-the-fly); the cost is a gather
    along the spatial axis (~sub-ms at B=256/N=16384), negligible vs the forward.
    Returns a tensor of the same shape and (channels-last) memory format as ``x``.
    """
    if mode == "none":
        return x
    B, C, H, W = x.shape
    channels_last = x.is_contiguous(memory_format=th.channels_last)
    if mode == "global":
        N = H * W
        perm = th.rand(B, N, device=x.device, generator=generator).argsort(dim=-1)  # (B, N)
        idx = perm.unsqueeze(1).expand(B, C, N)
        out = th.gather(x.reshape(B, C, N), 2, idx).reshape(B, C, H, W)
    elif mode == "patch":
        P = int(patch_size)
        if H % P or W % P:
            raise ValueError(f"patch permute needs H,W divisible by patch_size={P}; got {(H, W)}")
        nH, nW = H // P, W // P
        # (B,C,H,W) -> per-patch flattened points (B, nH, nW, C, P*P)
        xr = x.reshape(B, C, nH, P, nW, P).permute(0, 2, 4, 1, 3, 5).reshape(B, nH, nW, C, P * P)
        perm = th.rand(B, nH, nW, P * P, device=x.device, generator=generator).argsort(dim=-1)
        idx = perm.unsqueeze(3).expand(B, nH, nW, C, P * P)
        xr = th.gather(xr, 4, idx)
        out = xr.reshape(B, nH, nW, C, P, P).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
    else:
        raise ValueError(f"unknown permute_atlas mode: {mode!r} (expected none|patch|global)")
    return out.contiguous(memory_format=th.channels_last) if channels_last else out.contiguous()


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    Original ported from this codebase:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type
    ):

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def flow_matching_q_sample(self, x_start, t_value, noise=None):
        """JiT flow-matching interpolation.

        Convention (matches ``jit.sampling._jit_velocity_from_xstart``):
          * ``t_value == 0`` → pure noise.
          * ``t_value == 1`` → clean data.

        ``x_t = t * x_0 + (1 - t) * ε``.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        t_v = t_value.view(-1, *([1] * (x_start.ndim - 1))).to(dtype=x_start.dtype)
        return t_v * x_start + (1.0 - t_v) * noise

    def flow_matching_training_losses(
        self,
        model,
        x_start,
        t_value,
        t_discrete,
        model_kwargs=None,
        noise=None,
        channel_loss_weights=None,
        recon_loss="mse",
        chamfer_loss_weight=1.0,
        chamfer_subsample=0,
        chamfer_patch_size=8,
        chamfer_rev_weight=1.0,
        mse_hybrid_weight=0.0,
        mse_hybrid_lownoise_mult=1.0,
        sinkhorn_eps=0.05,
        sinkhorn_iters=50,
        compile_sinkhorn=False,
        huber_delta=0.0,
        residual_mad_diag=False,
        permute_mode="none",
        permute_patch_size=8,
        skip_recon=False,
    ):
        """JiT flow-matching loss with x₀ prediction.

        ``t_value`` is continuous in (0, 1) and drives the interpolation
        ``x_t = t * x_0 + (1 - t) * ε``. ``t_discrete`` is the integer timestep
        (typically ``round(t_value * (T-1))``) fed to the model's timestep
        embedding, keeping the sampler and trainer on the same grid. The model
        predicts x₀ directly.

        ``recon_loss`` selects the reconstruction term:
          * ``"mse"`` (default) — index-aligned ``MSE(pred, x_0)``.
          * ``"chamfer_feature"`` / ``"chamfer_geometric"`` — permutation-invariant
            Chamfer over the per-Gaussian point set (see ``_chamfer_recon_loss``),
            scaled by ``chamfer_loss_weight``. This removes the dependence on the
            (near-random) atlas ordering that the index-aligned MSE assumes.
          * ``"chamfer_patch"`` — Chamfer restricted to within each
            ``chamfer_patch_size``-sized patch (see ``_chamfer_recon_loss``); a middle
            ground that bounds matches to a patch and composes with
            ``permute_mode="patch"``.
          * ``"sinkhorn_patch"`` — same per-patch grouping but the OPTIMAL ASSIGNMENT
            (entropic-OT EMD via log-Sinkhorn, ``sinkhorn_eps``/``sinkhorn_iters``) instead of
            hard nearest-neighbour: forces a within-patch bijection (collision → 0) with a
            dense gradient. ``chamfer_rev_weight`` is a no-op for this mode.
          * ``"sinkhorn_patch_hard"`` — same Sinkhorn plan, but HARD-rounded
            (``σ(i)=argmax_j Π[i,j]``) to a single matched target, then ``MSE(pred_i, tgt_σ(i))``
            (DETR-style hard matching). Gives a crisp per-point gradient with no soft blend (no
            mean-pull); the argmax converges in far fewer ``sinkhorn_iters`` than the soft plan,
            but the rounding is only ~bijective (monitor ``sinkhorn_collision_frac``).

        ``mse_hybrid_weight`` (>0, Chamfer modes only): add ``λ · MSE(pred, x_start)`` on
        top of the Chamfer term. Chamfer's optimum is permutation-degenerate so it gives no
        gradient toward copying the (near-clean) low-noise input; this index-aligned MSE
        supplies that gradient, dragging the low-noise loss toward 0 and steepening
        loss-vs-t. With ``permute_mode!="none"`` the target is the permuted ``x_start``, so
        it rewards equivariant copying (compatible with the augmentation).

        ``channel_loss_weights`` (optional) is a 1-D tensor of length C. For MSE it
        scales the per-channel squared error before spatial averaging (compensating
        channels whose post-normalization per-object spatial std is ≪ 1, which would
        otherwise get ~var² less gradient). For the Chamfer modes it re-weights
        channels inside the distance metric (see ``_chamfer_recon_loss``).
        """
        if self.model_mean_type != ModelMeanType.START_X:
            raise ValueError(
                "flow_matching_training_losses requires predict_xstart=True "
                "(ModelMeanType.START_X); got "
                f"{self.model_mean_type}"
            )
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        # Order-robustness augmentation (Chamfer only): randomize the per-Gaussian
        # ordering the model sees, so iterative sampling on its own non-canonical
        # output is in-distribution. Permutation-invariant Chamfer makes the target
        # loss unchanged; ε is i.i.d. so the externally-supplied noise still applies.
        if permute_mode != "none":
            if recon_loss == "mse":
                raise ValueError(
                    "permute_mode requires a Chamfer recon_loss; index-aligned MSE is "
                    "unlearnable under input permutation (ordering signal vanishes in noise)."
                )
            x_start = permute_atlas(x_start, permute_mode, permute_patch_size)

        x_t = self.flow_matching_q_sample(x_start, t_value, noise=noise)
        model_output = model(x_t, t_discrete, **model_kwargs)
        assert model_output.shape == x_start.shape

        # Caller-gated short-circuit: when the outer recon_loss_weight is 0 (render-only
        # mode), the recon dispatch (Sinkhorn / Chamfer / MSE) contributes nothing to
        # backward. Skip it entirely — saves the OT/forward compute. The model forward
        # MUST still run because the render path consumes pred_xstart downstream.
        if skip_recon:
            recon = th.zeros(x_start.shape[0], device=x_start.device, dtype=model_output.dtype)
            return {
                "loss": recon,
                "mse": recon,
                "pred_xstart": model_output,
                "x_t": x_t,
            }

        if channel_loss_weights is not None:
            if channel_loss_weights.ndim != 1 or channel_loss_weights.shape[0] != x_start.shape[1]:
                raise ValueError(
                    f"channel_loss_weights must be 1-D of length C={x_start.shape[1]}, "
                    f"got shape {tuple(channel_loss_weights.shape)}"
                )

        def _weighted_mse_flat(a, b):
            """Per-sample index-aligned MSE (mean over C,H,W) with channel weights -> (B,)."""
            sq = (a - b) ** 2
            if channel_loss_weights is not None:
                w = channel_loss_weights.to(device=sq.device, dtype=sq.dtype)
                sq = sq * w.view((1, -1) + (1,) * (sq.ndim - 2))
            return mean_flat(sq)

        mse_hybrid = None
        recon_diag = {}
        if recon_loss == "mse":
            recon = _weighted_mse_flat(x_start, model_output)
        elif recon_loss in ("chamfer_feature", "chamfer_geometric", "chamfer_patch",
                             "sinkhorn_patch", "sinkhorn_patch_hard"):
            recon, recon_diag = _chamfer_recon_loss(
                model_output,
                x_start,
                recon_loss,
                channel_loss_weights=channel_loss_weights,
                weight=chamfer_loss_weight,
                subsample=chamfer_subsample,
                patch_size=chamfer_patch_size,
                rev_weight=chamfer_rev_weight,
                sinkhorn_eps=sinkhorn_eps,
                sinkhorn_iters=sinkhorn_iters,
                compile_sinkhorn=compile_sinkhorn,
                huber_delta=huber_delta,
                residual_mad=residual_mad_diag,
                return_diag=True,
            )
            # Optional Chamfer+MSE hybrid: add an index-aligned MSE term. Chamfer's
            # nearest-match optimum is degenerate over within-patch permutations, so it
            # gives no gradient toward the input-copying solution; this MSE term supplies
            # that gradient (mandatory per-cell match), which collapses the low-noise loss
            # toward 0 and steepens loss-vs-t. Target is x_start (already permuted if
            # permute_mode!=none), so under permutation it rewards EQUIVARIANT copying.
            if mse_hybrid_weight > 0.0:
                mse_hybrid = _weighted_mse_flat(x_start, model_output)
                # Optionally UPWEIGHT the MSE hybrid toward the LOW-NOISE regime (high
                # t_value) with a linear ramp: w(t) = 1 + (mult-1)*t_value, so w=1 at the
                # noisy end (t=0, full mse_hybrid_weight) ramping to w=mult at the clean
                # end (t=1). mult=1 -> uniform across t (back-compat). Unlike a t**p
                # downweight, this KEEPS the t=0 baseline and amplifies toward clean.
                if mse_hybrid_lownoise_mult != 1.0:
                    ln_w = 1.0 + (mse_hybrid_lownoise_mult - 1.0) * t_value.to(mse_hybrid.dtype)
                    recon = recon + mse_hybrid_weight * ln_w * mse_hybrid
                else:
                    recon = recon + mse_hybrid_weight * mse_hybrid
        else:
            raise ValueError(
                "recon_loss must be one of "
                "mse|chamfer_feature|chamfer_geometric|chamfer_patch|sinkhorn_patch|"
                "sinkhorn_patch_hard, "
                f"got {recon_loss!r}"
            )

        # Key kept as "mse" for backward-compatible logging/keys downstream even
        # when the term is actually a Chamfer loss (+ optional MSE hybrid).
        terms = {
            "mse": recon,
        }
        terms["loss"] = recon
        terms["pred_xstart"] = model_output
        terms["x_t"] = x_t
        if mse_hybrid is not None:
            terms["mse_hybrid"] = mse_hybrid  # raw (un-weighted-by-lambda) per-sample MSE term
        if "sinkhorn_marginal_resid" in recon_diag:
            # Scalar convergence monitor (mean |plan row-marginal − 1/M|); see _chamfer_recon_loss.
            terms["sinkhorn_marginal_resid"] = recon_diag["sinkhorn_marginal_resid"]
        if "sinkhorn_collision_frac" in recon_diag:
            # Scalar bijection monitor for sinkhorn_patch_hard (argmax collision fraction).
            terms["sinkhorn_collision_frac"] = recon_diag["sinkhorn_collision_frac"]
        # Residual-radius diagnostics (opt-in via --log_residual_mad) for choosing huber_delta.
        for _k in ("recon_residual_median", "recon_residual_mad",
                   "recon_residual_p90", "recon_residual_p99"):
            if _k in recon_diag:
                terms[_k] = recon_diag[_k]
        return terms

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, t, **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
            # Expose x0 prediction so callers can reuse it (e.g. render loss) without a second forward pass.
            if self.model_mean_type == ModelMeanType.START_X:
                terms["pred_xstart"] = model_output
            else:
                terms["pred_xstart"] = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    key = (id(arr), str(timesteps.device))
    arr_t = _SCHEDULE_TENSOR_CACHE.get(key)
    if arr_t is None:
        arr_t = th.from_numpy(arr).float().to(device=timesteps.device)
        _SCHEDULE_TENSOR_CACHE[key] = arr_t
    res = arr_t[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
