#!/usr/bin/env python3
"""Temporary checks for 3DGS scale, opacity, and rotation parameterization."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataloaders.standard_3dgen_loader import Standard3DGenDataset
from jit.train import (
    _PipeConfig as LegacyPipeConfig,
    _build_gaussian_model_from_point_cloud,
    _load_reference_cameras as load_legacy_reference_cameras,
    _prepare_train_cameras as prepare_legacy_cameras,
    _try_import_renderer as try_import_legacy_renderer,
)
from utils.gsplat_render_util import (
    RENDER_OPACITY_RAW_MAX,
    RENDER_OPACITY_RAW_MIN,
    RENDER_SCALE_RAW_MAX,
    RENDER_SCALE_RAW_MIN,
    _load_reference_cameras as load_gsplat_reference_cameras,
    _normalize_quaternions_with_identity_fallback,
    _prepare_train_cameras as prepare_gsplat_cameras,
    _render_gsplat_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj_list", type=str, required=True)
    parser.add_argument("--gs_path", type=str, required=True)
    parser.add_argument("--ref_camera_tar", type=str, required=True)
    parser.add_argument("--num_objects", type=int, default=4)
    parser.add_argument("--num_cams", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render_size", type=int, default=256)
    parser.add_argument("--print_per_object_limit", type=int, default=4)
    return parser.parse_args()


def summarize(values: torch.Tensor) -> dict[str, float]:
    flat = values.reshape(-1).float().cpu()
    quantiles = torch.tensor([0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0], dtype=torch.float32)
    q = torch.quantile(flat, quantiles)
    return {
        "count": int(flat.numel()),
        "min": float(q[0]),
        "p001": float(q[1]),
        "p01": float(q[2]),
        "p05": float(q[3]),
        "median": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "p999": float(q[7]),
        "max": float(q[8]),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "frac_lt_0": float((flat < 0).float().mean()),
        "frac_in_0_1": float(((flat >= 0) & (flat <= 1)).float().mean()),
        "frac_le_0": float((flat <= 0).float().mean()),
        "frac_gt_1": float((flat > 1).float().mean()),
    }


def format_summary(name: str, summary: dict[str, float]) -> str:
    return (
        f"{name}: count={summary['count']} min={summary['min']:.6f} p001={summary['p001']:.6f} "
        f"p01={summary['p01']:.6f} p05={summary['p05']:.6f} median={summary['median']:.6f} "
        f"p95={summary['p95']:.6f} p99={summary['p99']:.6f} p999={summary['p999']:.6f} "
        f"max={summary['max']:.6f} mean={summary['mean']:.6f} std={summary['std']:.6f} "
        f"frac_lt_0={summary['frac_lt_0']:.6f} frac_le_0={summary['frac_le_0']:.6f} "
        f"frac_in_0_1={summary['frac_in_0_1']:.6f} "
        f"frac_gt_1={summary['frac_gt_1']:.6f}"
    )


def build_gsplat_inputs_from_raw(
    point_cloud: torch.Tensor,
    *,
    apply_exp: bool,
    apply_sigmoid: bool,
    apply_quat_normalize: bool,
) -> dict[str, torch.Tensor | int]:
    if point_cloud.ndim != 2 or point_cloud.shape[-1] < 59:
        raise ValueError(f"Expected raw point cloud shape (N, 59+), got {tuple(point_cloud.shape)}")

    pc = point_cloud[:, :59].to(dtype=torch.float32).contiguous()
    means = pc[:, 0:3]
    feat = pc[:, 4:52]
    feat_sh = feat.reshape(-1, 3, 16)
    features_dc = feat_sh[:, :, 0].unsqueeze(1).contiguous()
    features_rest = feat_sh[:, :, 1:].transpose(1, 2).contiguous()
    colors = torch.cat((features_dc, features_rest), dim=1).contiguous()

    raw_opacity = pc[:, 3]
    raw_scales = pc[:, 52:55]
    raw_rotation = pc[:, 55:59]

    if apply_sigmoid:
        opacities = torch.sigmoid(raw_opacity.clamp(RENDER_OPACITY_RAW_MIN, RENDER_OPACITY_RAW_MAX))
    else:
        opacities = raw_opacity
    if apply_exp:
        scales = torch.exp(raw_scales.clamp(RENDER_SCALE_RAW_MIN, RENDER_SCALE_RAW_MAX))
    else:
        scales = raw_scales
    if apply_quat_normalize:
        quats = _normalize_quaternions_with_identity_fallback(raw_rotation)
    else:
        quats = raw_rotation

    return {
        "means": means.unsqueeze(0),
        "quats": quats.unsqueeze(0),
        "scales": scales.unsqueeze(0),
        "opacities": opacities.unsqueeze(0),
        "colors": colors.unsqueeze(0),
        "sh_degree": 3,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = Standard3DGenDataset(
        obj_list=[args.obj_list],
        gs_path=args.gs_path,
        caption_path=None,
        mean_file=None,
        std_file=None,
    )
    sample_count = min(args.num_objects, len(dataset))
    sample_indices = random.sample(range(len(dataset)), sample_count)
    print(f"sample_indices={sample_indices}")

    per_object_scales: list[torch.Tensor] = []
    per_object_opacities: list[torch.Tensor] = []
    per_object_quats: list[torch.Tensor] = []
    per_object_quat_norms: list[torch.Tensor] = []
    for idx in sample_indices:
        raw_pc = dataset[idx]["point_cloud"].float()
        raw_scales = raw_pc[:, 52:55]
        raw_opacity = raw_pc[:, 3:4]
        raw_quat = raw_pc[:, 55:59]
        raw_quat_norm = raw_quat.norm(dim=-1, keepdim=True)
        per_object_scales.append(raw_scales)
        per_object_opacities.append(raw_opacity)
        per_object_quats.append(raw_quat)
        per_object_quat_norms.append(raw_quat_norm)
        if len(per_object_scales) <= args.print_per_object_limit:
            print(f"object_idx={idx}")
            print("  " + format_summary("raw_scale_slice", summarize(raw_scales)))
            print("  " + format_summary("exp(raw_scale_slice)", summarize(torch.exp(raw_scales.clamp(RENDER_SCALE_RAW_MIN, RENDER_SCALE_RAW_MAX)))))
            print("  " + format_summary("raw_opacity", summarize(raw_opacity)))
            print("  " + format_summary("sigmoid(raw_opacity)", summarize(torch.sigmoid(raw_opacity.clamp(RENDER_OPACITY_RAW_MIN, RENDER_OPACITY_RAW_MAX)))))
            print("  " + format_summary("raw_quat_norm", summarize(raw_quat_norm)))
            print("  " + format_summary("normalized_quat_norm", summarize(_normalize_quaternions_with_identity_fallback(raw_quat).norm(dim=-1, keepdim=True))))

    all_raw_scales = torch.cat(per_object_scales, dim=0)
    all_raw_opacity = torch.cat(per_object_opacities, dim=0)
    all_raw_quats = torch.cat(per_object_quats, dim=0)
    all_raw_quat_norms = torch.cat(per_object_quat_norms, dim=0)
    print("combined_distribution")
    print("  " + format_summary("raw_scale_slice", summarize(all_raw_scales)))
    print("  " + format_summary("exp(raw_scale_slice)", summarize(torch.exp(all_raw_scales.clamp(RENDER_SCALE_RAW_MIN, RENDER_SCALE_RAW_MAX)))))
    print("  " + format_summary("raw_opacity", summarize(all_raw_opacity)))
    print("  " + format_summary("sigmoid(raw_opacity)", summarize(torch.sigmoid(all_raw_opacity.clamp(RENDER_OPACITY_RAW_MIN, RENDER_OPACITY_RAW_MAX)))))
    print("  " + format_summary("raw_quat_norm", summarize(all_raw_quat_norms)))
    print("  " + format_summary("normalized_quat_norm", summarize(_normalize_quaternions_with_identity_fallback(all_raw_quats).norm(dim=-1, keepdim=True))))

    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping render comparison.")
        return

    legacy_renderer = try_import_legacy_renderer()
    if isinstance(legacy_renderer, Exception):
        raise legacy_renderer
    legacy_render, Camera, GaussianModel = legacy_renderer
    import gsplat

    device = torch.device("cuda")
    legacy_ref_cameras = load_legacy_reference_cameras(args.ref_camera_tar)
    legacy_cameras = prepare_legacy_cameras(legacy_ref_cameras, Camera, args.render_size)

    gsplat_ref_cameras = load_gsplat_reference_cameras(args.ref_camera_tar)
    gsplat_cameras = prepare_gsplat_cameras(gsplat_ref_cameras, args.render_size, device)

    num_cams = min(args.num_cams, len(legacy_cameras), int(gsplat_cameras["viewmats"].shape[0]))
    cam_indices = list(range(num_cams))
    background = torch.zeros(3, dtype=torch.float32, device=device)
    pipe = LegacyPipeConfig()

    exp_l1s: list[float] = []
    noexp_l1s: list[float] = []
    exp_mses: list[float] = []
    noexp_mses: list[float] = []
    sigmoid_l1s: list[float] = []
    nosigmoid_l1s: list[float] = []
    sigmoid_mses: list[float] = []
    nosigmoid_mses: list[float] = []
    normquat_l1s: list[float] = []
    nonormquat_l1s: list[float] = []
    normquat_mses: list[float] = []
    nonormquat_mses: list[float] = []

    for idx in sample_indices:
        raw_pc = dataset[idx]["point_cloud"].float()
        raw_pc_cuda = raw_pc.to(device)
        legacy_model = _build_gaussian_model_from_point_cloud(raw_pc_cuda, GaussianModel, detach_input=True)
        legacy_renders = []
        for cam in legacy_cameras[:num_cams]:
            legacy_renders.append(legacy_render(cam, legacy_model, pipe, background)["render"])
        legacy_renders_t = torch.stack(legacy_renders, dim=0)

        exp_render = _render_gsplat_batch(
            gsplat,
            build_gsplat_inputs_from_raw(
                raw_pc_cuda,
                apply_exp=True,
                apply_sigmoid=True,
                apply_quat_normalize=True,
            ),
            gsplat_cameras,
            cam_indices,
            device,
        )[0]
        noexp_render = _render_gsplat_batch(
            gsplat,
            build_gsplat_inputs_from_raw(
                raw_pc_cuda,
                apply_exp=False,
                apply_sigmoid=True,
                apply_quat_normalize=True,
            ),
            gsplat_cameras,
            cam_indices,
            device,
        )[0]
        nosigmoid_render = _render_gsplat_batch(
            gsplat,
            build_gsplat_inputs_from_raw(
                raw_pc_cuda,
                apply_exp=True,
                apply_sigmoid=False,
                apply_quat_normalize=True,
            ),
            gsplat_cameras,
            cam_indices,
            device,
        )[0]
        nonormquat_render = _render_gsplat_batch(
            gsplat,
            build_gsplat_inputs_from_raw(
                raw_pc_cuda,
                apply_exp=True,
                apply_sigmoid=True,
                apply_quat_normalize=False,
            ),
            gsplat_cameras,
            cam_indices,
            device,
        )[0]

        exp_diff = exp_render - legacy_renders_t
        noexp_diff = noexp_render - legacy_renders_t
        sigmoid_diff = exp_diff
        nosigmoid_diff = nosigmoid_render - legacy_renders_t
        normquat_diff = exp_diff
        nonormquat_diff = nonormquat_render - legacy_renders_t
        exp_l1 = float(exp_diff.abs().mean().item())
        noexp_l1 = float(noexp_diff.abs().mean().item())
        exp_mse = float(exp_diff.square().mean().item())
        noexp_mse = float(noexp_diff.square().mean().item())
        sigmoid_l1 = float(sigmoid_diff.abs().mean().item())
        nosigmoid_l1 = float(nosigmoid_diff.abs().mean().item())
        sigmoid_mse = float(sigmoid_diff.square().mean().item())
        nosigmoid_mse = float(nosigmoid_diff.square().mean().item())
        normquat_l1 = float(normquat_diff.abs().mean().item())
        nonormquat_l1 = float(nonormquat_diff.abs().mean().item())
        normquat_mse = float(normquat_diff.square().mean().item())
        nonormquat_mse = float(nonormquat_diff.square().mean().item())
        exp_l1s.append(exp_l1)
        noexp_l1s.append(noexp_l1)
        exp_mses.append(exp_mse)
        noexp_mses.append(noexp_mse)
        sigmoid_l1s.append(sigmoid_l1)
        nosigmoid_l1s.append(nosigmoid_l1)
        sigmoid_mses.append(sigmoid_mse)
        nosigmoid_mses.append(nosigmoid_mse)
        normquat_l1s.append(normquat_l1)
        nonormquat_l1s.append(nonormquat_l1)
        normquat_mses.append(normquat_mse)
        nonormquat_mses.append(nonormquat_mse)
        if len(sigmoid_l1s) <= args.print_per_object_limit:
            print(
                f"render_compare object_idx={idx} "
                f"l1(exp_vs_legacy)={exp_l1:.8f} l1(noexp_vs_legacy)={noexp_l1:.8f} "
                f"mse(exp_vs_legacy)={exp_mse:.8f} mse(noexp_vs_legacy)={noexp_mse:.8f} "
                f"l1(sigmoid_vs_legacy)={sigmoid_l1:.8f} l1(nosigmoid_vs_legacy)={nosigmoid_l1:.8f} "
                f"mse(sigmoid_vs_legacy)={sigmoid_mse:.8f} mse(nosigmoid_vs_legacy)={nosigmoid_mse:.8f} "
                f"l1(normquat_vs_legacy)={normquat_l1:.8f} l1(nonormquat_vs_legacy)={nonormquat_l1:.8f} "
                f"mse(normquat_vs_legacy)={normquat_mse:.8f} mse(nonormquat_vs_legacy)={nonormquat_mse:.8f}"
            )

    print("render_compare_summary")
    print(
        f"  mean_l1(exp_vs_legacy)={sum(exp_l1s) / len(exp_l1s):.8f} "
        f"mean_l1(noexp_vs_legacy)={sum(noexp_l1s) / len(noexp_l1s):.8f}"
    )
    print(
        f"  mean_mse(exp_vs_legacy)={sum(exp_mses) / len(exp_mses):.8f} "
        f"mean_mse(noexp_vs_legacy)={sum(noexp_mses) / len(noexp_mses):.8f}"
    )
    print(
        f"  mean_l1(sigmoid_vs_legacy)={sum(sigmoid_l1s) / len(sigmoid_l1s):.8f} "
        f"mean_l1(nosigmoid_vs_legacy)={sum(nosigmoid_l1s) / len(nosigmoid_l1s):.8f}"
    )
    print(
        f"  mean_mse(sigmoid_vs_legacy)={sum(sigmoid_mses) / len(sigmoid_mses):.8f} "
        f"mean_mse(nosigmoid_vs_legacy)={sum(nosigmoid_mses) / len(nosigmoid_mses):.8f}"
    )
    print(
        f"  mean_l1(normquat_vs_legacy)={sum(normquat_l1s) / len(normquat_l1s):.8f} "
        f"mean_l1(nonormquat_vs_legacy)={sum(nonormquat_l1s) / len(nonormquat_l1s):.8f}"
    )
    print(
        f"  mean_mse(normquat_vs_legacy)={sum(normquat_mses) / len(normquat_mses):.8f} "
        f"mean_mse(nonormquat_vs_legacy)={sum(nonormquat_mses) / len(nonormquat_mses):.8f}"
    )

    if sum(exp_l1s) / len(exp_l1s) < sum(noexp_l1s) / len(noexp_l1s):
        print("conclusion=torch.exp appears necessary: it matches the legacy renderer better.")
    else:
        print("conclusion=torch.exp does not improve the match to the legacy renderer in this sample.")
    if sum(sigmoid_l1s) / len(sigmoid_l1s) < sum(nosigmoid_l1s) / len(nosigmoid_l1s):
        print("conclusion=torch.sigmoid appears necessary: it matches the legacy renderer better.")
    else:
        print("conclusion=torch.sigmoid does not improve the match to the legacy renderer in this sample.")
    if sum(normquat_l1s) / len(normquat_l1s) < sum(nonormquat_l1s) / len(nonormquat_l1s):
        print("conclusion=quaternion normalization appears necessary: it matches the legacy renderer better.")
    else:
        print("conclusion=quaternion normalization does not improve the match to the legacy renderer in this sample.")


if __name__ == "__main__":
    main()
