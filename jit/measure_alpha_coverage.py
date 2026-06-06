#!/usr/bin/env python
"""Measure GT alpha coverage at different `render_zoom_factor` values.

The render losses (L1, alpha_L1, LPIPS) are pixel-mean over the full frame, so the gradient
signal density per object-pixel is roughly `alpha_mean` of the headline loss — i.e. if 80%
of pixels are background, render losses are running at ~20% of their effective signal density.
Zooming in via narrowed FOV pushes alpha_mean / bbox_fill up by approximately zoom².

Usage:
    PYTHONPATH=. .3dgen/bin/python jit/measure_alpha_coverage.py \\
        --zooms 1.0,1.6,2.0 --num_samples 32
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/gaussian-splatting")

import gsplat  # noqa: E402
from utils.gsplat_render_util import (  # noqa: E402
    _load_reference_cameras, _prepare_train_cameras, _render_gsplat_batch,
    _plane_to_point_cloud_batch, _denormalize_point_cloud, _point_clouds_to_gsplat_inputs,
)
from utils.plane_utils import load_sphere2plane  # noqa: E402
from dataloaders.standard_3dgen_loader import Standard3DGenDataset  # noqa: E402
from dataloaders.text_3dgen_loader import Text3DGenDataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ref_camera_tar", default="artifacts/ref_camera.tar.gz")
ap.add_argument("--obj_list", default="data/gaussianverse/all_obj_list_filtered.json")
ap.add_argument("--gs_path", default="data/gaussianverse/")
ap.add_argument("--mean_file", default="data/stats/all_mean_postfix.pt")
ap.add_argument("--std_file", default="data/stats/all_std_postfix.pt")
ap.add_argument("--sphere2plane_path", default="data/gaussianverse/sphere2plane.npy")
ap.add_argument("--exclude_keys_file", default="data/outlier_keys_8sigma.json")
ap.add_argument("--rank_transform_file", default="data/stats/rank_quantiles_8ch_clipped.pt")
ap.add_argument("--clip_thresholds_file", default="data/stats/clip_thresholds_opacity_scales.pt")
ap.add_argument("--text_embed_path", default="object_classification/text_tokens")
ap.add_argument("--render_size", type=int, default=224)
ap.add_argument("--num_cam", type=int, default=4)
ap.add_argument("--num_samples", type=int, default=32)
ap.add_argument("--zooms", type=str, default="1.0,1.6,2.0",
                help="Comma-separated GLOBAL zoom factors to sweep (ignored when "
                     "--per_sample_zoom_file is set).")
ap.add_argument("--per_sample_zoom_file", type=str, default=None,
                help="If set, ignore --zooms and instead render once using the per-(sample, "
                     "camera) zooms from this .pt (produced by data/build_per_sample_render_fov.py). "
                     "Reports the resulting alpha coverage so you can verify the precompute "
                     "without launching training.")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

device = torch.device("cuda")
torch.manual_seed(a.seed)

ref_cams = _load_reference_cameras(a.ref_camera_tar)[: a.num_cam]
print(f"[cams] {a.num_cam} of {len(_load_reference_cameras(a.ref_camera_tar))} ref cams; "
      f"base fovx={np.degrees(ref_cams[0]['fovx']):.1f}°  fovy={np.degrees(ref_cams[0]['fovy']):.1f}°")

s2p = load_sphere2plane(a.sphere2plane_path, expected_points=16384)
plane_to_sphere = torch.tensor(np.argsort(s2p), dtype=torch.long, device=device)

base = Standard3DGenDataset(
    obj_list=[a.obj_list], gs_path=a.gs_path, caption_path=None,
    mean_file=a.mean_file, std_file=a.std_file, sphere2plane_path=a.sphere2plane_path,
    exclude_keys_file=a.exclude_keys_file, rank_transform_file=a.rank_transform_file,
    clip_thresholds_file=a.clip_thresholds_file, text_embed_path=a.text_embed_path,
)
ds = Text3DGenDataset(base, feature_indices=None, return_full_for_render=True,
                      preload_to_cpu=False, lazy_cache_to_cpu=False)
loader = torch.utils.data.DataLoader(ds, batch_size=a.num_samples, shuffle=True, num_workers=0)
batch = next(iter(loader))
x_norm = batch[0].to(device)
batch_hashes = list(batch[-1])   # last element is the hash tuple from Text3DGenDataset

norm_mean = torch.load(a.mean_file, weights_only=False).to(device)
norm_std = torch.load(a.std_file, weights_only=False).to(device)
gt_pc_norm = _plane_to_point_cloud_batch(x_norm.float(), plane_to_sphere)
gt_pc_raw = _denormalize_point_cloud(gt_pc_norm, norm_mean, norm_std)
gt_gauss = _point_clouds_to_gsplat_inputs(gt_pc_raw, dc_only=False, detach_input=True)
gt_gauss = {**gt_gauss, "colors": gt_gauss["colors"][..., :1, :], "sh_degree": 0}

# Build the per-sample zoom tensor (shape (B, num_cam)) when the table is provided.
# This lets us probe the precompute output: render at exactly the zooms training will use
# and verify alpha_mean / bbox_fill / clip_frac.
per_sample_cam_zooms = None
if a.per_sample_zoom_file:
    payload = torch.load(a.per_sample_zoom_file, map_location="cpu", weights_only=False)
    table_hashes = list(payload["hash_keys"])
    zoom_factors = payload["zoom_factors"].float()
    h2r = {h: i for i, h in enumerate(table_hashes)}
    rows = torch.tensor([h2r[h] for h in batch_hashes], dtype=torch.long)
    per_sample_cam_zooms = zoom_factors[rows][:, : a.num_cam].to(device)  # (B, num_cam)
    print(f"[per_sample_zoom] loaded {a.per_sample_zoom_file}: "
          f"batch zooms mean={float(per_sample_cam_zooms.mean()):.3f} "
          f"min={float(per_sample_cam_zooms.min()):.3f} "
          f"max={float(per_sample_cam_zooms.max()):.3f}")
    zooms = [None]   # one pseudo-iteration; the global zoom_factor is irrelevant
else:
    zooms = [float(z) for z in a.zooms.split(",")]

print(f"\n  N={a.num_samples}  cams={a.num_cam}  size={a.render_size}x{a.render_size} = {a.render_size**2} pixels\n")
print(f"  {'zoom':>10} {'fovx°':>7} | {'alpha_mean':>10} {'frac>0.5':>9} {'frac>0.1':>9} | "
      f"{'bbox_fill':>9} {'clip_frac':>9} | {'signal_x':>9}")
base_alpha = None
for z in zooms:
    # Per-sample mode: keep camera bundle at zoom=1.0 and apply per-sample zooms downstream.
    # Global-sweep mode: bake the global zoom into the camera bundle (back-compat path).
    train_cams = _prepare_train_cameras(
        ref_cams, a.render_size, device,
        zoom_factor=1.0 if z is None else z,
    )
    with torch.no_grad():
        rgb, alpha = _render_gsplat_batch(
            gsplat, gt_gauss, train_cams, list(range(a.num_cam)), device, return_alpha=True,
            per_sample_zooms=per_sample_cam_zooms,
        )
    a_t = alpha[:, :, 0, :, :]  # (B, num_cam, H, W)
    am = float(a_t.mean())
    f05 = float((a_t > 0.5).float().mean())
    f01 = float((a_t > 0.1).float().mean())
    if base_alpha is None:
        base_alpha = am
    sig = am / base_alpha
    bbox_fills = []
    clip_count = 0
    total = 0
    for b in range(a_t.shape[0]):
        for c in range(a_t.shape[1]):
            m = (a_t[b, c] > 0.1).float()
            total += 1
            if m.sum() < 4:
                continue
            ys, xs = torch.where(m > 0)
            H = a_t.shape[2]; W = a_t.shape[3]
            y0, y1, x0, x1 = ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item()
            bbox_fills.append(((y1 - y0 + 1) * (x1 - x0 + 1)) / (H * W))
            # Clipping = bbox touches any frame edge -> object cut off by zoom
            if y0 == 0 or y1 == H - 1 or x0 == 0 or x1 == W - 1:
                clip_count += 1
    bb = float(np.mean(bbox_fills)) if bbox_fills else 0.0
    clip_frac = clip_count / max(total, 1)
    if z is None:
        # Per-sample mode: FOV is per (sample, cam), so just report the batch's mean zoom.
        label = "per-sample"
        new_fov_deg = float(np.degrees(2.0 * np.arctan(
            np.tan(ref_cams[0]['fovx'] / 2.0) / float(per_sample_cam_zooms.mean())
        )))
    else:
        label = f"{z:.2f}"
        new_fov_deg = float(np.degrees(2.0 * np.arctan(np.tan(ref_cams[0]['fovx'] / 2.0) / z)))
    print(f"  {label:>10} {new_fov_deg:>7.2f} | {am:>10.4f} {f05:>9.4f} {f01:>9.4f} | "
          f"{bb:>9.4f} {clip_frac*100:>8.2f}% | {sig:>8.2f}x")

print("\nkey: alpha_mean = avg alpha = object share of the L1/LPIPS denominator.")
print("     bbox_fill  = fraction of frame inside object's axis-aligned bbox.")
print("     clip_frac  = % of renders where the bbox touches a frame edge (object cut off).")
print("     signal_x   = alpha_mean(z) / alpha_mean(1.0) = relative effective gradient density.")
