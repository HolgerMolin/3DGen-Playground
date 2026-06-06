"""Diagnose the "small objects" symptom on a JiT checkpoint.

Three questions:
  1. Is it a render/camera framing artifact?  -> render GT objects through the
     EXACT same pipeline+camera as validation and see if they fill the frame.
  2. Does the model under-predict spatial extent (mean reversion / shrinkage)?
     -> compare denormalized xyz spread (std, p1-p99 span) of generated samples
     vs real objects, and normalized-space per-channel std vs the data's ~1.0.
  3. What does a ONE-STEP x0 prediction look like?  -> (a) from pure noise at the
     in-distribution max-noise step (t=0), and (b) denoising a noised GT at a few
     t levels, to see how extent collapses as noise rises.

Run (shares GPU with a live training job fine):
    python jit/diag_small_objects.py --ckpt output/.../0036000.pt
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from jit import train_gsplat as T
from jit.ab_x0renoise_vs_heun import build
from jit.sampling import _jit_predict_x0, resolve_sampling_shape, sample_model
from dataloaders.standard_3dgen_loader import Standard3DGenDataset
from dataloaders.text_3dgen_loader import Text3DGenDataset


def extent_stats(pc_raw, label):
    """pc_raw: (B, N, C) denormalized. Report xyz spatial extent (physical units)."""
    xyz = pc_raw[..., :3]                       # (B, N, 3)
    std = xyz.std(dim=1)                         # (B, 3) per-axis spread per object
    p99 = torch.quantile(xyz, 0.99, dim=1)
    p01 = torch.quantile(xyz, 0.01, dim=1)
    span = (p99 - p01)                           # (B, 3) robust bbox span
    # radius = mean distance of points from per-object centroid
    centroid = xyz.mean(dim=1, keepdim=True)
    radius = (xyz - centroid).norm(dim=-1).mean(dim=1)   # (B,)
    print(f"  [{label}]  n={xyz.shape[0]}")
    print(f"     xyz std  (per-axis, mean over objs): "
          f"{std.mean(0).tolist()}  | mean={std.mean().item():.4f}")
    print(f"     p1-p99 span (per-axis):              "
          f"{span.mean(0).tolist()}  | mean={span.mean().item():.4f}")
    print(f"     mean point radius from centroid:     {radius.mean().item():.4f}")
    return dict(std=std.mean().item(), span=span.mean().item(), radius=radius.mean().item())


def scale_stats(pc_raw, label, dc_only=True, rank_tables=None):
    """Report the activated gaussian scales (exp of scale channels) — per-gaussian size."""
    from utils.gsplat_render_util import _constrain_denormalized_point_cloud_for_render
    pc = _constrain_denormalized_point_cloud_for_render(
        pc_raw.float(), dc_only=dc_only, rank_transform_tables=rank_tables)
    scales = pc[..., 7:10] if dc_only else pc[..., 52:55]
    print(f"     [{label}] gaussian scale exp: mean={scales.mean().item():.5f} "
          f"median={scales.median().item():.5f} p99={torch.quantile(scales.flatten(),0.99).item():.5f}")
    return scales.mean().item()


def render_pc_raw(b, pc_raw, cam_idx=0):
    g = T._point_clouds_to_gsplat_inputs(
        pc_raw.to(next(b["model"].parameters()).device), dc_only=True,
        detach_input=True, rank_transform_tables=b["rank_tables"])
    with torch.no_grad():
        img = T._render_gsplat_batch(b["renderer"], g, b["train_cameras"],
                                     [cam_idx], next(b["model"].parameters()).device)
    return img[:, 0]  # (B,3,H,W)


def save_grid(imgs, path, cols=None):
    imgs = imgs.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    n, h, w, _ = imgs.shape
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = (imgs[i] * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(canvas).save(path)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="output/diag_small_objects")
    ap.add_argument("--n_gt", type=int, default=9)
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)

    b = build(args.ckpt, device)
    cfg = b["cfg"]
    print(f"[build] step={b['step']} model={cfg['model']} P_mean={cfg['P_mean']} "
          f"P_std={cfg['P_std']} val_sampler={cfg['val_sampler']} "
          f"val_steps={cfg['val_sampling_steps']} val_cfg={cfg['val_cfg_scale']}")
    norm_mean, norm_std = b["norm_mean"], b["norm_std"]
    p2s = b["plane_to_sphere"]
    feat_idx = torch.tensor(T.DC_ONLY_FEATURE_INDICES, dtype=torch.long)

    # ---- GT objects through the identical pipeline ----------------------------
    print("\n=== GROUND TRUTH (real objects, same render pipeline) ===")
    base = Standard3DGenDataset(
        obj_list=[cfg["obj_list"]], gs_path=cfg["gs_path"], caption_path=None,
        mean_file=cfg["mean_file"], std_file=cfg["std_file"],
        sphere2plane_path=cfg["sphere2plane_path"],
        exclude_keys_file=cfg.get("exclude_keys_file"),
        rank_transform_file=cfg.get("rank_transform_file"),
        clip_thresholds_file=cfg.get("clip_thresholds_file"),
        text_embed_path=cfg["text_embed_path"])
    ds = Text3DGenDataset(base, feature_indices=feat_idx, return_full_for_render=False)
    gt_planes = []
    for i in range(args.n_gt):
        item = ds[i]
        gt_planes.append(item[0].float())
    gt_plane = torch.stack(gt_planes).to(device)          # (B, C,128,128) or (B,16384,C)
    gt_pc = T._plane_to_point_cloud_batch(gt_plane, p2s)
    gt_pc_raw = T._denormalize_point_cloud(gt_pc, norm_mean, norm_std)
    gt_ext = extent_stats(gt_pc_raw, "GT")
    scale_stats(gt_pc_raw, "GT", rank_tables=b["rank_tables"])
    # normalized-space std (data should be ~1.0 per channel globally; per-object xyz varies)
    print(f"     GT normalized xyz std (per-axis): {gt_pc[..., :3].std(dim=1).mean(0).tolist()}")
    save_grid(render_pc_raw(b, gt_pc_raw), os.path.join(args.out, "gt.png"))

    # ---- One-step x0 from PURE NOISE at t=0 (in-dist max-noise step) ----------
    print("\n=== ONE-STEP x0 from pure noise (t=0, max-noise step) ===")
    n_tiles = int(cfg["val_grid_rows"]) * int(cfg["val_grid_cols"])
    shape = resolve_sampling_shape(model=b["model"], batch_size=n_tiles, in_channels=b["in_channels"])
    gen = torch.Generator(device="cpu").manual_seed(int(cfg["val_grid_seed"]))
    noise = torch.stack([torch.randn(shape[1:], generator=gen) for _ in range(n_tiles)]).to(device)
    cond = b["cond_pool"][:n_tiles].to(device)
    for cfg_scale in (1.0, float(cfg["val_cfg_scale"])):
        with torch.no_grad():
            x0 = _jit_predict_x0(model=b["model"], sample=noise,
                                 t_value=torch.tensor(0.0, device=device), cond_embeds=cond,
                                 cfg_scale=cfg_scale, cfg_interval=(0.0, 1.0))
        pc = T._plane_to_point_cloud_batch(x0.float(), p2s)
        pc_raw = T._denormalize_point_cloud(pc, norm_mean, norm_std)
        print(f"  -- cfg={cfg_scale} -- pred normalized RMS (all ch): {x0.float().square().mean().sqrt().item():.4f}")
        extent_stats(pc_raw, f"1step-t0 cfg{cfg_scale}")
        scale_stats(pc_raw, f"1step-t0 cfg{cfg_scale}", rank_tables=b["rank_tables"])
        save_grid(render_pc_raw(b, pc_raw), os.path.join(args.out, f"onestep_t0_cfg{cfg_scale}.png"))

    # ---- One-step x0 denoising a NOISED GT at several t (reconstruction) ------
    print("\n=== ONE-STEP x0 denoising noised GT (shrinkage vs noise level) ===")
    x0_gt = gt_plane if gt_plane.ndim == 4 else gt_plane
    # build x_t = t*x0 + (1-t)*eps in the model's plane space
    eps = torch.randn_like(gt_plane)
    for t_value in (0.05, 0.2, 0.5, 0.8):
        x_t = t_value * gt_plane + (1.0 - t_value) * eps
        with torch.no_grad():
            x0 = _jit_predict_x0(model=b["model"], sample=x_t,
                                 t_value=torch.tensor(float(t_value), device=device),
                                 cond_embeds=cond[:args.n_gt],
                                 cfg_scale=1.0, cfg_interval=(0.0, 1.0))
        pc = T._plane_to_point_cloud_batch(x0.float(), p2s)
        pc_raw = T._denormalize_point_cloud(pc, norm_mean, norm_std)
        st = extent_stats(pc_raw, f"recon t={t_value}")
        print(f"        -> xyz-std vs GT: {st['std']/gt_ext['std']*100:.0f}% of GT")

    # ---- Multi-step sampler (what validation actually shows) ------------------
    print("\n=== MULTI-STEP sample (validation sampler) ===")
    for sampler, steps, cfg_scale in [(cfg["val_sampler"], int(cfg["val_sampling_steps"]), float(cfg["val_cfg_scale"])),
                                       ("x0_renoise", 50, 3.0)]:
        s = sample_model(sampler=sampler, model=b["model"], shape=shape, cond_embeds=cond,
                         num_inference_steps=steps, device=device, predict_xstart=True,
                         diffusion_steps=1000, noise_schedule=cfg["noise_schedule"],
                         cfg_scale=cfg_scale, P_mean=float(cfg["P_mean"]), P_std=float(cfg["P_std"]),
                         initial_noise=noise)
        pc = T._plane_to_point_cloud_batch(s.float(), p2s)
        pc_raw = T._denormalize_point_cloud(pc, norm_mean, norm_std)
        st = extent_stats(pc_raw, f"{sampler}:{steps} cfg{cfg_scale}")
        print(f"        -> xyz-std vs GT: {st['std']/gt_ext['std']*100:.0f}% of GT")
        save_grid(render_pc_raw(b, pc_raw), os.path.join(args.out, f"multistep_{sampler}_s{steps}_c{cfg_scale}.png"))

    print("\nDONE. Compare gt.png vs onestep_*.png / multistep_*.png in", args.out)


if __name__ == "__main__":
    main()
