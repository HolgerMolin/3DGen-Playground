#!/usr/bin/env python
"""Stage 1 of the GaussianCube-vs-ours head-to-head: text -> GaussianCube voxel grid ->
explicit 3D Gaussians -> TRELLIS-format .ply + manifest.json.

Runs under the .gaussiancube venv (torch 1.12.1+cu116, py3.8), NOT .3dgen:
    PYTHONPATH=../GaussianCube \
        python jit/gaussiancube_gen.py [--limit N] [--max_gpu_gb G]

We reuse GaussianCube's diffusion model + CLIP text encoder + DPM-Solver sampler directly
(model/*.py is import-clean: no mpi4py / dist_util / diff_gaussian_rasterization). We do NOT use
their gaussian_renderer (CUDA rasterizer) — instead we decode the sampled voxel grid into explicit
Gaussians ourselves (inlined parse_volume_data) and render later through OUR gsplat pipeline via
jit/eval_trellis_compare.py.

NOTE on v1.0: GaussianCube's own inference.py only enables text conditioning for objaverse_v1.1
(text_cond = model_name == 'objaverse_v1.1'). The v1.0 checkpoint is the paper's text-to-3D model
(same text-conditioned config, encoder_dim=768, unconditional_gen=False), so we enable text-cond for
it here too. The 4-caption smoke test must confirm the outputs actually track the prompts.

Activation handling: GaussianCube emits *activated* opacity/scale (clamp[0,1]) and a normalized
quaternion. Our downstream renderer (utils/gsplat_render_util) expects *raw* logit opacity (it applies
sigmoid) and *log* scale (it applies exp). So we write logit(opacity) and log(scale) into the .ply,
and DC color / quaternion verbatim, exactly mirroring the TRELLIS .ply convention that the Stage-2
loader already understands.
"""
import os
import json
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement
from tqdm import tqdm

# GaussianCube repo (added to sys.path via PYTHONPATH); imports are rasterizer/mpi4py-free.
from model.unet import UNetModel
from model.clip import FrozenCLIPEmbedder
from model.dpmsolver import NoiseScheduleVP, model_wrapper, DPM_Solver
from model.gaussian_diffusion import get_named_beta_schedule

GC_ROOT = "../GaussianCube"

# repo_id + relative paths, mirroring inference.py MODEL_REPOS
MODEL_REPOS = {
    "objaverse_v1.0": dict(repo="BwZhang/GaussianCube-Objaverse",
                           ckpt="v1.0/objaverse_ckpt.pt", mean="v1.0/mean.pt", std="v1.0/std.pt", bound=0.5),
    "objaverse_v1.1": dict(repo="BwZhang/GaussianCube-Objaverse",
                           ckpt="v1.1/objaverse_ckpt.pt", mean="v1.1/mean.pt", std="v1.1/std.pt", bound=0.5),
}
MODEL_TYPES = {"xstart": "x_start", "v": "v", "eps": "noise"}


def init_volume_grid(bound=0.5, num_pts_each_axis=32):
    """Per-voxel-center coordinate grid, (num^3, 3). Copied verbatim from GaussianCube
    utils.script_util to avoid importing it (it transitively pulls in mpi4py)."""
    g = np.linspace(-bound, bound, num_pts_each_axis)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T


def parse_volume_data(volume, std_volume_xyz, active_sh_degree=0):
    """Decode a (C,H,W,D) GaussianCube volume into explicit Gaussian params. Inlined (no CUDA)
    copy of gaussian_renderer.parse_volume_data. active_sh_degree=0 -> sh_dim=0, C=14."""
    sh_dim = 3 * ((active_sh_degree + 1) ** 2 - 1)
    C, H, W, D = volume.shape
    volume = volume.permute(1, 2, 3, 0).reshape(-1, C)
    xyz = volume[:, :3] + std_volume_xyz
    features_dc = volume[:, 3:6].reshape((xyz.shape[0], 3, 1)).transpose(1, 2)        # (N,1,3)
    opacities = volume[:, 6 + sh_dim:7 + sh_dim].reshape((xyz.shape[0], 1)).clamp(0, 1)
    scales = volume[:, 7 + sh_dim:10 + sh_dim].reshape((xyz.shape[0], 3)).clamp(0, 1)
    rots = torch.nn.functional.normalize(volume[:, 10 + sh_dim:].reshape((xyz.shape[0], 4)))
    return {"xyz": xyz, "features_dc": features_dc[:, 0, :], "opacities": opacities,
            "scales": scales, "rots": rots}


def save_trellis_ply(path, xyz, opacity_act, f_dc, scale_act, rots):
    """Write a TRELLIS-format .ply (x,y,z,opacity,f_dc_0..2,scale_0..2,rot_0..3). Opacity/scale are
    GaussianCube's *activated* values; store logit/log so the downstream renderer's sigmoid/exp
    recover them. f_dc (SH-DC) and rots (wxyz) are written verbatim."""
    op = np.clip(opacity_act.reshape(-1).astype(np.float64), 1e-4, 1 - 1e-4)
    opacity_logit = np.log(op / (1.0 - op)).astype(np.float32)
    scale_log = np.log(np.clip(scale_act.astype(np.float64), 1e-6, None)).astype(np.float32)
    xyz = xyz.astype(np.float32)
    f_dc = f_dc.astype(np.float32)
    rots = rots.astype(np.float32)
    names = ["x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    cols = [xyz[:, 0], xyz[:, 1], xyz[:, 2], opacity_logit,
            f_dc[:, 0], f_dc[:, 1], f_dc[:, 2],
            scale_log[:, 0], scale_log[:, 1], scale_log[:, 2],
            rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]]
    dtype = [(n, "f4") for n in names]
    arr = np.empty(xyz.shape[0], dtype=dtype)
    for n, c in zip(names, cols):
        arr[n] = c.astype(np.float32)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(path)


def load_captions(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data[k] for k in sorted(data)]
    out = []
    for x in data:
        out.append(x if isinstance(x, str) else (x.get("caption") or x.get("text") or x.get("prompt")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="data/baseline_captions_500.json")
    ap.add_argument("--out_dir", default="output/gaussiancube_compare/objaverse_v1.0_seed0")
    ap.add_argument("--model_name", default="objaverse_v1.0", choices=list(MODEL_REPOS))
    ap.add_argument("--config", default=os.path.join(GC_ROOT, "configs/objaverse_text_cond.yml"))
    ap.add_argument("--guidance_scale", type=float, default=3.5)
    ap.add_argument("--steps", type=int, default=100, help="DPM-Solver steps (== rescale_timesteps)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--active_sh_degree", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_gpu_gb", type=float, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.max_gpu_gb is not None and device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gb / total), device.index or 0)
        print(f"[gc-gen] GPU memory capped at {args.max_gpu_gb} GB ({args.max_gpu_gb/total*100:.0f}%)")

    info = MODEL_REPOS[args.model_name]
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download(repo_id=info["repo"], filename=info["ckpt"], revision="main")
    mean_file = hf_hub_download(repo_id=info["repo"], filename=info["mean"], revision="main")
    std_file = hf_hub_download(repo_id=info["repo"], filename=info["std"], revision="main")
    bound = info["bound"]

    cfg = OmegaConf.load(args.config)
    cfg["model"]["precision"] = "32"
    image_size = cfg["model"]["image_size"]
    in_channels = cfg["model"]["in_channels"]

    print(f"[gc-gen] building UNet + loading {args.model_name} ckpt")
    model = UNetModel(**cfg["model"])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()

    betas = get_named_beta_schedule(cfg["diffusion"]["noise_schedule"], cfg["diffusion"]["steps"])
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=torch.from_numpy(betas).to(device))

    clip_text_encoder = FrozenCLIPEmbedder().eval().to(device)

    std_volume_xyz = torch.tensor(init_volume_grid(bound=bound, num_pts_each_axis=image_size),
                                  dtype=torch.float32, device=device).contiguous()
    mean = torch.load(mean_file).to(torch.float32).to(device).permute(3, 0, 1, 2).contiguous()
    std = torch.load(std_file).to(torch.float32).to(device).permute(3, 0, 1, 2).contiguous()

    # text conditioning is valid for both objaverse checkpoints (paper model = v1.0 is text-to-3D);
    # GaussianCube's stock inference.py only wired v1.1, which we correct here.
    text_cond = args.model_name in ("objaverse_v1.0", "objaverse_v1.1")
    model_type = MODEL_TYPES[cfg["diffusion"]["predict_type"]]

    captions = load_captions(args.captions)
    if args.limit is not None:
        captions = captions[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    manifest = []

    for idx, cap in enumerate(tqdm(captions, desc="GaussianCube gen")):
        ply = os.path.join(args.out_dir, f"{idx:04d}.ply")
        manifest.append({"idx": idx, "caption": cap, "ply": ply})
        if os.path.exists(ply):
            continue

        text_features = clip_text_encoder.encode(cap)                       # (1,77,768)
        condition = {"cond_text": text_features}
        unconditional_condition = {"cond_text": torch.zeros_like(text_features)}

        model_fn = model_wrapper(
            model, noise_schedule, model_type=model_type, model_kwargs={},
            guidance_type="classifier-free", guidance_scale=args.guidance_scale,
            condition=condition, unconditional_condition=unconditional_condition,
        )
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

        g = torch.Generator(device=device).manual_seed(args.seed * 1_000_003 + idx)
        with torch.no_grad():
            noise = torch.randn((1, in_channels, image_size, image_size, image_size),
                                device=device, generator=g) * args.temperature
            samples = dpm_solver.sample(x=noise, steps=args.steps, t_start=1.0, t_end=1 / 1000,
                                        order=2, skip_type="time_uniform", method="adaptive")
            samples_denorm = samples * std + mean                            # (1,C,H,W,D)
            pc = parse_volume_data(samples_denorm[0], std_volume_xyz, args.active_sh_degree)

        save_trellis_ply(
            ply,
            pc["xyz"].cpu().numpy(),
            pc["opacities"].cpu().numpy(),
            pc["features_dc"].cpu().numpy(),
            pc["scales"].cpu().numpy(),
            pc["rots"].cpu().numpy(),
        )
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[gc-gen] wrote {len(manifest)} entries -> {manifest_path}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
