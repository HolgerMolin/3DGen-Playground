"""A/B: heun vs x0_renoise on the SAME chamfer_feature checkpoint.

Decisive test for the chamfer sampling-collapse hypothesis (CLAUDE.md TODO):
permutation-invariant Chamfer discards the per-cell x_t<->x0 correspondence that
accumulating samplers (heun) integrate, so heun collapses; x0_renoise never
accumulates per-cell, so it should tolerate the permutation freedom.

Reproduces _run_validation_grid exactly (same EMA weights, 4 default prompts,
grid_seed, cfg, steps) and only swaps val_sampler. Renders both grids + a
one-shot direct-x0 sanity render. Run standalone (single GPU, shares with any
live training job):

    python jit/ab_x0renoise_vs_heun.py --ckpt output/.../0038500.pt
"""
import argparse
import os

import numpy as np
import torch

from jit import train_gsplat as T
from jit.sampling import _jit_predict_x0


def build(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = vars(ck["args"]) if hasattr(ck["args"], "__dict__") else ck["args"]
    assert cfg["sh_degree0_only"], "this driver assumes the DC-only (14ch) config"
    in_channels = len(T.DC_ONLY_FEATURE_INDICES)
    feature_indices = torch.tensor(T.DC_ONLY_FEATURE_INDICES, dtype=torch.long)

    n_tiles = int(cfg["val_grid_rows"]) * int(cfg["val_grid_cols"])
    cond_pool = T._load_val_prompts(cfg.get("val_prompts_file"), n_tiles, device)
    text_dim = int(cond_pool.shape[1])

    model = T.JiT_3DGS_models[cfg["model"]](
        input_size=128, in_channels=in_channels, text_dim=text_dim,
        class_dropout_prob=float(cfg["class_dropout_prob"]),
        learn_sigma=False, bottleneck=bool(cfg["bottleneck"]),
    )
    null_path = cfg["null_text_token_path"] or T._default_null_path(cfg["text_embed_path"])
    model.load_null_embeddings(torch.from_numpy(T.load_null_text_token(null_path).astype(np.float32)))
    missing, unexpected = model.load_state_dict(ck["ema"], strict=False)
    if missing or unexpected:
        print(f"[load] missing={list(missing)} unexpected={list(unexpected)}")
    model.to(device).eval()

    plane_to_sphere = T.load_sphere2plane(cfg["sphere2plane_path"], 128 * 128)
    rank_tables = T.load_rank_transform_payload_torch(cfg["rank_transform_file"], device=device)

    mean_full = torch.load(cfg["mean_file"], weights_only=True).float().cpu()
    std_full = torch.load(cfg["std_file"], weights_only=True).float().cpu()
    if rank_tables is not None:
        ridx = torch.tensor(rank_tables["channels"], dtype=torch.long)
        mean_full = mean_full.clone(); std_full = std_full.clone()
        mean_full[ridx] = 0.0; std_full[ridx] = 1.0
    norm_mean = mean_full[feature_indices]
    norm_std = std_full[feature_indices]

    ref_cameras = T._load_reference_cameras(cfg["ref_camera_tar"])
    renderer = T._try_import_renderer()
    assert not isinstance(renderer, Exception), f"renderer import failed: {renderer}"
    train_cameras = T._prepare_train_cameras(ref_cameras, int(cfg["train_render_size"]), device)

    return dict(
        model=model, cfg=cfg, in_channels=in_channels, cond_pool=cond_pool,
        plane_to_sphere=plane_to_sphere, norm_mean=norm_mean, norm_std=norm_std,
        rank_tables=rank_tables, train_cameras=train_cameras, renderer=renderer,
        step=int(ck.get("step", 0)),
    )


def run_grid(b, sampler, out_root, steps=None, cfg_scale=None, tag=None):
    cfg = b["cfg"]
    steps = int(cfg["val_sampling_steps"]) if steps is None else int(steps)
    cfg_scale = float(cfg["val_cfg_scale"]) if cfg_scale is None else float(cfg_scale)
    out_dir = os.path.join(out_root, tag or sampler)
    os.makedirs(out_dir, exist_ok=True)
    path = T._run_validation_grid(
        model=b["model"], plane_to_sphere=b["plane_to_sphere"],
        norm_mean=b["norm_mean"], norm_std=b["norm_std"],
        train_cameras=b["train_cameras"], renderer_tuple=b["renderer"],
        output_dir=out_dir, epoch=0, step=b["step"],
        device=next(b["model"].parameters()).device,
        in_channels=b["in_channels"], cond_pool=b["cond_pool"],
        grid_seed=int(cfg["val_grid_seed"]), camera_idx=int(cfg["val_grid_camera_idx"]),
        grid_rows=int(cfg["val_grid_rows"]), grid_cols=int(cfg["val_grid_cols"]),
        dc_only=True, predict_xstart=True, noise_schedule=cfg["noise_schedule"],
        diffusion_steps=1000, val_sampling_steps=steps,
        val_sampler=sampler, cfg_scale=cfg_scale,
        P_mean=float(cfg["P_mean"]), P_std=float(cfg["P_std"]),
        rank_transform_tables=b["rank_tables"],
    )
    print(f"[grid] {tag or sampler} (sampler={sampler} steps={steps} cfg={cfg_scale}): {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_root", default="output/ab_x0renoise")
    ap.add_argument("--experiments", default="heun:50:1.5,x0_renoise:50:1.5",
                    help="comma list of sampler:steps:cfg (steps/cfg optional, '-' = config default)")
    args = ap.parse_args()
    device = torch.device("cuda")
    b = build(args.ckpt, device)
    print(f"[build] step={b['step']} model={b['cfg']['model']} "
          f"P_mean={b['cfg']['P_mean']} P_std={b['cfg']['P_std']} seed={b['cfg']['val_grid_seed']}")
    for spec in args.experiments.split(","):
        parts = (spec.strip() + "::").split(":")
        sampler = parts[0]
        steps = None if parts[1] in ("", "-") else int(parts[1])
        cfgv = None if parts[2] in ("", "-") else float(parts[2])
        tag = f"{sampler}_s{steps or b['cfg']['val_sampling_steps']}_c{cfgv or b['cfg']['val_cfg_scale']}"
        run_grid(b, sampler, args.out_root, steps=steps, cfg_scale=cfgv, tag=tag)


if __name__ == "__main__":
    main()
