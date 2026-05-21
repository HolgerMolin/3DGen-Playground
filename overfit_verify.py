"""
overfit_verify.py — replay the four training losses against the fixed overfit
sample, across N random seeds, to verify what a checkpoint has actually learned.

Built for autoresearch verification: given a JiT checkpoint produced by an
agent, reconstruct the dataset/model/diffusion from the args saved inside the
checkpoint, take the first sample (the one --overfit 1 trains on), and
recompute the four losses we currently train on:

  1. flow-matching MSE
  2. render L1 (rendered RGB vs GT-rendered RGB)
  3. render alpha-mask L1
  4. render LPIPS

across `--num_seeds` seeds. Each seed reseeds python/numpy/torch so the
sampled (t, noise, render-camera) triple varies. Outputs per-seed values plus
mean/std/min/max to stdout AND a JSON file next to the checkpoint.

Usage:
    python overfit_verify.py --checkpoint output/overfit_smoke/0010000.pt
    python overfit_verify.py --checkpoint <path> --num_seeds 32 --weights ema

Notes:
  * The training-time `render_loss_noise_cutoff` mask is intentionally NOT
    applied — we want each seed to produce a value, not be skipped.
  * The training-time per-loss weights are NOT applied either — raw losses are
    reported so callers can apply their own weighting if needed.
  * Channel-loss weights (off in the standard overfit config) are not applied.
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GS_ROOT = REPO_ROOT / "submodules" / "gaussian-splatting"
if GS_ROOT.is_dir() and str(GS_ROOT) not in sys.path:
    sys.path.insert(0, str(GS_ROOT))

from dataloaders.standard_3dgen_loader import Standard3DGenDataset
from dataloaders.class_3dgen_loader import (
    Class3DGenDataset, DC_ONLY_FEATURE_INDICES, FULL_3DGS_FEATURE_DIM,
)
from jit.models import JiT_3DGS_models
from jit.diffusion import create_diffusion
from utils.plane_utils import load_sphere2plane
from utils.gsplat_render_util import (
    _compute_render_loss_for_batch,
    _load_reference_cameras,
    _prepare_train_cameras,
    _try_import_renderer,
    _try_import_lpips,
    load_rank_transform_payload_torch,
)


# Defaults mirror jit/train_gsplat.py's argparse for any keys absent from older
# checkpoints, so we can rebuild a faithful args namespace from ckpt['args'].
_ARG_DEFAULTS: dict[str, Any] = {
    "class_dropout_prob": 0.1,
    "label_embed_init_std": 0.1,
    "bottleneck": False,
    "predict_xstart": True,
    "noise_schedule": "squaredcos_cap_v2",
    "aux_classifier": False,
    "sh_degree0_only": True,
    "gradient_checkpointing": False,
    "P_mean": -0.8,
    "P_std": 0.8,
    "render_loss_num_cam": 1,
    "train_render_size": 224,
    "lpips_net": "vgg",
    "perceptual_backend": "lpips",
    "rank_transform_file": None,
    "exclude_keys_file": None,
    "mean_file": None,
    "std_file": None,
}


def _args_from_ckpt(d: dict) -> argparse.Namespace:
    out = dict(_ARG_DEFAULTS)
    for k, v in d.items():
        if v is None and k in _ARG_DEFAULTS:
            continue
        out[k] = v
    return argparse.Namespace(**out)


def _strip_compile_prefix(state_dict: dict) -> dict:
    """torch.compile prepends `_orig_mod.` to keys; strip if uniformly present."""
    if not state_dict:
        return state_dict
    if all(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
    return state_dict


def _sample_jit_timesteps(batch_size: int, num_timesteps: int, device: torch.device,
                           p_mean: float, p_std: float):
    probs = torch.sigmoid(torch.randn(batch_size, device=device) * p_std + p_mean)
    eps = 1e-4
    t_value = probs.clamp(min=eps, max=1.0 - eps)
    t_discrete = torch.clamp(
        (t_value * (num_timesteps - 1)).round().long(),
        min=0, max=num_timesteps - 1,
    )
    return t_value, t_discrete


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_GT_NAME_RE = re.compile(
    r"^gt_idx(?P<idx>\d+)_class(?P<cls>\d+)_(?P<hash>.+)_cams(?P<cams>[\d\-]+)\.png$"
)


def _hash_key_from_gt_dir(gt_dir: Path, sample_index: int) -> Optional[tuple[str, int]]:
    """Parse `(hash_key, class)` from the training-time overfit-GT filename for
    `sample_index`. Returns None if the file is absent (older runs / no GT
    dump) or the name doesn't match the expected pattern."""
    if not gt_dir.is_dir():
        return None
    matches = sorted(glob.glob(str(gt_dir / f"gt_idx{sample_index:02d}_class*_cams*.png")))
    if not matches:
        return None
    name = Path(matches[0]).name
    m = _GT_NAME_RE.match(name)
    if not m:
        return None
    return m.group("hash"), int(m.group("cls"))


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a JiT overfit checkpoint by recomputing the four training "
            "losses (MSE, render L1, alpha L1, LPIPS) across N random seeds on "
            "the fixed overfit sample."
        ),
    )
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to a .pt checkpoint with keys {model|ema, args, step}.")
    parser.add_argument("--num_seeds", type=int, default=32)
    parser.add_argument("--seed_base", type=int, default=0,
                        help="Seeds used are seed_base + 0..num_seeds-1.")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema",
                        help="Which weight set to verify; falls back to the other if absent.")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="Dataset index of the overfit sample (matches --overfit 1: first).")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Where to write the result JSON (default: <ckpt_dir>/overfit_verify.json).")
    parser.add_argument("--device", type=str, default="cuda")
    cli = parser.parse_args()

    if cli.device == "cuda" and not torch.cuda.is_available():
        print("[overfit_verify] CUDA unavailable — render losses require CUDA, aborting.",
              file=sys.stderr)
        return 2
    device = torch.device(cli.device)

    ckpt_path = Path(cli.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        print(f"[overfit_verify] checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2
    print(f"[overfit_verify] checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "args" not in ckpt:
        print("[overfit_verify] checkpoint missing 'args' — cannot reconstruct config.",
              file=sys.stderr)
        return 2
    args = _args_from_ckpt(ckpt["args"])
    saved_step = int(ckpt.get("step", -1))
    print(f"[overfit_verify] saved_step={saved_step} model={args.model} "
          f"sh_degree0_only={args.sh_degree0_only}")

    # === Reconstruct dataset (matches train_gsplat.py: base → class wrap) ===
    print(f"[overfit_verify] loading base dataset (obj_list={args.obj_list})")
    base_dataset = Standard3DGenDataset(
        obj_list=[args.obj_list],
        gs_path=args.gs_path,
        caption_path=None,
        mean_file=args.mean_file,
        std_file=args.std_file,
        sphere2plane_path=args.sphere2plane_path,
        exclude_keys_file=getattr(args, "exclude_keys_file", None),
        rank_transform_file=getattr(args, "rank_transform_file", None),
    )

    with open(args.class_map, "r") as f:
        class_map = json.load(f)
    num_classes = max(v for v in class_map.values() if v >= 0) + 1
    print(f"[overfit_verify] num_classes={num_classes}")

    if bool(getattr(args, "sh_degree0_only", True)):
        feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
        in_channels = len(DC_ONLY_FEATURE_INDICES)
    else:
        feature_indices = None
        in_channels = FULL_3DGS_FEATURE_DIM

    point_cloud_shape = tuple(base_dataset[0]["point_cloud"].shape)
    num_points = (
        int(point_cloud_shape[-2] * point_cloud_shape[-1])
        if len(point_cloud_shape) == 3
        else int(point_cloud_shape[0])
    )
    plane_to_sphere = load_sphere2plane(args.sphere2plane_path, num_points)

    # The overfit train run uses return_full_for_render=False (feature_indices
    # is set when sh_degree0_only=True). Match that here so the sample shape
    # the model sees is identical to training.
    dataset = Class3DGenDataset(
        base_dataset, class_map,
        feature_indices=feature_indices,
        return_full_for_render=False,
        preload_to_cpu=False,
        lazy_cache_to_cpu=False,
    )

    pc, label, hash_key = dataset[cli.sample_index]
    print(f"[overfit_verify] overfit sample: idx={cli.sample_index} "
          f"class={int(label)} hash={hash_key}")

    # Integrity check: training's overfit-GT dump encodes the sample's hash_key
    # in the filename. If that file is present, verify it matches what we just
    # loaded — catches edits to obj_list / class_map / exclude_keys that would
    # silently shift dataset[0] to a different object between training and
    # verification.
    gt_dir = Path(args.results_dir).expanduser() / "overfit_gt"
    if not gt_dir.is_absolute():
        gt_dir = (REPO_ROOT / gt_dir).resolve()
    gt_info = _hash_key_from_gt_dir(gt_dir, cli.sample_index)
    sample_integrity: dict[str, Any]
    if gt_info is None:
        sample_integrity = {"status": "skipped", "reason": f"no GT dump at {gt_dir}"}
        print(f"[overfit_verify] WARNING: no overfit-GT file found in {gt_dir}; "
              f"cannot cross-check the sample identity. Trusting dataset[{cli.sample_index}].")
    else:
        gt_hash, gt_class = gt_info
        if gt_hash != str(hash_key) or gt_class != int(label):
            print(
                "[overfit_verify] FATAL: dataset sample does not match training GT.\n"
                f"  training GT (from {gt_dir}): hash={gt_hash} class={gt_class}\n"
                f"  this run's dataset[{cli.sample_index}]: hash={hash_key} class={int(label)}\n"
                "Likely cause: obj_list, exclude_keys_file, or class_map content changed "
                "between training and verification.",
                file=sys.stderr,
            )
            return 3
        sample_integrity = {
            "status": "matched",
            "gt_dir": str(gt_dir),
            "gt_hash": gt_hash,
            "gt_class": gt_class,
        }
        print(f"[overfit_verify] sample integrity OK (hash matches training GT in {gt_dir})")

    x = pc.float().unsqueeze(0).to(device)
    y = torch.tensor([int(label)], dtype=torch.long, device=device)

    # === Norm stats + rank-transform tables (same overrides as training) ===
    norm_mean_full = torch.load(args.mean_file, weights_only=True).float().to(device)
    norm_std_full = torch.load(args.std_file, weights_only=True).float().to(device)
    rank_transform_tables = load_rank_transform_payload_torch(
        getattr(args, "rank_transform_file", None), device=device,
    )
    if rank_transform_tables is not None:
        rank_idx = torch.tensor(
            rank_transform_tables["channels"], dtype=torch.long, device=device,
        )
        norm_mean_full = norm_mean_full.clone()
        norm_std_full = norm_std_full.clone()
        norm_mean_full[rank_idx] = 0.0
        norm_std_full[rank_idx] = 1.0
    if feature_indices is not None:
        norm_mean = norm_mean_full[feature_indices]
        norm_std = norm_std_full[feature_indices]
    else:
        norm_mean = norm_mean_full
        norm_std = norm_std_full

    # === Renderer + perceptual ===
    print(f"[overfit_verify] loading reference cameras: {args.ref_camera_tar}")
    ref_cameras = _load_reference_cameras(args.ref_camera_tar)
    renderer_probe = _try_import_renderer()
    if isinstance(renderer_probe, Exception):
        print(f"[overfit_verify] gsplat renderer import failed: {renderer_probe}",
              file=sys.stderr)
        return 2
    renderer_tuple = renderer_probe
    train_cameras = _prepare_train_cameras(
        ref_cameras, int(getattr(args, "train_render_size", 224)), device,
    )

    perceptual_backend = str(getattr(args, "perceptual_backend", "lpips"))
    if perceptual_backend == "dinov2":
        from utils.dinov2_perceptual import DinoV2Perceptual
        lpips_fn = DinoV2Perceptual().to(device).eval()
    else:
        lpips_probe = _try_import_lpips()
        if isinstance(lpips_probe, Exception):
            print(f"[overfit_verify] LPIPS import failed: {lpips_probe}", file=sys.stderr)
            return 2
        lpips_fn = lpips_probe.LPIPS(net=str(getattr(args, "lpips_net", "vgg"))).to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # === Model ===
    print(f"[overfit_verify] building model={args.model} "
          f"in_channels={in_channels} num_classes={num_classes}")
    model = JiT_3DGS_models[args.model](
        input_size=128,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=float(getattr(args, "class_dropout_prob", 0.1)),
        learn_sigma=False,
        gradient_checkpointing=False,
        bottleneck=bool(getattr(args, "bottleneck", False)),
        aux_classifier=bool(getattr(args, "aux_classifier", False)),
        label_embed_init_std=float(getattr(args, "label_embed_init_std", 0.1)),
    )

    weights_key = cli.weights
    if weights_key not in ckpt:
        fallback = "model" if weights_key == "ema" else "ema"
        if fallback not in ckpt:
            print("[overfit_verify] checkpoint has neither 'model' nor 'ema' weights.",
                  file=sys.stderr)
            return 2
        print(f"[overfit_verify] '{weights_key}' weights absent — using '{fallback}'.")
        weights_key = fallback
    state = _strip_compile_prefix(ckpt[weights_key])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[overfit_verify] state_dict mismatch ({weights_key}): "
              f"missing={list(missing)} unexpected={list(unexpected)}")
    model = model.to(device).eval()

    # === Diffusion ===
    diffusion = create_diffusion(
        timestep_respacing="",
        noise_schedule=str(args.noise_schedule),
        learn_sigma=False,
        predict_xstart=bool(getattr(args, "predict_xstart", True)),
    )

    # === Per-seed evaluation ===
    per_seed: list[dict[str, Any]] = []
    print(f"[overfit_verify] running {cli.num_seeds} seeds "
          f"(base={cli.seed_base}, P_mean={float(args.P_mean):.3f}, "
          f"P_std={float(args.P_std):.3f})")
    t_start = time.perf_counter()
    with torch.no_grad():
        for k in range(int(cli.num_seeds)):
            seed = int(cli.seed_base) + k
            _seed_everything(seed)

            t_value, t_discrete = _sample_jit_timesteps(
                x.shape[0], diffusion.num_timesteps, device,
                float(args.P_mean), float(args.P_std),
            )
            noise = torch.randn_like(x)
            x_t = diffusion.flow_matching_q_sample(x, t_value, noise=noise)
            model_out = model(x_t, t_discrete, y=y)
            x0_pred = model_out.float()
            mse = ((x - x0_pred) ** 2).mean()

            # `_compute_render_loss_for_batch` consumes `random.sample` to
            # pick `num_cam` cameras; this is exactly the training path.
            render_l1, alpha_l1, lpips_loss = _compute_render_loss_for_batch(
                x0_pred=x0_pred,
                x_gt_full=x,
                norm_mean_pred=norm_mean,
                norm_std_pred=norm_std,
                norm_mean_full=norm_mean,
                norm_std_full=norm_std,
                train_cameras=train_cameras,
                renderer_tuple=renderer_tuple,
                lpips_fn=lpips_fn,
                num_cam=int(getattr(args, "render_loss_num_cam", 1)),
                device=device,
                dc_only=bool(getattr(args, "sh_degree0_only", True)),
                plane_to_sphere=plane_to_sphere,
                sample_weights=None,
                rank_transform_tables=rank_transform_tables,
            )

            entry = {
                "seed": seed,
                "t_value": float(t_value.item()),
                "t_discrete": int(t_discrete.item()),
                "mse": float(mse.item()),
                "render_l1": float(render_l1.item()),
                "render_alpha_l1": float(alpha_l1.item()),
                "render_lpips": float(lpips_loss.item()),
            }
            per_seed.append(entry)
            print(
                f"[overfit_verify] seed={seed:4d}  t={entry['t_value']:.4f}  "
                f"mse={entry['mse']:.4e}  render_l1={entry['render_l1']:.4e}  "
                f"alpha_l1={entry['render_alpha_l1']:.4e}  "
                f"lpips={entry['render_lpips']:.4e}"
            )
    elapsed = time.perf_counter() - t_start

    aggregate = {
        "mse": _stats([e["mse"] for e in per_seed]),
        "render_l1": _stats([e["render_l1"] for e in per_seed]),
        "render_alpha_l1": _stats([e["render_alpha_l1"] for e in per_seed]),
        "render_lpips": _stats([e["render_lpips"] for e in per_seed]),
    }

    summary = {
        "checkpoint": str(ckpt_path),
        "saved_step": saved_step,
        "weights": weights_key,
        "model": str(args.model),
        "num_seeds": int(cli.num_seeds),
        "seed_base": int(cli.seed_base),
        "sample_index": int(cli.sample_index),
        "sample_class": int(label),
        "sample_hash_key": str(hash_key),
        "sample_integrity": sample_integrity,
        "render_loss_num_cam": int(getattr(args, "render_loss_num_cam", 1)),
        "train_render_size": int(getattr(args, "train_render_size", 224)),
        "P_mean": float(args.P_mean),
        "P_std": float(args.P_std),
        "noise_schedule": str(args.noise_schedule),
        "sh_degree0_only": bool(getattr(args, "sh_degree0_only", True)),
        "elapsed_s": float(elapsed),
        "aggregate": aggregate,
        "per_seed": per_seed,
    }

    print("\n[overfit_verify] === aggregate over %d seeds ===" % cli.num_seeds)
    for name, st in aggregate.items():
        print(f"  {name:>15s}: mean={st['mean']:.4e}  std={st['std']:.4e}  "
              f"min={st['min']:.4e}  max={st['max']:.4e}")
    print(f"[overfit_verify] elapsed={elapsed:.2f}s")

    out_path = (
        Path(cli.output_json).expanduser().resolve()
        if cli.output_json
        else ckpt_path.parent / "overfit_verify.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[overfit_verify] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
