"""Per-class intra-class appearance-diversity diagnosis.

For each class in object_to_class.json:
  1. Sample N members from the training set
  2. Render each at K canonical views via gsplat
  3. Embed renders with DINOv2-S/14, average across views per object → 1 feature per member
  4. Class diversity = mean pairwise (1 − cosine) among the N member features

Then correlate per-class diversity with per-class cfg_signal (probe JSON dump),
partition the weak-20 classes into incoherent / under-trained / ambiguous, find
strong classes with high diversity (existence proofs), and write a visual sanity
grid for 3 weak + 3 strong classes.

Read-only: does not modify training code or labels.
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
GS_ROOT = os.path.join(REPO_ROOT, "submodules", "gaussian-splatting")
if GS_ROOT not in sys.path:
    sys.path.insert(0, GS_ROOT)

from dataloaders.standard_3dgen_loader import (
    Standard3DGenDataset,
    extract_directory_info,
)
from dataloaders.class_3dgen_loader import (
    DC_ONLY_FEATURE_INDICES,
    Class3DGenDataset,
)
from utils.plane_utils import load_sphere2plane
from utils.gsplat_render_util import (
    _load_reference_cameras,
    _prepare_train_cameras,
    _denormalize_point_cloud,
    _plane_to_point_cloud_batch,
    _point_clouds_to_gsplat_inputs,
    _render_gsplat_batch,
)
from jit._diag_paths import diversity_dir, dinov2_cache_paths, per_class_dump_path


def _load_dinov2(device):
    print("Loading DINOv2-S/14 from torch.hub …")
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        pretrained=True,
        source="github",
        trust_repo=True,
    )
    model.eval().to(device)
    return model


def _dinov2_normalize(x):
    """Apply ImageNet normalization expected by DINOv2."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def _dinov2_embed(model, imgs_uint8, batch_size, device):
    """imgs_uint8: (N, 3, H, W) torch uint8 tensor on CPU. Returns (N, 384) features on CPU."""
    feats = []
    n = imgs_uint8.shape[0]
    for i in range(0, n, batch_size):
        batch = imgs_uint8[i:i + batch_size].to(device=device, dtype=torch.float32) / 255.0
        # DINOv2-S/14 expects multiples of 14; resize to 224x224
        batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
        batch = _dinov2_normalize(batch)
        f = model(batch)  # (B, 384)
        feats.append(f.cpu().float())
    return torch.cat(feats, dim=0)


def _build_dataset(args):
    base = Standard3DGenDataset(
        obj_list=[args.obj_list],
        gs_path=args.gs_path,
        caption_path=None,
        rendering_path=None,
        num_images=1,
        mean_file=args.mean_file,
        std_file=args.std_file,
        sphere2plane_path=args.sphere2plane_path,
    )
    with open(args.class_map) as f:
        class_map = json.load(f)
    feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
    ds = Class3DGenDataset(
        base_dataset=base,
        class_map=class_map,
        feature_indices=feature_indices,
        return_full_for_render=False,
    )
    return base, ds


def _sample_member_indices(ds, n_per_class, seed):
    """Group ds.valid_indices by class, sample n_per_class per class."""
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for vi_pos, lbl in enumerate(ds.valid_labels):
        by_class.setdefault(int(lbl), []).append(vi_pos)
    sampled = {}
    for cls, positions in by_class.items():
        if len(positions) <= n_per_class:
            sampled[cls] = list(positions)
        else:
            sampled[cls] = list(rng.choice(positions, size=n_per_class, replace=False))
    return sampled


def _render_object(
    pc_plane14,                 # (14, 128, 128) torch on cpu, normalized
    base_dataset,
    plane_to_sphere,            # (16384,) torch.long
    train_cameras,
    renderer_module,
    device,
    cam_indices,
):
    """Render a single object at the given cam_indices. Returns (num_cam, 3, H, W) cpu uint8."""
    pc_plane14 = pc_plane14.to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,14,128,128)
    pc_flat = _plane_to_point_cloud_batch(pc_plane14, plane_to_sphere=plane_to_sphere)  # (1, 16384, 14)

    # base_dataset.mean/std may be saved as (59,), (1, 59), or (59, 1, 1). Flatten then slice.
    mean_full = np.asarray(base_dataset.mean).reshape(-1).astype(np.float32)
    std_full = np.asarray(base_dataset.std).reshape(-1).astype(np.float32)
    if mean_full.size != 59 or std_full.size != 59:
        raise ValueError(f"Unexpected mean/std size: mean={mean_full.size} std={std_full.size}")
    mean_t = torch.from_numpy(mean_full[list(DC_ONLY_FEATURE_INDICES)]).to(device)
    std_t = torch.from_numpy(std_full[list(DC_ONLY_FEATURE_INDICES)]).to(device)
    pc_denorm = _denormalize_point_cloud(pc_flat, mean_t, std_t)
    gs = _point_clouds_to_gsplat_inputs(pc_denorm, dc_only=True)

    rgb = _render_gsplat_batch(
        renderer_module=renderer_module,
        gaussian_inputs=gs,
        camera_bundle=train_cameras,
        cam_indices=cam_indices,
        device=device,
        return_alpha=False,
    )
    # rgb: (1, num_cam, 3, H, W) in [0,1]
    rgb = rgb[0]
    return (rgb * 255.0).clamp(0, 255).to(torch.uint8).cpu()


def _spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    return float((rx * ry).mean())


def _pearson(x, y):
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return float((x * y).mean())


def _bootstrap_ci(x, y, fn, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = fn(x[idx], y[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _composite_grid(images, classes_in_order, cfgs, divs, n_cols, out_path, title):
    """images: list of (n_cols, 3, H, W) torch uint8 tensors, one per class."""
    n_rows = len(images)
    H, W = images[0].shape[-2:]
    pad = 8
    canvas = np.full((n_rows * H + (n_rows - 1) * pad + 80, n_cols * W + (n_cols - 1) * pad, 3), 255, dtype=np.uint8)
    for r, row in enumerate(images):
        for c in range(min(n_cols, row.shape[0])):
            img = row[c].permute(1, 2, 0).numpy()
            y0 = r * (H + pad) + 80
            x0 = c * (W + pad)
            canvas[y0:y0 + H, x0:x0 + W] = img
    fig, ax = plt.subplots(figsize=(canvas.shape[1] / 100, canvas.shape[0] / 100), dpi=100)
    ax.imshow(canvas)
    ax.set_axis_off()
    for r, (cls, cfg, div) in enumerate(zip(classes_in_order, cfgs, divs)):
        y0 = r * (H + pad) + 80
        ax.text(2, y0 - 8, f"class {cls}  cfg={cfg:.3f}  div={div:.3f}", fontsize=10, color="black")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pt checkpoint. Used to derive default --out_dir and --probe_json.",
    )
    parser.add_argument("--obj_list", default=os.environ.get("ALL_OBJ_JSON",
        "/home/tiangexiang/gen3d/gaussianverse/all_obj_list_filtered.json"))
    parser.add_argument("--gs_path", default="/home/tiangexiang/gen3d/gaussianverse")
    parser.add_argument("--mean_file", default="data/stats/all_mean.pt")
    parser.add_argument("--std_file", default="data/stats/all_std.pt")
    parser.add_argument("--sphere2plane_path", default="/home/tiangexiang/gen3d/gaussianverse/sphere2plane.npy")
    parser.add_argument("--ref_camera_tar", default="artifacts/ref_camera.tar.gz")
    parser.add_argument("--class_map", default="object_labels/improved_labels/object_to_class.json")
    parser.add_argument(
        "--probe_json",
        default=None,
        help="Probe per-class JSON. Defaults to the t=0.5 dump for this checkpoint.",
    )
    parser.add_argument(
        "--probe_t",
        type=float,
        default=0.5,
        help="t value to use when auto-deriving --probe_json (ignored if explicit).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output dir. Defaults to output/diagnostics/<run_id>/diversity/.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Shared DINOv2 feature cache dir. Defaults to output/diagnostics/_cache/dinov2_features/. "
             "The cache key includes n_per_class and seed so different sampling configs don't collide.",
    )
    parser.add_argument("--n_per_class", type=int, default=24)
    parser.add_argument("--cam_indices", type=int, nargs="+", default=[0, 13, 26, 39])
    parser.add_argument("--render_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render_batch", type=int, default=8,
                        help="Objects per gsplat render batch (each object has len(cam_indices) views).")
    parser.add_argument("--dino_batch", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else diversity_dir(args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir.resolve()}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base_ds, ds = _build_dataset(args)
    sampled = _sample_member_indices(ds, args.n_per_class, args.seed)
    classes = sorted(sampled.keys())
    print(f"num classes: {len(classes)}  members per class (avg): "
          f"{np.mean([len(v) for v in sampled.values()]):.1f}")

    # Plane->sphere permutation. utils.plane_utils.load_sphere2plane returns a torch
    # tensor (the same one passed as `plane_to_sphere` to the gsplat helpers).
    plane_to_sphere = load_sphere2plane(args.sphere2plane_path, 16384).long()

    # Cameras
    ref_cams = _load_reference_cameras(args.ref_camera_tar)
    train_cameras = _prepare_train_cameras(ref_cams, args.render_size, device)
    cam_indices = list(args.cam_indices)
    n_views = len(cam_indices)
    print(f"render at cams {cam_indices} ({n_views} views) at {args.render_size}x{args.render_size}")

    # Renderer
    import gsplat as renderer_module

    # ---- Compute features per class with disk cache ----
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        features_path = cache_dir / f"dinov2_n{args.n_per_class}_seed{args.seed}.npz"
        member_keys_path = cache_dir / f"dinov2_n{args.n_per_class}_seed{args.seed}_keys.json"
    else:
        features_path, member_keys_path = dinov2_cache_paths(args.n_per_class, args.seed)
    print(f"Cache: features={features_path}  keys={member_keys_path}")
    diversity_path = out_dir / "per_class_diversity.json"

    if features_path.exists() and member_keys_path.exists():
        print(f"Loading cached features from {features_path}")
        z = np.load(features_path, allow_pickle=True)
        feats_by_class = {int(k): z[k] for k in z.files}
        with open(member_keys_path) as f:
            member_keys = {int(k): v for k, v in json.load(f).items()}
    else:
        dino = _load_dinov2(device)
        feats_by_class: dict[int, np.ndarray] = {}
        member_keys: dict[int, list[str]] = {}

        # Pre-compute base_dataset.mean/std slices on cpu once
        t0 = time.time()
        for ci, cls in enumerate(classes):
            positions = sampled[cls]
            n = len(positions)
            # Render in mini-batches of args.render_batch objects to keep memory small
            all_renders = []
            keys = []
            for i in range(0, n, args.render_batch):
                batch_pos = positions[i:i + args.render_batch]
                rendered_batch = []
                for pos in batch_pos:
                    real_idx = ds.valid_indices[pos]
                    sample = base_ds[real_idx]
                    pc_plane = sample["point_cloud"]  # (C, H, W) numpy or tensor, sh_degree0_only=False here
                    if isinstance(pc_plane, np.ndarray):
                        pc_plane = torch.from_numpy(pc_plane)
                    if pc_plane.ndim == 4 and pc_plane.shape[0] == 1:
                        pc_plane = pc_plane[0]
                    # Slice to DC_ONLY 14 channels
                    pc14 = pc_plane[list(DC_ONLY_FEATURE_INDICES)]
                    keys.append(sample["hash_key"])
                    rgb = _render_object(
                        pc_plane14=pc14,
                        base_dataset=base_ds,
                        plane_to_sphere=plane_to_sphere,
                        train_cameras=train_cameras,
                        renderer_module=renderer_module,
                        device=device,
                        cam_indices=cam_indices,
                    )  # (n_views, 3, H, W) cpu uint8
                    rendered_batch.append(rgb)
                all_renders.extend(rendered_batch)
            # Stack: (n * n_views, 3, H, W)
            stacked = torch.stack([r for r in all_renders], dim=0)  # (n, n_views, 3, H, W)
            n_actual = stacked.shape[0]
            stacked_flat = stacked.view(n_actual * n_views, 3, args.render_size, args.render_size)

            # DINOv2 embed
            f = _dinov2_embed(dino, stacked_flat, batch_size=args.dino_batch, device=device)
            f = f.view(n_actual, n_views, -1).mean(dim=1)  # average over views per object
            feats_by_class[int(cls)] = f.numpy().astype(np.float32)
            member_keys[int(cls)] = keys

            if (ci + 1) % 10 == 0 or ci == len(classes) - 1:
                elapsed = time.time() - t0
                eta = elapsed / (ci + 1) * (len(classes) - ci - 1)
                print(f"  [{ci+1:3d}/{len(classes)}] class {cls:3d}: n={n_actual:2d}  "
                      f"elapsed {elapsed:.0f}s  eta {eta:.0f}s")

        # Cache to disk
        np.savez_compressed(
            features_path,
            **{str(k): v for k, v in feats_by_class.items()},
        )
        with open(member_keys_path, "w") as f:
            json.dump({str(k): v for k, v in member_keys.items()}, f)
        print(f"Cached features → {features_path}")

    # ---- Compute per-class diversity ----
    diversity = {}
    for cls in classes:
        feats = feats_by_class[cls]  # (M, D)
        n = feats.shape[0]
        if n < 2:
            diversity[cls] = float("nan")
            continue
        # Normalize for cosine
        fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        # Pairwise cosine distance: 1 - fn @ fn.T (off-diagonal only)
        cos = fn @ fn.T
        triu = cos[np.triu_indices(n, k=1)]
        # diversity = mean of (1 - cos) over distinct pairs
        diversity[cls] = float(np.mean(1.0 - triu))

    with open(diversity_path, "w") as f:
        json.dump({str(k): v for k, v in diversity.items()}, f, indent=2)
    print(f"Wrote diversity → {diversity_path}")

    # ---- Load probe per-class cfg_signal ----
    probe_path = (
        Path(args.probe_json)
        if args.probe_json
        else per_class_dump_path(args.checkpoint, args.probe_t)
    )
    if not probe_path.exists():
        raise SystemExit(
            f"Probe per-class JSON not found at {probe_path}. "
            f"Run jit/probe_conditioning.py with --per_class_dump_dir first, "
            f"or pass --probe_json explicitly."
        )
    print(f"Loading probe per-class cfg_signal from: {probe_path}")
    with open(probe_path) as f:
        probe = json.load(f)
    cfg_per_class = {int(k): float(v) for k, v in probe["cfg_signal_per_class"].items()}
    t_value = probe.get("t_value", "?")
    print(f"Loaded cfg_signal_per_class for {len(cfg_per_class)} classes at t={t_value}")

    common = sorted(set(diversity.keys()) & set(cfg_per_class.keys()))
    div_arr = np.array([diversity[c] for c in common])
    cfg_arr = np.array([cfg_per_class[c] for c in common])

    pearson = _pearson(div_arr, cfg_arr)
    spearman = _spearman(div_arr, cfg_arr)
    p_lo, p_hi = _bootstrap_ci(div_arr, cfg_arr, _pearson, n_boot=2000, seed=args.seed)
    s_lo, s_hi = _bootstrap_ci(div_arr, cfg_arr, _spearman, n_boot=2000, seed=args.seed)
    print(f"Pearson r(div, cfg)  = {pearson:+.3f}  CI95 [{p_lo:+.3f}, {p_hi:+.3f}]  n={len(common)}")
    print(f"Spearman ρ(div, cfg) = {spearman:+.3f}  CI95 [{s_lo:+.3f}, {s_hi:+.3f}]  n={len(common)}")

    corr_summary = {
        "n_classes": len(common),
        "t_value_for_cfg_signal": t_value,
        "pearson_r": pearson,
        "pearson_ci95": [p_lo, p_hi],
        "spearman_rho": spearman,
        "spearman_ci95": [s_lo, s_hi],
    }
    with open(out_dir / "correlation_summary.json", "w") as f:
        json.dump(corr_summary, f, indent=2)

    # ---- Scatter plot ----
    sort_by_cfg = sorted(common, key=lambda c: cfg_per_class[c])
    weak10 = sort_by_cfg[:10]
    strong10 = sort_by_cfg[-10:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(div_arr, cfg_arr, s=18, alpha=0.7, c="#4f7cd0")
    for c in weak10:
        ax.annotate(f"{c}", (diversity[c], cfg_per_class[c]),
                    fontsize=8, color="#c44")
    for c in strong10:
        ax.annotate(f"{c}", (diversity[c], cfg_per_class[c]),
                    fontsize=8, color="#1a8")
    ax.set_xlabel("intra-class DINOv2 diversity (mean pairwise 1 − cos)")
    ax.set_ylabel(f"cfg_signal at t={t_value}")
    ax.set_title(f"Per-class diversity vs cfg_signal  (n={len(common)})\n"
                 f"Pearson r={pearson:+.3f} [{p_lo:+.3f}, {p_hi:+.3f}]   "
                 f"Spearman ρ={spearman:+.3f} [{s_lo:+.3f}, {s_hi:+.3f}]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "diversity_vs_cfg_scatter.png", dpi=140)
    plt.close(fig)

    # ---- Partition weak-20 classes by diversity quartile ----
    weak20 = sort_by_cfg[:20]
    div_med = float(np.median(div_arr))
    div_q1 = float(np.quantile(div_arr, 0.25))
    div_q3 = float(np.quantile(div_arr, 0.75))
    print(f"diversity quartiles  q1={div_q1:.3f}  median={div_med:.3f}  q3={div_q3:.3f}")

    partition = {"incoherent_high_div": [], "under_trained_low_div": [], "ambiguous_mid": []}
    for c in weak20:
        d = diversity[c]
        if d >= div_q3:
            partition["incoherent_high_div"].append(c)
        elif d <= div_q1:
            partition["under_trained_low_div"].append(c)
        else:
            partition["ambiguous_mid"].append(c)

    weak_rows = [
        {
            "class": int(c),
            "cfg_signal": cfg_per_class[c],
            "diversity": diversity[c],
            "diversity_quartile": (
                "high" if diversity[c] >= div_q3 else
                "low" if diversity[c] <= div_q1 else "mid"
            ),
        }
        for c in weak20
    ]

    print("\nWeak-20 partition (by diversity quartile, using all-class q1/q3):")
    for k, v in partition.items():
        print(f"  {k:>22}: {len(v):>2}  →  {v}")

    # ---- Strong classes with high diversity (existence proofs) ----
    strong_high_div = [
        c for c in strong10 if diversity[c] >= div_med
    ]
    strong_high_div_above_q3 = [c for c in strong10 if diversity[c] >= div_q3]
    print(f"\nStrong-10 with above-MEDIAN diversity (existence proofs): {strong_high_div}")
    print(f"Strong-10 with above-Q3 diversity:                          {strong_high_div_above_q3}")

    with open(out_dir / "weak_partition.json", "w") as f:
        json.dump(
            {
                "weak20_rows": weak_rows,
                "partition": partition,
                "strong_high_div_above_median": strong_high_div,
                "strong_high_div_above_q3": strong_high_div_above_q3,
                "diversity_quartiles": {"q1": div_q1, "median": div_med, "q3": div_q3},
            },
            f,
            indent=2,
        )

    # ---- Visual sanity grid: 3 weak + 3 strong, 6 members each ----
    pick_weak = weak10[:3]
    pick_strong = strong10[-3:][::-1]  # top three
    pick = list(pick_weak) + list(pick_strong)
    print(f"\nVisual sanity grid for classes: weak={pick_weak}  strong={pick_strong}")

    # We need to re-render 6 members per chosen class. Use the same sampled member
    # positions that were used to compute features (member_keys[cls]) — first 6.
    grid_imgs = []
    grid_cfgs = []
    grid_divs = []
    for cls in pick:
        member_hashes = member_keys[cls][:6]
        # Map each hash back to its real_idx
        wanted_keys = set(member_hashes)
        member_renders = []
        for vi_pos in sampled[cls]:
            real_idx = ds.valid_indices[vi_pos]
            sample = base_ds[real_idx]
            if sample["hash_key"] not in wanted_keys:
                continue
            pc_plane = sample["point_cloud"]
            if isinstance(pc_plane, np.ndarray):
                pc_plane = torch.from_numpy(pc_plane)
            if pc_plane.ndim == 4 and pc_plane.shape[0] == 1:
                pc_plane = pc_plane[0]
            pc14 = pc_plane[list(DC_ONLY_FEATURE_INDICES)]
            rgb = _render_object(
                pc_plane14=pc14,
                base_dataset=base_ds,
                plane_to_sphere=plane_to_sphere,
                train_cameras=train_cameras,
                renderer_module=renderer_module,
                device=device,
                cam_indices=[cam_indices[0]],  # single canonical view for the grid
            )  # (1, 3, H, W)
            member_renders.append(rgb[0])
            if len(member_renders) >= 6:
                break
        if len(member_renders) < 6:
            # pad with the last image to keep grid alignment
            while len(member_renders) < 6:
                member_renders.append(member_renders[-1])
        grid_imgs.append(torch.stack(member_renders, dim=0))
        grid_cfgs.append(cfg_per_class[cls])
        grid_divs.append(diversity[cls])
    _composite_grid(
        grid_imgs,
        pick,
        grid_cfgs,
        grid_divs,
        n_cols=6,
        out_path=out_dir / "visual_sanity_grid.png",
        title="Sanity grid: top 3 rows = weakest cfg classes, bottom 3 rows = strongest cfg classes",
    )
    print(f"Wrote visual sanity grid → {out_dir / 'visual_sanity_grid.png'}")

    # ---- Bundle ----
    bundle = {
        "checkpoint_step": 260000,
        "n_per_class_target": args.n_per_class,
        "n_views_per_object": n_views,
        "cam_indices": cam_indices,
        "correlation": corr_summary,
        "weak20_partition": {k: list(map(int, v)) for k, v in partition.items()},
        "strong_high_div_above_median": list(map(int, strong_high_div)),
        "strong_high_div_above_q3": list(map(int, strong_high_div_above_q3)),
        "diversity_quartiles": {"q1": div_q1, "median": div_med, "q3": div_q3},
    }
    with open(out_dir / "diagnosis_bundle.json", "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"Wrote bundle → {out_dir / 'diagnosis_bundle.json'}")


if __name__ == "__main__":
    main()
