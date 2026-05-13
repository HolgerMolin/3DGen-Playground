"""
Render 4-view montages for sampled objects in selected k-means clusters.

For each chosen cluster (10 random + the 3 smallest by default), pick N
random member objects, render the same 4-view setup used during embedding
(azim 0/90/180/270, elev 30, dist 1.6, FOV 49.1, white BG, 224x224), and
save a single PNG per cluster: rows = objects, cols = 4 views.

Output: object_classification/cluster_inspection/cluster_<id>_size<N>.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse helpers from the embedding script.
from object_classification.encode_image_embeddings import (  # noqa: E402
    _build_cameras,
    _render_object_white_bg,
    _DEFAULT_AZIMUTHS_DEG,
    _DEFAULT_CAM_DISTANCE,
    _DEFAULT_ELEVATION_DEG,
    _DEFAULT_FOV_DEG,
    _RENDER_SIZE,
    _env_path,
)
from dataloaders.standard_3dgen_loader import (  # noqa: E402
    extract_directory_info,
    load_obj_list,
    load_ply,
)


def parse_args() -> argparse.Namespace:
    load_dotenv(_REPO_ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels",
                   default=str(_SCRIPT_DIR / "image_kmeans_labels__k120.npz"))
    p.add_argument("--obj-list", default=_env_path("ALL_OBJ_JSON"))
    p.add_argument("--gs-path", default=_env_path("GS_PATH"))
    p.add_argument("--output-dir",
                   default=str(_SCRIPT_DIR / "cluster_inspection"))
    p.add_argument("--n-random", type=int, default=10,
                   help="Number of random clusters to render.")
    p.add_argument("--n-smallest", type=int, default=3,
                   help="Number of smallest clusters to also render.")
    p.add_argument("--objects-per-cluster", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--world-up", default="y_down", choices=["y_down", "y_up"])
    return p.parse_args()


def _tensor_to_uint8(views: torch.Tensor) -> np.ndarray:
    """(V, 3, H, W) in [0,1] -> (V, H, W, 3) uint8."""
    arr = views.detach().cpu().clamp(0, 1).numpy()
    arr = (arr * 255.0).astype(np.uint8)
    return arr.transpose(0, 2, 3, 1)  # (V, H, W, 3)


def _build_cluster_montage(
    rendered: list[tuple[str, np.ndarray]],
    title: str,
    tile_size: int = _RENDER_SIZE,
    label_height: int = 22,
    title_height: int = 32,
) -> Image.Image:
    """rendered: list of (key, (V, H, W, 3) uint8). Output is one PIL image:
    rows = objects (with label on the left), cols = 4 views, plus a title bar."""
    if not rendered:
        # Empty placeholder
        img = Image.new("RGB", (tile_size * 4, title_height + 8), (240, 240, 240))
        ImageDraw.Draw(img).text((6, 4), title + "  (no objects rendered)",
                                 fill=(0, 0, 0))
        return img

    n = len(rendered)
    v = rendered[0][1].shape[0]
    label_width = 130
    grid_w = label_width + v * tile_size
    grid_h = title_height + n * (tile_size + 4) - 4
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    draw.rectangle([(0, 0), (grid_w, title_height)], fill=(40, 40, 40))
    draw.text((8, 6), title, font=title_font, fill=(255, 255, 255))

    for row, (key, views) in enumerate(rendered):
        y0 = title_height + row * (tile_size + 4)
        # Object key label
        draw.text((4, y0 + tile_size // 2 - 8), key, font=font, fill=(0, 0, 0))
        for col in range(v):
            tile = Image.fromarray(views[col])
            canvas.paste(tile, (label_width + col * tile_size, y0))
    return canvas


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load labels + keys
    print(f"Loading cluster labels from {args.labels}")
    data = np.load(args.labels, allow_pickle=True)
    labels = data["labels"]
    keys = data["keys"].tolist()
    print(f"  {len(keys):,} objects across {labels.max() + 1} clusters")

    # Load obj-list to map key -> data dir
    print(f"Loading obj_list from {args.obj_list}")
    obj_data = load_obj_list([args.obj_list])
    # Build key -> tar_gz_path map (key = caption-style stem 'chunk/filename')
    stem_to_tar = {v.removesuffix(".tar.gz"): v for v in obj_data.values()}
    gs_root = Path(args.gs_path)

    # Pick clusters
    rng = np.random.default_rng(args.seed)
    sizes = np.bincount(labels)
    n_clusters = len(sizes)
    smallest = list(np.argsort(sizes)[: args.n_smallest])
    remaining_ids = [c for c in range(n_clusters) if c not in smallest]
    random_pick = list(rng.choice(remaining_ids,
                                  size=min(args.n_random, len(remaining_ids)),
                                  replace=False))
    chosen = random_pick + smallest
    print(f"Random clusters: {sorted(random_pick)}")
    print(f"Smallest clusters: {smallest} (sizes {[int(sizes[c]) for c in smallest]})")

    # Build cameras once
    cameras = _build_cameras(
        azimuths_deg=list(_DEFAULT_AZIMUTHS_DEG),
        elevation_deg=_DEFAULT_ELEVATION_DEG,
        distance=_DEFAULT_CAM_DISTANCE,
        fov_deg=_DEFAULT_FOV_DEG,
        image_size=_RENDER_SIZE,
        device=device,
        world_up_axis=args.world_up,
    )
    import gsplat as renderer_module

    keys_by_cluster: dict[int, list[str]] = {}
    for c in chosen:
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        n_take = min(args.objects_per_cluster, len(idxs))
        sampled = rng.choice(idxs, size=n_take, replace=False)
        keys_by_cluster[int(c)] = [keys[i] for i in sampled]

    total = sum(len(v) for v in keys_by_cluster.values())
    print(f"\nRendering {total} objects across {len(keys_by_cluster)} clusters ...")
    t_start = time.time()

    for c in chosen:
        cluster_id = int(c)
        cluster_size = int(sizes[c])
        sample_keys = keys_by_cluster.get(cluster_id, [])
        rendered: list[tuple[str, np.ndarray]] = []
        for key in sample_keys:
            tar_path = stem_to_tar.get(key)
            if tar_path is None:
                print(f"  [WARN] no obj_list entry for key={key}, skipping")
                continue
            chunk, filename = extract_directory_info(tar_path)
            data_dir = gs_root / chunk / filename
            try:
                gs2sphere = np.load(data_dir / "gs2sphere.npy")
                pc = load_ply(str(data_dir / "point_cloud.ply"))[gs2sphere]
            except Exception as exc:
                print(f"  [WARN] load failed for {key}: {exc!r}")
                continue
            pc_t = torch.from_numpy(pc)
            try:
                with torch.no_grad():
                    views = _render_object_white_bg(
                        pc_t, cameras, renderer_module, device)
            except Exception as exc:
                print(f"  [WARN] render failed for {key}: {exc!r}")
                continue
            rendered.append((key, _tensor_to_uint8(views)))

        title = (f"Cluster {cluster_id} | size={cluster_size:,} | "
                 f"sampled={len(rendered)}/{len(sample_keys)}")
        montage = _build_cluster_montage(rendered, title)
        out_path = out_dir / f"cluster_{cluster_id:03d}_size{cluster_size}.png"
        montage.save(out_path)
        elapsed = time.time() - t_start
        print(f"  cluster {cluster_id:>3} (size {cluster_size:>5}): "
              f"{len(rendered)} objects -> {out_path.name}  [{elapsed:.1f}s total]")

    print(f"\nDone. Wrote {len(keys_by_cluster)} montages to {out_dir}")


if __name__ == "__main__":
    main()
