"""Render a clean 3x3 grid of randomly-sampled GROUND-TRUTH objects from the dataset.

Loads real GaussianVerse objects (their normalized atlas + caption) and renders them
through the EXACT same atlas->point-cloud->gsplat pipeline the model's own renders use
(see jit/eval_clip_alignment.gen_render), so the GT grid matches the report style and
camera. Same clean look as the other grids: white bg, object cropped to its alpha bbox,
Liberation Serif caption superimposed top-left. Tiles/font are bumped up for readability.

RUN UNDER .3dgen (PATH must include .3dgen/bin for gsplat):
    CUDA_VISIBLE_DEVICES=0 python jit/render_gt_grid.py --seed 0 --max_gpu_gb 18

Saves output/report/gt_grid_3x3.png (+ .pdf) and output/report/gt_grid_3x3.pt (img+alpha,
all views, for CPU re-layout) and gt_grid_3x3_captions.json.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from jit import train_gsplat as T
from jit.eval_clip_alignment import build
from jit.make_report_grids import make_tile, tile_grid
from dataloaders.standard_3dgen_loader import Standard3DGenDataset


def shorten(cap, max_chars=72):
    """Keep the first sentence (the object identity), length-capped, so captions render
    uniformly large/readable instead of shrinking to fit a 3-line paragraph."""
    first = cap.split(". ")[0].strip().rstrip(".")
    if len(first) > max_chars:
        first = first[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return first + "."


@torch.no_grad()
def render_gt_atlas(b, atlas14, cam_indices, device, return_alpha=True):
    """atlas14: (P, 14, 128, 128) normalized DC-only GT atlas. Mirrors gen_render's
    post-sampling decode so GT renders are pixel-comparable to the model's."""
    pc = T._plane_to_point_cloud_batch(atlas14.float(), b["plane_to_sphere"])
    pc_raw = T._denormalize_point_cloud(pc, b["norm_mean"], b["norm_std"])
    gauss = T._point_clouds_to_gsplat_inputs(
        pc_raw.to(device), dc_only=True, detach_input=True,
        rank_transform_tables=b["rank_tables"],
    )
    return T._render_gsplat_batch(b["renderer"], gauss, b["train_cameras"], cam_indices,
                                  device, return_alpha=return_alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(_REPO / "output/jit_final_sinkhorn_render_bs512_69k_20260529_071914/0099195.pt"),
                    help="only used for its training config (data paths, mean/std, rank, cameras)")
    ap.add_argument("--captions_json", default="data/gaussianverse/captions.json")
    ap.add_argument("--obj_list", default=None, help="default: the run's own training obj_list")
    ap.add_argument("--seed", type=int, default=0, help="which random 9 objects to draw")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--n_views", type=int, default=4)
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--render_size", type=int, default=224)
    ap.add_argument("--require_caption", action="store_true", default=True,
                    help="skip objects without a caption when drawing the random sample")
    ap.add_argument("--tile", type=int, default=420)
    ap.add_argument("--font", type=int, default=42)
    ap.add_argument("--full_captions", action="store_true",
                    help="superimpose the full (long) caption instead of the shortened first sentence")
    ap.add_argument("--max_gpu_gb", type=float, default=None)
    ap.add_argument("--out", default=str(_REPO / "output/report/gt_grid_3x3"))
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.max_gpu_gb:
        idx = torch.cuda.current_device()
        tot = torch.cuda.get_device_properties(idx).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gb / tot), idx)
        print(f"[mem] cap {args.max_gpu_gb:.0f}GiB of {tot:.0f}GiB")

    t0 = time.time()
    # build() gives us the exact render rig (cameras, renderer, mean/std, rank tables,
    # plane_to_sphere) for this run's config. We don't use its model, just the pipeline.
    b = build(args.ckpt, device, render_size=args.render_size,
              prompts_file=None, max_prompts=1)
    cfg = b["cfg"]
    ncam = int(b["train_cameras"]["viewmats"].shape[0])
    cam_indices = sorted({int(round(x)) for x in np.linspace(0, ncam - 1, args.n_views)})
    feature_indices = torch.tensor(T.DC_ONLY_FEATURE_INDICES, dtype=torch.long)

    # GT dataset with this run's exact preprocessing (clip -> rank -> normalize, atlas order).
    obj_list = args.obj_list or cfg["obj_list"]
    ds = Standard3DGenDataset(
        obj_list=[obj_list],
        gs_path=cfg["gs_path"],
        caption_path=args.captions_json,
        mean_file=cfg["mean_file"],
        std_file=cfg["std_file"],
        sphere2plane_path=cfg["sphere2plane_path"],
        rank_transform_file=cfg["rank_transform_file"],
        clip_thresholds_file=cfg["clip_thresholds_file"],
    )
    print(f"[gt] dataset {len(ds):,} objects | obj_list={Path(obj_list).name} ({time.time()-t0:.0f}s)")

    # Draw a random sample (seeded, reproducible); skip any object missing a caption.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ds))
    atlases, caps, keys = [], [], []
    for j in order:
        s = ds[int(j)]
        cap = (s.get("caption") or "").strip()
        if args.require_caption and not cap:
            continue
        atlases.append(s["point_cloud"][feature_indices])      # (14, 128, 128) normalized
        caps.append(cap)
        keys.append(s.get("hash_key", str(int(j))))
        if len(atlases) == args.n:
            break
    assert len(atlases) == args.n, f"only found {len(atlases)} captioned objects"
    atlas14 = torch.stack(atlases, 0)                          # (n, 14, 128, 128)
    print(f"[gt] sampled {args.n} captioned objects (seed {args.seed})")

    rgb, al = render_gt_atlas(b, atlas14, cam_indices, device, return_alpha=True)
    img = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).cpu()    # (n, V, 3, H, W)
    alpha = (al.clamp(0, 1) * 255).round().to(torch.uint8).cpu()   # (n, V, 1, H, W)
    print(f"[gt] rendered {args.n} objects in {time.time()-t0:.0f}s | cams={cam_indices}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"captions": caps, "keys": keys, "cam_indices": cam_indices,
                "seed": args.seed, "gt_img": img, "gt_alpha": alpha},
               args.out + ".pt")
    json.dump(caps, open(args.out + "_captions.json", "w"), indent=2, ensure_ascii=False)

    v = args.view
    disp = caps if args.full_captions else [shorten(c) for c in caps]
    cells = [(make_tile(img[i, v], alpha[i, v], bg=1.0, tile=args.tile), disp[i])
             for i in range(args.n)]
    tile_grid(cells, args.rows, args.cols, args.out + ".png", max_size=args.font)
    Image.open(args.out + ".png").convert("RGB").save(args.out + ".pdf", "PDF", resolution=200.0)
    print(f"[saved] {args.out}.png (+ .pdf, .pt)  tile={args.tile} font={args.font} view={cam_indices[v]}")


if __name__ == "__main__":
    main()
