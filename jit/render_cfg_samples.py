"""Render a showcase of validation samples at a fixed CFG scale.

Reuses the validated generate+render path from jit/eval_clip_alignment.py (EMA
weights, sphere2plane inverse, rank-transform inverse, denorm, gsplat @224px) and
produces two montages:

  showcase_<tag>_all.png        : every validation prompt, one canonical view (breadth)
  showcase_<tag>_multiview.png  : a curated subset, several views each (3D coherence)

Seed-round 0 reproduces the run's dit_validation grid noise. Usage:

    python jit/render_cfg_samples.py --ckpt output/<run>/0063000.pt --cfg 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jit.eval_clip_alignment import build, gen_render

# Curated, recognizable subset spanning categories (animal / tool / furniture /
# container / building / vehicle / accessory). Matched by text against the pool.
MULTIVIEW_PROMPTS = [
    "a dog", "a sword", "a blue chair", "a glass bottle",
    "a small house", "a rocket", "a hat", "a vase",
]


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def montage_grid(tiles01, labels, out_path, cols, font_size=12):
    """Flat grid; one label per tile (top-left)."""
    n = tiles01.shape[0]
    H, W = int(tiles01.shape[-2]), int(tiles01.shape[-1])
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
    arr = (tiles01.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = arr[i]
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = _font(font_size)
    for i in range(n):
        r, c = divmod(i, cols)
        draw.text((c * W + 3, r * H + 3), labels[i], fill=(255, 255, 0), font=font)
    img.save(out_path)


def montage_multiview(tiles01, row_labels, out_path, n_views, font_size=14):
    """rows = objects, cols = views; label the leftmost tile of each row."""
    n_rows = tiles01.shape[0] // n_views
    H, W = int(tiles01.shape[-2]), int(tiles01.shape[-1])
    canvas = np.zeros((n_rows * H, n_views * W, 3), dtype=np.uint8)
    arr = (tiles01.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    for i in range(tiles01.shape[0]):
        r, c = divmod(i, n_views)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = arr[i]
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = _font(font_size)
    for r in range(n_rows):
        draw.text((3, r * H + 3), row_labels[r], fill=(255, 255, 0), font=font)
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--seed_round", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="default: run's val_sampling_steps")
    ap.add_argument("--sampler", default=None, help="default: run's val_sampler (heun)")
    ap.add_argument("--cols", type=int, default=8, help="columns in the all-prompts grid")
    ap.add_argument("--n_prompts", type=int, default=None, help="render only first N prompts (e.g. 9 for a 3x3)")
    ap.add_argument("--no_multiview", action="store_true", help="skip the multi-view turntable")
    ap.add_argument("--multiview_n_views", type=int, default=6)
    ap.add_argument("--sub_batch", type=int, default=32)
    ap.add_argument("--cpu_encode", action="store_true",
                    help="encode text conditioning on CPU (keeps CLIP off the GPU; memory-safe)")
    ap.add_argument("--max_gpu_gb", type=float, default=8.0,
                    help="hard cap this process's GPU allocation; 0 disables. Protects a co-resident training job")
    ap.add_argument("--scores_json", default=None, help="clip_alignment.json to label grid w/ CLIP scores")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.max_gpu_gb and args.max_gpu_gb > 0:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        frac = min(0.95, float(args.max_gpu_gb) / total_gb)
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        print(f"[mem] capping this process at {args.max_gpu_gb:.1f} GB (frac={frac:.3f} of {total_gb:.0f} GB)")
    t0 = time.time()
    b = build(args.ckpt, device,
              encode_device=(torch.device("cpu") if args.cpu_encode else None),
              max_prompts=args.n_prompts)
    cfg = b["cfg"]
    sampler = args.sampler or cfg["val_sampler"]
    steps = int(args.steps or cfg["val_sampling_steps"])
    prompts = b["prompts"]
    P = len(prompts)
    tag = f"cfg{args.cfg:g}_step{b['step']:07d}"

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "cfg_showcase")
    os.makedirs(out_dir, exist_ok=True)

    # Optional CLIP-score labels (4-view-avg mean per prompt) from a prior eval JSON.
    score_by_prompt = {}
    sj = args.scores_json
    if sj is None:
        guess = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                             f"clip_eval_step{b['step']:07d}", "clip_alignment.json")
        sj = guess if os.path.exists(guess) else None
    if sj and os.path.exists(sj):
        d = json.load(open(sj))
        key = f"cfg{args.cfg:g}"
        if key in d.get("results", {}):
            score_by_prompt = {r["prompt"]: r["mean"] for r in d["results"][key]["per_prompt"]}
            print(f"[labels] using CLIP scores from {sj} ({key})")

    ncam = int(b["train_cameras"]["viewmats"].shape[0])
    print(f"[build] {time.time()-t0:.1f}s step={b['step']} model={cfg['model']} sampler={sampler} "
          f"steps={steps} cfg={args.cfg} seed_round={args.seed_round} ncam={ncam}")

    # ---- 1) all-prompts grid, single canonical view (cam 0) -------------------------
    cam0 = [0]
    tiles = []
    for s in range(0, P, args.sub_batch):
        idxs = list(range(s, min(P, s + args.sub_batch)))
        r = gen_render(b, idxs, args.seed_round, args.cfg, steps, sampler, cam0, device)
        tiles.append(r[:, 0].detach().cpu())  # (nb, 3, H, W)
    tiles = torch.cat(tiles, dim=0)
    labels = []
    for p in prompts:
        lab = p[:22]
        if p in score_by_prompt:
            lab = f"{p[:18]} {score_by_prompt[p]:.3f}"
        labels.append(lab)
    all_path = os.path.join(out_dir, f"showcase_{tag}_all.png")
    montage_grid(tiles, labels, all_path, cols=args.cols)
    print(f"[grid] {P} prompts x1 view -> {all_path}")

    # ---- 2) curated multi-view turntable --------------------------------------------
    if args.no_multiview or sum(p in prompts for p in MULTIVIEW_PROMPTS) < 2:
        print(f"[multiview] skipped (no_multiview={args.no_multiview})")
        print(f"[done] {time.time()-t0:.0f}s  out_dir={out_dir}")
        return
    nv = min(args.multiview_n_views, ncam)
    cam_indices = sorted({int(round(x)) for x in np.linspace(0, ncam - 1, nv)})
    mv_idxs = [prompts.index(p) for p in MULTIVIEW_PROMPTS if p in prompts]
    mv = gen_render(b, mv_idxs, args.seed_round, args.cfg, steps, sampler, cam_indices, device)
    # (n_obj, nv, 3, H, W) -> flatten rows=obj, cols=view
    n_obj = mv.shape[0]
    mv_flat = mv.reshape(n_obj * len(cam_indices), *mv.shape[2:]).detach().cpu()
    row_labels = []
    for i in mv_idxs:
        p = prompts[i]
        row_labels.append(f"{p[:18]} {score_by_prompt[p]:.3f}" if p in score_by_prompt else p[:20])
    mv_path = os.path.join(out_dir, f"showcase_{tag}_multiview.png")
    montage_multiview(mv_flat, row_labels, mv_path, n_views=len(cam_indices))
    print(f"[multiview] {n_obj} prompts x{len(cam_indices)} views (cams {cam_indices}) -> {mv_path}")
    print(f"[done] {time.time()-t0:.0f}s  out_dir={out_dir}")


if __name__ == "__main__":
    main()
