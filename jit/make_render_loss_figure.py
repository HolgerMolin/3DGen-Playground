"""Publication figure: effect of the render loss, controlled before/after.

Two parallel 4x4 panels in the report's clean style (white bg, object cropped to its
alpha bbox + centered, Liberation Serif caption above each tile), one panel per model:

  left  : step 22k, render loss not yet enabled (pure Sinkhorn recon)  -> "Without render loss"
  right : step 69k, +47k steps of render-loss fine-tuning              -> "With render loss"

Same 16 captions, identical initial noise (seed 0), cfg, sampler and decode pipeline; only
the render loss (and the fine-tuning steps it drives) differs.

    CUDA_VISIBLE_DEVICES=0 python jit/make_render_loss_figure.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from jit.eval_clip_alignment import build, gen_render
from jit.make_report_grids import make_tile, tile_grid, FONT
from jit.render_ours_simple import DEFAULT as SIMPLE_CAPTIONS

OFF_CKPT = "output/jit_sinkhorn_pmean_curriculum_20260526_021638/0022000.pt"
ON_CKPT = "output/jit_final_sinkhorn_render_cam2_22k_20260528_210521/0069000.pt"


def _title_font(size):
    for cand in (os.path.expanduser("~/.fonts/LiberationSerif-Bold.ttf"),
                 "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
                 FONT):
        if os.path.exists(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def render_panel(ckpt, caps, cfg, seed, view, render_size, device):
    """Render one checkpoint -> clean 4x4 PIL grid (object cropped, caption above tile)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(caps, f, ensure_ascii=False)
        capfile = f.name
    b = build(ckpt, device, render_size=render_size, prompts_file=capfile)
    b["amp_dtype"] = torch.bfloat16
    sampler = b["cfg"]["val_sampler"]
    steps = int(b["cfg"]["val_sampling_steps"])
    rgb, al = gen_render(b, list(range(len(caps))), seed, cfg, steps, sampler, [view], device,
                         return_alpha=True)
    img = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).cpu()    # (P,1,3,H,W)
    alpha = (al.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    os.unlink(capfile)
    n = len(caps)
    rows = cols = int(round(n ** 0.5))
    cells = [(make_tile(img[i, 0], alpha[i, 0], bg=1.0, tile=320), caps[i]) for i in range(n)]
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    tile_grid(cells, rows, cols, tmp)
    return Image.open(tmp).convert("RGB"), int(b["step"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off_ckpt", default=OFF_CKPT)
    ap.add_argument("--on_ckpt", default=ON_CKPT)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--render_size", type=int, default=224)
    ap.add_argument("--captions", default=None, help="JSON list; default = 16 simple captions")
    ap.add_argument("--title_size", type=int, default=46)
    ap.add_argument("--out", default="output/report/render_loss_comparison")
    args = ap.parse_args()

    caps = json.load(open(args.captions, encoding="utf-8")) if args.captions else list(SIMPLE_CAPTIONS)
    device = torch.device("cuda")

    off_grid, off_step = render_panel(args.off_ckpt, caps, args.cfg, args.seed, args.view,
                                      args.render_size, device)
    on_grid, on_step = render_panel(args.on_ckpt, caps, args.cfg, args.seed, args.view,
                                    args.render_size, device)

    panels = [(off_grid, "Without Render Loss"),
              (on_grid, "With Render Loss")]

    # ---- compose: two parallel panels, each with a centered title -------------------
    gap = 70
    pad = 20
    title_h = int(args.title_size * 1.5)
    tfont = _title_font(args.title_size)
    pw = max(p.width for p, _ in panels)
    ph = max(p.height for p, _ in panels)
    W = pad * 2 + 2 * pw + gap
    H = pad + title_h + ph + pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for i, (panel, title) in enumerate(panels):
        x0 = pad + i * (pw + gap)
        cx = x0 + pw // 2
        tb = d.textbbox((0, 0), title, font=tfont)
        ty = pad + (title_h - (tb[3] - tb[1])) // 2 - tb[1]
        d.text((cx - (tb[2] - tb[0]) // 2, ty), title, font=tfont, fill=(0, 0, 0))
        canvas.paste(panel, (x0, pad + title_h))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(f"{args.out}.png")
    canvas.save(f"{args.out}.pdf", "PDF", resolution=200.0)
    print(f"[saved] {args.out}.png  +  {args.out}.pdf  ({W}x{H})")
    print(f"[info] off=step{off_step} on=step{on_step} cfg={args.cfg} seed={args.seed} "
          f"view={args.view} n_caps={len(caps)}")


if __name__ == "__main__":
    main()
