"""Build clean report grids from saved render tensors (CPU only — iterate, no GPU).

CVPR-style: white background (alpha-composited), Liberation Serif (Times New Roman metric),
caption superimposed top-left (no scores/titles/headers), and each object CROPPED to its
alpha bounding box then resized to fill the tile ("camera as close as possible", uniform
across samples). Large font.

    python jit/make_report_grids.py [--view 0] [--tile 320] [--font 30]
    # GaussianCube comparison (renders_gc.pt has gc_* keys):
    python jit/make_report_grids.py --bundle output/report/renders_gc.pt \
        --baseline_prefix gc --baseline_label GaussianCube --out_dir output/report/GaussianCube

Reads a renders bundle (needs ours_alpha + <baseline>_alpha) -> grid_*.png + manifest.
"""
import argparse, json, os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

FONT = os.path.expanduser("~/.fonts/LiberationSerif-Regular.ttf")
if not os.path.exists(FONT):
    FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"


def composite(rgb, alpha, bg):
    r = rgb.float() / 255.0
    a = alpha.float() / 255.0
    out = (r + (1.0 - a) * bg).clamp(0, 1)
    return Image.fromarray((out * 255).round().byte().permute(1, 2, 0).numpy())


def make_tile(rgb, alpha, *, bg=1.0, tile=320, margin=0.07, thr=12):
    """Crop to the object's alpha bbox (square, with margin), composite on bg, resize to tile."""
    a = alpha[0].numpy()
    ys, xs = np.where(a > thr)
    H, W = a.shape
    if len(xs) == 0:
        y0, y1, x0, x1 = 0, H, 0, W
    else:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = side // 2
    box = (max(0, cx - half), max(0, cy - half), min(W, cx + half), min(H, cy + half))
    crop = composite(rgb, alpha, bg).crop(box)
    s = max(crop.width, crop.height)
    sq = Image.new("RGB", (s, s), (int(bg * 255),) * 3)
    sq.paste(crop, ((s - crop.width) // 2, (s - crop.height) // 2))
    return sq.resize((tile, tile), Image.LANCZOS)


def wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def fit_caption(draw, text, max_w, max_lines=3, max_size=30, min_size=18):
    for sz in range(max_size, min_size - 1, -1):
        f = ImageFont.truetype(FONT, sz)
        lines = wrap(draw, text, f, max_w)
        if len(lines) <= max_lines and all(draw.textlength(ln, font=f) <= max_w for ln in lines):
            return lines, f
    f = ImageFont.truetype(FONT, min_size)
    return wrap(draw, text, f, max_w)[:max_lines], f


def draw_caption(draw, x, y, lines, font, fill=(0, 0, 0), halo=(255, 255, 255)):
    lh = font.size + 3
    for k, ln in enumerate(lines):
        yy = y + k * lh
        for dx in (-2, -1, 1, 2):
            for dy in (-2, -1, 1, 2):
                draw.text((x + dx, yy + dy), ln, font=font, fill=halo)
        draw.text((x, yy), ln, font=font, fill=fill)


def tile_grid(cells, rows, cols, out, *, gutter=14, cap_pad=8, max_size=30):
    """cells: row-major list of (PIL tile, caption). White canvas, caption superimposed."""
    W, H = cells[0][0].size
    cw, ch = W + gutter, H + gutter
    canvas = Image.new("RGB", (cols * cw + gutter, rows * ch + gutter), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for i, (img, cap) in enumerate(cells):
        r, c = divmod(i, cols)
        x, y = gutter + c * cw, gutter + r * ch
        canvas.paste(img, (x, y))
        lines, font = fit_caption(d, cap, W - 2 * cap_pad, max_size=max_size)
        draw_caption(d, x + cap_pad, y + cap_pad, lines, font)
    canvas.save(out)
    return out


def strat(scores, k):
    order = np.argsort(scores)
    ranks = np.linspace(0, len(order) - 1, k).round().astype(int)
    return [int(order[r]) for r in ranks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="output/report/renders.pt")
    ap.add_argument("--baseline_prefix", default="trellis",
                    help="bundle key prefix for the baseline model (e.g. 'trellis', 'gc')")
    ap.add_argument("--baseline_label", default="TRELLIS",
                    help="display label / filename token for the baseline model")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--bg", type=float, default=255.0)
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--font", type=int, default=30)
    ap.add_argument("--out_dir", default="output/report")
    ap.add_argument("--ours_score_json", default=None,
                    help="per_object JSON (e.g. scores_vqa_ours.json) to rank/stratify ours by, instead of the bundle's score")
    ap.add_argument("--baseline_score_json", default=None,
                    help="per_object JSON (e.g. scores_vqa_gc.json) to stratify the baseline by")
    ap.add_argument("--score_label", default="CLIP", help="ranking-score name (recorded in the manifest)")
    args = ap.parse_args()

    d = torch.load(args.bundle, weights_only=False)
    caps = d["captions"]
    bp, blabel = args.baseline_prefix, args.baseline_label
    bname = blabel.lower()
    oi, ti, oa, ta = d["ours_img"], d[f"{bp}_img"], d["ours_alpha"], d[f"{bp}_alpha"]

    def _scores(path, fallback):  # rank/stratify by an external per_object JSON (e.g. VQAScore) or the bundle's score
        if not path:
            return np.array(fallback, dtype=np.float64)
        per = sorted(json.load(open(path, encoding="utf-8"))["per_object"], key=lambda r: r["idx"])
        arr = np.array([r["score"] for r in per], dtype=np.float64)[:len(caps)]
        assert all(per[i]["caption"].strip() == caps[i].strip() for i in range(len(arr))), "caption/order mismatch"
        return arr
    os_ = _scores(args.ours_score_json, d["ours_score"])
    ts_ = _scores(args.baseline_score_json, d[f"{bp}_score"])
    v, bg = args.view, args.bg / 255.0
    os.makedirs(args.out_dir, exist_ok=True)
    man = {}

    def to(i): return (make_tile(oi[i, v], oa[i, v], bg=bg, tile=args.tile), caps[i])
    def tt(i): return (make_tile(ti[i, v], ta[i, v], bg=bg, tile=args.tile), caps[i])

    idx = strat(ts_, 16)
    tile_grid([tt(i) for i in idx], 4, 4, f"{args.out_dir}/grid_{bname}_4x4.png", max_size=args.font)
    man[f"{bname}_4x4"] = [caps[i] for i in idx]

    idx = strat(os_, 16)
    tile_grid([to(i) for i in idx], 4, 4, f"{args.out_dir}/grid_ours_4x4.png", max_size=args.font)
    man["ours_4x4"] = [caps[i] for i in idx]

    for name, sel in [("best", np.argsort(os_)[::-1][:4]), ("worst", np.argsort(os_)[:4])]:
        sel = [int(i) for i in sel]
        tile_grid([to(i) for i in sel] + [tt(i) for i in sel], 2, 4,
                  f"{args.out_dir}/grid_{name}_4x2.png", max_size=args.font)
        man[f"{name}_4x2"] = {"captions": [caps[i] for i in sel],
                              "layout": f"row0=ours, row1={blabel}"}

    # head-to-head: 6 captions stratified across ours score, row0=ours / row1=baseline
    sel = strat(os_, 6)
    tile_grid([to(i) for i in sel] + [tt(i) for i in sel], 2, 6,
              f"{args.out_dir}/grid_compare_2x6.png", max_size=args.font)
    man["compare_2x6"] = {"captions": [caps[i] for i in sel],
                          "layout": f"row0=ours, row1={blabel} (stratified by ours score, low->high)"}

    man["_meta"] = {"score_label": args.score_label, "baseline": blabel,
                    "best_worst_ranked_by": f"ours {args.score_label}"}
    json.dump(man, open(f"{args.out_dir}/grids_manifest.json", "w"), indent=2)
    print(f"wrote grids to {args.out_dir} | baseline={blabel} ranked_by={args.score_label} "
          f"view={d['cam_indices'][v]} tile={args.tile} font={args.font} bg={args.bg}")


if __name__ == "__main__":
    main()
