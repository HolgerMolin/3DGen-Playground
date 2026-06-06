"""CFG sweep for a *class-conditioned* JiT checkpoint.

The text-conditioned showcase (jit/render_cfg_samples.py / eval_clip_alignment.py)
can't drive a class-conditioned run (LabelEmbedder, no CLIP text pool). This mirrors
the same validated generate+render pipeline but conditions on class ids and lays the
output out as a grid:

    rows  = classes (largest-first, named from class_names.json)
    cols  = cfg scales

Each row shares its initial noise across all cfg columns, so the only variable along
a row is the guidance scale. Seed convention matches jit/train_gsplat.py's
_run_validation_grid (val_grid_seed + class_id at seed_round 0), so cfg=val_cfg_scale
reproduces the run's own validation tiles.

    python jit/render_classcond_cfg.py --ckpt output/<run>/0048000.pt --cfg_scales 2,4,6,8
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

from jit import train_gsplat as T
from jit.sampling import sample_model


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def build(ckpt_path: str, device: torch.device, render_size: int | None = None) -> dict:
    """Mirror eval_clip_alignment.build but for the class-conditioned path."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = vars(ck["args"]) if hasattr(ck["args"], "__dict__") else ck["args"]
    assert cfg["sh_degree0_only"], "this tool assumes the DC-only (14ch) config"
    in_channels = len(T.DC_ONLY_FEATURE_INDICES)
    feature_indices = torch.tensor(T.DC_ONLY_FEATURE_INDICES, dtype=torch.long)

    state = ck["ema"]
    # num_classes is the LabelEmbedder table minus its learnable null row.
    table = state["y_embedder.embedding_table.weight"]
    num_classes = int(table.shape[0]) - 1

    model = T.JiT_3DGS_models[cfg["model"]](
        input_size=128, in_channels=in_channels,
        text_dim=int(cfg.get("text_dim") or 768),
        num_classes=num_classes,
        class_dropout_prob=float(cfg["class_dropout_prob"]),
        learn_sigma=False, gradient_checkpointing=False,
        bottleneck=bool(cfg["bottleneck"]),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
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
    rsize = int(render_size or cfg["train_render_size"])
    train_cameras = T._prepare_train_cameras(ref_cameras, rsize, device)

    return dict(
        model=model, cfg=cfg, in_channels=in_channels, num_classes=num_classes,
        plane_to_sphere=plane_to_sphere, norm_mean=norm_mean, norm_std=norm_std,
        rank_tables=rank_tables, train_cameras=train_cameras, renderer=renderer,
        render_size=rsize, step=int(ck.get("step", 0)),
    )


@torch.no_grad()
def gen_render(b, class_ids, seed_round, cfg_scale, steps, sampler, cam_indices, device):
    """class_ids: list[int]. Returns (N, n_views, 3, H, W) in [0, 1]."""
    cfg = b["cfg"]
    N = len(class_ids)
    shape = T.resolve_sampling_shape(model=b["model"], batch_size=N, in_channels=b["in_channels"])
    _, C, H, W = shape

    # Per-class noise; matches _run_validation_grid (val_grid_seed + id) at seed_round 0.
    base = int(cfg["val_grid_seed"])
    initial_noise = torch.empty((N, C, H, W), dtype=torch.float32)
    g = torch.Generator(device="cpu")
    for j, cid in enumerate(class_ids):
        g.manual_seed(base + int(seed_round) * 100_000 + int(cid))
        initial_noise[j] = torch.randn((C, H, W), generator=g, dtype=torch.float32)
    initial_noise = initial_noise.to(device)
    cond_embeds = torch.tensor(class_ids, dtype=torch.long, device=device)

    sample = sample_model(
        sampler=sampler, model=b["model"], shape=(N, C, H, W),
        cond_embeds=cond_embeds, num_inference_steps=steps, device=device,
        predict_xstart=True, diffusion_steps=1000, cfg_scale=float(cfg_scale),
        timestep_schedule="logit_normal",
        P_mean=float(cfg["P_mean"]), P_std=float(cfg["P_std"]),
        initial_noise=initial_noise,
    )
    pc = T._plane_to_point_cloud_batch(sample.float(), b["plane_to_sphere"])
    pc_raw = T._denormalize_point_cloud(pc, b["norm_mean"], b["norm_std"])
    gauss = T._point_clouds_to_gsplat_inputs(
        pc_raw.to(device), dc_only=True, detach_input=True,
        rank_transform_tables=b["rank_tables"],
    )
    return T._render_gsplat_batch(b["renderer"], gauss, b["train_cameras"], cam_indices, device)


def save_cfg_grid(tiles, labels, out_path, rows, cols, cfg, step, title_prefix, no_labels=False):
    """One clean rows x cols grid for a single cfg. By default adds a class-name caption
    strip under each tile and a title bar; with no_labels=True it's a tight, text-free grid."""
    H = int(tiles[0].shape[-2]); W = int(tiles[0].shape[-1])
    cap, title_h = (0, 0) if no_labels else (22, 30)
    canvas = np.zeros((title_h + rows * (H + cap), cols * W, 3), dtype=np.uint8)
    for i in range(min(rows * cols, len(tiles))):
        r, c = divmod(i, cols)
        tile = (tiles[i].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        y = title_h + r * (H + cap); x = c * W
        canvas[y:y + H, x:x + W] = tile
    img = Image.fromarray(canvas)
    if not no_labels:
        draw = ImageDraw.Draw(img)
        draw.text((8, 6), f"{title_prefix} · CFG {cfg:g}", fill=(255, 255, 255), font=_font(18))
        fcap = _font(13)
        for i in range(min(rows * cols, len(tiles))):
            r, c = divmod(i, cols)
            lab = labels[i]
            try:
                tw = draw.textlength(lab, font=fcap)
            except Exception:
                tw = len(lab) * 7
            x = c * W + max(3, int((W - tw) // 2))
            y = title_h + r * (H + cap) + H + 4
            draw.text((x, y), lab, fill=(225, 225, 225), font=fcap)
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg_scales", default="2,4,6,8")
    ap.add_argument("--classes", default=None,
                    help="comma list of class ids; default = first --n_classes (largest)")
    ap.add_argument("--n_classes", type=int, default=12)
    ap.add_argument("--per_cfg_grid", action="store_true",
                    help="emit one clean rows x cols grid PER cfg (separate files) instead of the combined grid")
    ap.add_argument("--grid_rows", type=int, default=4)
    ap.add_argument("--grid_cols", type=int, default=4)
    ap.add_argument("--no_labels", action="store_true",
                    help="per_cfg_grid: emit a tight, text-free grid (no title bar / captions)")
    ap.add_argument("--seed_round", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="default: run's val_sampling_steps")
    ap.add_argument("--sampler", default=None, help="default: run's val_sampler")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--sub_batch", type=int, default=8)
    ap.add_argument("--class_names", default="object_labels/hier_uniform_k1000/class_names.json")
    ap.add_argument("--max_gpu_gb", type=float, default=5.0,
                    help="hard cap this process's GPU allocation; 0 disables. Protects co-resident training")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.max_gpu_gb and args.max_gpu_gb > 0:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        frac = min(0.95, float(args.max_gpu_gb) / total_gb)
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        print(f"[mem] capping at {args.max_gpu_gb:.1f} GB (frac={frac:.3f} of {total_gb:.0f} GB)")

    t0 = time.time()
    b = build(args.ckpt, device)
    cfg = b["cfg"]
    sampler = args.sampler or cfg["val_sampler"]
    steps = int(args.steps or cfg["val_sampling_steps"])
    cfg_scales = [float(x) for x in args.cfg_scales.split(",") if x.strip()]

    n_default = (args.grid_rows * args.grid_cols) if args.per_cfg_grid else args.n_classes
    if args.classes:
        class_ids = [int(x) for x in args.classes.split(",") if x.strip()]
    else:
        class_ids = list(range(min(n_default, b["num_classes"])))

    names = {}
    if args.class_names and os.path.exists(args.class_names):
        names = {int(k): v for k, v in json.load(open(args.class_names)).items()}
    if args.per_cfg_grid:
        row_labels = [(names.get(cid, f"class {cid}") or f"class {cid}")[:26] for cid in class_ids]
    else:
        row_labels = [f"{cid}:{names.get(cid, '')}"[:22] for cid in class_ids]

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "cfg_showcase")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[build] {time.time()-t0:.1f}s step={b['step']} model={cfg['model']} num_classes={b['num_classes']} "
          f"sampler={sampler} steps={steps} render={b['render_size']}px")
    print(f"[plan] classes={class_ids}")
    print(f"[plan] cfg_scales={cfg_scales}")

    cam = [int(args.cam)]
    N = len(class_ids)

    # ---- per-cfg separate grids (presentation) --------------------------------------
    if args.per_cfg_grid:
        rows, cols = args.grid_rows, args.grid_cols
        title_prefix = f"JiT class-cond · step {b['step']//1000}k"
        outs = []
        for cs in cfg_scales:
            tiles = [None] * N
            for s in range(0, N, args.sub_batch):
                chunk = class_ids[s:s + args.sub_batch]
                r = gen_render(b, chunk, args.seed_round, cs, steps, sampler, cam, device)
                for k in range(len(chunk)):
                    tiles[s + k] = r[k, 0].detach().cpu()
            suffix = "_clean" if args.no_labels else ""
            out_path = os.path.join(out_dir, f"grid{rows}x{cols}_cfg{cs:g}_step{b['step']:07d}{suffix}.png")
            save_cfg_grid(tiles, row_labels, out_path, rows, cols, cs, b["step"], title_prefix,
                          no_labels=args.no_labels)
            outs.append(out_path)
            print(f"[cfg {cs:g}] {time.time()-t0:.0f}s -> {out_path}")
        print(f"[done] {time.time()-t0:.0f}s  {len(outs)} grids in {out_dir}")
        return

    # tiles[(cfg_idx, row)] -> (3,H,W). Sample one cfg at a time, sub-batching classes.
    grid = [[None] * N for _ in cfg_scales]
    for ci, cs in enumerate(cfg_scales):
        for s in range(0, N, args.sub_batch):
            chunk = class_ids[s:s + args.sub_batch]
            r = gen_render(b, chunk, args.seed_round, cs, steps, sampler, cam, device)
            for k in range(len(chunk)):
                grid[ci][s + k] = r[k, 0].detach().cpu()
        print(f"[cfg {cs:g}] {time.time()-t0:.0f}s done")

    H = int(grid[0][0].shape[-2]); W = int(grid[0][0].shape[-1])
    pad_l, pad_t = 150, 22  # left gutter for class names, top gutter for cfg headers
    canvas = np.zeros((pad_t + N * H, pad_l + len(cfg_scales) * W, 3), dtype=np.uint8)
    for ci in range(len(cfg_scales)):
        for r in range(N):
            tile = (grid[ci][r].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            y = pad_t + r * H; x = pad_l + ci * W
            canvas[y:y + H, x:x + W] = tile
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    f_hdr, f_row = _font(16), _font(13)
    for ci, cs in enumerate(cfg_scales):
        draw.text((pad_l + ci * W + 4, 3), f"cfg {cs:g}", fill=(255, 255, 0), font=f_hdr)
    for r in range(N):
        draw.text((4, pad_t + r * H + H // 2 - 7), row_labels[r], fill=(255, 255, 0), font=f_row)

    tag = f"classcfg_step{b['step']:07d}_seed{args.seed_round}"
    out_path = os.path.join(out_dir, f"showcase_{tag}.png")
    img.save(out_path)
    print(f"[grid] {N} classes x {len(cfg_scales)} cfg -> {out_path}")
    print(f"[done] {time.time()-t0:.0f}s  out_dir={out_dir}")


if __name__ == "__main__":
    main()
