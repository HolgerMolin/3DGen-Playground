"""Render OUR model on a set of SIMPLE, plausible single-object captions and build a clean
4x4 grid — shows our model in a favorable setting (it handles simple common objects far
better than the surreal multi-attribute combos in the 500-set baseline).

RUN UNDER .3dgen (PATH must include .3dgen/bin for gsplat):
    CUDA_VISIBLE_DEVICES=0 python jit/render_ours_simple.py --max_gpu_gb 18

Saves output/report/ours_simple.pt (img+alpha, all 4 views — iterate bg/view on CPU) and
output/report/grid_ours_simple_4x4.png. Same style as the other grids (white bg, Times,
caption superimposed, no chrome).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from jit.eval_clip_alignment import build, gen_render
from jit.make_report_grids import make_tile, tile_grid

# Simple, common, plausible single objects (in-distribution for GaussianVerse/Objaverse).
DEFAULT = [
    "A wooden chair.", "A red apple.", "A blue ceramic vase.", "A green frog.",
    "A yellow rubber duck.", "A brown teddy bear.", "A potted cactus.", "A red sports car.",
    "A white teapot.", "A leather boot.", "An orange pumpkin.", "An acoustic guitar.",
    "A wooden sailboat.", "A wooden barrel.", "A red rose.", "A stone statue.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default=None, help="JSON list of strings; default = built-in 16 simple")
    ap.add_argument("--ckpt", default=str(_REPO / "output/jit_final_sinkhorn_render_bs512_69k_20260529_071914/0099195.pt"))
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_views", type=int, default=4)
    ap.add_argument("--render_size", type=int, default=224)
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--max_gpu_gb", type=float, default=None)
    ap.add_argument("--tag", default="simple", help="output filename tag: grid_ours_<tag>_4x4.png / ours_<tag>.pt")
    ap.add_argument("--out_dir", default=str(_REPO / "output/report"))
    args = ap.parse_args()

    caps = json.load(open(args.captions, encoding="utf-8")) if args.captions else DEFAULT
    os.makedirs(args.out_dir, exist_ok=True)
    capfile = os.path.join(args.out_dir, f"ours_{args.tag}_captions.json")
    json.dump(caps, open(capfile, "w"), indent=2, ensure_ascii=False)

    device = torch.device("cuda")
    if args.max_gpu_gb:
        idx = torch.cuda.current_device()
        tot = torch.cuda.get_device_properties(idx).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gb / tot), idx)
        print(f"[mem] cap {args.max_gpu_gb:.0f}GiB of {tot:.0f}GiB")

    t0 = time.time()
    b = build(args.ckpt, device, render_size=args.render_size, prompts_file=capfile)
    b["amp_dtype"] = torch.bfloat16
    ncam = int(b["train_cameras"]["viewmats"].shape[0])
    cam_indices = sorted({int(round(x)) for x in np.linspace(0, ncam - 1, args.n_views)})
    sampler = b["cfg"]["val_sampler"]; steps = int(b["cfg"]["val_sampling_steps"])
    rgb, al = gen_render(b, list(range(len(caps))), args.seed, args.cfg, steps, sampler, cam_indices, device, return_alpha=True)
    img = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).cpu()       # (P,V,3,H,W)
    alpha = (al.clamp(0, 1) * 255).round().to(torch.uint8).cpu()       # (P,V,1,H,W)
    print(f"[ours-simple] rendered {len(caps)} in {time.time()-t0:.0f}s | cfg={args.cfg} cams={cam_indices}")

    torch.save({"captions": caps, "cam_indices": cam_indices, "cfg": args.cfg, "seed": args.seed,
                "ours_img": img, "ours_alpha": alpha}, os.path.join(args.out_dir, f"ours_{args.tag}.pt"))

    v = args.view
    cells = [(make_tile(img[i, v], alpha[i, v], bg=1.0, tile=320), caps[i]) for i in range(len(caps))]
    out = tile_grid(cells, 4, 4, os.path.join(args.out_dir, f"grid_ours_{args.tag}_4x4.png"))
    print("saved", out)


if __name__ == "__main__":
    main()
