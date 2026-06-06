"""Render two checkpoints on a shared caption set into a VQAScore bundle (run under .3dgen).

Mirrors jit/render_ours_simple.py's render path (build + gen_render, return_alpha) and saves a
bundle that jit/vqascore_pair.py scores. Identical captions / views / cfg / seed for both models,
so only the checkpoint differs.

    CUDA_VISIBLE_DEVICES=0 python jit/render_vqa_bundle.py --max_gpu_gb 18
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from jit.eval_clip_alignment import build, gen_render
from jit.render_ours_simple import DEFAULT as SIMPLE_CAPTIONS

OFF = "output/jit_sinkhorn_pmean_curriculum_20260526_021638/0022000.pt"
ON = "output/jit_final_sinkhorn_render_cam2_22k_20260528_210521/0069000.pt"


def render_one(ckpt, caps_file, cfg, seed, n_views, render_size, device, sub_batch=32):
    b = build(ckpt, device, render_size=render_size, prompts_file=caps_file)
    b["amp_dtype"] = torch.bfloat16
    ncam = int(b["train_cameras"]["viewmats"].shape[0])
    cams = sorted({int(round(x)) for x in np.linspace(0, ncam - 1, n_views)})
    sampler = b["cfg"]["val_sampler"]; steps = int(b["cfg"]["val_sampling_steps"])
    P = len(b["prompts"])
    imgs, alphas = [], []
    for s in range(0, P, sub_batch):
        idxs = list(range(s, min(P, s + sub_batch)))
        rgb, al = gen_render(b, idxs, seed, cfg, steps, sampler, cams, device, return_alpha=True)
        imgs.append((rgb.clamp(0, 1) * 255).round().to(torch.uint8).cpu())
        alphas.append((al.clamp(0, 1) * 255).round().to(torch.uint8).cpu())
        print(f"  [{ckpt.split('/')[-2][:24]}] rendered {min(P, s+sub_batch)}/{P}", flush=True)
    img = torch.cat(imgs, 0)      # (P,V,3,H,W)
    alpha = torch.cat(alphas, 0)  # (P,V,1,H,W)
    return img, alpha, cams, int(b["step"]), list(b["prompts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off_ckpt", default=OFF)
    ap.add_argument("--on_ckpt", default=ON)
    ap.add_argument("--captions", default=None, help="JSON list; default = 16 simple captions")
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_views", type=int, default=4)
    ap.add_argument("--render_size", type=int, default=224)
    ap.add_argument("--sub_batch", type=int, default=32)
    ap.add_argument("--max_gpu_gb", type=float, default=None)
    ap.add_argument("--out", default="output/report/eval_logs/renders_render_loss.pt")
    args = ap.parse_args()

    caps = json.load(open(args.captions, encoding="utf-8")) if args.captions else list(SIMPLE_CAPTIONS)
    capfile = str(_REPO / "output/report/eval_logs/_vqa_caps.json")
    json.dump(caps, open(capfile, "w"), ensure_ascii=False)

    device = torch.device("cuda")
    if args.max_gpu_gb:
        idx = torch.cuda.current_device()
        tot = torch.cuda.get_device_properties(idx).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gb / tot), idx)
        print(f"[mem] cap {args.max_gpu_gb:.0f}GiB of {tot:.0f}GiB")

    off_img, off_al, cams0, off_step, p0 = render_one(args.off_ckpt, capfile, args.cfg, args.seed,
                                                      args.n_views, args.render_size, device, args.sub_batch)
    on_img, on_al, cams1, on_step, p1 = render_one(args.on_ckpt, capfile, args.cfg, args.seed,
                                                    args.n_views, args.render_size, device, args.sub_batch)
    assert cams0 == cams1 and p0 == p1, "view/prompt mismatch between checkpoints"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"captions": caps, "cam_indices": cams0, "cfg": args.cfg, "seed": args.seed,
                "off_step": off_step, "on_step": on_step,
                "off_img": off_img, "off_alpha": off_al,
                "on_img": on_img, "on_alpha": on_al}, args.out)
    print(f"[saved] {args.out} | {len(caps)} caps x {len(cams0)} views | off=step{off_step} on=step{on_step} "
          f"cfg={args.cfg} cams={cams0}")


if __name__ == "__main__":
    main()
