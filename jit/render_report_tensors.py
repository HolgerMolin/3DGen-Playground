"""Render BOTH models (ours @cfg4/seed0, TRELLIS from PLYs) for the 150 shared captions
and save the image tensors, so report grids can be (re)built on CPU without re-sampling.

RUN UNDER .3dgen (with PATH including .3dgen/bin for gsplat). GPU only for this step;
grid layout/iteration is done by jit/make_report_grids.py on CPU.

    python jit/render_report_tensors.py --max_gpu_gb 18   # co-resident-safe cap

Saves output/report/renders.pt:
  {captions:[150], cam_indices:[..], cfg, seed,
   ours_img:    uint8 (150, V, 3, H, W),  ours_score:[150]   (official 500-baseline cfg4),
   trellis_img: uint8 (150, V, 3, H, W),  trellis_score:[150] (clip_trellis.json)}
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from jit import train_gsplat as T
from jit.eval_clip_alignment import build, gen_render
from jit.eval_trellis_compare import render_trellis_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default=str(_REPO / "data/baseline_captions_150_seed0.json"))
    ap.add_argument("--ckpt", default=str(_REPO / "output/jit_final_sinkhorn_render_bs512_69k_20260529_071914/0099195.pt"))
    ap.add_argument("--manifest", default=str(_REPO / "output/trellis_compare/text_base_seed0/manifest.json"))
    ap.add_argument("--baseline500", default=str(_REPO / "output/jit_final_sinkhorn_render_bs512_69k_20260529_071914/clip_eval_step0099195/clip_alignment.json"))
    ap.add_argument("--trellis_clip", default=str(_REPO / "output/trellis_compare/text_base_seed0/clip_trellis.json"))
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_views", type=int, default=4)
    ap.add_argument("--render_size", type=int, default=224)
    ap.add_argument("--axis", default="x_-90")
    ap.add_argument("--target_radius", type=float, default=0.6)
    ap.add_argument("--sub_batch", type=int, default=30)
    ap.add_argument("--max_gpu_gb", type=float, default=None)
    ap.add_argument("--out_dir", default=str(_REPO / "output/report"))
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.max_gpu_gb:
        idx = torch.cuda.current_device()
        tot = torch.cuda.get_device_properties(idx).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gb / tot), idx)
        print(f"[mem] cap {args.max_gpu_gb:.0f}GiB of {tot:.0f}GiB")
    os.makedirs(args.out_dir, exist_ok=True)
    captions = json.load(open(args.captions, encoding="utf-8"))
    P = len(captions)

    # ---- OURS: build flagship, sample @cfg/seed, render -----------------------------
    t0 = time.time()
    b = build(args.ckpt, device, render_size=args.render_size, prompts_file=args.captions)
    b["amp_dtype"] = torch.bfloat16
    ncam = int(b["train_cameras"]["viewmats"].shape[0])
    cam_indices = sorted({int(round(x)) for x in np.linspace(0, ncam - 1, args.n_views)})
    H = W = args.render_size
    sampler = b["cfg"]["val_sampler"]; steps = int(b["cfg"]["val_sampling_steps"])
    print(f"[ours] build {time.time()-t0:.0f}s | cfg={args.cfg} seed={args.seed} sampler={sampler}:{steps} cams={cam_indices}")
    ours = torch.empty((P, len(cam_indices), 3, H, W), dtype=torch.uint8)
    ours_a = torch.empty((P, len(cam_indices), 1, H, W), dtype=torch.uint8)
    for s in range(0, P, args.sub_batch):
        idxs = list(range(s, min(P, s + args.sub_batch)))
        rgb, al = gen_render(b, idxs, args.seed, args.cfg, steps, sampler, cam_indices, device, return_alpha=True)
        ours[idxs] = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        ours_a[idxs] = (al.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        print(f"  ours {min(P, s+args.sub_batch)}/{P} ({time.time()-t0:.0f}s)", flush=True)
    del b
    torch.cuda.empty_cache()

    # ---- TRELLIS: render saved PLYs through the same pipeline ------------------------
    t1 = time.time()
    man = {m["caption"]: m["ply"] for m in json.load(open(args.manifest, encoding="utf-8"))}
    ref = T._load_reference_cameras(str(_REPO / "artifacts/ref_camera.tar.gz"))
    cams = T._prepare_train_cameras(ref, args.render_size, device)
    renderer = T._try_import_renderer()
    trellis = torch.empty((P, len(cam_indices), 3, H, W), dtype=torch.uint8)
    trellis_a = torch.empty((P, len(cam_indices), 1, H, W), dtype=torch.uint8)
    for i, cap in enumerate(captions):
        rgb, al = render_trellis_ply(man[cap], target_radius=args.target_radius, pct=97.0, axis=args.axis,
                                     renderer=renderer, cams=cams, cam_indices=cam_indices, device=device,
                                     return_alpha=True)
        trellis[i] = (rgb[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        trellis_a[i] = (al[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        if (i + 1) % 30 == 0:
            print(f"  trellis {i+1}/{P} ({time.time()-t1:.0f}s)", flush=True)

    # ---- official scores (for selection + labels) -----------------------------------
    bl = json.load(open(args.baseline500))
    ocfg = {p["prompt"]: p["mean"] for p in bl["results"][f"cfg{args.cfg:g}"]["per_prompt"]}
    tcl = {o["caption"]: o["score"] for o in json.load(open(args.trellis_clip))["per_object"]}
    ours_score = [float(ocfg[c]) for c in captions]
    trellis_score = [float(tcl[c]) for c in captions]

    out = os.path.join(args.out_dir, "renders.pt")
    torch.save({
        "captions": captions, "cam_indices": cam_indices, "cfg": args.cfg, "seed": args.seed,
        "render_size": args.render_size, "axis": args.axis, "target_radius": args.target_radius,
        "ours_img": ours, "trellis_img": trellis,
        "ours_alpha": ours_a, "trellis_alpha": trellis_a,
        "ours_score": ours_score, "trellis_score": trellis_score,
        "ours_label": f"Ours (130M, cfg{args.cfg:g})", "trellis_label": "TRELLIS-text-base",
    }, out)
    print(f"\n[done] saved {out}  ours{tuple(ours.shape)} trellis{tuple(trellis.shape)}  "
          f"({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
