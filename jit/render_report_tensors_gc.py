"""Render BOTH models (ours @cfg/seed, GaussianCube from its .ply manifest) for the shared
caption set and SAVE the image+alpha tensors AND CLIP scores to disk, so report figures can be
(re)built on CPU without re-sampling/re-rendering. GaussianCube analogue of render_report_tensors.py.

RUN UNDER .3dgen (activate so .3dgen/bin is on PATH -> gsplat finds ninja).

    # ours is the expensive part (diffusion sampling); can run while GC gen is still going:
    python jit/render_report_tensors_gc.py --stage ours --max_gpu_gb 25
    # after GaussianCube generation finishes (manifest complete):
    python jit/render_report_tensors_gc.py --stage gc   --max_gpu_gb 25
    # (or --stage both in one shot once the manifest is ready)

Incrementally updates output/report/renders_gc.pt:
  {captions:[N], cam_indices:[V], cfg, seed, render_size, axis, target_radius,
   ours_img/ours_alpha: uint8 (N,V,3,H,W)/(N,V,1,H,W),  ours_score:[N],
   gc_img/gc_alpha:     uint8 (N,V,3,H,W)/(N,V,1,H,W),  gc_score:[N],
   ours_label, gc_label}
Also writes output/report/scores_ours_cfg{cfg}.json and output/report/scores_gc.json
(formats consumed by jit/compare_gaussiancube_ours.py).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from jit import train_gsplat as T
from jit.eval_clip_alignment import (build, gen_render, load_clip,
                                     clip_text_features, clip_image_features)
from jit.eval_trellis_compare import render_trellis_ply


def _load_captions(path):
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):
        return [d[k] for k in sorted(d)]
    return [x if isinstance(x, str) else (x.get("caption") or x.get("text") or x.get("prompt")) for x in d]


def _score(clip, cmean, cstd, text_feats, rgb_pv3hw, device):
    """CLIP alignment per object: mean_v cos(CLIP_img(view_v), CLIP_txt). rgb_pv3hw: (V,3,H,W) in [0,1]."""
    img = clip_image_features(clip, cmean, cstd, rgb_pv3hw, device).mean(dim=0, keepdim=True)  # (1,768)
    return float((img * text_feats).sum(-1).item())


def _merge_save(out_path, updates, captions, meta):
    blob = torch.load(out_path) if os.path.exists(out_path) else {}
    if blob.get("captions") not in (None, captions):
        raise SystemExit("[merge] existing renders_gc.pt has different captions; delete it to rebuild.")
    blob["captions"] = captions
    blob.update(meta)
    blob.update(updates)
    torch.save(blob, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ours", "gc", "both"], default="both")
    ap.add_argument("--captions", default=str(_REPO / "data/baseline_captions_500.json"))
    ap.add_argument("--ckpt", default=str(_REPO / "output/jit_final_sinkhorn_render_bs512_69k_20260529_071914/0099195.pt"))
    ap.add_argument("--manifest", default=str(_REPO / "output/gaussiancube_compare/objaverse_v1.0_seed0/manifest.json"))
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_views", type=int, default=4)
    ap.add_argument("--render_size", type=int, default=224)
    ap.add_argument("--axis", default="identity", help="GaussianCube canonical frame (calibrated)")
    ap.add_argument("--target_radius", type=float, default=0.6)
    ap.add_argument("--pct", type=float, default=97.0)
    ap.add_argument("--sub_batch", type=int, default=16)
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
    out_pt = os.path.join(args.out_dir, "renders_gc.pt")
    captions = _load_captions(args.captions)
    P = len(captions)
    H = W = args.render_size
    meta = {"cfg": args.cfg, "seed": args.seed, "render_size": args.render_size,
            "axis": args.axis, "target_radius": args.target_radius, "ours_ckpt": os.path.abspath(args.ckpt),
            "ours_label": f"Ours (130M, cfg{args.cfg:g})", "gc_label": "GaussianCube objaverse_v1.0"}

    clip, tok, cmean, cstd = load_clip(device)
    text_feats = clip_text_features(clip, tok, captions, device)  # (P,768)

    cam_indices = None
    if os.path.exists(out_pt):
        cam_indices = torch.load(out_pt).get("cam_indices")

    # ---------------- OURS: build flagship, sample @cfg/seed, render + score ----------------
    if args.stage in ("ours", "both"):
        t0 = time.time()
        b = build(args.ckpt, device, render_size=args.render_size, prompts_file=args.captions)
        b["amp_dtype"] = torch.bfloat16
        ncam = int(b["train_cameras"]["viewmats"].shape[0])
        cam_indices = sorted({int(round(x)) for x in np.linspace(0, ncam - 1, args.n_views)})
        sampler = b["cfg"]["val_sampler"]; steps = int(b["cfg"]["val_sampling_steps"])
        print(f"[ours] build {time.time()-t0:.0f}s | cfg={args.cfg} seed={args.seed} "
              f"sampler={sampler}:{steps} cams={cam_indices}")
        ours = torch.empty((P, len(cam_indices), 3, H, W), dtype=torch.uint8)
        ours_a = torch.empty((P, len(cam_indices), 1, H, W), dtype=torch.uint8)
        ours_score = [0.0] * P
        for s in range(0, P, args.sub_batch):
            idxs = list(range(s, min(P, s + args.sub_batch)))
            rgb, al = gen_render(b, idxs, args.seed, args.cfg, steps, sampler, cam_indices, device, return_alpha=True)
            rgb = rgb.float().clamp(0, 1)
            for j, gi in enumerate(idxs):
                ours_score[gi] = _score(clip, cmean, cstd, text_feats[gi:gi+1], rgb[j], device)
            ours[idxs] = (rgb * 255).round().to(torch.uint8).cpu()
            ours_a[idxs] = (al.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu()
            print(f"  ours {min(P, s+args.sub_batch)}/{P}  mean_so_far={np.mean(ours_score[:min(P,s+args.sub_batch)]):.4f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
        del b; torch.cuda.empty_cache()
        _merge_save(out_pt, {"ours_img": ours, "ours_alpha": ours_a, "ours_score": ours_score,
                             "cam_indices": cam_indices}, captions, meta)
        # ours scores JSON in eval_clip_alignment format (compare script consumes results[cfg].per_prompt)
        json.dump({"results": {f"cfg{args.cfg:g}": {"cfg_scale": args.cfg,
                   "per_prompt": [{"prompt": c, "mean": ours_score[i]} for i, c in enumerate(captions)]}},
                   "summary": {f"cfg{args.cfg:g}": {"mean": float(np.mean(ours_score)),
                   "sem": float(np.std(ours_score)/np.sqrt(P))}}},
                  open(os.path.join(args.out_dir, f"scores_ours_cfg{args.cfg:g}.json"), "w"), indent=2)
        print(f"[ours] mean CLIP = {np.mean(ours_score):.4f} +- {np.std(ours_score)/np.sqrt(P):.4f}")

    # ---------------- GaussianCube: render saved .ply through the same pipeline + score -------
    if args.stage in ("gc", "both"):
        man = json.load(open(args.manifest, encoding="utf-8"))
        man = {m["caption"]: m["ply"] for m in man}
        missing = [c for c in captions if c not in man or not os.path.exists(man[c])]
        if missing:
            raise SystemExit(f"[gc] manifest missing/not-rendered for {len(missing)} captions "
                             f"(gen not finished?). First: {missing[0]!r}")
        if cam_indices is None:
            ref0 = T._load_reference_cameras(str(_REPO / "artifacts/ref_camera.tar.gz"))
            ncam = len(T._prepare_train_cameras(ref0, args.render_size, device)["viewmats"])
            cam_indices = sorted({int(round(x)) for x in np.linspace(0, ncam - 1, args.n_views)})
        t1 = time.time()
        ref = T._load_reference_cameras(str(_REPO / "artifacts/ref_camera.tar.gz"))
        cams = T._prepare_train_cameras(ref, args.render_size, device)
        renderer = T._try_import_renderer()
        gc = torch.empty((P, len(cam_indices), 3, H, W), dtype=torch.uint8)
        gc_a = torch.empty((P, len(cam_indices), 1, H, W), dtype=torch.uint8)
        gc_score = [0.0] * P
        for i, cap in enumerate(captions):
            rgb, al = render_trellis_ply(man[cap], target_radius=args.target_radius, pct=args.pct,
                                         axis=args.axis, renderer=renderer, cams=cams,
                                         cam_indices=cam_indices, device=device, return_alpha=True)
            rgb = rgb[0].float().clamp(0, 1)  # (V,3,H,W)
            gc_score[i] = _score(clip, cmean, cstd, text_feats[i:i+1], rgb, device)
            gc[i] = (rgb * 255).round().to(torch.uint8).cpu()
            gc_a[i] = (al[0].float().clamp(0, 1) * 255).round().to(torch.uint8).cpu()
            if (i + 1) % 50 == 0:
                print(f"  gc {i+1}/{P}  mean_so_far={np.mean(gc_score[:i+1]):.4f}  ({time.time()-t1:.0f}s)", flush=True)
        _merge_save(out_pt, {"gc_img": gc, "gc_alpha": gc_a, "gc_score": gc_score,
                             "cam_indices": cam_indices}, captions, meta)
        json.dump({"source": "GaussianCube-objaverse-v1.0", "axis": args.axis,
                   "target_radius": args.target_radius, "n_objects": P,
                   "mean": float(np.mean(gc_score)), "std": float(np.std(gc_score)),
                   "sem": float(np.std(gc_score)/np.sqrt(P)),
                   "per_object": [{"idx": i, "caption": c, "score": gc_score[i]} for i, c in enumerate(captions)]},
                  open(os.path.join(args.out_dir, "scores_gc.json"), "w"), indent=2)
        print(f"[gc] mean CLIP = {np.mean(gc_score):.4f} +- {np.std(gc_score)/np.sqrt(P):.4f}")

    print(f"\n[done] updated {out_pt} (stage={args.stage})")


if __name__ == "__main__":
    main()
