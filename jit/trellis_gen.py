"""Stage 1 of the TRELLIS-vs-ours comparison: generate 3D Gaussians for a caption set.

RUN UNDER THE TRELLIS VENV, NOT .3dgen:
    python jit/trellis_gen.py [--limit N] [--max_gpu_gb G]

For each caption it runs TRELLIS-text-base (text->3D) decoding ONLY the Gaussian format
(skips mesh/RF decoders), and saves a standard 3DGS .ply per caption (raw logit opacity,
log scale, SH-DC color — the layout our renderer ingests). Resumable: existing .ply are
skipped. A manifest.json (idx, caption, ply) is written for Stage 2.

TRELLIS sampler settings are left at the model's own recommended defaults (25-step
FlowEulerGuidanceInterval, cfg 7.5) — i.e. TRELLIS at its best, vs ours at ours.
"""
import argparse
import json
import os
import sys
import time

REPO = "."
TRELLIS_ROOT = "../TRELLIS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default=f"{REPO}/data/baseline_captions_150_seed0.json")
    ap.add_argument("--out_dir", default=f"{REPO}/output/trellis_compare/text_base_seed0")
    ap.add_argument("--model", default="microsoft/TRELLIS-text-base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only first N captions (smoke)")
    ap.add_argument("--max_gpu_gb", type=float, default=None,
                    help="hard per-process GPU cap (GiB) so a co-resident smoke can't OOM training")
    args = ap.parse_args()

    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    sys.path.insert(0, TRELLIS_ROOT)

    import torch
    if args.max_gpu_gb:
        idx = torch.cuda.current_device()
        tot = torch.cuda.get_device_properties(idx).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(args.max_gpu_gb) / tot), idx)
        print(f"[mem] cap {args.max_gpu_gb:.0f}GiB of {tot:.0f}GiB total", flush=True)

    from trellis.pipelines import TrellisTextTo3DPipeline

    caps = json.load(open(args.captions, encoding="utf-8"))
    if args.limit:
        caps = caps[:int(args.limit)]
    os.makedirs(args.out_dir, exist_ok=True)

    t_load = time.time()
    pipe = TrellisTextTo3DPipeline.from_pretrained(args.model)
    pipe.cuda()
    print(f"[load] pipeline ready in {time.time()-t_load:.0f}s | {len(caps)} captions -> {args.out_dir}", flush=True)

    manifest = []
    t0 = time.time()
    for i, cap in enumerate(caps):
        ply = os.path.join(args.out_dir, f"{i:04d}.ply")
        rec = {"idx": i, "caption": cap, "ply": ply}
        if os.path.exists(ply):
            manifest.append(rec)
            continue
        ts = time.time()
        out = pipe.run(cap, seed=args.seed, formats=["gaussian"])
        out["gaussian"][0].save_ply(ply)
        manifest.append(rec)
        json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=2, ensure_ascii=False)
        print(f"[{i+1}/{len(caps)}] {time.time()-ts:.1f}s  {cap[:48]!r}", flush=True)

    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[done] {len(manifest)} objects in {time.time()-t0:.0f}s -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
