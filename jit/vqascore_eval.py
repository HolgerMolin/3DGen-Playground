"""VQAScore (Lin et al., CVPR 2024) for the GaussianCube-vs-ours comparison, computed straight from
the SAVED renders in output/report/renders_gc.pt — no re-rendering, no re-sampling.

VQAScore = P("Yes" | image, 'Does this figure show "{caption}"? Please answer yes or no.') from a
CLIP-FlanT5 VLM. Its yes/no head is FlanT5 (a generative LM), independent of the CLIP text space the
generators are conditioned on — an independent check that the CLIP-alignment metric isn't "teaching
to the test".

RUN UNDER .vqascore (torch 2.1.2+cu121, transformers 4.36.1):
    CUDA_VISIBLE_DEVICES=1 python jit/vqascore_eval.py [--limit 8 --control]

Pipeline: composite each saved render (rgb+alpha) onto WHITE bg -> PNG; VQAScore each (image,caption)
pair (mean over the 4 views) -> per-object score in [0,1]; write per-model JSON + paired summary.
"""
import argparse, json, math, os, sys, types
import numpy as np
from PIL import Image

# t2v-metrics 1.2's top __init__ pulls in ITMScore -> `import ImageReward` (a metric we never use,
# whose package won't build in this env). Stub that single module so the import succeeds; we only
# ever construct VQAScore(clip-flant5-xl). The vqascore backends (incl. llava) are vendored in 1.2.
sys.modules.setdefault("ImageReward", types.ModuleType("ImageReward"))


def composite_white(rgb_u8, alpha_u8):
    """(3,H,W)+(1,H,W) uint8 tensors -> PIL RGB on white bg (object over white)."""
    r = rgb_u8.float() / 255.0
    a = alpha_u8.float() / 255.0
    out = (r + (1.0 - a) * 1.0).clamp(0, 1)
    return Image.fromarray((out * 255).round().byte().permute(1, 2, 0).numpy())


def paired_summary(caps, a, b, a_name, b_name):
    """Paired stats reusing the logic of jit/compare_gaussiancube_ours.py (a=ours, b=gc)."""
    n = len(caps)
    diff = a - b                                          # >0 => ours better
    sem = lambda x: float(np.std(x, ddof=1) / math.sqrt(n))
    lengths = np.array([len(c) for c in caps]); med = float(np.median(lengths))
    buckets = {}
    for name, mask in [("simple", lengths <= med), ("detailed", lengths > med)]:
        if mask.sum():
            buckets[name] = {"n": int(mask.sum()), f"{a_name}_mean": float(a[mask].mean()),
                             f"{b_name}_mean": float(b[mask].mean()),
                             f"{a_name}_win_rate": float((diff[mask] > 0).mean())}
    return {
        "n": n, "metric": "VQAScore (clip-flant5; P(Yes)) mean over 4 views",
        f"{a_name}_mean": float(a.mean()), f"{a_name}_sem": sem(a),
        f"{b_name}_mean": float(b.mean()), f"{b_name}_sem": sem(b),
        f"mean_paired_diff_{a_name}_minus_{b_name}": float(diff.mean()), "sem_paired_diff": sem(diff),
        "paired_tstat": float(diff.mean() / sem(diff)) if sem(diff) > 0 else float("nan"),
        f"{a_name}_win_rate": float((diff > 0).mean()), f"{b_name}_win_rate": float((diff < 0).mean()),
        "ties": int((diff == 0).sum()), "buckets_by_caption_length": buckets,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="output/report/renders_gc.pt")
    ap.add_argument("--model", default="clip-flant5-xl")
    ap.add_argument("--views", default="all", help="'all' (mean over the 4 saved views) or an int view index")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_dir", default="output/report/vqa_images")
    ap.add_argument("--out_dir", default="output/report")
    ap.add_argument("--limit", type=int, default=None, help="smoke: only first N objects")
    ap.add_argument("--control", action="store_true",
                    help="also score ours vs a SHUFFLED (wrong) caption — discrimination sanity check")
    args = ap.parse_args()

    import torch
    d = torch.load(args.bundle, weights_only=False)
    caps_all = d["captions"]
    P = len(caps_all) if args.limit is None else min(len(caps_all), args.limit)
    caps = caps_all[:P]
    V = len(d["cam_indices"])
    views = list(range(V)) if args.views == "all" else [int(args.views)]
    models = {"ours": ("ours_img", "ours_alpha"), "gc": ("gc_img", "gc_alpha")}

    # 1) dump PNGs (white bg), skip-existing
    paths = {m: [[None] * len(views) for _ in range(P)] for m in models}
    for m, (ik, ak) in models.items():
        mdir = os.path.join(args.img_dir, m); os.makedirs(mdir, exist_ok=True)
        img, al = d[ik], d[ak]
        for i in range(P):
            for vi, v in enumerate(views):
                p = os.path.join(mdir, f"{i:04d}_v{v}.png")
                paths[m][i][vi] = p
                if not os.path.exists(p):
                    composite_white(img[i, v], al[i, v]).save(p)
        print(f"[png] {m}: {P}x{len(views)} ready in {mdir}", flush=True)

    # 2) VQAScore
    import t2v_metrics
    print(f"[vqa] loading {args.model} ...", flush=True)
    scorer = t2v_metrics.VQAScore(model=args.model)

    def vqa_per_object(image_paths_2d, texts):
        """image_paths_2d: list[P] of list[V] png paths; texts: list[P]. Returns (P,) mean-over-view VQAScore."""
        ds, idxmap = [], []
        for i in range(len(texts)):
            for vi in range(len(image_paths_2d[i])):
                ds.append({"images": [image_paths_2d[i][vi]], "texts": [texts[i]]})
                idxmap.append((i, vi))
        out = scorer.batch_forward(dataset=ds, batch_size=args.batch_size)   # (n_sample,1,1)
        out = np.asarray(out.reshape(len(ds)).float().cpu())
        per = np.zeros((len(texts), len(image_paths_2d[0])))
        for k, (i, vi) in enumerate(idxmap):
            per[i, vi] = out[k]
        return per.mean(axis=1)

    scores = {}
    for m in models:
        print(f"[vqa] scoring {m} ({P}x{len(views)} pairs) ...", flush=True)
        scores[m] = vqa_per_object(paths[m], caps)
        print(f"[vqa] {m} mean VQAScore = {scores[m].mean():.4f} +- {scores[m].std(ddof=1)/math.sqrt(P):.4f}", flush=True)

    # optional discrimination control: ours vs a deterministically-shuffled (wrong) caption
    if args.control:
        wrong = [caps[(i + 1) % P] for i in range(P)]
        ctrl = vqa_per_object(paths["ours"], wrong)
        print(f"[control] ours vs WRONG caption: {ctrl.mean():.4f}  (should be << ours {scores['ours'].mean():.4f})", flush=True)

    # 3) write per-model JSON (per_object) + paired summary
    os.makedirs(args.out_dir, exist_ok=True)
    for m, src in [("ours", "Ours (130M, cfg4) VQAScore"), ("gc", "GaussianCube-objaverse-v1.0 VQAScore")]:
        s = scores[m]
        json.dump({"source": src, "model": args.model, "metric": "VQAScore", "n_objects": P,
                   "views": views, "mean": float(s.mean()), "std": float(s.std()),
                   "sem": float(s.std(ddof=1) / math.sqrt(P)),
                   "per_object": [{"idx": i, "caption": caps[i], "score": float(s[i])} for i in range(P)]},
                  open(os.path.join(args.out_dir, f"scores_vqa_{m}.json"), "w"), indent=2)

    summary = paired_summary(caps, scores["ours"], scores["gc"], "ours", "gc")
    json.dump(summary, open(os.path.join("output/gaussiancube_compare", "summary_vqa_gc_vs_ours.json"), "w"), indent=2)

    print(f"\n==== VQAScore ({args.model}, mean over {len(views)} views) — {P} captions ====")
    print(f"GaussianCube  {summary['gc_mean']:.4f} +- {summary['gc_sem']:.4f}")
    print(f"ours          {summary['ours_mean']:.4f} +- {summary['ours_sem']:.4f}")
    print(f"paired diff (ours - gc): {summary['mean_paired_diff_ours_minus_gc']:+.4f} "
          f"+- {summary['sem_paired_diff']:.4f}  (t={summary['paired_tstat']:+.2f})")
    print(f"win-rate: ours {summary['ours_win_rate']*100:.1f}% | gc {summary['gc_win_rate']*100:.1f}%")
    print(f"saved: output/report/scores_vqa_{{ours,gc}}.json + output/gaussiancube_compare/summary_vqa_gc_vs_ours.json")


if __name__ == "__main__":
    main()
