"""VQAScore (clip-flant5-xl) for the render-loss before/after pair, from a saved render bundle.

Generalized copy of jit/vqascore_eval.py for a two-model bundle with keys {off,on}_{img,alpha};
writes to its own out_dir so it does NOT touch the GaussianCube comparison files / PNGs.

RUN UNDER .vqascore (transformers 4.36.1, t2v-metrics 1.2):
    CUDA_VISIBLE_DEVICES=1 python jit/vqascore_pair.py
"""
import argparse, json, math, os, sys, types
import numpy as np
from PIL import Image

# t2v-metrics 1.2 eager-imports ITMScore -> `import ImageReward` (unbuildable here, unused). Stub it.
sys.modules.setdefault("ImageReward", types.ModuleType("ImageReward"))


def composite_white(rgb_u8, alpha_u8):
    r = rgb_u8.float() / 255.0
    a = alpha_u8.float() / 255.0
    out = (r + (1.0 - a) * 1.0).clamp(0, 1)
    return Image.fromarray((out * 255).round().byte().permute(1, 2, 0).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="output/report/eval_logs/renders_render_loss.pt")
    ap.add_argument("--model", default="clip-flant5-xl")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_dir", default="output/report/eval_logs/vqa_images")
    ap.add_argument("--out_dir", default="output/report/eval_logs")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    d = torch.load(args.bundle, weights_only=False)
    caps_all = d["captions"]
    P = len(caps_all) if args.limit is None else min(len(caps_all), args.limit)
    caps = caps_all[:P]
    V = len(d["cam_indices"])
    views = list(range(V))
    models = {"off": ("off_img", "off_alpha"), "on": ("on_img", "on_alpha")}

    # 1) dump white-bg PNGs (fresh dir per model)
    paths = {m: [[None] * V for _ in range(P)] for m in models}
    for m, (ik, ak) in models.items():
        mdir = os.path.join(args.img_dir, m); os.makedirs(mdir, exist_ok=True)
        img, al = d[ik], d[ak]
        for i in range(P):
            for vi, v in enumerate(views):
                p = os.path.join(mdir, f"{i:04d}_v{v}.png")
                paths[m][i][vi] = p
                if not os.path.exists(p):
                    composite_white(img[i, v], al[i, v]).save(p)
        print(f"[png] {m}: {P}x{V} ready in {mdir}", flush=True)

    # 2) VQAScore
    import t2v_metrics
    print(f"[vqa] loading {args.model} ...", flush=True)
    scorer = t2v_metrics.VQAScore(model=args.model)

    def vqa_per_object(image_paths_2d, texts):
        ds, idxmap = [], []
        for i in range(len(texts)):
            for vi in range(len(image_paths_2d[i])):
                ds.append({"images": [image_paths_2d[i][vi]], "texts": [texts[i]]})
                idxmap.append((i, vi))
        out = scorer.batch_forward(dataset=ds, batch_size=args.batch_size)
        out = np.asarray(out.reshape(len(ds)).float().cpu())
        per = np.zeros((len(texts), len(image_paths_2d[0])))
        for k, (i, vi) in enumerate(idxmap):
            per[i, vi] = out[k]
        return per.mean(axis=1)

    scores = {}
    for m in models:
        print(f"[vqa] scoring {m} ({P}x{V} pairs) ...", flush=True)
        scores[m] = vqa_per_object(paths[m], caps)
        print(f"[vqa] {m} mean VQAScore = {scores[m].mean():.4f} "
              f"+- {scores[m].std(ddof=1)/math.sqrt(P):.4f}", flush=True)

    # 3) write per-model JSON + paired summary (diff = on - off, >0 => render helps)
    os.makedirs(args.out_dir, exist_ok=True)
    for m, src in [("off", "Without render loss (step 22k) VQAScore"),
                   ("on", "With render loss (step 69k) VQAScore")]:
        s = scores[m]
        json.dump({"source": src, "model": args.model, "metric": "VQAScore", "n_objects": P,
                   "views": d["cam_indices"], "mean": float(s.mean()), "std": float(s.std()),
                   "sem": float(s.std(ddof=1) / math.sqrt(P)),
                   "per_object": [{"idx": i, "caption": caps[i], "score": float(s[i])} for i in range(P)]},
                  open(os.path.join(args.out_dir, f"vqa_{m}.json"), "w"), indent=2)

    on, off = scores["on"], scores["off"]
    diff = on - off
    sem = lambda x: float(np.std(x, ddof=1) / math.sqrt(P))
    summary = {"n": P, "metric": "VQAScore (clip-flant5-xl; P(Yes)) mean over %d views" % V,
               "on_mean": float(on.mean()), "on_sem": sem(on),
               "off_mean": float(off.mean()), "off_sem": sem(off),
               "mean_paired_diff_on_minus_off": float(diff.mean()), "sem_paired_diff": sem(diff),
               "paired_tstat": float(diff.mean() / sem(diff)) if sem(diff) > 0 else float("nan"),
               "on_win_rate": float((diff > 0).mean()), "off_win_rate": float((diff < 0).mean()),
               "ties": int((diff == 0).sum())}
    json.dump(summary, open(os.path.join(args.out_dir, "summary_vqa_render_loss.json"), "w"), indent=2)

    print(f"\n==== VQAScore ({args.model}, mean over {V} views) — {P} captions ====")
    print(f"Without render loss (22k)  {summary['off_mean']:.4f} +- {summary['off_sem']:.4f}")
    print(f"With render loss    (69k)  {summary['on_mean']:.4f} +- {summary['on_sem']:.4f}")
    print(f"paired diff (on - off): {summary['mean_paired_diff_on_minus_off']:+.4f} "
          f"+- {summary['sem_paired_diff']:.4f}  (t={summary['paired_tstat']:+.2f})")
    print(f"win-rate: on {summary['on_win_rate']*100:.1f}% | off {summary['off_win_rate']*100:.1f}%")
    print("saved: output/report/eval_logs/vqa_{off,on}.json + summary_vqa_render_loss.json")


if __name__ == "__main__":
    main()
