#!/usr/bin/env python
"""Stage 4: paired GaussianCube-vs-ours comparison on the shared caption set.

Joins the Stage-2 GaussianCube scores (clip_gaussiancube.json: per_object[i]={idx,caption,score})
with the ours scores (clip_alignment.json: results[cfgTag].per_prompt[i]={prompt,mean}) by caption
index (both files are in baseline_captions order), and reports mean+-sem for each, the paired
difference, the per-caption win-rate, and a paired t-stat. Pure-CPU (numpy+json); no GPU/render.

    python jit/compare_gaussiancube_ours.py \
        --gc_json   output/gaussiancube_compare/objaverse_v1.0_seed0/clip_gaussiancube.json \
        --ours_json output/gaussiancube_compare/ours_cfg4/clip_alignment.json \
        --cfg 4 --out output/gaussiancube_compare/summary_gc_vs_ours.json
"""
import argparse
import json
import math

import numpy as np


def load_gc(path):
    d = json.load(open(path, encoding="utf-8"))
    per = sorted(d["per_object"], key=lambda r: r["idx"])
    caps = [r["caption"] for r in per]
    scr = np.array([r["score"] for r in per], dtype=np.float64)
    return caps, scr, d.get("source", "GaussianCube")


def load_ours(path, cfg):
    d = json.load(open(path, encoding="utf-8"))
    tag = f"cfg{cfg:g}"
    if tag not in d["results"]:
        avail = list(d["results"].keys())
        raise SystemExit(f"cfg tag {tag} not in ours results (have {avail})")
    per = d["results"][tag]["per_prompt"]              # already in prompt/caption order
    caps = [r["prompt"] for r in per]
    scr = np.array([r["mean"] for r in per], dtype=np.float64)
    return caps, scr


def fmt(m, s, n):
    return f"{m:.4f} +- {s/math.sqrt(n):.4f} (sem)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc_json", default="output/gaussiancube_compare/objaverse_v1.0_seed0/clip_gaussiancube.json")
    ap.add_argument("--ours_json", default="output/gaussiancube_compare/ours_cfg4/clip_alignment.json")
    ap.add_argument("--cfg", type=float, default=4)
    ap.add_argument("--out", default="output/gaussiancube_compare/summary_gc_vs_ours.json")
    args = ap.parse_args()

    gc_caps, gc, gc_src = load_gc(args.gc_json)
    ours_caps, ours = load_ours(args.ours_json, args.cfg)

    n = min(len(gc), len(ours))
    if len(gc) != len(ours):
        print(f"[warn] length mismatch GC={len(gc)} ours={len(ours)}; truncating to {n}")
    gc_caps, gc = gc_caps[:n], gc[:n]
    ours_caps, ours = ours_caps[:n], ours[:n]
    mismatch = sum(1 for a, b in zip(gc_caps, ours_caps) if a.strip() != b.strip())
    if mismatch:
        print(f"[warn] {mismatch}/{n} captions differ between files — are both on the same set/order?")

    diff = ours - gc                                   # >0 => ours better
    ours_wins = int((diff > 0).sum())
    gc_wins = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    mean_diff = float(diff.mean())
    sem_diff = float(diff.std(ddof=1) / math.sqrt(n))
    tstat = mean_diff / sem_diff if sem_diff > 0 else float("nan")

    # complexity buckets: split by caption length (chars) at the median -> simple vs detailed
    lengths = np.array([len(c) for c in gc_caps])
    med = float(np.median(lengths))
    buckets = {}
    for name, mask in [("simple", lengths <= med), ("detailed", lengths > med)]:
        if mask.sum() == 0:
            continue
        buckets[name] = {
            "n": int(mask.sum()),
            "gc_mean": float(gc[mask].mean()),
            "ours_mean": float(ours[mask].mean()),
            "ours_win_rate": float((diff[mask] > 0).mean()),
        }

    out = {
        "n": n,
        "gc_source": gc_src,
        "ours_cfg": args.cfg,
        "gc_mean": float(gc.mean()), "gc_sem": float(gc.std(ddof=1) / math.sqrt(n)),
        "ours_mean": float(ours.mean()), "ours_sem": float(ours.std(ddof=1) / math.sqrt(n)),
        "mean_paired_diff_ours_minus_gc": mean_diff, "sem_paired_diff": sem_diff,
        "paired_tstat": tstat,
        "ours_win_rate": ours_wins / n, "gc_win_rate": gc_wins / n, "ties": ties,
        "buckets_by_caption_length": buckets,
        "caption_length_median_chars": med,
    }
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\n==== {gc_src}  vs  ours (cfg{args.cfg:g})  on {n} captions ====")
    print(f"{gc_src:<28} {fmt(gc.mean(), gc.std(ddof=1), n)}")
    print(f"{'ours cfg'+format(args.cfg,'g'):<28} {fmt(ours.mean(), ours.std(ddof=1), n)}")
    print(f"\npaired diff (ours - {gc_src}): {mean_diff:+.4f} +- {sem_diff:.4f}  (t={tstat:+.2f})")
    print(f"win-rate: ours {ours_wins}/{n} ({ours_wins/n*100:.1f}%) | "
          f"{gc_src} {gc_wins}/{n} ({gc_wins/n*100:.1f}%) | ties {ties}")
    print("\nby caption length:")
    for name, b in buckets.items():
        print(f"  {name:<9} n={b['n']:<4} gc={b['gc_mean']:.4f}  ours={b['ours_mean']:.4f}  "
              f"ours_win={b['ours_win_rate']*100:.0f}%")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
