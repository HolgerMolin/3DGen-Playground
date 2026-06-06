"""
Hierarchical (semantic -> visual) dual-coherent taxonomy for GaussianVerse.

Fixes the failure of the CLIP-image-only taxonomy (visually tight, semantically
loose) by building structure from BOTH modalities:

  Stage 1  semantic super-categories  : MiniBatchKMeans on CLIP **caption** embeddings
                                        -> ~M clean semantic groups.
  Stage 2  visual refinement          : split each super-cat by **DINOv2** appearance
                                        into ~uniform leaves (tight on both axes).
  Stage 3  misc bucket                : route the worst-fitting tail (low joint-cosine
                                        to its leaf) into one capped 'misc' class.
  Stage 4  balance                    : merge-small / adjust-to-k in the JOINT
                                        (DINOv2 + caption) space -> ~k_target near-uniform.
  Stage 5  name + emit                : majority caption head-noun (+ nearest ImageNet
                                        cross-ref) -> object_to_class.json, class_names.json,
                                        class_hierarchy.json, cluster_summary.txt, meta.json.

Reuses the generic cluster algebra from data/build_imagenet_taxonomy.py.

Usage
-----
python data/build_hierarchical_taxonomy.py \
    --image-embeddings object_classification/image_embeddings.npz \
    --caption-embeddings object_classification/caption_clip_text_embeddings.npz \
    --output-dir object_labels/hier_uniform_k1000
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.build_imagenet_taxonomy import (  # noqa: E402  (reuse generic helpers)
    _unit, _centroid, _split_cluster, _merge_small, _adjust_to_k, _stats,
    build_text_prototypes, load_classnames, _env_path,
)


# ---------------------------------------------------------------------------
# Head-noun naming (captions are ~92% "A <adjs> NOUN ...")
# ---------------------------------------------------------------------------
_ARTICLES = {"a", "an", "the"}
_ADJ_STOP = set("""small large big tall short long thin wide narrow round square rectangular
detailed realistic unique simple stylized colorful color colored dark light bright old new modern
vintage rustic white black red green blue yellow orange purple pink brown gray grey gold golden
silver metallic wooden wood metal plastic stone glass ceramic clay marble shiny smooth rough curved
flat pixelated low poly cartoon beautiful intricate elegant sleek futuristic traditional ornate
decorative miniature cute realistic-looking textured angular sculpted 3d two three four multi""".split())
_PREPS = {"of", "with", "on", "in", "that", "featuring", "wearing", "holding", "for", "atop",
          "standing", "sitting", "resembling", "depicting", "containing", "made", "set", "displayed",
          "is", "has", "having", "to", "at", "near", "inside", "above", "below", "from", "and", "or"}
# Generic filler heads ("a 3D model of a X" -> we want X, not "model").
_FILLER = {"model", "representation", "rendering", "render", "depiction", "illustration", "version",
           "object", "piece", "scene", "image", "figure", "style", "design", "shape", "structure",
           "thing", "3d", "2d", "cg", "cgi", "looking"}
_SKIP = _ADJ_STOP | _FILLER


def head_noun(caption: str) -> str:
    """Best-effort category noun: drop article/adjectives/fillers/prepositions and
    take the first 1-2 real noun tokens. Per-cluster majority vote denoises the
    residual errors, so this only needs to be right on average."""
    toks = [t for t in re.findall(r"[a-z0-9]+", caption.lower()) if len(t) > 1 and not t.isdigit()]
    i = 1 if toks and toks[0] in _ARTICLES else 0
    out, steps = [], 0
    while i < len(toks) and steps < 8:
        t = toks[i]; steps += 1; i += 1
        if t in _ARTICLES or t in _SKIP or t in _PREPS:
            if out:
                break            # already have a noun → stop at the next filler/prep
            continue             # still searching → skip leading fillers/preps/articles
        out.append(t)
        if len(out) >= 2:
            break
    return " ".join(out)


def cluster_name(members, keys, caps):
    c = Counter()
    for r in members:
        n = head_noun(caps.get(keys[r], ""))
        if n:
            c[n] += 1
    return c.most_common(1)[0][0] if c else "object"


# ---------------------------------------------------------------------------
# IO / alignment
# ---------------------------------------------------------------------------

def load_aligned(img_path, cap_path):
    di = np.load(img_path, allow_pickle=True)
    dc = np.load(cap_path, allow_pickle=True)
    irow = {str(k): i for i, k in enumerate(di["keys"].tolist())}
    crow = {str(k): i for i, k in enumerate(dc["keys"].tolist())}
    keys = sorted(set(irow) & set(crow))
    IMG = _unit(di["embeddings"][[irow[k] for k in keys]].astype(np.float32))
    TXT = _unit(dc["embeddings"][[crow[k] for k in keys]].astype(np.float32))
    return keys, IMG, TXT


def rebalance(E, clusters, k_target, target, floor, cap, max_iters, seed):
    for it in range(max_iters):
        changed = False
        nc = []
        for c in clusters:
            if len(c["members"]) > cap:
                parts = _split_cluster(E, c, max(2, round(len(c["members"]) / target)), seed + it)
                if len(parts) > 1:
                    nc += parts; changed = True
                else:
                    nc.append(c)
            else:
                nc.append(c)
        clusters = nc
        clusters, merged = _merge_small(E, clusters, floor)
        changed = changed or merged
        s = _stats([len(c["members"]) for c in clusters])
        print(f"    iter {it:>2}: k={s['k']:>5} min={s['min']:>4} max={s['max']:>5} "
              f"mean={s['mean']:.1f} cv={s['cv']:.3f}")
        if not changed:
            break
    return _adjust_to_k(E, clusters, k_target, seed)


def coherence(E, members):
    return float((E[members] @ _centroid(E, members)).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv(_REPO_ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-embeddings", default=str(_REPO_ROOT / "object_classification/image_embeddings.npz"))
    p.add_argument("--caption-embeddings", default=str(_REPO_ROOT / "object_classification/caption_clip_text_embeddings.npz"))
    p.add_argument("--captions", default=_env_path("CAPTIONS_PATH"))
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "object_labels/hier_uniform_k1000"))
    p.add_argument("--k-target", type=int, default=1000, help="Total classes incl. misc.")
    p.add_argument("--super-cats", type=int, default=250, help="Stage-1 semantic super-categories (M).")
    p.add_argument("--misc-frac", type=float, default=0.05, help="Fraction routed to 'misc' (worst joint-fit).")
    p.add_argument("--w-visual", type=float, default=1.0)
    p.add_argument("--w-text", type=float, default=1.0)
    p.add_argument("--floor-frac", type=float, default=0.5)
    p.add_argument("--cap-frac", type=float, default=1.5)
    p.add_argument("--max-iters", type=int, default=25)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    caps = json.loads(Path(args.captions).read_text()) if args.captions else {}

    print(f"Loading + aligning embeddings …")
    keys, IMG, TXT = load_aligned(args.image_embeddings, args.caption_embeddings)
    N = len(keys)
    k_real = args.k_target - 1                       # reserve one id for misc
    target = N / args.k_target
    print(f"  N={N:,}  visual={IMG.shape}  text={TXT.shape}  target≈{target:.0f}  k_real={k_real}")

    # --- Stage 1: semantic super-categories (caption space) -------------
    M = args.super_cats
    print(f"\nStage 1: {M} semantic super-categories (MiniBatchKMeans on captions) …")
    km = MiniBatchKMeans(n_clusters=M, random_state=args.seed, n_init=3, batch_size=8192, max_iter=200)
    super_lab = km.fit_predict(TXT)
    ss = _stats(np.bincount(super_lab, minlength=M))
    print(f"  super-cat sizes: {ss}")

    # --- Stage 2: visual split within each super-cat --------------------
    print(f"\nStage 2: visual (DINOv2) split within each super-cat …")
    leaves = []
    for s in range(M):
        mem = np.where(super_lab == s)[0]
        if len(mem) == 0:
            continue
        k = max(1, round(len(mem) / target))
        cdict = {"members": mem, "name": str(s), "cent": _centroid(IMG, mem)}
        leaves += _split_cluster(IMG, cdict, k, args.seed) if k >= 2 else [cdict]
    print(f"  {len(leaves)} leaves after visual split. {_stats([len(c['members']) for c in leaves])}")

    # --- joint space ----------------------------------------------------
    EJ = _unit(np.concatenate([args.w_visual * IMG, args.w_text * TXT], axis=1))

    # --- Stage 3: misc bucket (worst joint-fit tail) --------------------
    fit = np.empty(N, dtype=np.float32)
    for c in leaves:
        cent = _centroid(EJ, c["members"])
        fit[c["members"]] = EJ[c["members"]] @ cent
    n_misc = int(round(args.misc_frac * N))
    misc_idx = np.argsort(fit)[:n_misc] if n_misc > 0 else np.array([], dtype=int)
    is_misc = np.zeros(N, dtype=bool); is_misc[misc_idx] = True
    print(f"\nStage 3: misc bucket = {is_misc.sum():,} objects ({is_misc.mean()*100:.1f}%), "
          f"fit cutoff={fit[misc_idx].max() if n_misc else float('nan'):.3f}")

    # rebuild non-misc leaves in joint space
    jleaves = []
    for c in leaves:
        mem = c["members"][~is_misc[c["members"]]]
        if len(mem) > 0:
            jleaves.append({"members": mem, "name": c["name"], "cent": _centroid(EJ, mem)})

    # --- Stage 4: balance to k_real in joint space ----------------------
    floor = max(1, round(args.floor_frac * target))
    cap = max(floor + 1, round(args.cap_frac * target))
    print(f"\nStage 4: balance to k={k_real} in joint space (floor={floor} cap={cap}) …")
    t0 = time.time()
    final = rebalance(EJ, jleaves, k_real, target, floor, cap, args.max_iters, args.seed)
    print(f"  balanced: {_stats([len(c['members']) for c in final])}  ({time.time()-t0:.0f}s)")

    # --- assign ids (size desc; misc last) ------------------------------
    final.sort(key=lambda c: -len(c["members"]))
    assign = np.full(N, -1, dtype=np.int64)
    for cid, c in enumerate(final):
        assign[c["members"]] = cid
    misc_id = len(final)
    assign[is_misc] = misc_id
    assert (assign >= 0).all(), "unassigned objects remain"

    # --- Stage 5: name + cross-ref + coherence --------------------------
    print(f"\nStage 5: naming + coherence …")
    inames = load_classnames(None)
    P = build_text_prototypes(inames, device)         # (C,768) for nearest-ImageNet cross-ref
    used_names = {}
    rows = []
    for cid, c in enumerate(final):
        mem = c["members"]
        base = cluster_name(mem, keys, caps)
        nm = base if base not in used_names else f"{base}#{used_names[base]+1}"
        used_names[base] = used_names.get(base, 0) + 1
        c["display"] = nm
        c["super"] = int(Counter(super_lab[mem]).most_common(1)[0][0])
        txt_cent = _unit(TXT[mem].mean(0))
        c["near_in"] = inames[int((txt_cent @ P.T).argmax())]
        c["vcoh"] = coherence(IMG, mem)
        c["tcoh"] = coherence(TXT, mem)
        rows.append((cid, len(mem), nm, c["super"], c["near_in"], c["vcoh"], c["tcoh"]))

    # --- write artifacts ------------------------------------------------
    obj_to_class = {keys[r]: int(assign[r]) for r in range(N)}
    (out_dir / "object_to_class.json").write_text(json.dumps(obj_to_class))
    class_names = {str(cid): final[cid]["display"] for cid in range(len(final))}
    class_names[str(misc_id)] = "misc"
    (out_dir / "class_names.json").write_text(json.dumps(class_names, indent=0))
    hierarchy = {str(cid): {"name": final[cid]["display"], "super_cat": final[cid]["super"],
                            "nearest_imagenet": final[cid]["near_in"], "is_misc": False}
                 for cid in range(len(final))}
    hierarchy[str(misc_id)] = {"name": "misc", "super_cat": -1, "nearest_imagenet": "", "is_misc": True}
    (out_dir / "class_hierarchy.json").write_text(json.dumps(hierarchy, indent=0))

    sizes = np.bincount(assign, minlength=misc_id + 1)
    real_sizes = sizes[:misc_id]
    lines = [
        f"Hierarchical dual-coherent taxonomy: {len(final)} real classes + misc, N={N:,}",
        f"real-class sizes: {_stats(real_sizes)}",
        f"misc size: {int(sizes[misc_id]):,} ({sizes[misc_id]/N*100:.1f}%)",
        f"coherence (mean over real classes): visual={np.mean([c['vcoh'] for c in final]):.3f} "
        f"caption={np.mean([c['tcoh'] for c in final]):.3f}",
        "",
        f"  {'cid':>4} {'size':>5} {'vcoh':>5} {'tcoh':>5}  {'name':<24} {'~imagenet':<18} examples",
        "  " + "-" * 116,
    ]
    order = sorted(range(len(final)), key=lambda i: -len(final[i]["members"]))
    for cid in order:
        c = final[cid]; mem = c["members"]
        ex = " | ".join(caps.get(keys[r], "")[:42].replace("\n", " ") for r in mem[:2])
        lines.append(f"  {cid:>4} {len(mem):>5} {c['vcoh']:.3f} {c['tcoh']:.3f}  "
                     f"{c['display'][:24]:<24} {c['near_in'][:18]:<18} {ex}")
    (out_dir / "cluster_summary.txt").write_text("\n".join(lines) + "\n")

    meta = dict(
        method="hierarchical_semantic_then_visual", N=N,
        image_embeddings=args.image_embeddings, caption_embeddings=args.caption_embeddings,
        k_target=args.k_target, k_real=len(final), super_cats=M, misc_frac=args.misc_frac,
        misc_count=int(sizes[misc_id]), w_visual=args.w_visual, w_text=args.w_text,
        floor=floor, cap=cap, target=target, seed=args.seed,
        real_size_stats=_stats(real_sizes),
        mean_visual_coh=float(np.mean([c["vcoh"] for c in final])),
        mean_caption_coh=float(np.mean([c["tcoh"] for c in final])),
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote taxonomy to {out_dir} ({len(final)} real + 1 misc = {len(final)+1} classes)")
    print(f"  real sizes: {_stats(real_sizes)}")
    print(f"  mean coherence  visual={meta['mean_visual_coh']:.3f}  caption={meta['mean_caption_coh']:.3f}")


if __name__ == "__main__":
    main()
