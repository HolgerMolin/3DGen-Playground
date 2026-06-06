"""
Build an ImageNet-seeded, balance-to-uniform class taxonomy for GaussianVerse.

Pipeline
--------
1. Load per-object CLIP **image** features (clip_image_embeddings.npz, produced by
   object_classification/encode_clip_image_embeddings.py) -- unit-norm 768-d
   vectors in CLIP's joint image-text space.
2. Build 1000 ImageNet class **text** prototypes via CLIP's text projection head
   with a prompt ensemble (canonical zero-shot recipe). Class names come from
   torchvision's ImageNet-1k category list by default.
3. Zero-shot assign every object to its argmax-cosine class -> 1000 initial
   clusters.
4. Rebalance toward uniform: iteratively SPLIT clusters above `cap` (MiniBatchKMeans
   in CLIP-image space) and MERGE clusters below `floor` into their nearest
   centroid neighbour, until sizes settle inside [floor, cap]; then nudge the
   cluster count to exactly --k-target. Target size = N / k_target (~250).
5. Emit:
     object_to_class.json   {"<chunk>/<file>": int}   (same format as the legacy map)
     class_names.json       {int: name}
     cluster_summary.txt     per-class size + name + nearest ImageNet class + example captions
     meta.json               all parameters + final size-distribution stats

No label augmentation is performed (people/characters etc. fall into whatever
ImageNet class is nearest, and the split step separates large buckets).

Usage
-----
python data/build_imagenet_taxonomy.py \
    --embeddings object_classification/clip_image_embeddings.npz \
    --output-dir object_labels/imagenet_uniform_k1000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CLIP_HF_ID = "openai/clip-vit-large-patch14"
_CLIP_DIM = 768
# Prompt ensemble: averaged then renormalized per class (OpenAI zero-shot recipe,
# trimmed and biased toward the white-bg 3D-render domain of our data).
_TEMPLATES = (
    "a photo of a {}.",
    "a 3D render of a {}.",
    "a rendering of a {}.",
    "a 3D model of a {}.",
    "a photo of a {} on a white background.",
    "a {}.",
)


def _env_path(name, fallback=None):
    val = os.environ.get(name)
    return os.path.expandvars(val) if val is not None else fallback


def parse_args():
    load_dotenv(_REPO_ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings",
                   default=str(_REPO_ROOT / "object_classification/clip_image_embeddings.npz"),
                   help="CLIP image-embedding .npz (embeddings (N,768) + keys).")
    p.add_argument("--output-dir",
                   default=str(_REPO_ROOT / "object_labels/imagenet_uniform_k1000"))
    p.add_argument("--classnames-json", default=None,
                   help="Optional class-name source. Accepts imagenet_class_index.json "
                        "({idx:[synset,name]}), {idx:name}, or a JSON list. Defaults to "
                        "torchvision's ImageNet-1k categories.")
    p.add_argument("--captions", default=_env_path("CAPTIONS_PATH"),
                   help="captions.json for the summary (optional).")
    p.add_argument("--k-target", type=int, default=1000,
                   help="Final number of clusters.")
    p.add_argument("--floor-frac", type=float, default=0.5,
                   help="Merge clusters smaller than floor_frac * target.")
    p.add_argument("--cap-frac", type=float, default=1.5,
                   help="Split clusters larger than cap_frac * target.")
    p.add_argument("--max-iters", type=int, default=25)
    p.add_argument("--device", default=None, help="cuda/cpu for the CLIP text encode.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Class names + text prototypes
# ---------------------------------------------------------------------------

def load_classnames(path):
    if path:
        obj = json.loads(Path(path).read_text())
        if isinstance(obj, dict):
            items = sorted(obj.items(), key=lambda kv: int(kv[0]))
            names = []
            for _, v in items:
                name = v[-1] if isinstance(v, (list, tuple)) else v
                names.append(str(name).replace("_", " "))
            return names
        if isinstance(obj, list):
            return [str(x).replace("_", " ") for x in obj]
        raise ValueError(f"Unrecognized classnames JSON structure in {path}")
    from torchvision.models import ResNet50_Weights
    return list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])


@torch.no_grad()
def build_text_prototypes(names, device, dtype=torch.float16):
    from transformers import CLIPModel, CLIPTokenizer
    print(f"Encoding {len(names)} class names x {len(_TEMPLATES)} templates with CLIP text tower …")
    m = CLIPModel.from_pretrained(_CLIP_HF_ID, torch_dtype=dtype).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(_CLIP_HF_ID)
    protos = []
    for i in range(0, len(names), 256):
        chunk = names[i:i + 256]
        acc = torch.zeros(len(chunk), _CLIP_DIM, device=device)
        for t in _TEMPLATES:
            txt = [t.format(n) for n in chunk]
            tt = tok(txt, padding=True, truncation=True, max_length=77,
                     return_tensors="pt").to(device)
            out = m.text_model(input_ids=tt["input_ids"], attention_mask=tt["attention_mask"])
            e = m.text_projection(out.pooler_output).float()
            acc += F.normalize(e, dim=-1)
        protos.append(F.normalize(acc / len(_TEMPLATES), dim=-1).cpu())
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(protos, 0).numpy().astype(np.float32)  # (C, 768)


# ---------------------------------------------------------------------------
# Cluster algebra (everything in unit-norm CLIP-image space)
# ---------------------------------------------------------------------------

def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _centroid(E, members):
    return _unit(E[members].mean(axis=0))


def _split_cluster(E, c, k, seed):
    """KMeans-split cluster `c` into <=k subclusters in CLIP-image space."""
    k = min(k, len(c["members"]))
    if k < 2:
        return [c]
    km = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=seed,
                         batch_size=4096, max_iter=100)
    lab = km.fit_predict(E[c["members"]])
    out = []
    for j in range(k):
        sel = c["members"][lab == j]
        if len(sel) == 0:
            continue
        out.append({"members": sel, "name": f"{c['name']}#{j}", "cent": _centroid(E, sel)})
    return out


def _merge_into_nearest(E, clusters, i):
    """Merge clusters[i] into its nearest-centroid neighbour. Returns new list."""
    cents = np.stack([c["cent"] for c in clusters])     # (K, 768)
    sims = cents @ clusters[i]["cent"]
    sims[i] = -2.0
    j = int(sims.argmax())
    a, b = clusters[i], clusters[j]
    mem = np.concatenate([a["members"], b["members"]])
    name = a["name"] if len(a["members"]) >= len(b["members"]) else b["name"]
    clusters[j] = {"members": mem, "name": name, "cent": _centroid(E, mem)}
    clusters.pop(i)
    return clusters


def _merge_small(E, clusters, floor):
    merged = False
    while len(clusters) > 1:
        sizes = np.array([len(c["members"]) for c in clusters])
        i = int(sizes.argmin())
        if sizes[i] >= floor:
            break
        clusters = _merge_into_nearest(E, clusters, i)
        merged = True
    return clusters, merged


def _adjust_to_k(E, clusters, k_target, seed):
    """Force exactly k_target clusters: merge smallest / split largest."""
    while len(clusters) > k_target:
        sizes = np.array([len(c["members"]) for c in clusters])
        clusters = _merge_into_nearest(E, clusters, int(sizes.argmin()))
    while len(clusters) < k_target:
        sizes = np.array([len(c["members"]) for c in clusters])
        i = int(sizes.argmax())
        big = clusters.pop(i)
        parts = _split_cluster(E, big, 2, seed)
        if len(parts) < 2:               # cannot split further (degenerate)
            clusters.append(big)
            break
        clusters += parts
    return clusters


def _stats(sizes):
    sizes = np.asarray(sizes)
    return dict(
        k=int(len(sizes)), total=int(sizes.sum()),
        min=int(sizes.min()), max=int(sizes.max()),
        mean=float(sizes.mean()), median=float(np.median(sizes)),
        std=float(sizes.std()), cv=float(sizes.std() / max(1e-9, sizes.mean())),
        p10=int(np.percentile(sizes, 10)), p90=int(np.percentile(sizes, 90)),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng_seed = args.seed

    # --- Load embeddings -------------------------------------------------
    print(f"Loading CLIP image embeddings from {args.embeddings}")
    data = np.load(args.embeddings, allow_pickle=True)
    E = _unit(data["embeddings"].astype(np.float32))      # (N, 768)
    keys = [str(k) for k in data["keys"].tolist()]
    N = len(keys)
    print(f"  {N:,} objects, dim {E.shape[1]}")

    # --- Text prototypes + zero-shot assignment -------------------------
    names = load_classnames(args.classnames_json)
    P = build_text_prototypes(names, device)              # (C, 768)
    print(f"Assigning {N:,} objects to {len(names)} ImageNet classes (argmax cosine) …")
    init = np.empty(N, dtype=np.int64)
    for i in range(0, N, 16384):                          # chunked matmul (N x C)
        sims = E[i:i + 16384] @ P.T
        init[i:i + 16384] = sims.argmax(axis=1)
    used = np.unique(init)
    print(f"  {len(used)} / {len(names)} ImageNet classes received >=1 object "
          f"({len(names) - len(used)} empty).")

    clusters = []
    for c in used:
        members = np.where(init == c)[0]
        clusters.append({"members": members, "name": names[int(c)],
                         "cent": _centroid(E, members)})
    init_stats = _stats([len(c["members"]) for c in clusters])
    print(f"  initial: {init_stats}")

    # --- Rebalance: split big / merge small ------------------------------
    target = N / args.k_target
    floor = max(1, round(args.floor_frac * target))
    cap = max(floor + 1, round(args.cap_frac * target))
    print(f"\nRebalancing toward k={args.k_target}: target={target:.1f} "
          f"floor={floor} cap={cap}")
    t0 = time.time()
    for it in range(args.max_iters):
        changed = False
        # SPLIT
        nc = []
        for c in clusters:
            if len(c["members"]) > cap:
                k = max(2, round(len(c["members"]) / target))
                parts = _split_cluster(E, c, k, rng_seed + it)
                if len(parts) > 1:
                    nc += parts
                    changed = True
                else:
                    nc.append(c)
            else:
                nc.append(c)
        clusters = nc
        # MERGE
        clusters, merged = _merge_small(E, clusters, floor)
        changed = changed or merged
        s = _stats([len(c["members"]) for c in clusters])
        print(f"  iter {it:>2}: k={s['k']:>5} min={s['min']:>4} max={s['max']:>5} "
              f"mean={s['mean']:.1f} cv={s['cv']:.3f}")
        if not changed:
            print("  converged (no split/merge this iter).")
            break

    # --- Force exactly k_target ------------------------------------------
    clusters = _adjust_to_k(E, clusters, args.k_target, rng_seed)
    final_stats = _stats([len(c["members"]) for c in clusters])
    print(f"\nFinal (k={args.k_target}): {final_stats}  ({time.time()-t0:.0f}s)")

    # --- Order clusters by size (desc) and build assignment --------------
    order = sorted(range(len(clusters)), key=lambda i: -len(clusters[i]["members"]))
    clusters = [clusters[i] for i in order]
    assign = np.empty(N, dtype=np.int64)
    for cid, c in enumerate(clusters):
        assign[c["members"]] = cid

    # Nearest ImageNet class per final cluster (for the summary).
    final_cents = np.stack([c["cent"] for c in clusters])     # (K, 768)
    nearest_imagenet = (final_cents @ P.T).argmax(axis=1)

    # --- Write artifacts -------------------------------------------------
    obj_to_class = {keys[r]: int(assign[r]) for r in range(N)}
    (out_dir / "object_to_class.json").write_text(json.dumps(obj_to_class))
    class_names = {str(cid): clusters[cid]["name"] for cid in range(len(clusters))}
    (out_dir / "class_names.json").write_text(json.dumps(class_names, indent=0))

    captions = {}
    if args.captions and Path(args.captions).exists():
        captions = json.loads(Path(args.captions).read_text())

    lines = [
        f"ImageNet-seeded uniform taxonomy: k={len(clusters)}  N={N:,}",
        f"sizes: {final_stats}",
        f"templates: {list(_TEMPLATES)}",
        "",
        f"  {'cid':>4}  {'size':>5}  {'name':<34}  {'nearest_imagenet':<22}  examples",
        "  " + "-" * 110,
    ]
    for cid, c in enumerate(clusters):
        mem = c["members"]
        ex = []
        for r in mem[:3]:
            cap = captions.get(keys[r], "")
            if cap:
                ex.append(cap[:50].replace("\n", " "))
        lines.append(
            f"  {cid:>4}  {len(mem):>5}  {c['name'][:34]:<34}  "
            f"{names[int(nearest_imagenet[cid])][:22]:<22}  {' | '.join(ex)}"
        )
    (out_dir / "cluster_summary.txt").write_text("\n".join(lines) + "\n")

    meta = dict(
        embeddings=str(args.embeddings), N=N, dim=int(E.shape[1]),
        assignment="clip_zeroshot_image_vs_imagenet_text",
        classnames_source=(args.classnames_json or "torchvision:ImageNet1k"),
        num_classes_seeded=len(names), num_classes_used=int(len(used)),
        k_target=args.k_target, target=target, floor=floor, cap=cap,
        floor_frac=args.floor_frac, cap_frac=args.cap_frac,
        templates=list(_TEMPLATES), seed=rng_seed,
        initial_stats=init_stats, final_stats=final_stats,
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nWrote:")
    for f in ("object_to_class.json", "class_names.json", "cluster_summary.txt", "meta.json"):
        print(f"  {out_dir / f}")


if __name__ == "__main__":
    main()
