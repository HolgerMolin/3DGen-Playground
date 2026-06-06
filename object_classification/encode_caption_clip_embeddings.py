"""
Encode every object's caption as a single CLIP **joint-space** (text_projection)
embedding, for the semantic axis of the hierarchical taxonomy.

This is DIFFERENT from `text_tokens/pooled.npy`: that file is the penultimate-
normed EOS token used for AdaLN training conditioning. Here we apply CLIP's
`text_projection` head (`text_model(...).pooler_output` → `text_projection`),
i.e. the `get_text_features`-equivalent, so the result lives in the SAME joint
image-text space as `clip_image_embeddings.npz` and is directly comparable to
the class-name prototypes.

Output (.npz, atomic, resumable — same scheme as encode_clip_image_embeddings.py):
  embeddings (N, 768) float32  — unit-norm joint-space CLIP caption features
  keys       (N,)     object   — "<chunk>/<filename>" stems (join key)

Usage
-----
python object_classification/encode_caption_clip_embeddings.py            # full run (~5-10 min, A100)
python object_classification/encode_caption_clip_embeddings.py --limit 64 --output /tmp/cap_smoke.npz
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
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CLIP_HF_ID = "openai/clip-vit-large-patch14"
_CLIP_DIM = 768
_CLIP_MAX_TOKENS = 77
_DEFAULT_OUTPUT = _SCRIPT_DIR / "caption_clip_text_embeddings.npz"


def _env_path(name, fallback=None):
    val = os.environ.get(name)
    return os.path.expandvars(val) if val is not None else fallback


def parse_args():
    load_dotenv(_REPO_ROOT / ".env")
    p = argparse.ArgumentParser(description="Encode captions as CLIP joint-space (projected) features.")
    p.add_argument("--obj-list", default=_env_path("ALL_OBJ_JSON"),
                   help="obj_list JSON (hash -> 'chunk/file.tar.gz'). Defaults to $ALL_OBJ_JSON.")
    p.add_argument("--captions", default=_env_path("CAPTIONS_PATH"),
                   help="captions.json (dict 'chunk/file' -> caption). Defaults to $CAPTIONS_PATH.")
    p.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    p.add_argument("--flush-every", type=int, default=20000)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    if not args.obj_list:
        p.error("--obj-list is required (or set ALL_OBJ_JSON in .env).")
    if not args.captions:
        p.error("--captions is required (or set CAPTIONS_PATH in .env).")
    return args


class _ClipCaptionEncoder:
    def __init__(self, device, dtype):
        from transformers import CLIPModel, CLIPTokenizer
        print(f"Loading {_CLIP_HF_ID} text tower in {dtype} …")
        self.model = CLIPModel.from_pretrained(_CLIP_HF_ID, torch_dtype=dtype).to(device).eval()
        self.tok = CLIPTokenizer.from_pretrained(_CLIP_HF_ID)
        self.device = device

    @torch.no_grad()
    def encode(self, texts):
        tt = self.tok(texts, padding=True, truncation=True, max_length=_CLIP_MAX_TOKENS,
                      return_tensors="pt").to(self.device)
        out = self.model.text_model(input_ids=tt["input_ids"], attention_mask=tt["attention_mask"])
        e = self.model.text_projection(out.pooler_output).float()   # (B, 768) joint space
        return F.normalize(e, dim=-1).cpu().numpy().astype(np.float32)


def _load_existing(path):
    if not path.exists():
        return [], []
    print(f"Resuming from {path} …")
    d = np.load(path, allow_pickle=True)
    keys = d["keys"].tolist()
    emb = d["embeddings"]
    if emb.shape[0] != len(keys) or emb.shape[1] != _CLIP_DIM:
        raise ValueError(f"Corrupt/incompatible cache {path}; delete it to restart.")
    print(f"  {len(keys):,} already done (skipped).")
    return keys, list(emb.astype(np.float32))


def _atomic_save(path, keys, embs):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.stack(embs, 0).astype(np.float32) if embs else np.zeros((0, _CLIP_DIM), np.float32)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, embeddings=arr, keys=np.array(keys, dtype=object))
    os.replace(tmp, path)


def main():
    args = parse_args()
    out = Path(args.output)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    print(f"Using device: {device}")

    print(f"Loading obj list {args.obj_list}")
    obj = json.load(open(args.obj_list))
    obj_stems = {v.removesuffix(".tar.gz") for v in obj.values()}
    print(f"Loading captions {args.captions}")
    caps = json.load(open(args.captions))
    keys = sorted(obj_stems & caps.keys())
    print(f"caption coverage: {len(keys):,}/{len(obj_stems):,}")

    saved_keys, saved_embs = _load_existing(out)
    done = set(saved_keys)
    todo = [k for k in keys if k not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"To encode: {len(todo):,}")
    if not todo:
        _atomic_save(out, saved_keys, saved_embs)
        print("Nothing to do."); return

    enc = _ClipCaptionEncoder(device, dtype)
    new_keys, new_embs = list(saved_keys), list(saved_embs)
    since = 0
    t0 = time.time()
    for i in tqdm(range(0, len(todo), args.batch_size), desc="Encoding captions", unit="batch"):
        batch = todo[i:i + args.batch_size]
        vecs = enc.encode([caps[k] for k in batch])
        for k, v in zip(batch, vecs):
            new_keys.append(k); new_embs.append(v); since += 1
        if since >= args.flush_every:
            _atomic_save(out, new_keys, new_embs); since = 0
    _atomic_save(out, new_keys, new_embs)
    print(f"\nDone. {len(new_keys):,} caption embeddings -> {out} "
          f"({len(new_keys)-len(saved_keys):,} new in {time.time()-t0:.0f}s).")


if __name__ == "__main__":
    main()
