"""
Encode object captions into dense vectors for clustering analysis.

Two models are available via --model:

  bge-small (default)
    BAAI/bge-small-en-v1.5 — 384-dim, fast, no instruction prefix needed.

  gte-qwen2
    Alibaba-NLP/gte-Qwen2-1.5B-instruct — 1536-dim, higher quality but
    significantly slower (~1.5B parameters). Requires trust_remote_code=True
    and a GPU / MPS device for practical throughput. Each caption is prefixed
    with the task instruction before encoding.

Two caption sources are available via --source:

  captions (default)
    Reads the preprocessed captions.json (81,826 entries filtered to the
    GaussianVerse aesthetic list). Optionally restrict further with
    --filter-list.

  3dtopia
    Reads the raw 3DTopia-objaverse-caption-361k.json directly, giving
    access to all ~361k captioned Objaverse objects regardless of whether
    they are in the aesthetic list. --filter-list is ignored in this mode.
    Keys are stored as the bare Objaverse UUID (obj_id).

Usage examples
--------------
# All 81k captions with bge-small (default):
python object_classification/encode_captions.py

# All 81k captions with gte-qwen2 (saves to a separate file by default):
python object_classification/encode_captions.py --model gte-qwen2

# All ~361k 3DTopia captions:
python object_classification/encode_captions.py --source 3dtopia

# Only objects present on disk:
python object_classification/encode_captions.py \
    --filter-list downloaded/aesthetic_chunk/filtered_aesthetic_list.json

# Custom paths / settings:
python object_classification/encode_captions.py \
    --captions-path downloaded/captions.json \
    --filter-list downloaded/aesthetic_chunk/filtered_aesthetic_list.json \
    --output object_classification/caption_embeddings.npz \
    --batch-size 128 \
    --device cpu
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Compatibility patch: transformers>=4.45 removed DynamicCache.get_usable_length
# (renamed to get_seq_length). GTE-Qwen2's custom modeling code still calls the
# old name, so we restore it here if missing.
# ---------------------------------------------------------------------------
try:
    from transformers import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = (
            lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
        )
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Defaults (overridden by .env, then by CLI flags)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CAPTIONS = _REPO_ROOT / "downloaded" / "captions.json"
_DEFAULT_3DTOPIA = _REPO_ROOT / "downloaded" / "3DTopia-objaverse-caption-361k.json"
_DEFAULT_FILTER = _REPO_ROOT / "downloaded" / "aesthetic_chunk" / "filtered_aesthetic_list.json"

_MODELS: dict[str, dict] = {
    "bge-small": {
        "hf_id": "BAAI/bge-small-en-v1.5",
        "trust_remote_code": False,
        # No instruction prefix for symmetric tasks with BGE models.
        "prompt": None,
        "default_batch_size": 256,
        "default_output": _SCRIPT_DIR / "caption_embeddings.npz",
    },
    "gte-qwen2": {
        "hf_id": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "trust_remote_code": True,
        # eager attention avoids the DynamicCache.get_usable_length() call that
        # was removed in transformers>=4.45, keeping compatibility with Python 3.13
        # environments where downgrading transformers is not possible.
        "model_kwargs": {"attn_implementation": "eager"},
        "prompt": (
            "Instruct: Represent 3D objects by their geometric silhouette and structural topology. Ignore stylistic adjectives. Map brand names to their generic visual class (e.g., Lamborghini to Sports Car). Differentiate aircraft by propulsion type and humans by articulation (Natural vs. Action Figure).\n"
            "Query: "
        ),
        # Qwen2 is much larger — smaller batches avoid OOM on most GPUs.
        "default_batch_size": 16,
        "default_output": _SCRIPT_DIR / "caption_embeddings_gte_qwen2.npz",
    },
}


def parse_args() -> argparse.Namespace:
    load_dotenv(_REPO_ROOT / ".env")

    captions_env = os.getenv("CAPTIONS_PATH", str(_DEFAULT_CAPTIONS))
    tdtopia_env = os.getenv("TDTOPIA_CAPTION_PATH", str(_DEFAULT_3DTOPIA))
    filter_env = os.getenv("ALL_OBJ_JSON", str(_DEFAULT_FILTER))

    parser = argparse.ArgumentParser(
        description="Encode captions into dense vectors for clustering analysis."
    )
    parser.add_argument(
        "--model",
        choices=list(_MODELS.keys()),
        default="bge-small",
        help=(
            "Embedding model. 'bge-small' (default): BAAI/bge-small-en-v1.5, 384-dim, fast. "
            "'gte-qwen2': Alibaba-NLP/gte-Qwen2-1.5B-instruct, 1536-dim, higher quality, "
            "GPU recommended."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["captions", "3dtopia"],
        default="captions",
        help=(
            "Caption source. 'captions' (default): read the preprocessed captions.json "
            "(81k aesthetic objects). '3dtopia': read the raw 3DTopia JSON directly "
            "(~361k Objaverse objects, ignores --filter-list)."
        ),
    )
    parser.add_argument(
        "--captions-path",
        default=captions_env,
        help=f"Path to captions.json used when --source captions (default: {captions_env})",
    )
    parser.add_argument(
        "--3dtopia-path",
        dest="tdtopia_path",
        default=tdtopia_env,
        help=f"Path to 3DTopia JSON used when --source 3dtopia (default: {tdtopia_env})",
    )
    parser.add_argument(
        "--filter-list",
        default=None,
        help=(
            "Optional path to an object-list JSON (e.g. filtered_aesthetic_list.json). "
            "Only used with --source captions. "
            f"Env default would be: {filter_env}"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Destination .npz file. Defaults to caption_embeddings.npz for bge-small "
            "or caption_embeddings_gte_qwen2.npz for gte-qwen2."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Encoding batch size. Defaults to 256 for bge-small, 16 for gte-qwen2.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device string, e.g. 'cpu', 'cuda', 'mps'. Auto-detected if omitted.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_from_captions_json(captions_path: str) -> dict[str, str]:
    path = Path(captions_path)
    if not path.exists():
        sys.exit(f"[ERROR] captions file not found: {path}")
    print(f"Loading captions from {path} …")
    with open(path, "r", encoding="utf-8") as f:
        captions: dict[str, str] = json.load(f)
    print(f"  {len(captions):,} total entries loaded.")
    return captions


def load_from_3dtopia_json(tdtopia_path: str) -> dict[str, str]:
    """
    3DTopia JSON is a list of {"obj_id": "<uuid>", "3dtopia": "<caption>"}.
    Returns a dict mapping obj_id -> caption.
    """
    path = Path(tdtopia_path)
    if not path.exists():
        sys.exit(f"[ERROR] 3DTopia caption file not found: {path}")
    print(f"Loading 3DTopia captions from {path} …")
    with open(path, "r", encoding="utf-8") as f:
        raw: list[dict] = json.load(f)
    captions = {entry["obj_id"]: entry["3dtopia"] for entry in raw if entry.get("3dtopia")}
    print(f"  {len(captions):,} entries with captions loaded.")
    return captions


def load_filter_keys(filter_list_path: str) -> set[str]:
    """
    filtered_aesthetic_list.json maps hash_key -> "chunk/filename.tar.gz".
    Returns the "chunk/filename" stems to match captions.json keys.
    """
    path = Path(filter_list_path)
    if not path.exists():
        sys.exit(f"[ERROR] filter-list file not found: {path}")
    print(f"Loading filter list from {path} …")
    with open(path, "r", encoding="utf-8") as f:
        obj_list: dict[str, str] = json.load(f)

    keys: set[str] = set()
    for tar_path in obj_list.values():
        stem = tar_path.removesuffix(".tar.gz")
        keys.add(stem)

    print(f"  {len(keys):,} objects in filter list.")
    return keys


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_in_batches(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    prompt: str | None = None,
) -> np.ndarray:
    """Encode texts in batches, returning a float32 array of shape (N, D)."""
    all_embeddings: list[np.ndarray] = []
    encode_kwargs: dict = dict(
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    if prompt is not None:
        encode_kwargs["prompt"] = prompt

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", unit="batch"):
        batch = texts[start : start + batch_size]
        vecs = model.encode(batch, batch_size=len(batch), **encode_kwargs)
        all_embeddings.append(vecs.astype(np.float32))
    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    model_cfg = _MODELS[args.model]

    # Resolve defaults that depend on model choice
    batch_size = args.batch_size if args.batch_size is not None else model_cfg["default_batch_size"]
    output_path = Path(args.output) if args.output is not None else model_cfg["default_output"]

    # --- Load captions -------------------------------------------------------
    if args.source == "3dtopia":
        captions_dict = load_from_3dtopia_json(args.tdtopia_path)
        print("  (--filter-list ignored for --source 3dtopia)")
    else:
        captions_dict = load_from_captions_json(args.captions_path)
        if args.filter_list is not None:
            filter_keys = load_filter_keys(args.filter_list)
            before = len(captions_dict)
            captions_dict = {k: v for k, v in captions_dict.items() if k in filter_keys}
            print(f"  Filtered {before:,} → {len(captions_dict):,} captions.")

    if not captions_dict:
        sys.exit("[ERROR] No captions remain after filtering. Check your paths.")

    keys: list[str] = list(captions_dict.keys())
    texts: list[str] = [captions_dict[k] for k in keys]

    # --- Load model ----------------------------------------------------------
    hf_id = model_cfg["hf_id"]
    print(f"\nLoading model {hf_id!r} …")
    st_kwargs: dict = {}
    if args.device:
        st_kwargs["device"] = args.device
    if model_cfg["trust_remote_code"]:
        st_kwargs["trust_remote_code"] = True
    if model_cfg.get("model_kwargs"):
        st_kwargs["model_kwargs"] = model_cfg["model_kwargs"]
    model = SentenceTransformer(hf_id, **st_kwargs)
    print(f"  Embedding dimension : {model.get_sentence_embedding_dimension()}")
    if args.device:
        print(f"  Device              : {args.device}")

    prompt = model_cfg["prompt"]
    if prompt:
        print(f"  Instruction prompt  : {prompt!r}")

    # --- Encode --------------------------------------------------------------
    print(f"\nEncoding {len(texts):,} captions (batch size {batch_size}) …")
    embeddings = encode_in_batches(model, texts, batch_size, prompt=prompt)
    print(f"  Done. Embedding matrix shape: {embeddings.shape}")

    # --- Save ----------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        embeddings=embeddings,
        keys=np.array(keys, dtype=object),
        captions=np.array(texts, dtype=object),
    )
    print(f"\nSaved embeddings to {output_path}")
    print("  Arrays stored:")
    print(f"    embeddings  — shape {embeddings.shape}, dtype float32")
    print(f"    keys        — shape ({len(keys)},), dtype str")
    print(f"    captions    — shape ({len(texts)},), dtype str")
    print("\nLoad with:  data = np.load('caption_embeddings.npz', allow_pickle=True)")


if __name__ == "__main__":
    main()

