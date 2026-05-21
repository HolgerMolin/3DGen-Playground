"""
Encode every caption as a CLIP-L/14 *penultimate-normed* token sequence for
text-conditioned JiT training with both AdaLN (pooled) and cross-attention
(per-token KV) pathways.

Output layout (directory, mmap-friendly)
----------------------------------------
Default output is a directory containing:
  keys.json                — JSON list of caption stems "<chunk>/<filename>"
                             (joins on image_embeddings.npz / captions.json).
  tokens.npy   (N, 77, 768) fp16 — CLIP penultimate hidden state with CLIP's
                                   final_layer_norm applied manually
                                   ("clip-skip 1"). Cross-attention KV.
  pooled.npy   (N, 768)     fp16 — EOS-position token of `tokens`. AdaLN input.
                                   Same geometry as the cross-attn KV.
  mask.npy     (N, 77)      uint8 — 1 valid, 0 padding (cross-attn -inf bias).

All three arrays are pre-allocated and written incrementally via numpy memmap,
so per-step work is O(B) instead of the O(N) full-file rewrite of the old
single-npz layout. The dataloader opens these with ``mmap_mode='r'`` so worker
processes share a single page cache instead of duplicating ~30 GB per worker.

A separate ``null_text_token.npz`` (written once at startup) caches:
  null_token (1, 768)        fp16 — penultimate-normed EOS token of the empty
                                    string, used as the length-1 KV sequence
                                    on the CFG unconditional branch.

Resume
------
If the output directory already contains tokens.npy/pooled.npy/mask.npy with
the expected (N, L, D) shape AND a keys.json listing the prefix of completed
keys, the script continues from where it left off. Mismatched shapes or
caption ordering raise — delete the directory to start fresh.

Examples
--------
# Single-GPU full run (~10 min on an A100 with batch=1024):
python object_classification/encode_text_embeddings.py

# Smoke test (32 captions, separate output dir):
python object_classification/encode_text_embeddings.py --limit 32 \
    --output /tmp/text_tokens_smoke \
    --null-output /tmp/null_text_token_smoke.npz

# 2-way shard (one process per GPU; separate output dirs):
CUDA_VISIBLE_DEVICES=0 python object_classification/encode_text_embeddings.py \
    --num-shards 2 --shard-id 0 \
    --output object_classification/text_tokens_shard0 &
CUDA_VISIBLE_DEVICES=1 python object_classification/encode_text_embeddings.py \
    --num-shards 2 --shard-id 1 \
    --output object_classification/text_tokens_shard1 &
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_CLIP_HF_ID = "openai/clip-vit-large-patch14"
_CLIP_TEXT_DIM = 768
_CLIP_MAX_TOKENS = 77
_DEFAULT_OUTPUT = _SCRIPT_DIR / "text_tokens"
_DEFAULT_NULL_OUTPUT = _SCRIPT_DIR / "null_text_token.npz"


def _env_path(name: str, fallback: "str | None" = None) -> "str | None":
    val = os.environ.get(name)
    if val is None:
        return fallback
    return os.path.expandvars(val)


def parse_args() -> argparse.Namespace:
    load_dotenv(_REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Encode captions with frozen CLIP-L/14 text tower "
                    "(penultimate-normed tokens + EOS-pooled + mask) to a "
                    "mmap-friendly directory layout."
    )
    parser.add_argument(
        "--obj-list",
        default=_env_path("ALL_OBJ_JSON"),
        help="Path to obj_list JSON (dict: hash -> 'chunk/file.tar.gz'). "
             "Caption keys outside this list are skipped.",
    )
    parser.add_argument(
        "--captions",
        default=_env_path("CAPTIONS_PATH"),
        help="Path to preprocessed captions.json (dict: 'chunk/file' -> caption).",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help=f"Output directory (default: {_DEFAULT_OUTPUT}). Will contain "
             f"keys.json, tokens.npy, pooled.npy, mask.npy.",
    )
    parser.add_argument(
        "--null-output",
        default=str(_DEFAULT_NULL_OUTPUT),
        help=f"Destination for the empty-string null token (default: "
             f"{_DEFAULT_NULL_OUTPUT}). Re-written on every run (cheap).",
    )
    parser.add_argument("--device", default=None,
                        help="Torch device. Auto-detected if omitted.")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Captions per CLIP forward pass. CLIP-L/14 text "
                             "in fp16 at batch=1024 uses ~2 GB of activations; "
                             "fits comfortably on an A100/L40.")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"),
                        help="Model weight dtype.")
    parser.add_argument("--flush-every", type=int, default=20000,
                        help="Persist keys.json + flush memmaps every N captions. "
                             "Per-row writes already go through mmap and reach disk "
                             "via page cache; this only controls resume granularity.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap captions processed (after filtering and skip-already-done).")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Output management
# ---------------------------------------------------------------------------

def _open_outputs(out_dir: Path, N_total: int):
    """Open or create the three mmap'd arrays sized for exactly N_total rows.

    Returns ``(tokens_mm, pooled_mm, mask_mm, done_keys)`` where ``done_keys``
    is the prefix of completed caption keys (empty list on a fresh run).
    Mismatched existing shapes raise — delete the directory to start fresh.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = out_dir / "tokens.npy"
    pooled_path = out_dir / "pooled.npy"
    mask_path = out_dir / "mask.npy"
    keys_path = out_dir / "keys.json"

    expected = {
        tokens_path: (np.float16, (N_total, _CLIP_MAX_TOKENS, _CLIP_TEXT_DIM)),
        pooled_path: (np.float16, (N_total, _CLIP_TEXT_DIM)),
        mask_path:   (np.uint8,   (N_total, _CLIP_MAX_TOKENS)),
    }
    all_present = all(p.exists() for p in expected)
    done_keys: list[str] = []
    if all_present:
        for path, (dtype, shape) in expected.items():
            existing = np.load(path, mmap_mode="r")
            if existing.dtype != dtype or existing.shape != shape:
                raise ValueError(
                    f"Existing {path} has dtype={existing.dtype}, shape={existing.shape}; "
                    f"expected dtype={dtype}, shape={shape}. Delete {out_dir} to start fresh."
                )
            del existing
        if keys_path.exists():
            done_keys = json.loads(keys_path.read_text())
        mode = "r+"
    elif any(p.exists() for p in expected):
        raise ValueError(
            f"Output directory {out_dir} is partially populated. "
            f"Delete it to start fresh."
        )
    else:
        mode = "w+"

    tokens_mm = np.lib.format.open_memmap(
        tokens_path, mode=mode, dtype=np.float16,
        shape=(N_total, _CLIP_MAX_TOKENS, _CLIP_TEXT_DIM),
    )
    pooled_mm = np.lib.format.open_memmap(
        pooled_path, mode=mode, dtype=np.float16, shape=(N_total, _CLIP_TEXT_DIM),
    )
    mask_mm = np.lib.format.open_memmap(
        mask_path, mode=mode, dtype=np.uint8, shape=(N_total, _CLIP_MAX_TOKENS),
    )
    return tokens_mm, pooled_mm, mask_mm, done_keys


def _persist_keys(out_dir: Path, keys: list[str]) -> None:
    keys_path = out_dir / "keys.json"
    tmp = keys_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(keys, f)
    os.replace(tmp, keys_path)


def _save_null(output_path: Path, null_token: np.ndarray) -> None:
    """Write the empty-string EOS token (1, 768) to its own small file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if null_token.shape != (1, _CLIP_TEXT_DIM):
        raise ValueError(
            f"null_token shape {null_token.shape} != (1, {_CLIP_TEXT_DIM})."
        )
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, null_token=null_token.astype(np.float16))
    os.replace(tmp, output_path)


# ---------------------------------------------------------------------------
# CLIP wrapper
# ---------------------------------------------------------------------------

class _ClipTextEncoder:
    def __init__(self, device: torch.device, dtype: torch.dtype):
        from transformers import CLIPTextModel, CLIPTokenizer
        print(f"Loading {_CLIP_HF_ID} text tower in {dtype} …")
        self.tokenizer = CLIPTokenizer.from_pretrained(_CLIP_HF_ID)
        self.model = (
            CLIPTextModel.from_pretrained(_CLIP_HF_ID, torch_dtype=dtype)
            .to(device).eval()
        )
        self.device = device
        # Approx truncation count: number of sequences whose final position is
        # non-padding (== filled all 77 slots). Includes captions exactly 77
        # tokens long, not just truncated ones, so this is a slight overestimate.
        # Much cheaper than the legacy second tokenization pass.
        self.truncations = 0

    @torch.no_grad()
    def encode(self, texts: list[str]) -> dict:
        """Encode a batch of texts. Returns:
            tokens : (B, 77, 768) fp16 — penultimate-normed sequence, on CPU
            pooled : (B, 768)     fp16 — EOS-position of tokens, on CPU
            mask   : (B, 77)      uint8 — 1 valid, 0 padding, on CPU
        """
        tok = self.tokenizer(
            texts,
            padding="max_length",
            max_length=_CLIP_MAX_TOKENS,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(self.device, non_blocking=True)
        attention_mask = tok["attention_mask"].to(self.device, non_blocking=True)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # "clip-skip 1": penultimate hidden state through CLIP's own final LN.
        penultimate = out.hidden_states[-2]  # (B, 77, 768), pre-LN
        tokens = self.model.final_layer_norm(penultimate)  # (B, 77, 768)

        eos_positions = input_ids.to(torch.int).argmax(dim=-1)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = tokens[batch_idx, eos_positions]  # (B, 768)

        # Truncation proxy: last position has a non-zero attention mask.
        self.truncations += int(attention_mask[:, -1].sum().item())

        return {
            "tokens": tokens.detach().to(torch.float16).cpu().numpy(),
            "pooled": pooled.detach().to(torch.float16).cpu().numpy(),
            "mask":   attention_mask.detach().to(torch.uint8).cpu().numpy(),
        }

    @torch.no_grad()
    def encode_null(self) -> np.ndarray:
        """EOS-position penultimate-normed token of the empty string. (1, 768) fp16."""
        out = self.encode([""])
        return out["pooled"]  # (1, 768) fp16


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    null_path = Path(args.null_output)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    model_dtype = dtype_map[args.dtype]

    # --- Load obj_list and captions -------------------------------------
    print(f"Loading obj list from {args.obj_list}")
    with open(args.obj_list, "r", encoding="utf-8") as f:
        obj_data: dict = json.load(f)
    obj_keys = {tar_path.removesuffix(".tar.gz") for tar_path in obj_data.values()}
    print(f"  {len(obj_keys):,} object stems.")

    print(f"Loading captions from {args.captions}")
    with open(args.captions, "r", encoding="utf-8") as f:
        captions: dict = json.load(f)
    print(f"  {len(captions):,} caption entries.")

    selected_keys = sorted(obj_keys & captions.keys())
    missing = len(obj_keys) - len(selected_keys)
    print(f"Caption coverage: {len(selected_keys):,}/{len(obj_keys):,} "
          f"({len(selected_keys)/max(1,len(obj_keys))*100:.2f}%)  missing={missing:,}")

    # --- Sharding --------------------------------------------------------
    if args.num_shards > 1:
        before = len(selected_keys)
        selected_keys = [k for i, k in enumerate(selected_keys) if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: {before:,} -> {len(selected_keys):,}")

    if args.limit is not None:
        selected_keys = selected_keys[: args.limit]

    N_total = len(selected_keys)
    if N_total == 0:
        raise ValueError("No captions to encode after filtering / sharding.")

    # --- Open output memmaps + maybe attach resume state ----------------
    tokens_mm, pooled_mm, mask_mm, done_keys = _open_outputs(out_dir, N_total)
    if done_keys:
        if done_keys != selected_keys[: len(done_keys)]:
            raise ValueError(
                f"Resume mismatch: existing keys.json prefix doesn't match the "
                f"start of the selected key list. Delete {out_dir} to start fresh."
            )
        start = len(done_keys)
        print(f"Resuming from index {start:,} / {N_total:,}")
    else:
        start = 0
        print(f"Fresh run: encoding {N_total:,} captions into {out_dir}")

    encoder = _ClipTextEncoder(device, dtype=model_dtype)

    # --- Null token (cheap, always rewrite) -----------------------------
    null_token = encoder.encode_null()
    _save_null(null_path, null_token)
    print(f"Wrote null token (1, {_CLIP_TEXT_DIM}) to {null_path}")

    if start >= N_total:
        print("Nothing to encode.")
        return

    since_flush = 0
    t0 = time.time()
    i = start

    pbar = tqdm(range(start, N_total, args.batch_size), desc="Encoding", unit="batch")
    for batch_start in pbar:
        batch_end = min(batch_start + args.batch_size, N_total)
        batch_keys = selected_keys[batch_start:batch_end]
        batch_texts = [captions[k] for k in batch_keys]
        out = encoder.encode(batch_texts)
        B = len(batch_keys)
        tokens_mm[i:i + B] = out["tokens"]
        pooled_mm[i:i + B] = out["pooled"]
        mask_mm[i:i + B] = out["mask"]
        i += B
        done_keys.extend(batch_keys)
        since_flush += B
        if since_flush >= args.flush_every:
            tokens_mm.flush(); pooled_mm.flush(); mask_mm.flush()
            _persist_keys(out_dir, done_keys)
            since_flush = 0
            pbar.set_postfix(written=f"{len(done_keys):,}/{N_total:,}")

    tokens_mm.flush(); pooled_mm.flush(); mask_mm.flush()
    _persist_keys(out_dir, done_keys)
    dt = time.time() - t0

    written = N_total - start
    print(f"\nDone. Wrote {len(done_keys):,} entries to {out_dir}")
    print(f"  Elapsed: {dt:.1f}s  ({written/max(1e-6, dt):.0f} captions/s)")
    print(f"  Approx truncations (>={_CLIP_MAX_TOKENS} tokens, includes exact-77): "
          f"{encoder.truncations:,} ({encoder.truncations/max(1,written)*100:.2f}%)")


if __name__ == "__main__":
    main()
