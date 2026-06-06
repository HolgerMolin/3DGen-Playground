"""
Encode each object as a single 4-view-averaged **CLIP image feature** for
zero-shot ImageNet classification.

This mirrors ``encode_image_embeddings.py`` (same gsplat rendering: 4 views at
azimuths {0,90,180,270}, elevation 30, white background, 224x224) but swaps the
DINOv2 backbone for the CLIP (openai/clip-vit-large-patch14) **image projection
head** (``CLIPModel.get_image_features``). The resulting 768-d vectors live in
CLIP's joint image-text space, so they can be cosine-compared against
``CLIPModel.get_text_features(<class names>)`` for zero-shot classification --
which is exactly what ``data/build_imagenet_taxonomy.py`` does.

Per object we render 4 views, project each through CLIP's image head, L2-norm
each view, average them, and L2-norm again -> a single (768,) vector. This is a
*different* representation from ``image_embeddings.npz`` (DINOv2 CLS, 3072-d), so
it is written to its own file ``clip_image_embeddings.npz``.

Output (.npz, atomic, resumable -- same scheme as encode_image_embeddings.py):
  embeddings (N, 768) float32   -- unit-norm joint-space CLIP image features
  keys       (N,)     object    -- "<chunk>/<filename>" stems (join key)

Usage
-----
# Defaults pulled from .env (GS_PATH, ALL_OBJ_JSON):
python object_classification/encode_clip_image_embeddings.py

# Smoke test on 8 objects with debug PNGs:
python object_classification/encode_clip_image_embeddings.py --limit 8 \
    --debug-render-dir object_classification/debug_renders_clip \
    --output /tmp/clip_image_embeddings_smoke.npz
"""

import argparse
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

# Reuse the proven rendering / camera / dataloading machinery verbatim.
from object_classification.encode_image_embeddings import (  # noqa: E402
    _build_cameras,
    _render_object_white_bg,
    _PCDataset,
    _identity_collate,
    _load_filter_keys,
    _env_path,
    _NUM_VIEWS,
    _RENDER_SIZE,
    _DEFAULT_AZIMUTHS_DEG,
    _DEFAULT_ELEVATION_DEG,
    _DEFAULT_CAM_DISTANCE,
    _DEFAULT_FOV_DEG,
)
from dataloaders.standard_3dgen_loader import load_obj_list  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_OUTPUT = _SCRIPT_DIR / "clip_image_embeddings.npz"
_CLIP_HF_ID = "openai/clip-vit-large-patch14"
_CLIP_DIM = 768  # ViT-L/14 joint-space projection dim
# CLIP's own image-preprocessing normalization (NOT the ImageNet stats DINOv2 uses).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def parse_args() -> argparse.Namespace:
    load_dotenv(_REPO_ROOT / ".env")
    p = argparse.ArgumentParser(
        description="Encode each object as a 4-view-averaged CLIP image feature."
    )
    p.add_argument("--obj-list", default=_env_path("ALL_OBJ_JSON"),
                   help="Path to obj_list JSON. Defaults to $ALL_OBJ_JSON from .env.")
    p.add_argument("--gs-path", default=_env_path("GS_PATH"),
                   help="Root of unzipped 3DGS chunks. Defaults to $GS_PATH from .env.")
    p.add_argument("--filter-list", default=None,
                   help="Optional obj_list-shaped JSON; only its stems are encoded.")
    p.add_argument("--output", default=str(_DEFAULT_OUTPUT),
                   help=f"Destination .npz (default: {_DEFAULT_OUTPUT}).")
    p.add_argument("--device", default=None, help="Torch device. Auto-detected if omitted.")
    p.add_argument("--clip-batch-size", type=int, default=64,
                   help="Images per CLIP forward pass (4 views/object).")
    p.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    p.add_argument("--flush-every", type=int, default=500,
                   help="Rewrite the output .npz atomically every N new objects.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap objects processed (after filter + skip-done). Smoke testing.")
    p.add_argument("--azimuths-deg", type=float, nargs="+",
                   default=list(_DEFAULT_AZIMUTHS_DEG))
    p.add_argument("--elevation-deg", type=float, default=_DEFAULT_ELEVATION_DEG)
    p.add_argument("--cam-distance", type=float, default=_DEFAULT_CAM_DISTANCE)
    p.add_argument("--fov-deg", type=float, default=_DEFAULT_FOV_DEG)
    p.add_argument("--world-up", choices=["y_down", "y_up"], default="y_down")
    p.add_argument("--debug-render-dir", default=None,
                   help="Save the 4 renders for the first --debug-render-count objects.")
    p.add_argument("--debug-render-count", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        p.error("--shard-id must be in [0, --num-shards).")
    if not args.obj_list:
        p.error("--obj-list is required (or set ALL_OBJ_JSON in .env).")
    if not args.gs_path:
        p.error("--gs-path is required (or set GS_PATH in .env).")
    if len(args.azimuths_deg) != _NUM_VIEWS:
        p.error(f"--azimuths-deg must have exactly {_NUM_VIEWS} entries.")
    return args


# ---------------------------------------------------------------------------
# CLIP image wrapper
# ---------------------------------------------------------------------------

class _ClipImageEncoder:
    def __init__(self, device: torch.device, dtype: torch.dtype):
        from transformers import CLIPModel
        print(f"Loading {_CLIP_HF_ID} image tower in {dtype} …")
        self.model = (
            CLIPModel.from_pretrained(_CLIP_HF_ID, torch_dtype=dtype).to(device).eval()
        )
        self.device = device
        self.dtype = dtype
        self.mean = torch.tensor(_CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_CLIP_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) in [0,1] on self.device. Returns (B, 768) unit-norm.

        Applies CLIP's vision tower + visual_projection directly (the joint
        image-text space). We bypass ``get_image_features`` because in
        transformers>=5 it returns a BaseModelOutputWithPooling rather than the
        projected tensor; the component path is stable across versions.
        """
        if images.shape[-1] != _RENDER_SIZE or images.shape[-2] != _RENDER_SIZE:
            images = F.interpolate(
                images, size=(_RENDER_SIZE, _RENDER_SIZE),
                mode="bicubic", align_corners=False,
            )
        x = ((images - self.mean) / self.std).to(self.dtype)
        pooled = self.model.vision_model(pixel_values=x).pooler_output  # (B, 1024)
        feats = self.model.visual_projection(pooled)                    # (B, 768)
        return F.normalize(feats.float(), dim=-1)


# ---------------------------------------------------------------------------
# Output / resume helpers (768-d)
# ---------------------------------------------------------------------------

def _load_existing(output_path: Path) -> "tuple[list[str], list[np.ndarray]]":
    if not output_path.exists():
        return [], []
    print(f"Resuming from existing output {output_path} …")
    data = np.load(output_path, allow_pickle=True)
    keys = data["keys"].tolist()
    embeddings = data["embeddings"]
    if embeddings.shape[0] != len(keys):
        raise ValueError(
            f"Corrupt cache: embeddings ({embeddings.shape[0]}) and keys ({len(keys)}) "
            f"differ in length in {output_path}")
    if embeddings.shape[1] != _CLIP_DIM:
        raise ValueError(
            f"Cache feature dim {embeddings.shape[1]} != expected {_CLIP_DIM}. "
            f"Delete/move {output_path} before re-running.")
    print(f"  Found {len(keys):,} previously encoded objects (skipping those).")
    return keys, list(embeddings.astype(np.float32))


def _atomic_save(output_path: Path, keys: "list[str]", embeddings: "list[np.ndarray]") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.stack(embeddings, axis=0).astype(np.float32)
           if embeddings else np.zeros((0, _CLIP_DIM), dtype=np.float32))
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, embeddings=arr, keys=np.array(keys, dtype=object))
    os.replace(tmp, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    # --- Object list (direct PLY load, bypassing Standard3DGenDataset) ----
    print(f"Loading obj list from {args.obj_list}  (gs_path={args.gs_path})")
    obj_data = load_obj_list([args.obj_list])
    hash_keys = list(obj_data.keys())
    all_keys = [obj_data[h].removesuffix(".tar.gz") for h in hash_keys]
    gs_root = Path(args.gs_path)

    selected = list(range(len(hash_keys)))
    if args.filter_list:
        keep = _load_filter_keys(args.filter_list)
        selected = [i for i in selected if all_keys[i] in keep]
        print(f"  Filtered {len(hash_keys):,} -> {len(selected):,} via {args.filter_list}")
    if args.num_shards > 1:
        before = len(selected)
        selected = [i for i in selected if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: {before:,} -> {len(selected):,}")

    saved_keys, saved_embeddings = _load_existing(output_path)
    done = set(saved_keys)
    remaining = [i for i in selected if all_keys[i] not in done]
    if args.limit is not None:
        remaining = remaining[: args.limit]
    print(f"To encode: {len(remaining):,} objects (already done: {len(done):,}).")
    if not remaining:
        print("Nothing to do.")
        _atomic_save(output_path, saved_keys, saved_embeddings)
        return

    cameras = _build_cameras(
        azimuths_deg=args.azimuths_deg, elevation_deg=args.elevation_deg,
        distance=args.cam_distance, fov_deg=args.fov_deg,
        image_size=_RENDER_SIZE, device=device, world_up_axis=args.world_up,
    )
    print(f"Cameras: V={_NUM_VIEWS} azim={args.azimuths_deg} elev={args.elevation_deg} "
          f"dist={args.cam_distance} fov={args.fov_deg} size={_RENDER_SIZE} up={args.world_up}")

    import gsplat as renderer_module
    encoder = _ClipImageEncoder(device, dtype)

    debug_render_dir = Path(args.debug_render_dir) if args.debug_render_dir else None
    if debug_render_dir is not None:
        debug_render_dir.mkdir(parents=True, exist_ok=True)
    debug_saved = 0

    objects_per_batch = max(1, args.clip_batch_size // _NUM_VIEWS)
    pending_keys: list[str] = []
    pending_imgs: list[torch.Tensor] = []

    new_keys: list[str] = list(saved_keys)
    new_embs: list[np.ndarray] = list(saved_embeddings)
    new_since_flush = 0
    t0 = time.time()

    def flush_batch() -> None:
        nonlocal new_since_flush
        if not pending_imgs:
            return
        imgs = torch.cat(pending_imgs, dim=0)             # (B*V, 3, H, W)
        feats = encoder.encode(imgs)                       # (B*V, 768) unit-norm per view
        feats = feats.view(len(pending_imgs), _NUM_VIEWS, _CLIP_DIM).mean(dim=1)  # avg views
        feats = F.normalize(feats, dim=-1)                 # renorm the averaged vector
        feats_np = feats.cpu().numpy().astype(np.float32)
        for k, vec in zip(pending_keys, feats_np):
            new_keys.append(k)
            new_embs.append(vec)
            new_since_flush += 1
        pending_keys.clear()
        pending_imgs.clear()

    pc_dataset = _PCDataset(
        indices=remaining, all_keys=all_keys, hash_keys=hash_keys,
        obj_data=obj_data, gs_root=gs_root,
    )
    loader = torch.utils.data.DataLoader(
        pc_dataset, batch_size=1, num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=_identity_collate, pin_memory=False,
    )
    print(f"DataLoader: num_workers={args.num_workers}  objects/batch={objects_per_batch}")

    pbar = tqdm(loader, total=len(remaining), desc="CLIP-encoding objects", unit="obj")
    for sample in pbar:
        key = sample["key"]
        if sample["point_cloud"] is None:
            print(f"\n[WARN] Failed to load key={key}: {sample['error']}. Skipping.")
            continue
        try:
            with torch.no_grad():
                rgb_views = _render_object_white_bg(
                    sample["point_cloud"], cameras, renderer_module, device)  # (V,3,H,W)
        except Exception as exc:
            print(f"\n[WARN] Render failed for key={key}: {exc!r}. Skipping.")
            continue

        if debug_render_dir is not None and debug_saved < args.debug_render_count:
            from PIL import Image
            grid = (rgb_views.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            strip = np.concatenate([g.transpose(1, 2, 0) for g in grid], axis=1)
            Image.fromarray(strip).save(debug_render_dir / f"{key.replace('/', '_')}.png")
            debug_saved += 1

        pending_keys.append(key)
        pending_imgs.append(rgb_views)
        if len(pending_imgs) >= objects_per_batch:
            flush_batch()
        if new_since_flush >= args.flush_every:
            _atomic_save(output_path, new_keys, new_embs)
            pbar.set_postfix_str(
                f"saved {len(new_keys):,} ({len(new_keys)-len(saved_keys):,} new "
                f"in {time.time()-t0:.0f}s)")
            new_since_flush = 0

    flush_batch()
    _atomic_save(output_path, new_keys, new_embs)
    print(f"\nDone. Saved {len(new_keys):,} embeddings to {output_path} "
          f"({len(new_keys)-len(saved_keys):,} new in {time.time()-t0:.0f}s).")
    print(f"  embeddings shape ({len(new_keys)}, {_CLIP_DIM}) float32; keys ({len(new_keys)},).")


if __name__ == "__main__":
    main()
