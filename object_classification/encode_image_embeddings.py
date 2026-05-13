"""
Encode each object as a multi-view image embedding for clustering analysis.

For every object in the dataset we:
  1. Load its 16,384-point 3DGS representation via Standard3DGenDataset.
  2. Render 4 views with gsplat at azimuths {0, 90, 180, 270}, elevation 30,
     a fixed camera distance, 224x224, on a white background.
  3. Pass each view through DINOv2-base (facebook/dinov2-base) and read the
     CLS token from last_hidden_state.
  4. L2-normalize each per-view CLS vector and concatenate them into a single
     feature vector of shape (4 * 768,) = (3072,) per object.

Output mirrors object_classification/encode_captions.py: a single .npz
containing 'embeddings' (N, 3072) float32 and 'keys' (N,) object-dtype, where
each key is the captions.json-style stem "<chunk>/<filename>" (i.e. the same
string that encode_captions.py stores so the two embedding sets can be joined
on key downstream).

Renderings are NEVER cached to disk -- they are produced in-memory per object
and discarded as soon as DINOv2 has consumed them. Embeddings, on the other
hand, ARE cached: the output .npz is rewritten atomically every --flush-every
objects, and on startup any keys already present in it are skipped, so the
script is fully resumable.

Coordinate convention
---------------------
GaussianVerse PLY files are stored Y-down (OpenCV-style). The interactive
viewer (viewer/index.html) confirms this by applying a 180-deg rotation
around X to the splat before display. The default --world-up=y_down places
the camera elevation along data -Y and azim=0 along data -Z (the canonical
front view). Use --debug-render-dir to dump PNGs of the first few renders
and visually verify before running on the full ~250K dataset.

Usage examples
--------------
# Defaults pulled from .env (GS_PATH, ALL_OBJ_JSON):
python object_classification/encode_image_embeddings.py

# Visual sanity check on 4 objects (saves PNGs, then exits via --limit):
python object_classification/encode_image_embeddings.py \
    --limit 4 --debug-render-dir object_classification/debug_renders

# Explicit paths and a smaller test run:
python object_classification/encode_image_embeddings.py \
    --obj-list /path/to/all_obj_list.json \
    --gs-path /path/to/3dgs \
    --output object_classification/image_embeddings.npz \
    --limit 1000

# Restrict to a subset and use a different DINOv2 batch size:
python object_classification/encode_image_embeddings.py \
    --filter-list downloaded/aesthetic_chunk/filtered_aesthetic_list.json \
    --dino-batch-size 32
"""

import argparse
import math
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

from dataloaders.standard_3dgen_loader import (  # noqa: E402
    extract_directory_info,
    load_obj_list,
    load_ply,
)


# ---------------------------------------------------------------------------
# Vendored gsplat helpers
# ---------------------------------------------------------------------------
# These are inlined from utils/gsplat_render_util.py because that module is
# not tracked on every branch (e.g. class_embedding has only utils/__pycache__).
# Behaviour matches the canonical implementation: opacities pass through a
# clamped sigmoid, scales through a clamped exp, and quaternions are L2-
# normalized with an identity fallback for near-zero norms.
_RENDER_OPACITY_RAW_MIN = -12.0
_RENDER_OPACITY_RAW_MAX = 12.0
_RENDER_SCALE_RAW_MIN = -12.0
_RENDER_SCALE_RAW_MAX = 8.0
_RENDER_QUAT_EPS = 1e-8


def _normalize_quaternions_with_identity_fallback(rotations_raw: torch.Tensor) -> torch.Tensor:
    quat_norms = rotations_raw.norm(dim=-1, keepdim=True)
    quats = rotations_raw / quat_norms.clamp_min(_RENDER_QUAT_EPS)
    identity_quat = torch.zeros_like(quats)
    identity_quat[..., 0] = 1.0
    return torch.where(quat_norms > _RENDER_QUAT_EPS, quats, identity_quat)


def _constrain_denormalized_point_cloud_for_render(
    point_cloud: torch.Tensor,
    *,
    dc_only: bool = False,
) -> torch.Tensor:
    if dc_only:
        scale_slice = slice(7, 10)
        rotation_slice = slice(10, 14)
    else:
        scale_slice = slice(52, 55)
        rotation_slice = slice(55, 59)

    safe_pc = torch.nan_to_num(point_cloud, nan=0.0, posinf=0.0, neginf=0.0)
    xyz = safe_pc[..., :3]
    opacity = torch.sigmoid(
        safe_pc[..., 3:4].clamp(_RENDER_OPACITY_RAW_MIN, _RENDER_OPACITY_RAW_MAX)
    )
    middle = safe_pc[..., 4:scale_slice.start]
    scales = torch.exp(
        safe_pc[..., scale_slice].clamp(_RENDER_SCALE_RAW_MIN, _RENDER_SCALE_RAW_MAX)
    )
    rotations = _normalize_quaternions_with_identity_fallback(safe_pc[..., rotation_slice])
    tail = safe_pc[..., rotation_slice.stop:]
    return torch.cat((xyz, opacity, middle, scales, rotations, tail), dim=-1)


def _point_clouds_to_gsplat_inputs(
    point_clouds: torch.Tensor,
    *,
    dc_only: bool = False,
    detach_input: bool = False,
) -> dict:
    if detach_input:
        point_clouds = point_clouds.detach()
    if point_clouds.ndim == 2:
        point_clouds = point_clouds.unsqueeze(0)
    if point_clouds.ndim != 3:
        raise ValueError(f"Expected (B, N, D) point clouds, got {tuple(point_clouds.shape)}")

    feature_dim = point_clouds.shape[-1]
    expected_dim = 14 if dc_only else 59
    if dc_only and feature_dim != expected_dim:
        raise ValueError(f"Expected {expected_dim} DC-only features, got {feature_dim}")
    if not dc_only and feature_dim < expected_dim:
        raise ValueError(f"Expected at least {expected_dim} full features, got {feature_dim}")

    pc = _constrain_denormalized_point_cloud_for_render(
        point_clouds[..., :expected_dim].to(dtype=torch.float32).contiguous(),
        dc_only=dc_only,
    )
    means = pc[..., 0:3]
    if dc_only:
        colors = pc[..., 4:7].unsqueeze(-2).contiguous()
        scales = pc[..., 7:10]
        quats = pc[..., 10:14]
        sh_degree = 0
    else:
        features = pc[..., 4:52]
        sh_coeffs = features.reshape(*pc.shape[:-1], 3, 16)
        features_dc = sh_coeffs[..., 0].unsqueeze(-2)
        features_rest = sh_coeffs[..., 1:].transpose(-1, -2)
        colors = torch.cat((features_dc, features_rest), dim=-2).contiguous()
        scales = pc[..., 52:55]
        quats = pc[..., 55:59]
        sh_degree = 3

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": pc[..., 3],
        "colors": colors,
        "sh_degree": sh_degree,
    }


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_OUTPUT = _SCRIPT_DIR / "image_embeddings.npz"
_DINOV2_HF_ID = "facebook/dinov2-base"
_DINOV2_DIM = 768          # base model hidden size
_NUM_VIEWS = 4
_FEATURE_DIM = _NUM_VIEWS * _DINOV2_DIM  # 3072
_RENDER_SIZE = 224

# Synthetic camera config. These are reasonable Objaverse-style defaults; the
# CLI exposes them so callers can sweep if needed.
_DEFAULT_AZIMUTHS_DEG = (0.0, 90.0, 180.0, 270.0)
_DEFAULT_ELEVATION_DEG = 30.0
_DEFAULT_CAM_DISTANCE = 1.6  # tuned so a unit-scale object fills ~70% of frame at the default FOV
_DEFAULT_FOV_DEG = 49.1


def _env_path(name: str, fallback: "str | None" = None) -> "str | None":
    """Read an env var and apply shell-style $VAR expansion. The repo's .env
    contains shell-only lines (e.g. an `if ... export PATH ...`) that cause
    python-dotenv to bail mid-file, so values like ALL_OBJ_JSON=$GS_PATH/...
    can land in os.environ unexpanded. expandvars resolves them against the
    current process environment."""
    val = os.environ.get(name)
    if val is None:
        return fallback
    return os.path.expandvars(val)


def parse_args() -> argparse.Namespace:
    load_dotenv(_REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Encode each object as a 4-view DINOv2 image embedding."
    )
    parser.add_argument(
        "--obj-list",
        default=_env_path("ALL_OBJ_JSON"),
        help="Path to obj_list JSON. Defaults to $ALL_OBJ_JSON from .env.",
    )
    parser.add_argument(
        "--gs-path",
        default=_env_path("GS_PATH"),
        help="Root path containing the unzipped 3DGS chunks. Defaults to $GS_PATH from .env.",
    )
    parser.add_argument(
        "--filter-list",
        default=None,
        help=(
            "Optional path to filtered_aesthetic_list.json (or any obj_list-shaped JSON). "
            "When provided, only objects whose stem appears in this list are encoded."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help=f"Destination .npz file (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. 'cuda', 'cpu'. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--dino-batch-size",
        type=int,
        default=64,
        help="Images per DINOv2 forward pass. 4 views per object, so the effective "
             "object throughput is dino-batch-size / 4.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=500,
        help="Rewrite the output .npz atomically every N newly encoded objects.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of objects processed (after filtering and skip-already-done). "
             "Useful for smoke-testing.",
    )
    parser.add_argument(
        "--azimuths-deg",
        type=float,
        nargs="+",
        default=list(_DEFAULT_AZIMUTHS_DEG),
        help=f"Azimuth angles in degrees for the 4 views (default: {list(_DEFAULT_AZIMUTHS_DEG)}).",
    )
    parser.add_argument(
        "--elevation-deg",
        type=float,
        default=_DEFAULT_ELEVATION_DEG,
        help=f"Elevation angle in degrees (default: {_DEFAULT_ELEVATION_DEG}).",
    )
    parser.add_argument(
        "--cam-distance",
        type=float,
        default=_DEFAULT_CAM_DISTANCE,
        help=f"Distance from origin to camera (default: {_DEFAULT_CAM_DISTANCE}).",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=_DEFAULT_FOV_DEG,
        help=f"Camera vertical FOV in degrees (default: {_DEFAULT_FOV_DEG}).",
    )
    parser.add_argument(
        "--world-up",
        choices=["y_down", "y_up"],
        default="y_down",
        help=(
            "Up-axis convention of the raw 3DGS data. The viewer (viewer/index.html) "
            "applies a 180-deg X-rotation to the splat for display, which means the "
            "GaussianVerse PLY files are stored Y-down (OpenCV-style). 'y_down' "
            "(default) places the camera elevation along -Y in data coords; switch to "
            "'y_up' if visual inspection shows objects rendered upside-down."
        ),
    )
    parser.add_argument(
        "--debug-render-dir",
        default=None,
        help=(
            "If set, save the 4 white-bg renders for the first --debug-render-count "
            "objects as PNGs into this directory. Use this to validate the camera "
            "convention before kicking off the full encode."
        ),
    )
    parser.add_argument(
        "--debug-render-count",
        type=int,
        default=4,
        help="Number of objects to save debug renders for (default: 4).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for prefetching PLY files. PLY parsing is the "
             "main CPU bottleneck; raise this until GPU stays >80%% busy.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="0-based shard index for multi-GPU runs. Each shard processes "
             "indices where idx %% num_shards == shard_id. Combine with "
             "CUDA_VISIBLE_DEVICES and a per-shard --output to fan across GPUs.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards (default 1 = no sharding).",
    )
    args = parser.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        parser.error("--shard-id must be in [0, --num-shards).")
    if not args.obj_list:
        parser.error("--obj-list is required (or set ALL_OBJ_JSON in .env).")
    if not args.gs_path:
        parser.error("--gs-path is required (or set GS_PATH in .env).")
    if len(args.azimuths_deg) != _NUM_VIEWS:
        parser.error(f"--azimuths-deg must have exactly {_NUM_VIEWS} entries.")
    return args


# ---------------------------------------------------------------------------
# Camera construction
# ---------------------------------------------------------------------------

def _look_at_c2w(cam_pos: np.ndarray, target: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """Build a 4x4 camera-to-world matrix in gsplat's OpenCV convention:
    cam +X = right, cam +Y = down, cam +Z = forward (into the scene). This
    is right-handed and matches w2c = inv(c2w) viewmats that gsplat expects.
    Note: an OpenGL-style frame (cam +Y = up) renders correctly-positioned
    cameras with the image flipped vertically, which looks identical to
    a from-below view."""
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-12)
    down = np.cross(forward, right)  # cam +Y direction in world coords

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = cam_pos
    return c2w


def _build_cameras(
    azimuths_deg: list[float],
    elevation_deg: float,
    distance: float,
    fov_deg: float,
    image_size: int,
    device: torch.device,
    world_up_axis: str = "y_down",
) -> dict:
    """Construct a (V, 4, 4) viewmat stack and a (V, 3, 3) intrinsics stack
    suitable for gsplat.rasterization.

    With world_up_axis='y_down' (the GaussianVerse default per the viewer's
    180-deg X-flip), the up direction in data coords is (0, -1, 0), and the
    canonical 'front' view (azim=0, elev=0) is placed along data -Z.
    """
    elev = math.radians(elevation_deg)
    fov = math.radians(fov_deg)
    focal = image_size / (2.0 * math.tan(fov / 2.0))
    K = np.array(
        [
            [focal, 0.0, image_size * 0.5],
            [0.0, focal, image_size * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    target = np.zeros(3, dtype=np.float32)
    if world_up_axis == "y_down":
        world_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        up_sign = -1.0   # cam_y = -R*sin(elev) so +elev moves cam in +world_up
        front_z_sign = -1.0  # azim=0 places camera at data -Z
    elif world_up_axis == "y_up":
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up_sign = 1.0
        front_z_sign = 1.0
    else:
        raise ValueError(f"Unknown world_up_axis: {world_up_axis}")

    viewmats = []
    intrinsics = []
    for azim_deg in azimuths_deg:
        azim = math.radians(azim_deg)
        cam_pos = np.array(
            [
                distance * math.cos(elev) * math.sin(azim),
                up_sign * distance * math.sin(elev),
                front_z_sign * distance * math.cos(elev) * math.cos(azim),
            ],
            dtype=np.float32,
        )
        c2w = _look_at_c2w(cam_pos, target, world_up)
        w2c = np.linalg.inv(c2w).astype(np.float32)
        viewmats.append(torch.from_numpy(w2c))
        intrinsics.append(torch.from_numpy(K.copy()))

    return {
        "viewmats": torch.stack(viewmats, dim=0).to(device),   # (V, 4, 4)
        "Ks": torch.stack(intrinsics, dim=0).to(device),       # (V, 3, 3)
        "width": int(image_size),
        "height": int(image_size),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_object_white_bg(
    point_cloud_raw: torch.Tensor,
    cameras: dict,
    renderer_module,
    device: torch.device,
) -> torch.Tensor:
    """Render one object at all cameras with a white background.

    Args:
        point_cloud_raw: (16384, 59) raw (un-normalized) point cloud as loaded
            from the PLY (after gs2sphere reordering).

    Returns:
        Tensor of shape (V, 3, H, W) in [0, 1] on `device`, white-composited.
    """
    pc = point_cloud_raw.to(device=device, dtype=torch.float32).unsqueeze(0)  # (1, N, 59)
    gs = _point_clouds_to_gsplat_inputs(pc, dc_only=False, detach_input=True)

    V = int(cameras["viewmats"].shape[0])
    viewmats = cameras["viewmats"].unsqueeze(0)         # (1, V, 4, 4)
    Ks = cameras["Ks"].unsqueeze(0)                     # (1, V, 3, 3)
    backgrounds = torch.zeros((1, V, 3), dtype=torch.float32, device=device)

    renders, alphas, _ = renderer_module.rasterization(
        means=gs["means"],
        quats=gs["quats"],
        scales=gs["scales"],
        opacities=gs["opacities"],
        colors=gs["colors"],
        viewmats=viewmats,
        Ks=Ks,
        width=cameras["width"],
        height=cameras["height"],
        sh_degree=gs["sh_degree"],
        backgrounds=backgrounds,
        packed=False,
        render_mode="RGB",
    )
    # renders: (1, V, H, W, 3) in [0, 1] (with black bg)
    # alphas:  (1, V, H, W, 1)
    rgb = renders.permute(0, 1, 4, 2, 3).clamp(0.0, 1.0)   # (1, V, 3, H, W)
    alpha = alphas.permute(0, 1, 4, 2, 3).clamp(0.0, 1.0)  # (1, V, 1, H, W)
    # rgb was rendered onto black, so rgb = alpha * fg. Composite onto white:
    rgb_white = rgb + (1.0 - alpha)
    rgb_white = rgb_white.clamp(0.0, 1.0)
    return rgb_white[0]  # (V, 3, H, W)


# ---------------------------------------------------------------------------
# DINOv2 wrapper
# ---------------------------------------------------------------------------

class _DinoV2Encoder:
    def __init__(self, device: torch.device):
        from transformers import AutoModel
        print(f"Loading {_DINOV2_HF_ID} …")
        self.model = AutoModel.from_pretrained(_DINOV2_HF_ID).to(device).eval()
        self.device = device
        # ImageNet normalization (DINOv2 was trained with these stats).
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) in [0, 1] on self.device. Returns (B, 768)."""
        if images.shape[-1] != _RENDER_SIZE or images.shape[-2] != _RENDER_SIZE:
            images = F.interpolate(
                images, size=(_RENDER_SIZE, _RENDER_SIZE), mode="bilinear", align_corners=False
            )
        x = (images - self.mean) / self.std
        out = self.model(pixel_values=x)
        cls = out.last_hidden_state[:, 0, :]  # (B, 768)
        return cls.float()


# ---------------------------------------------------------------------------
# Output / resume helpers
# ---------------------------------------------------------------------------

def _load_existing(output_path: Path) -> tuple[list[str], list[np.ndarray]]:
    if not output_path.exists():
        return [], []
    print(f"Resuming from existing output {output_path} …")
    data = np.load(output_path, allow_pickle=True)
    keys = data["keys"].tolist()
    embeddings = data["embeddings"]
    if embeddings.shape[0] != len(keys):
        raise ValueError(
            f"Corrupt cache: embeddings ({embeddings.shape[0]}) and keys ({len(keys)}) "
            f"have different lengths in {output_path}"
        )
    if embeddings.shape[1] != _FEATURE_DIM:
        raise ValueError(
            f"Cache feature dim {embeddings.shape[1]} != expected {_FEATURE_DIM} in {output_path}. "
            f"Delete or move the file before re-running."
        )
    print(f"  Found {len(keys):,} previously encoded objects (skipping those).")
    return keys, list(embeddings.astype(np.float32))


def _atomic_save(output_path: Path, keys: list[str], embeddings: list[np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.stack(embeddings, axis=0).astype(np.float32) if embeddings else np.zeros((0, _FEATURE_DIM), dtype=np.float32)
    # np.savez auto-appends '.npz' when given a string path without that suffix,
    # which would silently rewrite our temp filename. Passing a file handle
    # bypasses that quirk.
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(
            f,
            embeddings=arr,
            keys=np.array(keys, dtype=object),
        )
    os.replace(tmp, output_path)


class _PCDataset(torch.utils.data.Dataset):
    """Streams (key, point_cloud) tuples by reading PLY + gs2sphere on CPU
    workers. Bottleneck is plyfile parsing -- multi-worker loading is what
    keeps the GPU fed."""

    def __init__(
        self,
        indices: list[int],
        all_keys: list[str],
        hash_keys: list[str],
        obj_data: dict,
        gs_root: Path,
    ):
        self.indices = indices
        self.all_keys = all_keys
        self.hash_keys = hash_keys
        self.obj_data = obj_data
        self.gs_root = gs_root

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        key = self.all_keys[idx]
        tar_gz_path = self.obj_data[self.hash_keys[idx]]
        chunk, filename = extract_directory_info(tar_gz_path)
        data_dir = self.gs_root / chunk / filename
        try:
            gs2sphere = np.load(data_dir / "gs2sphere.npy")
            point_cloud = load_ply(str(data_dir / "point_cloud.ply"))[gs2sphere]
        except Exception as exc:
            return {"key": key, "point_cloud": None, "error": repr(exc)}
        return {
            "key": key,
            "point_cloud": torch.from_numpy(point_cloud),  # (16384, 59) float32
            "error": None,
        }


def _identity_collate(batch):
    """Workers return one sample per call; we consume them one-by-one."""
    return batch[0]


def _load_filter_keys(filter_list_path: str) -> set[str]:
    """Mirror encode_captions.load_filter_keys: read a JSON dict whose values are
    'chunk/filename.tar.gz' and return the bare 'chunk/filename' stems."""
    import json
    with open(filter_list_path, "r", encoding="utf-8") as f:
        obj_list: dict = json.load(f)
    return {tar_path.removesuffix(".tar.gz") for tar_path in obj_list.values()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Object list -----------------------------------------------------
    # We bypass Standard3DGenDataset because it requires a captions file we
    # don't need, and it normalizes the PC only for us to denormalize it
    # again. Direct PLY load via the helper functions is simpler.
    print(f"Loading obj list from {args.obj_list}  (gs_path={args.gs_path})")
    obj_data = load_obj_list([args.obj_list])
    hash_keys = list(obj_data.keys())
    # caption-style stem "chunk/filename" -- matches encode_captions.py keys
    all_keys = [obj_data[h].removesuffix(".tar.gz") for h in hash_keys]
    gs_root = Path(args.gs_path)

    # --- Filter ----------------------------------------------------------
    selected: list[int] = list(range(len(hash_keys)))
    if args.filter_list:
        keep = _load_filter_keys(args.filter_list)
        selected = [i for i in selected if all_keys[i] in keep]
        print(f"  Filtered {len(hash_keys):,} -> {len(selected):,} via {args.filter_list}")

    # --- Sharding --------------------------------------------------------
    if args.num_shards > 1:
        before = len(selected)
        selected = [i for i in selected if i % args.num_shards == args.shard_id]
        print(
            f"Shard {args.shard_id}/{args.num_shards}: {before:,} -> {len(selected):,} "
            f"objects assigned to this shard."
        )

    # --- Resume ----------------------------------------------------------
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

    # --- Build cameras --------------------------------------------------
    cameras = _build_cameras(
        azimuths_deg=args.azimuths_deg,
        elevation_deg=args.elevation_deg,
        distance=args.cam_distance,
        fov_deg=args.fov_deg,
        image_size=_RENDER_SIZE,
        device=device,
        world_up_axis=args.world_up,
    )
    print(
        f"Cameras: V={_NUM_VIEWS}  azim={args.azimuths_deg}  elev={args.elevation_deg}deg  "
        f"distance={args.cam_distance}  fov={args.fov_deg}deg  size={_RENDER_SIZE}  "
        f"world_up={args.world_up}"
    )

    # --- Renderer + DINOv2 ----------------------------------------------
    import gsplat as renderer_module
    encoder = _DinoV2Encoder(device)

    debug_render_dir = Path(args.debug_render_dir) if args.debug_render_dir else None
    if debug_render_dir is not None:
        debug_render_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Debug renders will be saved for the first {args.debug_render_count} "
            f"objects in: {debug_render_dir}"
        )
    debug_saved = 0

    # Buffers for batched DINOv2 forward passes.
    objects_per_dino_batch = max(1, args.dino_batch_size // _NUM_VIEWS)
    pending_keys: list[str] = []
    pending_imgs: list[torch.Tensor] = []  # each (V, 3, H, W) on device

    new_keys: list[str] = list(saved_keys)
    new_embs: list[np.ndarray] = list(saved_embeddings)
    new_since_flush = 0
    t0 = time.time()

    def flush_dino_batch() -> None:
        nonlocal new_since_flush
        if not pending_imgs:
            return
        imgs = torch.cat(pending_imgs, dim=0)        # (B*V, 3, H, W)
        feats = encoder.encode(imgs)                  # (B*V, 768)
        feats = F.normalize(feats, dim=-1)            # L2 normalize each view
        feats = feats.view(len(pending_imgs), _NUM_VIEWS * _DINOV2_DIM)  # concat 4 views
        feats_np = feats.cpu().numpy().astype(np.float32)
        for k, vec in zip(pending_keys, feats_np):
            new_keys.append(k)
            new_embs.append(vec)
            new_since_flush += 1
        pending_keys.clear()
        pending_imgs.clear()

    pc_dataset = _PCDataset(
        indices=remaining,
        all_keys=all_keys,
        hash_keys=hash_keys,
        obj_data=obj_data,
        gs_root=gs_root,
    )
    loader = torch.utils.data.DataLoader(
        pc_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=_identity_collate,
        pin_memory=False,  # PCs are small; pinning adds copy overhead
    )
    print(f"DataLoader: num_workers={args.num_workers}")

    pbar = tqdm(loader, total=len(remaining), desc="Encoding objects", unit="obj")
    for sample in pbar:
        key = sample["key"]
        if sample["point_cloud"] is None:
            print(f"\n[WARN] Failed to load key={key}: {sample['error']}. Skipping.")
            continue
        pc_raw = sample["point_cloud"]  # (16384, 59) torch.float32, on CPU
        try:
            with torch.no_grad():
                rgb_views = _render_object_white_bg(
                    pc_raw, cameras, renderer_module, device
                )  # (V, 3, H, W) in [0,1]
        except Exception as exc:
            print(f"\n[WARN] Render failed for key={key}: {exc!r}. Skipping.")
            continue

        if debug_render_dir is not None and debug_saved < args.debug_render_count:
            from PIL import Image
            grid = (rgb_views.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            # (V, 3, H, W) -> single horizontal strip (H, V*W, 3)
            strip = np.concatenate([g.transpose(1, 2, 0) for g in grid], axis=1)
            safe_name = key.replace("/", "_")
            Image.fromarray(strip).save(debug_render_dir / f"{safe_name}.png")
            debug_saved += 1

        pending_keys.append(key)
        pending_imgs.append(rgb_views)

        if len(pending_imgs) >= objects_per_dino_batch:
            flush_dino_batch()

        if new_since_flush >= args.flush_every:
            _atomic_save(output_path, new_keys, new_embs)
            elapsed = time.time() - t0
            pbar.set_postfix_str(
                f"saved {len(new_keys):,}  ({len(new_keys) - len(saved_keys):,} new in {elapsed:.0f}s)"
            )
            new_since_flush = 0

    # Drain final partial batch and flush.
    flush_dino_batch()
    _atomic_save(output_path, new_keys, new_embs)

    elapsed = time.time() - t0
    print(
        f"\nDone. Saved {len(new_keys):,} embeddings to {output_path}  "
        f"({len(new_keys) - len(saved_keys):,} new in {elapsed:.0f}s)."
    )
    print("  Arrays stored:")
    print(f"    embeddings  -- shape ({len(new_keys)}, {_FEATURE_DIM}), dtype float32")
    print(f"    keys        -- shape ({len(new_keys)},), dtype object")
    print(f"\nLoad with:  data = np.load('{output_path}', allow_pickle=True)")


if __name__ == "__main__":
    main()
