"""Stage 2: render TRELLIS .ply outputs through OUR gsplat pipeline + score CLIP alignment.

RUN UNDER .3dgen. Reuses the exact renderer/cameras/CLIP metric used for our own model
(jit/eval_clip_alignment.py), so the only difference vs our model's numbers is the model
that produced the Gaussians. Per-object normalization (center + isotropic fit to a target
radius, scaling the Gaussian scales with positions) neutralizes TRELLIS's different
canonical scale/centering so framing matches our cameras.

    python jit/eval_trellis_compare.py \
        --manifest output/trellis_compare/text_base_seed0/manifest.json \
        --ref_camera_tar artifacts/ref_camera.tar.gz \
        --n_views 4 --render_size 224 --target_radius 0.6 [--max_gpu_gb 40] [--limit N]

Writes clip_trellis.json (+ montage) next to the manifest. Score == our metric:
mean_v cos(CLIP_img(view_v), CLIP_txt(caption)) in ViT-L/14 joint space.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from jit import train_gsplat as T
from jit.eval_clip_alignment import load_clip, clip_text_features, clip_image_features, save_montage
from utils.gsplat_render_util import _point_clouds_to_gsplat_inputs, _render_gsplat_batch


# --------------------------------------------------------------------------------------
def load_trellis_ply(path: str):
    """Standard 3DGS .ply (TRELLIS, sh_degree 0) -> raw attrs matching our DC layout.

    Returns (xyz, opacity_raw, f_dc, scale_log, quat) as float32 numpy. All RAW
    (logit opacity, log scale, SH-DC color, unnormalized quats) — our renderer applies
    sigmoid/exp/normalize, so DO NOT pre-activate here.
    """
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    op = np.asarray(v["opacity"], dtype=np.float32)[:, None]
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    return xyz, op, fdc, scale, rot


def _quat_mul(a, b):  # (...,4) wxyz
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


_AXIS = {  # 3x3 applied to xyz (and as a rotation to the quats); world-frame alignment knobs
    "identity": np.eye(3, dtype=np.float32),
    "flip_y": np.diag([1, -1, 1]).astype(np.float32),
    "flip_z": np.diag([1, 1, -1]).astype(np.float32),
    "x_-90": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32),   # rotate -90 about X
    "x_+90": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
    "yz_swap": np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32),
}


def normalize_object(xyz, scale_log, rot, *, target_radius: float, pct: float, axis: str):
    """Center (median) + isotropic fit so the pct-th percentile radius == target_radius;
    scale the Gaussian log-scales with positions; optionally re-orient via an axis transform."""
    Rm = _AXIS[axis]
    if axis != "identity":
        xyz = xyz @ Rm.T
        # rotate quats so anisotropic splats stay consistent with rotated positions
        from numpy import trace
        t = trace(Rm)
        qw = np.sqrt(max(0.0, 1.0 + t)) / 2.0
        if qw > 1e-6:
            qx = (Rm[2, 1] - Rm[1, 2]) / (4 * qw)
            qy = (Rm[0, 2] - Rm[2, 0]) / (4 * qw)
            qz = (Rm[1, 0] - Rm[0, 1]) / (4 * qw)
            qrot = np.array([qw, qx, qy, qz], dtype=np.float32)
            rot = _quat_mul(qrot[None, :], rot)
    c = np.median(xyz, axis=0)
    xyz = xyz - c
    r = np.linalg.norm(xyz, axis=1)
    rad = float(np.percentile(r, pct))
    s = target_radius / max(rad, 1e-6)
    xyz = xyz * s
    scale_log = scale_log + np.log(s)
    return xyz.astype(np.float32), scale_log.astype(np.float32), rot.astype(np.float32)


@torch.no_grad()
def render_trellis_ply(path, *, target_radius, pct, axis, renderer, cams, cam_indices, device,
                       return_alpha=False):
    xyz, op, fdc, scale_log, rot = load_trellis_ply(path)
    xyz, scale_log, rot = normalize_object(
        xyz, scale_log, rot, target_radius=target_radius, pct=pct, axis=axis)
    pc14 = np.concatenate([xyz, op, fdc, scale_log, rot], axis=1)  # (N,14): xyz,op,dc,scale,quat
    pc = torch.from_numpy(pc14).float().unsqueeze(0).to(device)    # (1,N,14)
    gi = _point_clouds_to_gsplat_inputs(pc, dc_only=True, detach_input=True)
    return _render_gsplat_batch(renderer, gi, cams, cam_indices, device, return_alpha=return_alpha)  # (1,V,3,H,W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Stage-1 manifest.json")
    ap.add_argument("--ref_camera_tar", default=str(_REPO / "artifacts/ref_camera.tar.gz"))
    ap.add_argument("--n_views", type=int, default=4)
    ap.add_argument("--render_size", type=int, default=224)
    ap.add_argument("--target_radius", type=float, default=0.6, help="isotropic fit radius (frame match)")
    ap.add_argument("--pct", type=float, default=97.0, help="percentile radius used for the fit (robust to floaters)")
    ap.add_argument("--axis", default="identity", choices=list(_AXIS.keys()))
    ap.add_argument("--max_gpu_gb", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="default: clip_trellis.json next to manifest")
    ap.add_argument("--source", default="TRELLIS-text-base", help="model label for the output JSON / banner")
    ap.add_argument("--montage_name", default="montage_trellis.png", help="montage filename written next to manifest")
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.max_gpu_gb:
        idx = torch.cuda.current_device()
        tot = torch.cuda.get_device_properties(idx).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gb / tot), idx)
        print(f"[mem] cap {args.max_gpu_gb:.0f}GiB of {tot:.0f}GiB")

    man = json.load(open(args.manifest, encoding="utf-8"))
    if args.limit:
        man = man[:int(args.limit)]
    out_dir = os.path.dirname(os.path.abspath(args.manifest))
    out_json = args.out or os.path.join(out_dir, "clip_trellis.json")

    # Cameras + renderer + CLIP — identical to jit/eval_clip_alignment.
    ref = T._load_reference_cameras(args.ref_camera_tar)
    cams = T._prepare_train_cameras(ref, args.render_size, device)
    renderer = T._try_import_renderer()
    assert not isinstance(renderer, Exception), f"renderer import failed: {renderer}"
    ncam = int(cams["viewmats"].shape[0])
    if args.n_views >= ncam:
        cam_indices = list(range(ncam))
    else:
        cam_indices = sorted({int(round(x)) for x in np.linspace(0, ncam - 1, args.n_views)})
    print(f"[setup] {len(man)} objs | views={cam_indices} render={args.render_size}px "
          f"target_radius={args.target_radius} pct={args.pct} axis={args.axis}")

    clip, tok, cmean, cstd = load_clip(device)
    prompts = [m["caption"] for m in man]
    text_feats = clip_text_features(clip, tok, prompts, device)  # (P,768)

    scores, per = np.zeros(len(man)), []
    montage_tiles, montage_scores = [], []
    t0 = time.time()
    for i, m in enumerate(man):
        rendered = render_trellis_ply(
            m["ply"], target_radius=args.target_radius, pct=args.pct, axis=args.axis,
            renderer=renderer, cams=cams, cam_indices=cam_indices, device=device)  # (1,V,3,H,W)
        V = rendered.shape[1]
        flat = rendered.reshape(V, 3, args.render_size, args.render_size)
        img = clip_image_features(clip, cmean, cstd, flat, device).mean(dim=0, keepdim=True)  # (1,768)
        sc = float((img * text_feats[i:i + 1]).sum(-1).item())
        scores[i] = sc
        per.append({"idx": m["idx"], "caption": m["caption"], "score": sc})
        if i < 16:
            montage_tiles.append(rendered[0, 0].cpu())
            montage_scores.append(sc)
        if (i + 1) % 10 == 0 or i == len(man) - 1:
            print(f"[{i+1}/{len(man)}] mean so far {scores[:i+1].mean():.4f} ({time.time()-t0:.0f}s)", flush=True)

    payload = {
        "source": args.source, "manifest": os.path.abspath(args.manifest),
        "n_objects": len(man), "n_views": len(cam_indices), "cam_indices": cam_indices,
        "render_size": args.render_size, "target_radius": args.target_radius, "pct": args.pct,
        "axis": args.axis, "clip_model": "openai/clip-vit-large-patch14",
        "score_def": "mean_v cos(CLIP_img(view_v), CLIP_txt(caption))",
        "mean": float(scores.mean()), "std": float(scores.std()),
        "sem": float(scores.std() / np.sqrt(len(scores))),
        "per_object": per,
    }
    json.dump(payload, open(out_json, "w"), indent=2)
    montage_path = os.path.join(out_dir, args.montage_name)
    if montage_tiles:
        save_montage(torch.stack(montage_tiles), [m["caption"] for m in man[:len(montage_tiles)]],
                     montage_scores, montage_path)
    print(f"\n==== {args.source} CLIP alignment ({len(man)} objs) ====")
    print(f"mean={payload['mean']:.4f}  std={payload['std']:.4f}  sem={payload['sem']:.4f}")
    print(f"saved: {out_json}\nmontage: {montage_path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
