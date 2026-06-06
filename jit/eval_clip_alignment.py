"""Evaluate CLIP text-image alignment of generated 3DGS objects from a JiT checkpoint.

For each of the run's validation prompts we sample ``n_seeds`` objects (deterministic
per-(prompt, seed) noise), render each from ``n_views`` cameras, and compute the
multi-view CLIP score

    score(prompt, seed) = mean_v  cos( CLIP_img(view_v) , CLIP_txt(prompt) )

in the CLIP ViT-L/14 *joint* embedding space (get_image_features / get_text_features +
final projection) — NOT the penultimate text token used for AdaLN conditioning.
Because each per-view image feature is L2-normalized, averaging the view features and
dotting with the unit text feature equals the mean of the per-view cosine similarities.

The generation + render path reuses the exact training/validation helpers
(`jit.train_gsplat` -> `utils.gsplat_render_util`): EMA weights, sphere2plane atlas
inversion, Gaussian rank-transform inverse, per-channel denorm, gsplat rasterization on
a black background at `train_render_size`. So renders match the run's own
`dit_validation` grids; seed-round 0 reproduces the validation-grid noise exactly.

Usage (shares the GPU with a live training job, like jit/ab_x0renoise_vs_heun.py):

    python jit/eval_clip_alignment.py --ckpt output/<run>/0063000.pt \
        --n_seeds 4 --n_views 4 --cfg_scales 1,2,3,5
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jit import train_gsplat as T
from jit.sampling import sample_model

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


def _load_caption_prompts(path: str) -> list[str]:
    """Load caption strings from a JSON file (e.g. the Toys4k validation set).

    Accepts, in order of attempt:
      - a JSON list of strings:               ["a red toy car", ...]
      - a JSON dict {id: caption}:            {"toy_0001": "a red toy car", ...}
        (iterated in sorted-key order for reproducibility)
      - a JSON list of objects, each with a   [{"caption": "..."}, ...]
        "caption"/"text"/"prompt"/"description" string field.

    Returns a flat list[str]; raises a clear error on anything else.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _coerce(v):
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            for k in ("caption", "text", "prompt", "description", "desc"):
                if isinstance(v.get(k), str):
                    return v[k]
        return None

    if isinstance(data, dict):
        items = [data[k] for k in sorted(data.keys())]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"{path}: expected a JSON list or dict, got {type(data).__name__}")

    prompts = [_coerce(v) for v in items]
    if not prompts:
        raise ValueError(f"{path}: no captions found.")
    if any(p is None for p in prompts):
        bad = next(i for i, p in enumerate(prompts) if p is None)
        raise ValueError(
            f"{path}: could not extract a caption string from item {bad} "
            f"({items[bad]!r}); expected a str or a dict with a "
            "caption/text/prompt/description field."
        )
    return prompts


# --------------------------------------------------------------------------------------
# Build model + render pipeline from a checkpoint (mirrors jit/ab_x0renoise_vs_heun.build)
# --------------------------------------------------------------------------------------
def build(ckpt_path: str, device: torch.device, render_size: int | None = None,
          encode_device: torch.device | None = None, max_prompts: int | None = None,
          prompts_file: str | None = None, shuffle_seed: int | None = None) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = vars(ck["args"]) if hasattr(ck["args"], "__dict__") else ck["args"]
    assert cfg["sh_degree0_only"], "this eval assumes the DC-only (14ch) config"
    in_channels = len(T.DC_ONLY_FEATURE_INDICES)
    feature_indices = torch.tensor(T.DC_ONLY_FEATURE_INDICES, dtype=torch.long)

    # Prompt source priority: explicit --prompts_file (e.g. Toys4k captions) >
    # the run's own val_prompts_file > the built-in 64-prompt pool.
    if prompts_file:
        prompts = _load_caption_prompts(prompts_file)
    elif cfg.get("val_prompts_file"):
        with open(cfg["val_prompts_file"], "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if not (isinstance(prompts, list) and all(isinstance(p, str) for p in prompts)):
            raise ValueError(f"{cfg['val_prompts_file']} must be a JSON list of strings")
    else:
        prompts = list(T._DEFAULT_VAL_PROMPTS)
    # Optional deterministic subsample of a large validation set (shuffle, then cap).
    if shuffle_seed is not None:
        perm = np.random.default_rng(int(shuffle_seed)).permutation(len(prompts))
        prompts = [prompts[i] for i in perm]
    if max_prompts:
        prompts = prompts[:int(max_prompts)]

    # Conditioning embeddings: penultimate+LN+EOS pool (matches training/offline encoder).
    # encode_device='cpu' keeps the transient CLIPTextModel off the GPU (memory-safe when a
    # heavy training job is co-resident); cond_pool is moved to the GPU per sub-batch downstream.
    enc_dev = encode_device if encode_device is not None else device
    cond_pool = T._encode_clip_penultimate(prompts, enc_dev)  # (P, text_dim)
    text_dim = int(cond_pool.shape[1])

    model = T.JiT_3DGS_models[cfg["model"]](
        input_size=128, in_channels=in_channels, text_dim=text_dim,
        class_dropout_prob=float(cfg["class_dropout_prob"]),
        learn_sigma=False, bottleneck=bool(cfg["bottleneck"]),
    )
    null_path = cfg["null_text_token_path"] or T._default_null_path(cfg["text_embed_path"])
    model.load_null_embeddings(torch.from_numpy(T.load_null_text_token(null_path).astype(np.float32)))
    missing, unexpected = model.load_state_dict(ck["ema"], strict=False)
    if missing or unexpected:
        print(f"[load] missing={list(missing)} unexpected={list(unexpected)}")
    model.to(device).eval()

    plane_to_sphere = T.load_sphere2plane(cfg["sphere2plane_path"], 128 * 128)
    rank_tables = T.load_rank_transform_payload_torch(cfg["rank_transform_file"], device=device)

    mean_full = torch.load(cfg["mean_file"], weights_only=True).float().cpu()
    std_full = torch.load(cfg["std_file"], weights_only=True).float().cpu()
    if rank_tables is not None:
        # Channels carried by the rank transform are stored standard-normal already;
        # neutralize their mean/std so denorm is an identity for them (see ab driver).
        ridx = torch.tensor(rank_tables["channels"], dtype=torch.long)
        mean_full = mean_full.clone(); std_full = std_full.clone()
        mean_full[ridx] = 0.0; std_full[ridx] = 1.0
    norm_mean = mean_full[feature_indices]
    norm_std = std_full[feature_indices]

    ref_cameras = T._load_reference_cameras(cfg["ref_camera_tar"])
    renderer = T._try_import_renderer()
    assert not isinstance(renderer, Exception), f"renderer import failed: {renderer}"
    rsize = int(render_size or cfg["train_render_size"])
    train_cameras = T._prepare_train_cameras(ref_cameras, rsize, device)

    return dict(
        model=model, cfg=cfg, in_channels=in_channels, prompts=prompts,
        cond_pool=cond_pool, plane_to_sphere=plane_to_sphere,
        norm_mean=norm_mean, norm_std=norm_std, rank_tables=rank_tables,
        train_cameras=train_cameras, renderer=renderer, render_size=rsize,
        step=int(ck.get("step", 0)),
    )


# --------------------------------------------------------------------------------------
# Generate + render one sub-batch of prompts at a fixed seed round and cfg scale
# --------------------------------------------------------------------------------------
@torch.no_grad()
def gen_render(
    b: dict, prompt_idxs: list[int], seed_round: int, cfg_scale: float,
    steps: int, sampler: str, cam_indices: list[int], device: torch.device,
    return_alpha: bool = False,
) -> torch.Tensor:
    """Returns rendered images (len(prompt_idxs), n_views, 3, H, W) in [0, 1].
    With return_alpha=True, returns (rgb, alpha) where alpha is (..., 1, H, W) coverage."""
    cfg = b["cfg"]
    P = len(prompt_idxs)
    shape = T.resolve_sampling_shape(model=b["model"], batch_size=P, in_channels=b["in_channels"])
    _, C, H, W = shape

    # Deterministic per-(prompt, seed_round) initial noise. seed_round 0 with the
    # default offset reproduces _run_validation_grid's per-tile noise exactly.
    base = int(cfg["val_grid_seed"])
    initial_noise = torch.empty((P, C, H, W), dtype=torch.float32)
    g = torch.Generator(device="cpu")
    for j, pi in enumerate(prompt_idxs):
        g.manual_seed(base + int(seed_round) * 100_000 + int(pi))
        initial_noise[j] = torch.randn((C, H, W), generator=g, dtype=torch.float32)
    initial_noise = initial_noise.to(device)
    cond_embeds = b["cond_pool"][torch.tensor(prompt_idxs, dtype=torch.long)].to(device)

    amp = b.get("amp_dtype")
    amp_ctx = torch.autocast("cuda", dtype=amp) if amp is not None else contextlib.nullcontext()
    with amp_ctx:
        sample = sample_model(
            sampler=sampler, model=b["model"], shape=(P, C, H, W),
            cond_embeds=cond_embeds, num_inference_steps=steps, device=device,
            predict_xstart=True, diffusion_steps=1000, cfg_scale=float(cfg_scale),
            timestep_schedule="logit_normal",
            P_mean=float(cfg["P_mean"]), P_std=float(cfg["P_std"]),
            initial_noise=initial_noise,
        )
    # back to fp32 for the permutation-invariant atlas->point-cloud + gsplat render (fp32).
    pc = T._plane_to_point_cloud_batch(sample.float(), b["plane_to_sphere"])
    pc_raw = T._denormalize_point_cloud(pc, b["norm_mean"], b["norm_std"])
    gauss = T._point_clouds_to_gsplat_inputs(
        pc_raw.to(device), dc_only=True, detach_input=True,
        rank_transform_tables=b["rank_tables"],
    )
    rendered = T._render_gsplat_batch(b["renderer"], gauss, b["train_cameras"], cam_indices, device,
                                      return_alpha=return_alpha)
    return rendered  # (P, n_views, 3, H, W), or (rgb, alpha) if return_alpha


# --------------------------------------------------------------------------------------
# CLIP joint-space scoring
# --------------------------------------------------------------------------------------
# Canonical OpenAI CLIP image normalization (identical across all openai/clip-* models;
# preprocessor_config.json is not in the local cache, so we use the known constants and
# do resize/normalize manually — equivalent for our already-224px renders).
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def load_clip(device: torch.device):
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, local_files_only=True).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
    mean = torch.tensor(CLIP_IMAGE_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_IMAGE_STD, device=device).view(1, 3, 1, 1)
    return model, tok, mean, std


@torch.no_grad()
def clip_text_features(clip, tok, prompts: list[str], device: torch.device) -> torch.Tensor:
    inputs = tok(
        prompts, padding=True, truncation=True, max_length=77, return_tensors="pt",
    ).to(device)
    out = clip.get_text_features(**inputs)
    feat = out if torch.is_tensor(out) else out.pooler_output  # transformers>=5 returns a pooled output
    return F.normalize(feat.float(), dim=-1)  # (P, 768) unit


@torch.no_grad()
def clip_image_features(clip, mean, std, imgs01: torch.Tensor, device: torch.device,
                        sub: int = 256) -> torch.Tensor:
    """imgs01: (N, 3, H, W) in [0,1]. Returns (N, 768) unit-norm joint features."""
    x = imgs01.to(device).float()
    if x.shape[-1] != 224 or x.shape[-2] != 224:
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
    out = []
    for s in range(0, x.shape[0], sub):
        chunk = (x[s:s + sub] - mean) / std
        res = clip.get_image_features(pixel_values=chunk)
        feat = res if torch.is_tensor(res) else res.pooler_output
        out.append(F.normalize(feat.float(), dim=-1))
    return torch.cat(out, dim=0)


# --------------------------------------------------------------------------------------
def save_montage(tiles01: torch.Tensor, prompts: list[str], scores: list[float],
                 out_path: str, cols: int = 4) -> None:
    """tiles01: (n, 3, H, W) in [0,1]. Saves a labeled grid for sanity-checking."""
    from PIL import ImageDraw, ImageFont
    n = tiles01.shape[0]
    H, W = int(tiles01.shape[-2]), int(tiles01.shape[-1])
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
    arr = (tiles01.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = arr[i]
    img = Image.fromarray(canvas)
    try:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        for i in range(n):
            r, c = divmod(i, cols)
            label = f"{prompts[i][:24]}  {scores[i]:.3f}"
            draw.text((c * W + 3, r * H + 3), label, fill=(255, 255, 0), font=font)
    except Exception:
        pass
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--n_views", type=int, default=4)
    ap.add_argument("--cfg_scales", default="1,2,3,5")
    ap.add_argument("--sampler", default=None, help="default: run's val_sampler")
    ap.add_argument("--steps", type=int, default=None, help="default: run's val_sampling_steps")
    ap.add_argument("--sub_batch", type=int, default=32, help="prompts generated per sampling call")
    ap.add_argument("--render_size", type=int, default=None, help="default: run's train_render_size (224)")
    ap.add_argument("--prompts_file", default=None,
                    help="JSON captions to condition on (e.g. data/baseline_captions_500.json). "
                         "Overrides the run's val prompts. Accepts a list of strings, a "
                         "{id: caption} dict, or a list of {caption/text/prompt} objects.")
    ap.add_argument("--max_prompts", type=int, default=None, help="cap prompt count (smoke testing / subsample)")
    ap.add_argument("--shuffle_seed", type=int, default=None,
                    help="deterministically shuffle prompts before --max_prompts (random subsample)")
    ap.add_argument("--max_gpu_gb", type=float, default=None,
                    help="hard cap on THIS process's GPU memory (GiB). Makes the eval OOM "
                         "itself rather than starving a co-resident training job. Set below "
                         "the GPU's currently-free memory. ~0.5GB CUDA context sits outside it.")
    ap.add_argument("--bf16", action="store_true",
                    help="run sampling under bf16 autocast (~2x on tensor cores; the model "
                         "was trained in bf16 so this is in-distribution).")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg_scales = [float(x) for x in args.cfg_scales.split(",") if x.strip()]

    # Hard-cap this process's GPU memory BEFORE any allocation so a co-resident training
    # job can't be OOM-crashed by the eval — the eval OOMs itself instead. The fraction
    # is relative to total device memory; the ~0.5GB CUDA context sits outside the cap.
    if args.max_gpu_gb:
        dev_idx = torch.cuda.current_device()  # explicit int index (set_per_process_* needs it)
        free_b, total_b = torch.cuda.mem_get_info(dev_idx)
        total_gb, free_gb = total_b / 2**30, free_b / 2**30
        frac = min(1.0, float(args.max_gpu_gb) / total_gb)
        torch.cuda.set_per_process_memory_fraction(frac, dev_idx)
        print(f"[mem] cap={args.max_gpu_gb:.1f}GiB of {total_gb:.0f}GiB total "
              f"(fraction={frac:.3f}); {free_gb:.1f}GiB currently free on this GPU")
        if args.max_gpu_gb > free_gb - 1.0:
            print(f"[mem] WARNING: cap ({args.max_gpu_gb:.1f}GiB) is at/above free memory "
                  f"({free_gb:.1f}GiB) — lower --max_gpu_gb or --sub_batch so you don't "
                  f"starve the co-resident job.")

    t0 = time.time()
    b = build(args.ckpt, device, render_size=args.render_size,
              max_prompts=args.max_prompts, prompts_file=args.prompts_file,
              shuffle_seed=args.shuffle_seed)
    b["amp_dtype"] = torch.bfloat16 if args.bf16 else None
    cfg = b["cfg"]
    sampler = args.sampler or cfg["val_sampler"]
    steps = int(args.steps or cfg["val_sampling_steps"])
    prompts = b["prompts"]  # already loaded/shuffled/capped in build(); cond_pool stays aligned
    P = len(prompts)
    n_seeds = int(args.n_seeds)
    n_views = int(args.n_views)

    # Choose n_views evenly-spaced cameras from the available ref cameras.
    ncam = int(b["train_cameras"]["viewmats"].shape[0])
    if n_views >= ncam:
        cam_indices = list(range(ncam))
    else:
        cam_indices = sorted({int(round(x)) for x in np.linspace(0, ncam - 1, n_views)})
        # de-dup may shrink the set; top up with the next unused indices
        k = 0
        while len(cam_indices) < n_views and k < ncam:
            if k not in cam_indices:
                cam_indices.append(k)
            k += 1
        cam_indices = sorted(cam_indices)[:n_views]
    n_views = len(cam_indices)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)),
        f"clip_eval_step{b['step']:07d}",
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"[build] {time.time() - t0:.1f}s  step={b['step']} model={cfg['model']} "
          f"sampler={sampler} steps={steps} P_mean={cfg['P_mean']} P_std={cfg['P_std']}")
    print(f"[setup] prompts={P} seeds={n_seeds} -> {P * n_seeds} samples/cfg | "
          f"views={n_views} cams={cam_indices} render={b['render_size']}px | cfg_scales={cfg_scales}")
    print(f"[setup] out_dir={out_dir}")

    clip, tok, cmean, cstd = load_clip(device)
    text_feats = clip_text_features(clip, tok, prompts, device)  # (P, 768)

    H = W = b["render_size"]
    summary = {}
    results_full = {}

    for cfg_scale in cfg_scales:
        scores = np.zeros((P, n_seeds), dtype=np.float64)
        montage_tiles = None
        montage_scores = None
        ct = time.time()
        for r in range(n_seeds):
            for s in range(0, P, args.sub_batch):
                idxs = list(range(s, min(P, s + args.sub_batch)))
                rendered = gen_render(
                    b, idxs, r, cfg_scale, steps, sampler, cam_indices, device,
                )  # (nb, n_views, 3, H, W)
                nb = rendered.shape[0]
                flat = rendered.reshape(nb * n_views, 3, H, W)
                img_feats = clip_image_features(clip, cmean, cstd, flat, device)  # (nb*V, 768)
                img_feats = img_feats.reshape(nb, n_views, -1).mean(dim=1)  # mean over views
                tf = text_feats[torch.tensor(idxs, dtype=torch.long, device=device)]
                sc = (img_feats * tf).sum(-1)  # (nb,) = mean-view cosine
                scores[idxs, r] = sc.detach().cpu().numpy()
                # Sanity montage: seed 0, view 0, first 16 prompts.
                if r == 0 and s == 0:
                    k = min(16, nb)
                    montage_tiles = rendered[:k, 0].detach().cpu()
                    montage_scores = scores[:k, 0].tolist()
                del rendered, flat, img_feats
            torch.cuda.empty_cache()

        per_prompt_mean = scores.mean(axis=1)          # (P,)
        overall_mean = float(scores.mean())
        overall_std = float(scores.std())
        sem = float(scores.std() / np.sqrt(scores.size))
        order = np.argsort(per_prompt_mean)
        tag = f"cfg{cfg_scale:g}"
        summary[tag] = {
            "cfg_scale": cfg_scale, "mean": overall_mean, "std": overall_std, "sem": sem,
            "per_prompt_mean_min": float(per_prompt_mean.min()),
            "per_prompt_mean_max": float(per_prompt_mean.max()),
            "worst5": [(prompts[i], float(per_prompt_mean[i])) for i in order[:5]],
            "best5": [(prompts[i], float(per_prompt_mean[i])) for i in order[::-1][:5]],
            "seconds": time.time() - ct,
        }
        results_full[tag] = {
            "cfg_scale": cfg_scale,
            "per_prompt": [
                {"prompt": prompts[i], "mean": float(per_prompt_mean[i]),
                 "scores": [float(v) for v in scores[i]]}
                for i in range(P)
            ],
        }
        if montage_tiles is not None:
            save_montage(
                montage_tiles, prompts[:montage_tiles.shape[0]], montage_scores,
                os.path.join(out_dir, f"montage_{tag}.png"),
            )
        print(f"[{tag}] mean={overall_mean:.4f} +-{overall_std:.4f} (sem {sem:.4f})  "
              f"prompt-mean range [{per_prompt_mean.min():.3f}, {per_prompt_mean.max():.3f}]  "
              f"{summary[tag]['seconds']:.0f}s")

    payload = {
        "checkpoint": os.path.abspath(args.ckpt), "step": b["step"], "model": cfg["model"],
        "prompts_file": os.path.abspath(args.prompts_file) if args.prompts_file else None,
        "shuffle_seed": args.shuffle_seed, "bf16": bool(args.bf16), "sub_batch": int(args.sub_batch),
        "clip_model": CLIP_MODEL_ID, "sampler": sampler, "steps": steps,
        "n_prompts": P, "n_seeds": n_seeds, "n_samples_per_cfg": P * n_seeds,
        "n_views": n_views, "cam_indices": cam_indices, "render_size": b["render_size"],
        "P_mean": float(cfg["P_mean"]), "P_std": float(cfg["P_std"]),
        "background": "black", "score_def": "mean_v cos(CLIP_img(view_v), CLIP_txt(prompt)); cosine in [-1,1]",
        "cfg_scales": cfg_scales, "summary": summary, "results": results_full,
        "total_seconds": time.time() - t0,
    }
    out_json = os.path.join(out_dir, "clip_alignment.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n==== CLIP alignment (ViT-L/14 joint cosine; higher=better) ====")
    print(f"{'cfg':>5} {'mean':>8} {'std':>7} {'sem':>7}")
    for cfg_scale in cfg_scales:
        s = summary[f"cfg{cfg_scale:g}"]
        print(f"{cfg_scale:>5g} {s['mean']:>8.4f} {s['std']:>7.4f} {s['sem']:>7.4f}")
    print(f"\nsaved: {out_json}")
    print(f"montages: {out_dir}/montage_cfg*.png")
    print(f"[done] total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
