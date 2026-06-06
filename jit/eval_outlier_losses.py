#!/usr/bin/env python
"""Measure the four TRAINING losses (recon + render-L1 + alpha-mask + render-LPIPS) for one
checkpoint on an arbitrary object list — e.g. the 726 excluded 8-sigma outliers — by replaying
the exact training step (jit/train_gsplat.py): flow_matching_training_losses -> (recon, x0_pred),
then _compute_render_loss_for_batch on the low-noise (t_value>=cutoff) subset, no backprop.

  recon          : sinkhorn_patch_hard, training-t (sigmoid N(p_mean,p_std)), weighted xcw
  render_l1/alpha/lpips : on samples with t_value>=render_loss_noise_cutoff (the render regime),
                          un-normalized -> gsplat @train_render_size, num_cam cameras, vs GT renders.

    CUDA_VISIBLE_DEVICES=0 python jit/eval_outlier_losses.py \
        --resume output/jit_final_sinkhorn_render_cam2_22k_20260528_210521/0069000.pt \
        --obj_list data/outlier_obj_list_726.json --weights ema --passes 4
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dataloaders.standard_3dgen_loader import Standard3DGenDataset
from dataloaders.text_3dgen_loader import Text3DGenDataset, DC_ONLY_FEATURE_INDICES, FULL_3DGS_FEATURE_DIM
from jit.models import JiT_3DGS_models
from jit.diffusion import create_diffusion
from utils.plane_utils import load_sphere2plane
from utils.gsplat_render_util import (
    _compute_render_loss_for_batch, _prepare_train_cameras, _load_reference_cameras,
    _try_import_lpips, _try_import_renderer, load_rank_transform_payload_torch,
)


def p():
    a = argparse.ArgumentParser()
    a.add_argument("--resume", required=True)
    a.add_argument("--weights", default="ema", choices=("ema", "model"))
    a.add_argument("--model", default="JiT-B/8")
    a.add_argument("--obj_list", default="data/outlier_obj_list_726.json")
    a.add_argument("--gs_path", default="data/gaussianverse/")
    a.add_argument("--sphere2plane_path", default="data/gaussianverse/sphere2plane.npy")
    a.add_argument("--mean_file", default="data/stats/all_mean_postfix.pt")
    a.add_argument("--std_file", default="data/stats/all_std_postfix.pt")
    a.add_argument("--rank_transform_file", default="data/stats/rank_quantiles_8ch_clipped.pt")
    a.add_argument("--clip_thresholds_file", default="data/stats/clip_thresholds_opacity_scales.pt")
    a.add_argument("--text_embed_path", default="object_classification/text_tokens")
    a.add_argument("--ref_camera_tar", default="artifacts/ref_camera.tar.gz")
    a.add_argument("--noise_schedule", default="squaredcos_cap_v2")
    # recon config (match the earlier probe / cam2)
    a.add_argument("--recon_loss", default="sinkhorn_patch_hard")
    a.add_argument("--chamfer_loss_weight", type=float, default=0.07)
    a.add_argument("--chamfer_patch_size", type=int, default=8)
    a.add_argument("--sinkhorn_eps", type=float, default=0.05)
    a.add_argument("--sinkhorn_iters", type=int, default=100)
    a.add_argument("--huber_delta", type=float, default=1.7)
    a.add_argument("--p_mean", type=float, default=0.0)
    a.add_argument("--p_std", type=float, default=1.5)
    # render config (match cam2)
    a.add_argument("--render_num_cam", type=int, default=2)
    a.add_argument("--render_noise_cutoff", type=float, default=0.8)
    a.add_argument("--train_render_size", type=int, default=224)
    a.add_argument("--render_zoom_factor", type=float, default=1.0)
    a.add_argument("--lpips_net", default="vgg")
    a.add_argument("--passes", type=int, default=4, help="loops over the obj list (render keeps ~18%/pass)")
    a.add_argument("--batch_size", type=int, default=4)
    a.add_argument("--num_workers", type=int, default=4)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--output_dir", default="output/report/eval_logs/outlier_losses")
    return a.parse_args()


def main():
    args = p()
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    feat_idx = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
    in_ch = len(DC_ONLY_FEATURE_INDICES)

    # ---- data ----
    base = Standard3DGenDataset(
        obj_list=[args.obj_list], gs_path=args.gs_path, caption_path=None,
        mean_file=args.mean_file, std_file=args.std_file, sphere2plane_path=args.sphere2plane_path,
        exclude_keys_file=None, rank_transform_file=args.rank_transform_file,
        clip_thresholds_file=args.clip_thresholds_file, text_embed_path=args.text_embed_path)
    text_dim = int(base.text_pooled.shape[1])
    ds = Text3DGenDataset(base, feature_indices=feat_idx, return_full_for_render=True,
                          preload_to_cpu=False, lazy_cache_to_cpu=False, class_map=None)
    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, drop_last=False, generator=g)
    print(f"[data] {len(ds)} objects | {args.passes} passes | batch {args.batch_size}", flush=True)

    # ---- model ----
    model = JiT_3DGS_models[args.model](
        input_size=128, in_channels=in_ch, text_dim=text_dim, num_classes=None,
        class_dropout_prob=0.0, learn_sigma=False, gradient_checkpointing=False, bottleneck=False)
    ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    state = ck[args.weights]
    pfx = "_orig_mod."
    if any(k.startswith(pfx) for k in state):
        state = {(k[len(pfx):] if k.startswith(pfx) else k): v for k, v in state.items()}
    miss, unexp = model.load_state_dict(state, strict=False)
    if len(miss) > 5 or len(unexp) > 5:
        raise RuntimeError(f"weights not loaded: {len(miss)} missing / {len(unexp)} unexpected")
    model.to(dev).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    print(f"[load] {args.resume} step={ck.get('step')} weights={args.weights}", flush=True)

    diffusion = create_diffusion(timestep_respacing="", noise_schedule=args.noise_schedule,
                                 learn_sigma=False, predict_xstart=True)
    T = diffusion.num_timesteps

    # ---- render setup (mirror train_gsplat) ----
    rank_tables = load_rank_transform_payload_torch(args.rank_transform_file, device=dev)
    nm_full = torch.load(args.mean_file, weights_only=True).float().to(dev)
    ns_full = torch.load(args.std_file, weights_only=True).float().to(dev)
    if rank_tables is not None:
        ridx = torch.tensor(rank_tables["channels"], dtype=torch.long, device=dev)
        nm_full = nm_full.clone(); ns_full = ns_full.clone()
        nm_full[ridx] = 0.0; ns_full[ridx] = 1.0
    nm = nm_full[feat_idx.to(dev)]; ns = ns_full[feat_idx.to(dev)]
    plane_to_sphere = load_sphere2plane(args.sphere2plane_path, 128 * 128).to(dev)
    renderer = _try_import_renderer()
    if isinstance(renderer, Exception):
        raise RuntimeError(f"renderer import failed: {renderer}")
    ref_cams = _load_reference_cameras(args.ref_camera_tar)
    train_cameras = _prepare_train_cameras(ref_cams, args.train_render_size, dev,
                                           zoom_factor=float(args.render_zoom_factor))
    lp = _try_import_lpips()
    if isinstance(lp, Exception):
        raise RuntimeError(f"lpips import failed: {lp}")
    lpips_fn = lp.LPIPS(net=args.lpips_net).to(dev).eval()
    for q in lpips_fn.parameters():
        q.requires_grad_(False)
    print(f"[render] cams={len(ref_cams)} num_cam={args.render_num_cam} cutoff={args.render_noise_cutoff} "
          f"size={args.train_render_size} lpips={args.lpips_net}", flush=True)

    recon, rl1, ral, rlp, rn = [], 0.0, 0.0, 0.0, 0
    t0 = time.time()
    tgen = torch.Generator(device=dev); tgen.manual_seed(args.seed)
    with torch.no_grad():
        for ep in range(args.passes):
            for bi, batch in enumerate(loader):
                x, y_pooled, x_full, hash_keys = batch
                x = x.to(dev).float(); y_pooled = y_pooled.to(dev).float(); x_full = x_full.to(dev).float()
                B = x.shape[0]
                logit = args.p_mean + args.p_std * torch.randn(B, generator=tgen, device=dev)
                t_value = torch.sigmoid(logit)
                t = (t_value * (T - 1)).round().long()
                ld = diffusion.flow_matching_training_losses(
                    model, x, t_value, t, model_kwargs=dict(y_pooled=y_pooled),
                    noise=torch.randn(x.shape, generator=tgen, device=dev),
                    recon_loss=args.recon_loss, chamfer_loss_weight=args.chamfer_loss_weight,
                    chamfer_subsample=4096, chamfer_patch_size=args.chamfer_patch_size,
                    chamfer_rev_weight=12.0, mse_hybrid_weight=0.0, mse_hybrid_lownoise_mult=4.0,
                    sinkhorn_eps=args.sinkhorn_eps, sinkhorn_iters=args.sinkhorn_iters,
                    compile_sinkhorn=False, huber_delta=args.huber_delta, residual_mad_diag=False,
                    permute_mode="patch", permute_patch_size=args.chamfer_patch_size, skip_recon=False)
                recon.extend(ld["loss"].detach().cpu().tolist())
                x0_pred = ld["pred_xstart"].float()
                keep = (t_value >= args.render_noise_cutoff)
                k = int(keep.sum())
                if k > 0:
                    a_l1, a_al, a_lp = _compute_render_loss_for_batch(
                        x0_pred=x0_pred[keep], x_gt_full=x_full[keep],
                        norm_mean_pred=nm, norm_std_pred=ns, norm_mean_full=nm_full, norm_std_full=ns_full,
                        train_cameras=train_cameras, renderer_tuple=renderer, lpips_fn=lpips_fn,
                        num_cam=args.render_num_cam, device=dev, dc_only=True,
                        plane_to_sphere=plane_to_sphere, sample_weights=None,
                        rank_transform_tables=rank_tables, per_sample_cam_zooms=None)
                    rl1 += float(a_l1) * k; ral += float(a_al) * k; rlp += float(a_lp) * k; rn += k
            print(f"  pass {ep+1}/{args.passes}: recon_n={len(recon)} render_n={rn} ({time.time()-t0:.0f}s)", flush=True)

    recon = np.asarray(recon)
    qs = np.percentile(recon, [5, 50, 95])
    res = {"ckpt": args.resume, "step": int(ck.get("step", -1)), "n_objects": len(ds),
           "recon_n": int(len(recon)), "render_n": int(rn),
           "recon_mean": float(recon.mean()), "recon_median": float(qs[1]),
           "recon_p5": float(qs[0]), "recon_p95": float(qs[2]), "recon_std": float(recon.std()),
           "render_l1_mean": rl1 / max(rn, 1), "render_alpha_l1_mean": ral / max(rn, 1),
           "render_lpips_mean": rlp / max(rn, 1),
           "config": {"recon_loss": args.recon_loss, "cw": args.chamfer_loss_weight,
                      "huber": args.huber_delta, "sinkhorn_eps": args.sinkhorn_eps,
                      "sinkhorn_iters": args.sinkhorn_iters, "p_mean": args.p_mean, "p_std": args.p_std,
                      "render_num_cam": args.render_num_cam, "render_noise_cutoff": args.render_noise_cutoff,
                      "render_size": args.train_render_size, "lpips_net": args.lpips_net,
                      "per_sample_zoom": False}}
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(res, open(os.path.join(args.output_dir, "outlier_losses.json"), "w"), indent=2)
    print("\n==== TRAINING LOSSES on %d objects (step %s) ====" % (len(ds), res["step"]))
    print(f"  recon (sinkhorn, xcw, training-t):  mean={res['recon_mean']:.5f}  median={res['recon_median']:.5f}  (n={res['recon_n']})")
    print(f"  render L1   (t>= {args.render_noise_cutoff}):        {res['render_l1_mean']:.5f}")
    print(f"  render alpha-mask L1:               {res['render_alpha_l1_mean']:.5f}")
    print(f"  render LPIPS({args.lpips_net}):              {res['render_lpips_mean']:.5f}   (render_n={rn})")
    print(f"  saved: {os.path.join(args.output_dir, 'outlier_losses.json')}")


if __name__ == "__main__":
    main()
