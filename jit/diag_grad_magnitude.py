"""Gradient-magnitude profile across t for the recon losses.

Loads a checkpoint + a fixed real batch, then for each t in --t_values does ONE forward
and measures the parameter-gradient norm produced SEPARATELY by:
  * sinkhorn_patch        — soft entropic-OT, <Pi, sq> (the "blend → mean-pull" gradient)
  * sinkhorn_patch_hard   — argmax-rounded plan, MSE to the single matched target (DETR-style)
  * mse                   — index-aligned MSE (the bootstrap term)
Also reports the gradient w.r.t. the model output (`Gpred`) — the intrinsic per-loss scale,
backbone-free. All three are computed on the SAME pred/x batch so the comparison is apples-
to-apples; sinkhorn at weight cw, MSE at unit weight.

Run: PYTHONPATH=. ./.3dgen/bin/python jit/diag_grad_magnitude.py --resume <ckpt>
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import torch

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dataloaders.standard_3dgen_loader import Standard3DGenDataset, load_null_text_token
from dataloaders.text_3dgen_loader import Text3DGenDataset, DC_ONLY_FEATURE_INDICES
from jit.models import JiT_3DGS_models
from jit.diffusion import create_diffusion
from jit.diffusion.gaussian_diffusion import _chamfer_recon_loss

P = argparse.ArgumentParser()
P.add_argument("--resume", required=True)
P.add_argument("--model", default="JiT-B/8")
P.add_argument("--weights", default="model", choices=("model", "ema"))
P.add_argument("--batch_size", type=int, default=8)
P.add_argument("--t_values", default="0.05,0.15,0.3,0.5,0.7,0.85,0.95",
               help="FM convention: 0=noise, 1=clean.")
P.add_argument("--cw", type=float, default=1.0, help="chamfer_loss_weight applied to sinkhorn")
P.add_argument("--patch_size", type=int, default=8, help="Matches current run (chamfer_patch_size).")
P.add_argument("--sinkhorn_eps", type=float, default=0.05)
P.add_argument("--sinkhorn_iters", type=int, default=100, help="Matches current run.")
P.add_argument("--huber_delta", type=float, default=0.0,
               help="Pseudo-Huber radius for the within-patch ground cost (0 = off / squared-L2). "
                    ">0 robustifies the cost so outlier pairs stop dominating the gradient; compare "
                    "the high-noise grad magnitude with/without it.")
P.add_argument("--max_vram_gb", type=float, default=14.0)
P.add_argument("--obj_list", default="data/gaussianverse/all_obj_list_filtered.json")
P.add_argument("--gs_path", default="data/gaussianverse/")
P.add_argument("--mean_file", default=f"{REPO}/data/stats/all_mean_postfix.pt")
P.add_argument("--std_file", default=f"{REPO}/data/stats/all_std_postfix.pt")
P.add_argument("--sphere2plane_path", default="data/gaussianverse/sphere2plane.npy")
P.add_argument("--rank_transform_file", default=f"{REPO}/data/stats/rank_quantiles_8ch_clipped.pt")
P.add_argument("--clip_thresholds_file", default=f"{REPO}/data/stats/clip_thresholds_opacity_scales.pt")
P.add_argument("--exclude_keys_file", default=f"{REPO}/data/outlier_keys_8sigma.json")
P.add_argument("--text_embed_path", default="object_classification/text_tokens")
P.add_argument("--null_text_token_path", default="object_classification/null_text_token.npz")
a = P.parse_args()

dev = torch.device("cuda")
if a.max_vram_gb > 0:
    tot = torch.cuda.get_device_properties(0).total_memory / 1024**3
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_vram_gb / tot), 0)
torch.manual_seed(0)

fi = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
base = Standard3DGenDataset(obj_list=[a.obj_list], gs_path=a.gs_path, caption_path=None,
                            mean_file=a.mean_file, std_file=a.std_file,
                            sphere2plane_path=a.sphere2plane_path,
                            exclude_keys_file=a.exclude_keys_file,
                            rank_transform_file=a.rank_transform_file,
                            clip_thresholds_file=a.clip_thresholds_file,
                            text_embed_path=a.text_embed_path)
text_dim = int(base.text_pooled.shape[1])
ds = Text3DGenDataset(base, feature_indices=fi, return_full_for_render=False,
                      preload_to_cpu=False, lazy_cache_to_cpu=False)
ld = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=4)

model = JiT_3DGS_models[a.model](input_size=128, in_channels=len(DC_ONLY_FEATURE_INDICES),
                                 text_dim=text_dim, class_dropout_prob=0.0, learn_sigma=False,
                                 gradient_checkpointing=False, bottleneck=False)
if os.path.exists(a.null_text_token_path):
    import numpy as np
    model.load_null_embeddings(torch.from_numpy(load_null_text_token(a.null_text_token_path).astype(np.float32)))
ck = torch.load(a.resume, map_location="cpu", weights_only=False)
st = ck[a.weights]
def strip(k):
    for pre in ("_orig_mod.", "module."):
        while k.startswith(pre):
            k = k[len(pre):]
    return k
st = {strip(k): v for k, v in st.items()}
miss, unexp = model.load_state_dict(st, strict=False)
assert len(miss) <= 5 and len(unexp) <= 5, f"state mismatch miss={len(miss)} unexp={len(unexp)}"
print(f"[load] {a.resume} step={ck.get('step')} weights={a.weights} (miss={len(miss)} unexp={len(unexp)})")
model.to(dev).train()
for p in model.parameters():
    p.requires_grad_(True)
params = [p for p in model.parameters() if p.requires_grad]

diff = create_diffusion(timestep_respacing="", noise_schedule="squaredcos_cap_v2",
                        learn_sigma=False, predict_xstart=True)
T = diff.num_timesteps

batch = next(iter(ld))
x, y, _ = batch
x = x.to(dev).float(); y = y.to(dev).float()

def pgrad_norm():
    return (sum(p.grad.detach().float().norm().item() ** 2 for p in params if p.grad is not None)) ** 0.5

def zero():
    for p in params:
        p.grad = None

print(f"\n{'t':>5} | {'L_soft':>8} {'L_hard':>8} {'L_mse':>8} | "
      f"{'GN soft':>9} {'GN hard':>9} {'GN mse':>9} | "
      f"{'Gpred soft':>10} {'Gpred hard':>10} {'Gpred mse':>10} | "
      f"{'h/m':>5} {'s/m':>5} {'h/s':>5}")
rows = []
for t in [float(s) for s in a.t_values.split(",")]:
    tv = torch.full((x.shape[0],), t, device=dev)
    td = torch.round(tv * (T - 1)).long()
    noise = torch.randn_like(x)
    x_t = tv.view(-1, 1, 1, 1) * x + (1 - tv).view(-1, 1, 1, 1) * noise   # FM
    pred = model(x_t, td, y).float()
    pred.retain_grad()

    # SOFT sinkhorn
    L_soft = _chamfer_recon_loss(pred, x, "sinkhorn_patch", weight=a.cw, patch_size=a.patch_size,
                                 sinkhorn_eps=a.sinkhorn_eps, sinkhorn_iters=a.sinkhorn_iters,
                                 huber_delta=a.huber_delta).mean()
    zero(); pred.grad = None
    L_soft.backward(retain_graph=True)
    gn_s = pgrad_norm(); gp_s = pred.grad.detach().norm().item()

    # HARD sinkhorn (argmax-rounded)
    L_hard = _chamfer_recon_loss(pred, x, "sinkhorn_patch_hard", weight=a.cw, patch_size=a.patch_size,
                                 sinkhorn_eps=a.sinkhorn_eps, sinkhorn_iters=a.sinkhorn_iters,
                                 huber_delta=a.huber_delta).mean()
    zero(); pred.grad = None
    L_hard.backward(retain_graph=True)
    gn_h = pgrad_norm(); gp_h = pred.grad.detach().norm().item()

    # MSE
    L_mse = ((pred - x) ** 2).mean()
    zero(); pred.grad = None
    L_mse.backward()
    gn_m = pgrad_norm(); gp_m = pred.grad.detach().norm().item()

    rows.append((t, L_soft.item(), L_hard.item(), L_mse.item(),
                 gn_s, gn_h, gn_m, gp_s, gp_h, gp_m))
    print(f"{t:>5.2f} | {L_soft.item():>8.4f} {L_hard.item():>8.4f} {L_mse.item():>8.4f} | "
          f"{gn_s:>9.4f} {gn_h:>9.4f} {gn_m:>9.4f} | "
          f"{gp_s:>10.4f} {gp_h:>10.4f} {gp_m:>10.4f} | "
          f"{gn_h/gn_m:>5.2f} {gn_s/gn_m:>5.2f} {gn_h/gn_s:>5.2f}")
print("\n(FM: t=0 noise, t=1 clean. GN = ‖∂L/∂params‖ over all params. Gpred = ‖∂L/∂model_output‖.")
print(f" sink at weight cw={a.cw}, mse at weight 1. h/m = hard/mse ratio of GN; s/m soft/mse; h/s hard/soft.)")
