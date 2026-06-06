#!/usr/bin/env python
"""Is a more permutation-invariant recon loss LESS spatially correlated?

Controlled experiment: ONE fixed model + ONE fixed set of (pred, GT) pairs. We run the
model once per (object, t), then score the SAME predictions under increasing matching
permutation-freedom — sinkhorn_patch_hard within P×P patches for P ∈ {1,4,8,16,32}
(P=1 ⇒ index-aligned MSE; larger P ⇒ each GT cell can match a farther prediction). Only
the loss's patch size varies, so any change in the per-location error map is the LOSS,
not the model.

For each P we build the per-GT-cell cost map on the canonical 128×128 atlas (same as the
per-location probes) and measure how spatially structured it is:
  • Moran's I        — 4-neighbour spatial autocorrelation (≈0 ⇒ spatially random).
  • low-freq fraction — share of FFT power below a low spatial-frequency cutoff (the
                        equatorial band is low-freq, so this shrinks as error decorrelates).
  • layout-corr       — Pearson r between the error map and a GT layout-roughness map
                        (per-cell local |Δxyz| to atlas neighbours). HIGH r ⇒ the error is
                        driven by the (near-random, harmful) atlas layout; LOW r ⇒ decoupled.

HONEST framing: "less spatially correlated" = "less driven by the atlas layout artifact",
NOT automatically "better samples". Report these metrics as what they are. huber is OFF
(δ=0) for this sweep so the ONLY thing changing is the matching patch size.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from dataloaders.standard_3dgen_loader import Standard3DGenDataset  # noqa: E402
from dataloaders.text_3dgen_loader import Text3DGenDataset, DC_ONLY_FEATURE_INDICES, FULL_3DGS_FEATURE_DIM  # noqa: E402
from jit.models import JiT_3DGS_models  # noqa: E402
from jit.diffusion import create_diffusion  # noqa: E402
from jit.diffusion.gaussian_diffusion import _sinkhorn_log  # noqa: E402


def find_latest(root):
    c = [x for x in glob.glob(os.path.join(root, "**", "*.pt"), recursive=True) if Path(x).stem.isdigit()]
    return max(c, key=os.path.getmtime) if c else None


def to_patches(t, P):
    B, C, H, W = t.shape
    nH, nW = H // P, W // P
    return (t.float().reshape(B, C, nH, P, nW, P).permute(0, 2, 4, 1, 3, 5)
            .reshape(B, nH * nW, C, P * P).transpose(-1, -2).contiguous())


def unpatch(v, nH, nW, P):  # (B,nP,M,K)->(B,H,W,K)
    B, _, _, K = v.shape
    return v.reshape(B, nH, nW, P, P, K).permute(0, 1, 3, 2, 4, 5).reshape(B, nH * P, nW * P, K)


def per_cell_cost(pred, gt, P, eps, iters, device):
    """Per-GT-cell sinkhorn_patch_hard matched cost (huber off), localized to the atlas. -> (B,H,W) sum, (B,H,W) cnt."""
    B, C, H, W = pred.shape
    nH, nW = H // P, W // P; M = P * P
    xp = to_patches(pred, P); yp = to_patches(gt, P)
    aa = xp.pow(2).sum(-1); bb = yp.pow(2).sum(-1)
    ab = xp @ yp.transpose(-1, -2)
    sq = (aa.unsqueeze(-1) + bb.unsqueeze(-2) - 2 * ab).clamp_min(0)   # (B,nP,M,M)
    if M == 1:
        sigma = torch.zeros(B, nH * nW, 1, dtype=torch.long, device=device)
    else:
        sigma = _sinkhorn_log(sq, eps, iters).argmax(dim=-1)          # (B,nP,M)
    hard = sq.gather(-1, sigma.unsqueeze(-1))                          # (B,nP,M,1) matched cost
    cell_sum = torch.zeros(B, nH * nW, M, 1, device=device).scatter_add_(2, sigma.unsqueeze(-1), hard)
    cell_cnt = torch.zeros(B, nH * nW, M, 1, device=device).scatter_add_(2, sigma.unsqueeze(-1), torch.ones_like(hard))
    csum = unpatch(cell_sum, nH, nW, P).squeeze(-1).sum(0).double().cpu()
    ccnt = unpatch(cell_cnt, nH, nW, P).squeeze(-1).sum(0).double().cpu()
    return csum, ccnt


def morans_i(x):
    z = x - x.mean()
    vp = (z[:-1, :] * z[1:, :]).sum(); hp = (z[:, :-1] * z[:, 1:]).sum()
    P_pairs = (x.shape[0] - 1) * x.shape[1] + x.shape[0] * (x.shape[1] - 1)
    den = (z * z).sum()
    return float((x.size / P_pairs) * (vp + hp) / (den + 1e-12))


def lowfreq_frac(x, cutoff=0.05):
    z = x - x.mean()
    P = np.abs(np.fft.fft2(z)) ** 2; P[0, 0] = 0.0
    fy = np.fft.fftfreq(x.shape[0])[:, None]; fx = np.fft.fftfreq(x.shape[1])[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    tot = P.sum()
    return float(P[r <= cutoff].sum() / (tot + 1e-12))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", required=True)
    p.add_argument("--model", default="JiT-B/8", choices=list(JiT_3DGS_models.keys()))
    p.add_argument("--weights", default="model", choices=("ema", "model"))
    p.add_argument("--noise_schedule", default="squaredcos_cap_v2")
    p.add_argument("--class_map_path", default=None)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--text_embed_path", default=None)
    p.add_argument("--obj_list", required=True); p.add_argument("--gs_path", required=True)
    p.add_argument("--mean_file"); p.add_argument("--std_file")
    p.add_argument("--rank_transform_file"); p.add_argument("--clip_thresholds_file")
    p.add_argument("--sphere2plane_path", required=True); p.add_argument("--exclude_keys_file")
    p.add_argument("--sh_degree0_only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--patch_sizes", default="1,4,8,16,32")
    p.add_argument("--t_values", default="0.3,0.5,0.7")
    p.add_argument("--sinkhorn_eps", type=float, default=0.05)
    p.add_argument("--sinkhorn_iters", type=int, default=100)
    p.add_argument("--lowfreq_cutoff", type=float, default=0.05)
    p.add_argument("--num_samples", type=int, default=48)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mixed_precision", default="bf16", choices=("bf16", "fp16", "none"))
    p.add_argument("--label", default="model")
    p.add_argument("--output_dir", default=None)
    return p


def main():
    a = build_parser().parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    Ps = [int(x) for x in a.patch_sizes.split(",")]
    ts = [float(x) for x in a.t_values.split(",")]
    out = a.output_dir or os.path.join(REPO_ROOT, "jit", "loss_spatial_corr_out",
                                       f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    print(f"[ckpt] {a.resume} (weights={a.weights}) | Ps={Ps} ts={ts} | out={out}")

    if a.sh_degree0_only:
        fidx = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long); in_ch = len(DC_ONLY_FEATURE_INDICES)
    else:
        fidx = None; in_ch = FULL_3DGS_FEATURE_DIM

    class_cond = bool(a.class_map_path); cmap = None; ncls = None
    if class_cond:
        cmap = json.load(open(a.class_map_path)); ncls = a.num_classes or (max(int(v) for v in cmap.values()) + 1)
    base = Standard3DGenDataset(obj_list=[a.obj_list], gs_path=a.gs_path, caption_path=None,
                                mean_file=a.mean_file, std_file=a.std_file, sphere2plane_path=a.sphere2plane_path,
                                exclude_keys_file=a.exclude_keys_file, rank_transform_file=a.rank_transform_file,
                                clip_thresholds_file=a.clip_thresholds_file,
                                text_embed_path=(None if class_cond else a.text_embed_path))
    if class_cond:
        base.obj_data = {h: p for h, p in base.obj_data.items() if p.split('.tar.gz')[0] in cmap}
        base.keys = list(base.obj_data.keys()); text_dim = None
    else:
        text_dim = int(base.text_pooled.shape[1])
    ds = Text3DGenDataset(base, feature_indices=fidx, return_full_for_render=False,
                          preload_to_cpu=False, lazy_cache_to_cpu=False, class_map=cmap)
    g = torch.Generator(); g.manual_seed(a.seed)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                        pin_memory=(dev.type == "cuda"), drop_last=False, generator=g)

    model = JiT_3DGS_models[a.model](input_size=128, in_channels=in_ch, text_dim=(text_dim or 768),
                                     num_classes=ncls, class_dropout_prob=0.0, learn_sigma=False,
                                     gradient_checkpointing=False, bottleneck=False)
    ck = torch.load(a.resume, map_location="cpu", weights_only=False); st = ck[a.weights]
    if any(k.startswith("_orig_mod.") for k in st):
        st = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in st.items()}
        print("[load] stripped _orig_mod. prefix")
    miss, unexp = model.load_state_dict(st, strict=False)
    if len(miss) > 5 or len(unexp) > 5:
        raise RuntimeError(f"state_dict mismatch: {len(miss)} missing / {len(unexp)} unexpected")
    step = int(ck.get("step", -1)); print(f"[load] step={step}")
    model.to(dev).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    diff = create_diffusion(timestep_respacing="", noise_schedule=a.noise_schedule, learn_sigma=False, predict_xstart=True)
    T = diff.num_timesteps
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.mixed_precision]

    H = W = 128
    cost_sum = {(t, P): np.zeros((H, W)) for t in ts for P in Ps}
    cost_cnt = {(t, P): np.zeros((H, W)) for t in ts for P in Ps}
    layout_sum = np.zeros((H, W)); n_obj = 0

    for bi, batch in enumerate(loader):
        if n_obj >= a.num_samples:
            break
        x, y, _ = batch
        rem = a.num_samples - n_obj
        if x.shape[0] > rem:
            x = x[:rem]; y = y[:rem]
        bsz = x.shape[0]
        x = x.to(dev).float(); y = (y.to(dev).long() if class_cond else y.to(dev).float())
        ng = torch.Generator(device=dev); ng.manual_seed(a.seed * 9973 + bi)
        noise = torch.randn(x.shape, generator=ng, device=dev, dtype=x.dtype)
        # GT layout roughness: local |Δxyz| to 4 atlas neighbours (channels 0,1,2)
        xyz = x[:, 0:3]
        rough = torch.zeros(bsz, H, W, device=dev)
        rough[:, :-1, :] += (xyz[:, :, :-1, :] - xyz[:, :, 1:, :]).pow(2).sum(1).sqrt()
        rough[:, 1:, :] += (xyz[:, :, 1:, :] - xyz[:, :, :-1, :]).pow(2).sum(1).sqrt()
        rough[:, :, :-1] += (xyz[:, :, :, :-1] - xyz[:, :, :, 1:]).pow(2).sum(1).sqrt()
        rough[:, :, 1:] += (xyz[:, :, :, 1:] - xyz[:, :, :, :-1]).pow(2).sum(1).sqrt()
        layout_sum += rough.sum(0).double().cpu().numpy()

        for t in ts:
            tv = torch.full((bsz,), float(t), device=dev)
            td = torch.clamp((tv * (T - 1)).round().long(), 0, T - 1)
            xt = diff.flow_matching_q_sample(x, tv, noise=noise)
            with torch.no_grad():
                if amp is not None and dev.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp):
                        pred = model(xt, td, y)
                else:
                    pred = model(xt, td, y)
            pred = pred.float()
            for P in Ps:
                cs, cc = per_cell_cost(pred, x, P, a.sinkhorn_eps, a.sinkhorn_iters, dev)
                cost_sum[(t, P)] += cs.numpy(); cost_cnt[(t, P)] += cc.numpy()
        n_obj += bsz
        print(f"  [{n_obj}/{a.num_samples}] batch {bi}")

    layout = layout_sum / max(n_obj, 1)
    maps = {(t, P): cost_sum[(t, P)] / np.maximum(cost_cnt[(t, P)], 1e-9) for t in ts for P in Ps}

    # metrics per (t,P)
    moran = {(t, P): morans_i(maps[(t, P)]) for t in ts for P in Ps}
    lowf = {(t, P): lowfreq_frac(maps[(t, P)], a.lowfreq_cutoff) for t in ts for P in Ps}
    lcorr = {(t, P): float(np.corrcoef(maps[(t, P)].ravel(), layout.ravel())[0, 1]) for t in ts for P in Ps}
    moran_mean = {P: float(np.mean([moran[(t, P)] for t in ts])) for P in Ps}
    lowf_mean = {P: float(np.mean([lowf[(t, P)] for t in ts])) for P in Ps}
    lcorr_mean = {P: float(np.mean([lcorr[(t, P)] for t in ts])) for P in Ps}

    print("\n[result] spatial correlation of the per-location loss map vs matching patch size")
    print(f"  (P=1 = index-MSE; larger P = more permutation freedom)  model={a.label} step={step}")
    print(f"  {'P':>4} {'M':>6}  {'MoranI':>8}  {'lowfreqFrac':>11}  {'layoutCorr':>10}   (mean over t={ts})")
    for P in Ps:
        print(f"  {P:>4} {P*P:>6}  {moran_mean[P]:>8.4f}  {lowf_mean[P]:>11.4f}  {lcorr_mean[P]:>10.4f}")

    np.savez(os.path.join(out, "spatial_corr.npz"),
             patch_sizes=np.array(Ps), t_values=np.array(ts),
             cost_maps=np.stack([np.stack([maps[(t, P)] for P in Ps]) for t in ts]),
             layout=layout,
             moran=np.array([[moran[(t, P)] for P in Ps] for t in ts]),
             lowfreq=np.array([[lowf[(t, P)] for P in Ps] for t in ts]),
             layoutcorr=np.array([[lcorr[(t, P)] for P in Ps] for t in ts]))
    json.dump({"checkpoint": a.resume, "step": step, "label": a.label, "weights": a.weights,
               "patch_sizes": Ps, "t_values": ts, "n_obj": n_obj, "huber": 0.0,
               "moran_mean": moran_mean, "lowfreq_mean": lowf_mean, "layoutcorr_mean": lcorr_mean},
              open(os.path.join(out, "metadata.json"), "w"), indent=2)

    # Fig 1: cost-map grid (rows=P, cols=t), per-panel scale
    nr, nc = len(Ps), len(ts)
    fig, ax = plt.subplots(nr, nc, figsize=(2.7 * nc, 2.55 * nr), squeeze=False)
    for r, P in enumerate(Ps):
        for c, t in enumerate(ts):
            im = ax[r, c].imshow(maps[(t, P)], cmap="magma")
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(f"t={t}", fontsize=10)
            if c == 0:
                ax[r, c].set_ylabel(f"P={P} (M={P*P})\nMoranI={moran_mean[P]:.3f}", fontsize=8)
            fig.colorbar(im, ax=ax[r, c], fraction=0.046, pad=0.04)
    fig.suptitle(f"Per-location loss map vs matching patch size — {a.label} step {step}\n"
                 f"P=1 is index-MSE; larger P = more permutation-invariant (huber off)", fontsize=11)
    fig.savefig(os.path.join(out, "cost_maps_vs_patch.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

    # Fig 2: spatial-correlation metrics vs P
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    xs = np.arange(len(Ps))
    for ax_, d, dm, name in [(axs[0], moran, moran_mean, "Moran's I (spatial autocorr)"),
                             (axs[1], lowf, lowf_mean, f"low-freq power frac (<{a.lowfreq_cutoff})"),
                             (axs[2], lcorr, lcorr_mean, "corr with GT layout roughness")]:
        for t in ts:
            ax_.plot(xs, [d[(t, P)] for P in Ps], "o-", alpha=0.5, label=f"t={t}")
        ax_.plot(xs, [dm[P] for P in Ps], "s-", color="k", lw=2.5, label="mean")
        ax_.set_xticks(xs); ax_.set_xticklabels([f"{P}\n(M={P*P})" for P in Ps])
        ax_.set_xlabel("matching patch size  (1=index-MSE → more permutation-invariant →)")
        ax_.set_title(name); ax_.grid(alpha=0.3); ax_.legend(fontsize=8)
    fig.suptitle(f"Is the loss less spatially correlated as it becomes more permutation-invariant?  "
                 f"{a.label} step {step}, N={n_obj}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out, "spatial_corr_vs_patch.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\n[done] wrote figures + spatial_corr.npz + metadata.json to:\n  {out}")


if __name__ == "__main__":
    main()
