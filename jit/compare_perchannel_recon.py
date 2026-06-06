#!/usr/bin/env python
"""Overlay two sinkhorn_perchannel_probe runs (e.g. sinkhorn-recon vs MSE-recon) so the
per-channel difficulty + loss-vs-t SHAPE can be compared side by side. Each run is probed
with the recon loss it was trained on, so absolute magnitudes carry a metric caveat (the
sinkhorn run reassigns within 8x8 patches → its residual is a lower bound on index error;
the MSE run is index-aligned). Compare RANKING and t-SHAPE, not raw magnitudes."""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CHANNEL_NAMES = ["x","y","z","opacity","color_dc_R","color_dc_G","color_dc_B",
                 "log_scale_0","log_scale_1","log_scale_2","quat_w","quat_x","quat_y","quat_z"]
GROUPS = [("xyz",[0,1,2]),("opacity",[3]),("color",[4,5,6]),("scale",[7,8,9]),("rotation",[10,11,12,13])]


def load(run_dir):
    d = np.load(os.path.join(run_dir, "maps.npz"), allow_pickle=True)
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    return d, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="run dir A (e.g. sinkhorn)")
    p.add_argument("--b", required=True, help="run dir B (e.g. mse)")
    p.add_argument("--label_a", default="A")
    p.add_argument("--label_b", default="B")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    da, ma = load(args.a); db, mb = load(args.b)
    ta, tb = da["t_values"], db["t_values"]
    pma, pmb = da["perchannel_mean_vs_t"], db["perchannel_mean_vs_t"]   # (n_t, C)
    la = da["hard_loss_per_t"]; lb = db["hard_loss_per_t"]
    cmap = plt.get_cmap("tab10")
    gcol = {g: cmap(i) for i, (g, _) in enumerate(GROUPS)}
    La = f"{args.label_a} (step {ma['checkpoint_step']}, {ma['recon_loss']} P={ma['patch_size']})"
    Lb = f"{args.label_b} (step {mb['checkpoint_step']}, {mb['recon_loss']} P={mb['patch_size']})"

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])

    # (0,0) total recon loss vs t — the flat-vs-steep headline
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ta, la, "o-", color="C3", lw=2, label=args.label_a)
    ax.plot(tb, lb, "s--", color="C0", lw=2, label=args.label_b)
    ax.set_xlabel("t (0=noise, 1=clean)"); ax.set_ylabel("total per-cell recon loss (Σ channels)")
    ax.set_title("Loss vs t — shape (flat=sinkhorn, steep=MSE)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (0,1) per-group mean vs t, both models
    ax = fig.add_subplot(gs[0, 1])
    for g, idxs in GROUPS:
        ax.plot(ta, pma[:, idxs].mean(1), "o-", color=gcol[g], lw=1.8)
        ax.plot(tb, pmb[:, idxs].mean(1), "s--", color=gcol[g], lw=1.8, alpha=0.8)
    ax.set_yscale("log"); ax.set_xlabel("t (0=noise, 1=clean)"); ax.set_ylabel("group-mean residual")
    ax.set_title("Per-group residual vs t  (solid=A, dashed=B)"); ax.grid(alpha=0.3)
    ax.legend(handles=[Line2D([0],[0],color=gcol[g],lw=2,label=g) for g,_ in GROUPS], fontsize=8)

    # (0,2) clean-end (last t) per-channel bars
    ax = fig.add_subplot(gs[0, 2])
    order = np.argsort(-(pma[-1] + pmb[-1]) / 2)
    yy = np.arange(len(order))
    ax.barh(yy - 0.2, pma[-1][order], 0.4, color="C3", label=args.label_a)
    ax.barh(yy + 0.2, pmb[-1][order], 0.4, color="C0", label=args.label_b)
    ax.set_yticks(yy); ax.set_yticklabels([CHANNEL_NAMES[i] for i in order], fontsize=7)
    ax.invert_yaxis(); ax.set_title(f"Per-channel residual @ CLEAN (t={ta[-1]})")
    ax.set_xlabel("mean matched residual"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

    # (1,0) noisy-end per-channel bars
    ax = fig.add_subplot(gs[1, 0])
    order = np.argsort(-(pma[0] + pmb[0]) / 2)
    yy = np.arange(len(order))
    ax.barh(yy - 0.2, pma[0][order], 0.4, color="C3", label=args.label_a)
    ax.barh(yy + 0.2, pmb[0][order], 0.4, color="C0", label=args.label_b)
    ax.set_yticks(yy); ax.set_yticklabels([CHANNEL_NAMES[i] for i in order], fontsize=7)
    ax.invert_yaxis(); ax.set_title(f"Per-channel residual @ NOISE (t={ta[0]})")
    ax.set_xlabel("mean matched residual"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

    # (1,1) overall (t-averaged) per-channel bars
    ax = fig.add_subplot(gs[1, 1])
    oa, ob = pma.mean(0), pmb.mean(0)
    order = np.argsort(-(oa + ob) / 2)
    yy = np.arange(len(order))
    ax.barh(yy - 0.2, oa[order], 0.4, color="C3", label=args.label_a)
    ax.barh(yy + 0.2, ob[order], 0.4, color="C0", label=args.label_b)
    ax.set_yticks(yy); ax.set_yticklabels([CHANNEL_NAMES[i] for i in order], fontsize=7)
    ax.invert_yaxis(); ax.set_title("Per-channel residual (t-averaged)")
    ax.set_xlabel("mean matched residual"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

    # (1,2) text panel
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    lines = [f"A = {La}", f"B = {Lb}", "",
             "Both: JiT-B/8, pooled-CLIP AdaLN, render OFF,", "identical 14-ch preprocessing.", "",
             "CAVEAT: each probed with its OWN recon loss →", "compare ranking & t-shape, not raw magnitude",
             "(sinkhorn reassigns within 8x8 → lower bound).", "",
             "clean-end total loss:  A=%.3f  B=%.3f" % (la[-1], lb[-1]),
             "noise-end total loss:  A=%.3f  B=%.3f" % (la[0], lb[0])]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

    fig.suptitle(f"Per-channel recon difficulty: {args.label_a} vs {args.label_b}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
