#!/usr/bin/env python
"""How spatially correlated is the per-splat-location loss, per noise regime?

Operates on the CACHED channel-summed HARD-cost atlases already written by
jit/sinkhorn_perchannel_probe.py (maps.npz key `summed_hard_atlas`, shape
(n_t, 128, 128)) — NO GPU, no model. For each run × noise level it measures how
much the loss field clusters spatially (nearby atlas cells → similar loss):

  • Moran's I        — 4-neighbour spatial autocorrelation (≈0 ⇒ spatially random,
                       →1 ⇒ strongly clustered). Copied verbatim from
                       jit/loss_spatial_correlation.py for codebase consistency.
  • low-freq frac    — share of FFT power below a low spatial-frequency cutoff
                       (large-scale banding/blob structure). Same source.
  • corr length ξ    — lag (atlas pixels) where the radially-averaged, overlap-
                       normalised autocorrelation first decays to 1/e. The physical
                       scale of the correlation.

Noise regime is 100·(1−t)% (FM: t=0 noise → t=1 clean), matching the figure.

    .3dgen/bin/python jit/measure_loss_spatial_corr.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_OUT = REPO_ROOT / "jit" / "sinkhorn_perchannel_probe_out"
REPORT_OUT = REPO_ROOT / "output" / "report"

RUNS = [("huber_d1.7_step5k", "Sinkhorn"), ("mse_match5k_b8_5k", "MSE")]
EXCLUDE_T = {0.9}
LOWFREQ_CUTOFF = 0.05


# ── Metrics (morans_i / lowfreq_frac copied from jit/loss_spatial_correlation.py) ──
def morans_i(x):
    z = x - x.mean()
    vp = (z[:-1, :] * z[1:, :]).sum(); hp = (z[:, :-1] * z[:, 1:]).sum()
    P_pairs = (x.shape[0] - 1) * x.shape[1] + x.shape[0] * (x.shape[1] - 1)
    den = (z * z).sum()
    return float((x.size / P_pairs) * (vp + hp) / (den + 1e-12))


def lowfreq_frac(x, cutoff=LOWFREQ_CUTOFF):
    z = x - x.mean()
    P = np.abs(np.fft.fft2(z)) ** 2; P[0, 0] = 0.0
    fy = np.fft.fftfreq(x.shape[0])[:, None]; fx = np.fft.fftfreq(x.shape[1])[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    tot = P.sum()
    return float(P[r <= cutoff].sum() / (tot + 1e-12))


def corr_length(x):
    """1/e lag of the radially-averaged, overlap-normalised 2D autocorrelation."""
    z = x - x.mean()
    H, W = z.shape
    sH, sW = 2 * H, 2 * W
    F = np.fft.rfft2(z, s=(sH, sW))
    ac = np.fft.irfft2(F * np.conj(F), s=(sH, sW))
    Fo = np.fft.rfft2(np.ones_like(z), s=(sH, sW))
    cnt = np.fft.irfft2(Fo * np.conj(Fo), s=(sH, sW))
    ac = np.fft.fftshift(ac / np.maximum(cnt, 1.0))
    cy, cx = sH // 2, sW // 2
    ac = ac / ac[cy, cx]
    yy, xx = np.indices(ac.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bins = np.arange(0, H // 2 + 1)
    prof = np.array([ac[(rr >= b - 0.5) & (rr < b + 0.5)].mean() for b in bins])
    thr = 1.0 / np.e
    below = np.where(prof <= thr)[0]
    if len(below) == 0:
        return float(bins[-1]), bins, prof          # never decays in range
    i = below[0]
    if i == 0:
        return 0.0, bins, prof
    x0, x1, y0, y1 = bins[i - 1], bins[i], prof[i - 1], prof[i]
    xi = x0 + (thr - y0) * (x1 - x0) / (y1 - y0)
    return float(xi), bins, prof


def _use_times_like_font():
    import glob
    for d in [str(Path.home() / ".fonts"), "/usr/share/fonts"]:
        for f in glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True):
            if any(k in os.path.basename(f).lower() for k in ("times", "tinos", "liberationserif")):
                try: fm.fontManager.addfont(f)
                except Exception: pass
    avail = {f.name for f in fm.fontManager.ttflist}
    for fam in ("Times New Roman", "Tinos", "Liberation Serif"):
        if fam in avail:
            plt.rcParams["font.family"] = "serif"; plt.rcParams["font.serif"] = [fam]
            return fam
    return "serif"


def main():
    _use_times_like_font()
    results = {}
    curves = {}
    for name, label in RUNS:
        d = np.load(PROBE_OUT / name / "maps.npz", allow_pickle=True)
        summed, t_values = d["summed_hard_atlas"], d["t_values"]
        keep = [i for i, t in enumerate(t_values)
                if not any(abs(float(t) - x) < 1e-6 for x in EXCLUDE_T)]
        for i in keep:
            t = float(t_values[i]); noise = round(100 * (1 - t))
            m = morans_i(summed[i]); lf = lowfreq_frac(summed[i])
            xi, bins, prof = corr_length(summed[i])
            results[(label, noise)] = {"t": t, "noise_pct": noise, "morans_I": m,
                                       "lowfreq_frac": lf, "corr_length_px": xi}
            curves[(label, noise)] = (bins, prof)

    # ── Print table ──────────────────────────────────────────────────────────
    noises = sorted({n for (_, n) in results}, reverse=True)
    print(f"\n[spatial correlation of the per-splat-location loss]  (cutoff={LOWFREQ_CUTOFF})")
    print(f"  {'loss':<9}{'noise':>7}   {'MoranI':>8}  {'lowfreqFrac':>12}  {'corrLen(px)':>12}")
    for _, label in RUNS:
        for n in noises:
            r = results[(label, n)]
            print(f"  {label:<9}{n:>6}%   {r['morans_I']:>8.4f}  "
                  f"{r['lowfreq_frac']:>12.4f}  {r['corr_length_px']:>12.2f}")

    # ── Save numbers ─────────────────────────────────────────────────────────
    os.makedirs(REPORT_OUT, exist_ok=True)
    flat = [{"loss": lbl, **v} for (lbl, n), v in results.items()]
    with open(REPORT_OUT / "loss_spatial_corr.json", "w") as f:
        json.dump({"cutoff": LOWFREQ_CUTOFF, "metric_source": "jit/loss_spatial_correlation.py",
                   "rows": flat}, f, indent=2)

    # ── Companion figure: Moran's I vs noise, both losses ────────────────────
    fig, ax = plt.subplots(figsize=(5.0, 3.6), layout="constrained")
    marker = {"Sinkhorn": "o-", "MSE": "s--"}
    for _, label in RUNS:
        xs = noises
        ys = [results[(label, n)]["morans_I"] for n in xs]
        ax.plot(xs, ys, marker[label], lw=2, ms=7, label=label)
    ax.set_xlabel("noise level (%)", fontsize=13)
    ax.set_ylabel("Moran's I  (spatial autocorrelation)", fontsize=12)
    ax.set_title("Spatial correlation of the loss vs noise", fontsize=14)
    ax.invert_xaxis()                       # 90% (high noise) → 30% reads left→right as denoising
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    fig.savefig(REPORT_OUT / "loss_spatial_corr.png", dpi=150)
    plt.close(fig)
    print(f"\n[done] wrote {REPORT_OUT/'loss_spatial_corr.json'} and "
          f"{REPORT_OUT/'loss_spatial_corr.png'}")


if __name__ == "__main__":
    main()
