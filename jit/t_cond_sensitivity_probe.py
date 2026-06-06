"""t-vs-text conditioning sensitivity probe for JiT (pooled-AdaLN).

Question: is the timestep signal `t` being drowned by the pooled-CLIP text signal
in the additive conditioning vector `c = t_emb + y_pool` (jit/models.py:548), so
that the AdaLN modulation barely reconfigures across noise levels?

This reconstructs ONLY the conditioning path straight from a checkpoint's
state_dict (no full model build, no RoPE/bottleneck matching needed):

    t_emb  = L2( SiLU( L0( sinusoid(t) ) ) )            # t_embedder.mlp
    y_pool = P2( SiLU( P0( pooled_clip ) ) )            # y_embedder.proj
    c      = t_emb + y_pool
    m_i    = W_i @ SiLU(c) + b_i                        # blocks.i.adaLN_modulation[1]

Metrics
  (1) magnitude: mean ||t_emb|| vs mean ||y_pool||, and the per-dim variation each
      source injects into c (std over the t-schedule vs std over prompts).
  (2) AdaLN sensitivity: for each block, how much the modulation output swings when
      sweeping t (text fixed) vs sweeping text (t fixed). Ratio<1 => t under-drives.
      Broken out for shift / scale / gate, since `gate` decides whether a sub-block
      fires at all.

Usage:
  source .3dgen/bin/activate
  python jit/t_cond_sensitivity_probe.py output/<run>/0003000.pt [--weights ema|model]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
POOLED_NPY = REPO / "object_classification" / "text_tokens" / "pooled.npy"


def sinusoid(t, dim=256, max_period=10000.0):
    """OpenAI/DiT timestep embedding, matching TimestepEmbedder.timestep_embedding."""
    half = dim // 2
    freqs = torch.exp(-np.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def lin(x, W, b):
    return x @ W.t() + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--n_text", type=int, default=512)
    ap.add_argument("--n_t", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    print(f"Loading {args.ckpt} [{args.weights}] ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt[args.weights]
    sd = {k: v.float() for k, v in sd.items() if v.dtype.is_floating_point}
    D = sd["t_embedder.mlp.0.weight"].shape[0]
    n_blocks = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    step = ckpt.get("step", "?")
    print(f"hidden={D}  blocks={n_blocks}  step={step}")

    g = lambda k: sd[k].to(dev)

    # ---- build t_emb over the full schedule [0, 999] ----
    t_grid = torch.linspace(0, 999, args.n_t).round()
    te = sinusoid(t_grid).to(dev)
    t_emb = lin(F.silu(lin(te, g("t_embedder.mlp.0.weight"), g("t_embedder.mlp.0.bias"))),
                g("t_embedder.mlp.2.weight"), g("t_embedder.mlp.2.bias"))            # (n_t, D)

    # ---- build y_pool over a sample of real prompts ----
    pooled = np.load(POOLED_NPY, mmap_mode="r")
    idx = np.random.default_rng(args.seed).choice(pooled.shape[0], size=args.n_text, replace=False)
    idx.sort()
    yp_in = torch.from_numpy(np.asarray(pooled[idx], dtype=np.float32)).to(dev)        # (n_text, 768)
    y_pool = lin(F.silu(lin(yp_in, g("y_embedder.proj.0.weight"), g("y_embedder.proj.0.bias"))),
                 g("y_embedder.proj.2.weight"), g("y_embedder.proj.2.bias"))          # (n_text, D)

    # =========================== Metric 1: magnitudes ===========================
    print("\n=== (1) embedding magnitudes / variation injected into c = t_emb + y_pool ===")
    print(f"  mean ||t_emb||  (over t-schedule) : {t_emb.norm(dim=1).mean():.3f}")
    print(f"  mean ||y_pool|| (over prompts)    : {y_pool.norm(dim=1).mean():.3f}")
    # per-dim variation each source contributes to c:
    Svar_t = t_emb.std(dim=0).norm().item()       # ||std over t|| (L2 across dims)
    Svar_x = y_pool.std(dim=0).norm().item()       # ||std over text||
    print(f"  variation of c from t   ||std_t(t_emb)||   : {Svar_t:.3f}")
    print(f"  variation of c from text ||std_x(y_pool)|| : {Svar_x:.3f}")
    print(f"  --> t/text variation ratio               : {Svar_t / Svar_x:.3f}")

    # =================== Metric 2: AdaLN modulation sensitivity ===================
    # For a 2D grid (t x text) compute modulation, then:
    #   M_t   = avg over text of [ std over t of modulation ]   (per-dim, L2 aggregated)
    #   M_txt = avg over t    of [ std over text of modulation ]
    # subsample text for the dense grid to keep it light
    nt_g = min(args.n_t, 64)
    nx_g = min(args.n_text, 128)
    t_emb_g = t_emb[torch.linspace(0, t_emb.shape[0] - 1, nt_g).long()]                # (nt_g, D)
    y_pool_g = y_pool[torch.linspace(0, y_pool.shape[0] - 1, nx_g).long()]             # (nx_g, D)
    # c grid: (nt_g, nx_g, D)
    c = F.silu(t_emb_g[:, None, :] + y_pool_g[None, :, :])

    def chunk_stats(W, b, n_chunks, names):
        m = lin(c, W, b)                                                               # (nt, nx, n_chunks*D)
        parts = m.chunk(n_chunks, dim=-1)
        out = {}
        for name, p in zip(names, parts):
            # variation across t (text fixed), averaged over text:
            Mt = p.std(dim=0).norm(dim=-1).mean().item()
            # variation across text (t fixed), averaged over t:
            Mx = p.std(dim=1).norm(dim=-1).mean().item()
            out[name] = (Mt, Mx)
        return out

    print("\n=== (2) AdaLN modulation: swing from t (text fixed) vs text (t fixed) ===")
    print("    ratio = (t-driven swing) / (text-driven swing);  <1 means t under-drives")
    hdr = f"  {'block':>6} | {'shift':>17} | {'scale':>17} | {'gate':>17}"
    print(hdr)
    print(f"  {'':>6} | {'  Mt    Mx  ratio':>17} | {'  Mt    Mx  ratio':>17} | {'  Mt    Mx  ratio':>17}")
    agg = {"shift": [], "scale": [], "gate": []}
    names6 = ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]
    for i in range(n_blocks):
        W = g(f"blocks.{i}.adaLN_modulation.1.weight")
        b = g(f"blocks.{i}.adaLN_modulation.1.bias")
        s = chunk_stats(W, b, 6, names6)
        # average the msa/mlp pair for each of shift/scale/gate for a compact view
        def avg2(a, bn):
            return ((s[a][0] + s[bn][0]) / 2, (s[a][1] + s[bn][1]) / 2)
        row = {"shift": avg2("shift_msa", "shift_mlp"),
               "scale": avg2("scale_msa", "scale_mlp"),
               "gate": avg2("gate_msa", "gate_mlp")}
        cells = []
        for kk in ("shift", "scale", "gate"):
            Mt, Mx = row[kk]
            r = Mt / Mx if Mx > 1e-9 else float("nan")
            agg[kk].append(r)
            cells.append(f"{Mt:5.2f} {Mx:5.2f} {r:5.2f}")
        print(f"  {i:>6} | {cells[0]:>17} | {cells[1]:>17} | {cells[2]:>17}")

    # final layer (shift, scale only)
    Wf = g("final_layer.adaLN_modulation.1.weight")
    bf = g("final_layer.adaLN_modulation.1.bias")
    sf = chunk_stats(Wf, bf, 2, ["shift", "scale"])
    print("  final  | " + " | ".join(
        f"{sf[k][0]:5.2f} {sf[k][1]:5.2f} {sf[k][0]/sf[k][1]:5.2f}".rjust(17)
        for k in ("shift", "scale")) + " |        --        ")

    print("\n=== summary (median over the 12 blocks) ===")
    for kk in ("shift", "scale", "gate"):
        arr = np.array(agg[kk])
        print(f"  {kk:>6}: median t/text ratio = {np.median(arr):.3f}   "
              f"(min {arr.min():.3f}, max {arr.max():.3f})")
    allr = np.array(agg["shift"] + agg["scale"] + agg["gate"])
    print(f"  OVERALL median t/text modulation ratio = {np.median(allr):.3f}")
    print("\n  reading: ratio >> 1  -> t dominates the modulation")
    print("           ratio ~  1  -> balanced")
    print("           ratio << 1  -> text dominates, t is being drowned")


if __name__ == "__main__":
    main()
