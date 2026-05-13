"""Read-only weight inspection for a trained JiT checkpoint.

Diagnoses the class-conditioning bottleneck by looking at:
  1. LabelEmbedder geometry (cosine sim, distance to null, norms, PCA).
  2. Per-block AdaLN gate statistics from forwarding class embeddings through
     each block's adaLN_modulation.
  3. Per-block AdaLN modulation projection weight/bias norms.

Cross-references per-class cfg_signal (loaded from a probe JSON dump) against
the per-class final-block gate magnitude.

All outputs land under ``--out_dir``. Does not modify any training code.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
GS_ROOT = os.path.join(REPO_ROOT, "submodules", "gaussian-splatting")
if GS_ROOT not in sys.path:
    sys.path.insert(0, GS_ROOT)

from dataloaders.class_3dgen_loader import DC_ONLY_FEATURE_INDICES, FULL_3DGS_FEATURE_DIM
from jit._diag_paths import per_class_dump_path, weights_dir
from jit.models import JiT_3DGS_models


def _build_model(args_dict, num_classes, in_channels):
    return JiT_3DGS_models[args_dict["model"]](
        input_size=128,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=args_dict.get("class_dropout_prob", 0.1),
        learn_sigma=False,
        gradient_checkpointing=False,
        aux_classifier=args_dict.get("aux_classifier", False),
        label_embed_init_std=args_dict.get("label_embed_init_std", 0.02),
    )


def _infer_num_classes_from_state(state_dict, has_dropout):
    rows = state_dict["y_embedder.embedding_table.weight"].shape[0]
    return rows - 1 if has_dropout else rows


@torch.no_grad()
def section1_embedder_geometry(model, num_classes, out_dir, cfg_per_class):
    print("\n=== Section 1: LabelEmbedder geometry ===")
    W = model.y_embedder.embedding_table.weight.detach().cpu().numpy()
    rows, dim = W.shape
    print(f"embedding_table.weight: shape={W.shape}  num_classes={num_classes}  null_id={num_classes}")
    null_idx = num_classes  # last row is the CFG null token
    np.save(out_dir / "embedding_table.npy", W)

    # L2 norms
    norms = np.linalg.norm(W, axis=1)
    print(
        f"L2 norms: mean={norms.mean():.4f}  std={norms.std():.4f}  "
        f"min={norms.min():.4f}  max={norms.max():.4f}  "
        f"null_norm={norms[null_idx]:.4f}"
    )
    norm_summary = {
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "min": float(norms.min()),
        "max": float(norms.max()),
        "null_norm": float(norms[null_idx]),
        "per_class_norm": [float(v) for v in norms[:num_classes]],
    }
    with open(out_dir / "embedding_norms.json", "w") as f:
        json.dump(norm_summary, f, indent=2)

    # Cosine similarity matrix
    Wn = W / np.maximum(norms[:, None], 1e-12)
    cos_full = Wn @ Wn.T  # (rows, rows) — includes null at row/col `null_idx`
    np.save(out_dir / "cosine_similarity_full.npy", cos_full)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cos_full, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_title(f"Cosine similarity matrix ({rows}x{rows}, null at idx {null_idx})")
    ax.axhline(null_idx, color="k", lw=0.5, ls="--")
    ax.axvline(null_idx, color="k", lw=0.5, ls="--")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "cosine_similarity_heatmap.png", dpi=140)
    plt.close(fig)

    # Off-diagonal class-class cosine stats (exclude null row/col)
    cos_classes = cos_full[:num_classes, :num_classes]
    triu = cos_classes[np.triu_indices(num_classes, k=1)]
    cos_to_null = cos_full[:num_classes, null_idx]
    print(
        f"class-class cosine (off-diag): mean={triu.mean():.4f} std={triu.std():.4f} "
        f"min={triu.min():.4f} max={triu.max():.4f}"
    )
    print(
        f"class-to-null cosine: mean={cos_to_null.mean():.4f} std={cos_to_null.std():.4f} "
        f"min={cos_to_null.min():.4f} max={cos_to_null.max():.4f}"
    )

    # L2 distance to null
    dist_to_null = np.linalg.norm(W[:num_classes] - W[null_idx][None, :], axis=1)
    print(
        f"L2 distance to null: mean={dist_to_null.mean():.4f} std={dist_to_null.std():.4f} "
        f"min={dist_to_null.min():.4f} max={dist_to_null.max():.4f}"
    )

    order = np.argsort(dist_to_null)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(np.arange(num_classes), dist_to_null[order], width=1.0, color="#444")
    ax.set_xlabel("class (sorted by distance to null, ascending)")
    ax.set_ylabel("L2 distance to null embedding")
    ax.set_title("Class embedding distance to null token")
    fig.tight_layout()
    fig.savefig(out_dir / "distance_to_null_sorted.png", dpi=140)
    plt.close(fig)

    np.save(out_dir / "distance_to_null.npy", dist_to_null)
    np.save(out_dir / "cos_to_null.npy", cos_to_null)

    # PCA to 2D (numpy SVD)
    Wc = W - W.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
    coords = (U[:, :2] * S[:2])  # equivalent to projecting Wc onto top-2 PCs
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:num_classes, 0], coords[:num_classes, 1], s=14, c="#1f77b4", label="classes")
    ax.scatter(coords[null_idx, 0], coords[null_idx, 1], s=120, c="red", marker="*", label="null", zorder=5)
    explained = (S[:2] ** 2 / (S ** 2).sum()).round(3)
    ax.set_title(f"Embedding PCA (PC1={explained[0]}, PC2={explained[1]} var explained)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "embedding_pca_2d.png", dpi=140)
    plt.close(fig)
    np.save(out_dir / "embedding_pca_2d.npy", coords)

    # Per-class summary used for cross-reference in section 4
    per_class = {
        "norm": [float(v) for v in norms[:num_classes]],
        "dist_to_null": [float(v) for v in dist_to_null],
        "cos_to_null": [float(v) for v in cos_to_null],
        "mean_cos_to_others": [
            float((cos_classes[i].sum() - 1.0) / (num_classes - 1)) for i in range(num_classes)
        ],
    }
    with open(out_dir / "embedder_per_class.json", "w") as f:
        json.dump(per_class, f, indent=2)

    return {
        "norm_mean": float(norms.mean()),
        "null_norm": float(norms[null_idx]),
        "cos_class_class_mean": float(triu.mean()),
        "cos_to_null_mean": float(cos_to_null.mean()),
        "dist_to_null_mean": float(dist_to_null.mean()),
        "per_class": per_class,
    }


@torch.no_grad()
def section2_block_gates(model, num_classes, device, out_dir, cfg_per_class):
    print("\n=== Section 2: Per-block AdaLN gate statistics ===")
    W = model.y_embedder.embedding_table.weight.detach().to(device)
    null_idx = num_classes
    H = W.shape[1]
    rows = W.shape[0]

    block_stats = []
    final_block_per_class_gate_mag = None  # filled for last JiTBlock

    n_blocks = len(model.blocks)
    for bi, block in enumerate(model.blocks):
        out = block.adaLN_modulation(W).detach().cpu()  # (rows, 6*H)
        chunks = out.chunk(6, dim=1)
        # Order: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        gate_msa = chunks[2].numpy()
        gate_mlp = chunks[5].numpy()
        gates_concat = np.concatenate([gate_msa, gate_mlp], axis=1)  # (rows, 2H)

        abs_g = np.abs(gates_concat)
        stats = {
            "block": bi,
            "mean_abs_gate": float(abs_g.mean()),
            "std_gate": float(gates_concat.std()),
            "max_abs_gate": float(abs_g.max()),
            "frac_lt_0.01": float((abs_g < 0.01).mean()),
            "mean_abs_gate_msa": float(np.abs(gate_msa).mean()),
            "mean_abs_gate_mlp": float(np.abs(gate_mlp).mean()),
            "mean_abs_gate_null": float(np.abs(gates_concat[null_idx]).mean()),
            "mean_abs_gate_classes": float(np.abs(gates_concat[:num_classes]).mean()),
        }
        block_stats.append(stats)

        if bi == n_blocks - 1:
            # Per-class mean |gate| (over both positions and hidden dim, classes only)
            final_block_per_class_gate_mag = np.abs(gates_concat[:num_classes]).mean(axis=1)
            np.save(out_dir / "final_block_gate_concat.npy", gates_concat)

    with open(out_dir / "block_gate_stats.json", "w") as f:
        json.dump(block_stats, f, indent=2)

    # Print quick table
    print(f"{'blk':>3} | {'mean|g|':>10} {'std':>10} {'max|g|':>10} {'<0.01':>8} | "
          f"{'msa':>10} {'mlp':>10} | {'null':>10} {'classes':>10}")
    for s in block_stats:
        print(f"{s['block']:>3} | {s['mean_abs_gate']:>10.5f} {s['std_gate']:>10.5f} "
              f"{s['max_abs_gate']:>10.4f} {s['frac_lt_0.01']:>8.3f} | "
              f"{s['mean_abs_gate_msa']:>10.5f} {s['mean_abs_gate_mlp']:>10.5f} | "
              f"{s['mean_abs_gate_null']:>10.5f} {s['mean_abs_gate_classes']:>10.5f}")

    # Plot mean |gate| vs block
    bx = [s["block"] for s in block_stats]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bx, [s["mean_abs_gate"] for s in block_stats], marker="o", label="mean |gate| (all)")
    ax.plot(bx, [s["mean_abs_gate_msa"] for s in block_stats], marker="s", alpha=0.7, label="mean |gate_msa|")
    ax.plot(bx, [s["mean_abs_gate_mlp"] for s in block_stats], marker="^", alpha=0.7, label="mean |gate_mlp|")
    ax.set_xlabel("block index")
    ax.set_ylabel("mean |gate|")
    ax.set_title(f"Per-block AdaLN gate magnitude (averaged over {rows} embeddings, hidden={H})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gate_magnitude_per_block.png", dpi=140)
    plt.close(fig)

    # Final block gate distribution (histogram of all values across classes & hidden dim)
    final_concat = np.load(out_dir / "final_block_gate_concat.npy")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(final_concat[:num_classes].ravel(), bins=120, color="#2c7fb8", alpha=0.85, label="class gates")
    ax.hist(final_concat[null_idx].ravel(), bins=120, color="red", alpha=0.5, label="null gates")
    ax.set_xlabel("gate value (final block)")
    ax.set_ylabel("count")
    ax.set_title(f"Final block (idx {n_blocks-1}) gate distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "final_block_gate_distribution.png", dpi=140)
    plt.close(fig)

    # Per-class final-block gate magnitude vs cfg_signal
    pearson_r = None
    if cfg_per_class is not None and final_block_per_class_gate_mag is not None:
        cfg_arr = np.array([cfg_per_class.get(c, np.nan) for c in range(num_classes)], dtype=float)
        gate_arr = final_block_per_class_gate_mag
        valid = ~np.isnan(cfg_arr)
        if valid.sum() >= 2:
            cv, gv = cfg_arr[valid], gate_arr[valid]
            # Pearson r
            cv_z = (cv - cv.mean()) / (cv.std() + 1e-12)
            gv_z = (gv - gv.mean()) / (gv.std() + 1e-12)
            pearson_r = float((cv_z * gv_z).mean())
            print(f"Pearson r(per-class cfg_signal, final-block mean |gate|) = {pearson_r:+.4f} (n={int(valid.sum())})")

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(gv, cv, s=18, c="#1f77b4", alpha=0.8)
            ax.set_xlabel("final block mean |gate| (per class)")
            ax.set_ylabel("cfg_signal (per class, t=0.3)")
            ax.set_title(f"Per-class: final-block gate vs cfg_signal  Pearson r={pearson_r:+.3f}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "final_block_gate_vs_cfg.png", dpi=140)
            plt.close(fig)
        np.save(out_dir / "final_block_per_class_gate_mag.npy", gate_arr)

    # Also forward through FinalLayer's adaLN_modulation (no gate — just shift, scale)
    final_layer = model.final_layer
    out_fl = final_layer.adaLN_modulation(W).detach().cpu()
    shift_fl, scale_fl = out_fl.chunk(2, dim=1)
    fl_stats = {
        "mean_abs_shift": float(np.abs(shift_fl.numpy()).mean()),
        "mean_abs_scale": float(np.abs(scale_fl.numpy()).mean()),
        "std_shift": float(shift_fl.numpy().std()),
        "std_scale": float(scale_fl.numpy().std()),
        "frac_shift_lt_0.01": float((np.abs(shift_fl.numpy()) < 0.01).mean()),
        "frac_scale_lt_0.01": float((np.abs(scale_fl.numpy()) < 0.01).mean()),
        "mean_abs_shift_null": float(np.abs(shift_fl[null_idx].numpy()).mean()),
        "mean_abs_scale_null": float(np.abs(scale_fl[null_idx].numpy()).mean()),
    }
    with open(out_dir / "final_layer_adaln_stats.json", "w") as f:
        json.dump(fl_stats, f, indent=2)
    print(f"FinalLayer adaLN: mean|shift|={fl_stats['mean_abs_shift']:.5f}  "
          f"mean|scale|={fl_stats['mean_abs_scale']:.5f}  "
          f"frac<0.01 shift/scale={fl_stats['frac_shift_lt_0.01']:.3f}/"
          f"{fl_stats['frac_scale_lt_0.01']:.3f}")

    return {
        "block_stats": block_stats,
        "final_layer_stats": fl_stats,
        "final_block_per_class_gate_mag": (
            None if final_block_per_class_gate_mag is None else final_block_per_class_gate_mag.tolist()
        ),
        "pearson_r_gate_vs_cfg": pearson_r,
    }


@torch.no_grad()
def section3_proj_norms(model, out_dir):
    print("\n=== Section 3: AdaLN projection weight/bias norms ===")
    rows = []
    for bi, block in enumerate(model.blocks):
        lin = block.adaLN_modulation[-1]
        wn = float(lin.weight.detach().cpu().norm().item())
        bn = float(lin.bias.detach().cpu().norm().item())
        rows.append({"block": bi, "weight_fro": wn, "bias_l2": bn})

    fl = model.final_layer.adaLN_modulation[-1]
    fl_wn = float(fl.weight.detach().cpu().norm().item())
    fl_bn = float(fl.bias.detach().cpu().norm().item())
    fl_lin_wn = float(model.final_layer.linear.weight.detach().cpu().norm().item())
    fl_lin_bn = float(model.final_layer.linear.bias.detach().cpu().norm().item())

    with open(out_dir / "adaln_proj_norms.json", "w") as f:
        json.dump(
            {
                "blocks": rows,
                "final_layer_adaln_weight_fro": fl_wn,
                "final_layer_adaln_bias_l2": fl_bn,
                "final_layer_output_linear_weight_fro": fl_lin_wn,
                "final_layer_output_linear_bias_l2": fl_lin_bn,
            },
            f,
            indent=2,
        )

    print(f"{'blk':>3} | {'||W||_F':>10} {'||b||_2':>10}")
    for r in rows:
        print(f"{r['block']:>3} | {r['weight_fro']:>10.4f} {r['bias_l2']:>10.4f}")
    print(
        f"FinalLayer adaLN: ||W||_F={fl_wn:.4f}  ||b||_2={fl_bn:.4f}    "
        f"output linear: ||W||_F={fl_lin_wn:.4f}  ||b||_2={fl_lin_bn:.4f}"
    )

    bx = [r["block"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bx, [r["weight_fro"] for r in rows], marker="o", label="||W_adaLN||_F")
    ax.plot(bx, [r["bias_l2"] for r in rows], marker="s", label="||b_adaLN||_2")
    ax.axhline(fl_wn, color="purple", ls="--", alpha=0.7, label=f"final ||W||_F={fl_wn:.2f}")
    ax.set_xlabel("block index")
    ax.set_ylabel("norm")
    ax.set_title("AdaLN projection norms per block (zero-init at start)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "adaln_proj_norms.png", dpi=140)
    plt.close(fig)
    return {
        "blocks": rows,
        "final_layer_adaln_weight_fro": fl_wn,
        "final_layer_adaln_bias_l2": fl_bn,
        "final_layer_output_linear_weight_fro": fl_lin_wn,
        "final_layer_output_linear_bias_l2": fl_lin_bn,
    }


def section4_cross_reference(s1, s2, cfg_per_class, num_classes, out_dir):
    print("\n=== Section 4: Cross-reference cfg_signal vs embedder & gates ===")
    if cfg_per_class is None:
        print("No cfg per-class data provided; skipping cross-reference.")
        return None
    cfg_arr = np.array([cfg_per_class.get(c, np.nan) for c in range(num_classes)], dtype=float)
    sort_idx = np.argsort(cfg_arr)
    weakest = sort_idx[:10]
    strongest = sort_idx[-10:][::-1]

    per_class = s1["per_class"]
    gate_mag = s2.get("final_block_per_class_gate_mag")
    gate_arr = np.array(gate_mag, dtype=float) if gate_mag is not None else None

    def row(c):
        return {
            "class": int(c),
            "cfg_signal": float(cfg_arr[c]),
            "embed_norm": float(per_class["norm"][c]),
            "dist_to_null": float(per_class["dist_to_null"][c]),
            "cos_to_null": float(per_class["cos_to_null"][c]),
            "mean_cos_to_others": float(per_class["mean_cos_to_others"][c]),
            "final_block_mean_abs_gate": (
                float(gate_arr[c]) if gate_arr is not None else None
            ),
        }

    weak_rows = [row(c) for c in weakest]
    strong_rows = [row(c) for c in strongest]
    with open(out_dir / "weak_vs_strong_classes.json", "w") as f:
        json.dump({"weakest_10": weak_rows, "strongest_10": strong_rows}, f, indent=2)

    print("\nWeakest 10 classes by cfg_signal:")
    print(f"{'class':>5} {'cfg':>7} {'norm':>8} {'dnull':>8} {'cosnull':>9} {'cosOthers':>10} "
          f"{'gateMag':>9}")
    for r in weak_rows:
        gm = "-" if r["final_block_mean_abs_gate"] is None else f"{r['final_block_mean_abs_gate']:.4f}"
        print(
            f"{r['class']:>5} {r['cfg_signal']:>7.4f} {r['embed_norm']:>8.4f} "
            f"{r['dist_to_null']:>8.4f} {r['cos_to_null']:>9.4f} "
            f"{r['mean_cos_to_others']:>10.4f} {gm:>9}"
        )

    print("\nStrongest 10 classes by cfg_signal:")
    print(f"{'class':>5} {'cfg':>7} {'norm':>8} {'dnull':>8} {'cosnull':>9} {'cosOthers':>10} "
          f"{'gateMag':>9}")
    for r in strong_rows:
        gm = "-" if r["final_block_mean_abs_gate"] is None else f"{r['final_block_mean_abs_gate']:.4f}"
        print(
            f"{r['class']:>5} {r['cfg_signal']:>7.4f} {r['embed_norm']:>8.4f} "
            f"{r['dist_to_null']:>8.4f} {r['cos_to_null']:>9.4f} "
            f"{r['mean_cos_to_others']:>10.4f} {gm:>9}"
        )

    def mean_over(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "weak_mean": {
            "cfg_signal": mean_over(weak_rows, "cfg_signal"),
            "embed_norm": mean_over(weak_rows, "embed_norm"),
            "dist_to_null": mean_over(weak_rows, "dist_to_null"),
            "cos_to_null": mean_over(weak_rows, "cos_to_null"),
            "mean_cos_to_others": mean_over(weak_rows, "mean_cos_to_others"),
            "final_block_mean_abs_gate": mean_over(weak_rows, "final_block_mean_abs_gate"),
        },
        "strong_mean": {
            "cfg_signal": mean_over(strong_rows, "cfg_signal"),
            "embed_norm": mean_over(strong_rows, "embed_norm"),
            "dist_to_null": mean_over(strong_rows, "dist_to_null"),
            "cos_to_null": mean_over(strong_rows, "cos_to_null"),
            "mean_cos_to_others": mean_over(strong_rows, "mean_cos_to_others"),
            "final_block_mean_abs_gate": mean_over(strong_rows, "final_block_mean_abs_gate"),
        },
    }
    print("\nGroup means (weak vs strong):")
    for k in summary["weak_mean"]:
        wv = summary["weak_mean"][k]
        sv = summary["strong_mean"][k]
        if wv is None or sv is None:
            continue
        print(f"  {k:>30}: weak={wv:>9.4f}   strong={sv:>9.4f}   ratio={sv / (wv + 1e-12):.2f}x")
    with open(out_dir / "weak_vs_strong_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pt checkpoint to inspect.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output dir. Defaults to output/diagnostics/<run_id>/weights/.",
    )
    parser.add_argument(
        "--cfg_per_class_json",
        default=None,
        help="Probe JSON with cfg_signal_per_class field. Defaults to the t=0.5 dump "
             "at output/diagnostics/<run_id>/probe/cond_probe_t050_per_class.json.",
    )
    parser.add_argument(
        "--cfg_per_class_t",
        type=float,
        default=0.5,
        help="t value to use when auto-deriving --cfg_per_class_json (ignored if explicit).",
    )
    parser.add_argument(
        "--use_weights",
        choices=["ema", "model"],
        default="ema",
        help="Which set of weights to inspect (sampling-time = ema).",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else weights_dir(args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir.resolve()}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  top-level keys: {sorted(ckpt.keys())}")
    saved_args = ckpt.get("args", {})
    step = ckpt.get("step", "?")
    print(f"  step={step}  model={saved_args.get('model')}  "
          f"sh_degree0_only={saved_args.get('sh_degree0_only')}  "
          f"class_dropout_prob={saved_args.get('class_dropout_prob')}  "
          f"label_embed_init_std={saved_args.get('label_embed_init_std')}")

    in_channels = (
        len(DC_ONLY_FEATURE_INDICES)
        if saved_args.get("sh_degree0_only", True)
        else FULL_3DGS_FEATURE_DIM
    )
    has_dropout = float(saved_args.get("class_dropout_prob", 0.1)) > 0.0
    state_key = args.use_weights
    if state_key not in ckpt:
        raise SystemExit(f"Checkpoint has no '{state_key}' key. Found: {sorted(ckpt.keys())}")
    state_dict = ckpt[state_key]
    num_classes = _infer_num_classes_from_state(state_dict, has_dropout=has_dropout)
    print(f"  inferred num_classes={num_classes}  in_channels={in_channels}  weights={state_key}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _build_model(saved_args, num_classes=num_classes, in_channels=in_channels).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  non-strict load: missing={missing} unexpected={unexpected}")
    model.eval()

    cfg_json_path = (
        Path(args.cfg_per_class_json)
        if args.cfg_per_class_json
        else per_class_dump_path(args.checkpoint, args.cfg_per_class_t)
    )
    cfg_per_class = None
    if cfg_json_path.exists():
        with open(cfg_json_path) as f:
            probe = json.load(f)
        cfg_per_class = {int(k): float(v) for k, v in probe.get("cfg_signal_per_class", {}).items()}
        print(f"Loaded per-class cfg_signal for {len(cfg_per_class)} classes from {cfg_json_path}")
    else:
        print(f"WARNING: cfg per-class JSON not found at {cfg_json_path}; cross-reference disabled.")

    s1 = section1_embedder_geometry(model, num_classes, out_dir, cfg_per_class)
    s2 = section2_block_gates(model, num_classes, device, out_dir, cfg_per_class)
    s3 = section3_proj_norms(model, out_dir)
    s4 = section4_cross_reference(s1, s2, cfg_per_class, num_classes, out_dir)

    bundle = {
        "checkpoint": args.checkpoint,
        "step": step,
        "use_weights": state_key,
        "num_classes": num_classes,
        "section1_summary": {k: v for k, v in s1.items() if k != "per_class"},
        "section2_final_layer": s2["final_layer_stats"],
        "section2_pearson_r_gate_vs_cfg": s2["pearson_r_gate_vs_cfg"],
        "section3_final_layer_adaln_weight_fro": s3["final_layer_adaln_weight_fro"],
        "section3_final_layer_adaln_bias_l2": s3["final_layer_adaln_bias_l2"],
        "section3_final_layer_output_linear_weight_fro": s3["final_layer_output_linear_weight_fro"],
        "section3_final_layer_output_linear_bias_l2": s3["final_layer_output_linear_bias_l2"],
        "section4_summary": s4,
    }
    with open(out_dir / "inspection_bundle.json", "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"\nWrote inspection bundle: {out_dir / 'inspection_bundle.json'}")


if __name__ == "__main__":
    main()
