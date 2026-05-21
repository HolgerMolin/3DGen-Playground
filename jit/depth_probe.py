"""Depth-wise linear probe diagnostic for JiT checkpoints.

DISABLED on the text_embeddings branch: the probe trains an `nn.Linear(d_model,
num_classes)` per block, which has no analog under continuous text-embedding
conditioning. Rework to a per-block retrieval-against-prompts probe before
re-enabling.
"""

raise NotImplementedError(
    "jit/depth_probe.py is class-only and not ported to the text_embeddings "
    "branch. Rework the linear-classifier-per-block scheme into a retrieval@k "
    "probe over held-out captions before re-enabling."
)

_LEGACY_DOC = """For each transformer block in a frozen JiT trunk, this script reports:

  (1) probe_eval_acc — accuracy of a fresh ``nn.Linear(d_model, num_classes)``
      trained with plain CE to predict the class label from the mean-pooled
      residual stream output of that block. Generalises the aux-classifier
      result (``aux_classifier.md``) from the readout to every depth.

  (2) class_variance_fraction — the fraction of total residual stream variance
      that is attributable to varying the class label at fixed x_t. This
      separates "class is *decodable*" from "class *drives output magnitude*".

The two metrics together distinguish the three mechanistic stories about why
``cfg_signal`` is collapsed near 0.06 on the renderprobe checkpoint:

    Story A — class info is preserved through depth, modulation does not use it.
              probe acc stays high; class_variance_fraction also stays high.
    Story B — class info is dropped early, partially reconstructed from x_t.
              probe acc dips mid-trunk and recovers near the readout.
    Story C — class info is preserved but at negligible magnitude.
              probe acc stays high; class_variance_fraction is tiny.

The script is read-only:
  - never modifies the checkpoint
  - never writes anything under the model's output directory
  - dumps results under ``runs/depth_probe_<run>_<ckpt>_<weights>_t<NNN>_<ts>.{json,png}``

Typical use:

    source .3dgen/bin/activate
    python jit/depth_probe.py \\
        --checkpoint output/jit_JiT-B_8_full_improved_renderprobe/0260000.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
GS_ROOT = os.path.join(REPO_ROOT, "submodules", "gaussian-splatting")
if GS_ROOT not in sys.path:
    sys.path.insert(0, GS_ROOT)

from dataloaders.class_3dgen_loader import DC_ONLY_FEATURE_INDICES, FULL_3DGS_FEATURE_DIM
from jit.diffusion import create_diffusion
from jit.probe_conditioning import (
    _build_model,
    _infer_num_classes_from_state,
)


def _resolve_input_shape(model: nn.Module, batch_size: int, in_channels: int):
    return (batch_size, in_channels, model.sample_size, model.sample_size)


def _attach_block_hooks(model: nn.Module):
    """Register forward hooks on every JiT block to capture its residual stream output.

    Returns ``(storage, handles)``. ``storage[i]`` is overwritten on every forward
    pass with the output of ``model.blocks[i]`` (post-attn-residual, post-mlp-residual,
    shape ``(B, T, D)``). Caller must release the handles.
    """
    depth = len(model.blocks)
    storage = [None] * depth
    handles = []

    def make_hook(idx: int):
        def hook(_module, _inp, out):
            storage[idx] = out.detach()
        return hook

    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))
    return storage, handles


@torch.no_grad()
def collect_pooled_features(
    model: nn.Module,
    *,
    num_samples: int,
    in_channels: int,
    num_classes: int,
    diffusion_num_timesteps: int,
    t_value: float,
    device: torch.device,
    batch_size: int,
    seed: int,
):
    """Forward ``num_samples`` (random x_t, random y, fixed t) examples through the
    frozen trunk and return mean-pooled residual stream features at every block.

    Returns:
        features : (num_samples, depth, D) float32 on CPU
        labels   : (num_samples,) long on CPU
    """
    storage, handles = _attach_block_hooks(model)
    try:
        feat_chunks = []
        label_chunks = []
        g = torch.Generator(device=device).manual_seed(int(seed))
        g_cpu = torch.Generator(device="cpu").manual_seed(int(seed) + 1)

        t_disc_value = int(round(t_value * (diffusion_num_timesteps - 1)))
        n_done = 0
        while n_done < num_samples:
            bs = min(batch_size, num_samples - n_done)
            x_t = torch.randn(
                *_resolve_input_shape(model, bs, in_channels),
                generator=g, device=device,
            )
            t = torch.full((bs,), t_disc_value, dtype=torch.long, device=device)
            y = torch.randint(0, num_classes, (bs,), generator=g_cpu).to(
                device=device, dtype=torch.long
            )
            _ = model(x_t, t, y)
            # Mean-pool tokens per block, stack to (bs, depth, D), move to CPU.
            pooled = torch.stack(
                [s.mean(dim=1).float().cpu() for s in storage], dim=1
            )
            feat_chunks.append(pooled)
            label_chunks.append(y.cpu())
            n_done += bs

        features = torch.cat(feat_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)
        return features, labels
    finally:
        for h in handles:
            h.remove()


def train_linear_probe_per_block(
    train_features: torch.Tensor,   # (N, depth, D) cpu
    train_labels: torch.Tensor,     # (N,) cpu long
    eval_features: torch.Tensor,    # (M, depth, D) cpu
    eval_labels: torch.Tensor,      # (M,) cpu long
    num_classes: int,
    *,
    num_steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    seed: int,
):
    """Train a fresh linear probe per block; return per-block train/eval acc + losses."""
    depth = train_features.shape[1]
    D = train_features.shape[2]

    train_feats_dev = train_features.to(device)
    train_labels_dev = train_labels.to(device)
    eval_feats_dev = eval_features.to(device)
    eval_labels_dev = eval_labels.to(device)
    n_train = train_feats_dev.shape[0]

    results = []
    for k in range(depth):
        torch.manual_seed(seed + k)
        probe = nn.Linear(D, num_classes).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

        last_loss = float("nan")
        for _ in range(num_steps):
            idx = torch.randint(0, n_train, (batch_size,), device=device)
            xb = train_feats_dev[idx, k]
            yb = train_labels_dev[idx]
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().item())

        probe.eval()
        with torch.no_grad():
            train_logits = probe(train_feats_dev[:, k])
            train_acc = (train_logits.argmax(-1) == train_labels_dev).float().mean().item()
            eval_logits = probe(eval_feats_dev[:, k])
            eval_acc = (eval_logits.argmax(-1) == eval_labels_dev).float().mean().item()
            eval_loss = F.cross_entropy(eval_logits, eval_labels_dev).item()

        results.append({
            "block_idx": k,
            "probe_train_acc": float(train_acc),
            "probe_eval_acc": float(eval_acc),
            "probe_final_loss": float(last_loss),
            "probe_eval_loss": float(eval_loss),
        })
        print(
            f"  block {k:2d}: train_acc={train_acc:.4f}  eval_acc={eval_acc:.4f}  "
            f"final_train_loss={last_loss:.4f}  eval_loss={eval_loss:.4f}"
        )

    return results


@torch.no_grad()
def measure_class_variance_fraction(
    model: nn.Module,
    *,
    num_xt: int,
    num_classes_sample: int,
    num_classes: int,
    in_channels: int,
    diffusion_num_timesteps: int,
    t_value: float,
    device: torch.device,
    seed: int,
):
    """Per block, compute Var_y(features) / Var_total(features) on the full residual stream.

    Sample M=num_xt fixed x_t draws and N=num_classes_sample random class labels.
    Forward all M*N combinations (M outer iterations, batch of N), and per block:

        class_var = E_{m, t, d}[ Var_n( R[m, n, t, d] ) ]
        total_var = E_{t, d}[ Var_{m, n}( R[m, n, t, d] ) ]
        class_variance_fraction = class_var / total_var

    Both variances are population (unbiased=False). The class set is fixed across
    all M draws so the n-axis is comparable.
    """
    storage, handles = _attach_block_hooks(model)
    try:
        depth = len(model.blocks)
        g_xt = torch.Generator(device=device).manual_seed(int(seed) + 7)
        g_cpu = torch.Generator(device="cpu").manual_seed(int(seed) + 11)

        K = max(2, min(num_classes_sample, num_classes))
        class_pool = torch.randperm(num_classes, generator=g_cpu)[:K].to(
            device=device, dtype=torch.long
        )

        t_disc_value = int(round(t_value * (diffusion_num_timesteps - 1)))
        per_block_R = [[] for _ in range(depth)]

        for _ in range(num_xt):
            x_t_single = torch.randn(
                *_resolve_input_shape(model, 1, in_channels),
                generator=g_xt, device=device,
            )
            x_t = x_t_single.expand(K, -1, -1, -1).contiguous()
            t = torch.full((K,), t_disc_value, dtype=torch.long, device=device)
            _ = model(x_t, t, class_pool)
            for k in range(depth):
                per_block_R[k].append(storage[k].float().cpu())

        results = []
        for k in range(depth):
            R = torch.stack(per_block_R[k], dim=0)  # (M, N, T, D)
            class_var = R.var(dim=1, unbiased=False).mean().item()
            M, N, T, D = R.shape
            total_var = R.reshape(M * N, T, D).var(dim=0, unbiased=False).mean().item()
            frac = class_var / total_var if total_var > 0 else float("nan")
            results.append({
                "block_idx": k,
                "class_variance_fraction": float(frac),
                "residual_total_variance": float(total_var),
                "residual_class_variance": float(class_var),
            })
            print(
                f"  block {k:2d}: class_var={class_var:.4e}  "
                f"total_var={total_var:.4e}  fraction={frac:.4f}"
            )
        return results
    finally:
        for h in handles:
            h.remove()


def _safe_ckpt_name(ckpt_path: Path) -> str:
    parent = ckpt_path.parent.name or "ckpt"
    stem = ckpt_path.stem
    return f"{parent}_{stem}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .pt checkpoint. EMA weights are loaded by default.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--t_value", type=float, default=0.5,
        help="t value at which to probe (FM convention: 0=noise, 1=clean). "
             "0.5 matches the existing per-class probe in runs/.",
    )
    parser.add_argument("--num_train_samples", type=int, default=4096)
    parser.add_argument("--num_eval_samples", type=int, default=1024)
    parser.add_argument(
        "--forward_batch_size", type=int, default=32,
        help="Batch size for trunk forwards when collecting features.",
    )
    parser.add_argument("--probe_steps", type=int, default=1500)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=0.0)
    parser.add_argument("--probe_batch_size", type=int, default=128)
    parser.add_argument(
        "--num_xt", type=int, default=10,
        help="Number of distinct x_t draws for class_variance_fraction.",
    )
    parser.add_argument(
        "--num_classes_sample", type=int, default=20,
        help="Number of class labels to permute for class_variance_fraction.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--weights", type=str, choices=["ema", "live"], default="ema",
        help="Which set of weights to probe.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="runs",
        help="Directory for JSON and PNG output (created if missing).",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    step = ckpt.get("step", "?")
    print(
        f"  step={step}  model={saved_args.get('model')}  weights={args.weights}"
    )

    in_channels = (
        len(DC_ONLY_FEATURE_INDICES)
        if saved_args.get("sh_degree0_only", True)
        else FULL_3DGS_FEATURE_DIM
    )
    has_dropout = float(saved_args.get("class_dropout_prob", 0.1)) > 0.0
    num_classes = _infer_num_classes_from_state(ckpt["model"], has_dropout=has_dropout)
    print(f"  num_classes={num_classes}  in_channels={in_channels}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _build_model(
        saved_args, num_classes=num_classes, in_channels=in_channels
    ).to(device)

    state_key = "ema" if args.weights == "ema" else "model"
    if state_key not in ckpt:
        raise KeyError(
            f"Checkpoint has no '{state_key}' state dict. "
            f"Keys: {sorted(ckpt.keys())}"
        )
    missing, unexpected = model.load_state_dict(ckpt[state_key], strict=False)
    if missing or unexpected:
        print(
            f"  [{args.weights}] non-strict load: "
            f"missing={missing} unexpected={unexpected}"
        )

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    diffusion = create_diffusion(
        timestep_respacing="",
        predict_xstart=saved_args.get("predict_xstart", True),
        noise_schedule=saved_args.get("noise_schedule", "squaredcos_cap_v2"),
    )
    diffusion_num_timesteps = diffusion.num_timesteps
    depth = len(model.blocks)
    t_disc = int(round(args.t_value * (diffusion_num_timesteps - 1)))
    print(
        f"  trunk depth={depth}  diffusion_num_timesteps={diffusion_num_timesteps}\n"
        f"  t_value={args.t_value}  -> integer t={t_disc}"
    )

    print(f"\nCollecting probe-train features ({args.num_train_samples} samples)...")
    train_features, train_labels = collect_pooled_features(
        model,
        num_samples=args.num_train_samples,
        in_channels=in_channels,
        num_classes=num_classes,
        diffusion_num_timesteps=diffusion_num_timesteps,
        t_value=args.t_value,
        device=device,
        batch_size=args.forward_batch_size,
        seed=args.seed,
    )

    print(f"Collecting probe-eval features ({args.num_eval_samples} samples)...")
    eval_features, eval_labels = collect_pooled_features(
        model,
        num_samples=args.num_eval_samples,
        in_channels=in_channels,
        num_classes=num_classes,
        diffusion_num_timesteps=diffusion_num_timesteps,
        t_value=args.t_value,
        device=device,
        batch_size=args.forward_batch_size,
        seed=args.seed + 1000,
    )

    print(
        f"\nTraining linear probe per block "
        f"(num_steps={args.probe_steps}, lr={args.probe_lr})..."
    )
    probe_results = train_linear_probe_per_block(
        train_features, train_labels, eval_features, eval_labels,
        num_classes=num_classes,
        num_steps=args.probe_steps,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        batch_size=args.probe_batch_size,
        device=device,
        seed=args.seed,
    )

    print(
        f"\nMeasuring class_variance_fraction "
        f"(num_xt={args.num_xt}, num_classes_sample={args.num_classes_sample})..."
    )
    var_results = measure_class_variance_fraction(
        model,
        num_xt=args.num_xt,
        num_classes_sample=args.num_classes_sample,
        num_classes=num_classes,
        in_channels=in_channels,
        diffusion_num_timesteps=diffusion_num_timesteps,
        t_value=args.t_value,
        device=device,
        seed=args.seed,
    )

    by_block = {r["block_idx"]: dict(r) for r in probe_results}
    for r in var_results:
        by_block[r["block_idx"]].update(r)
    per_block = [by_block[k] for k in sorted(by_block.keys())]

    chance_acc = 1.0 / num_classes

    output = {
        "checkpoint": str(ckpt_path),
        "step": step,
        "weights": args.weights,
        "model": saved_args.get("model"),
        "num_classes": num_classes,
        "in_channels": in_channels,
        "depth": depth,
        "t_value": float(args.t_value),
        "diffusion_t_disc": t_disc,
        "chance_accuracy": float(chance_acc),
        "probe": {
            "num_train_samples": args.num_train_samples,
            "num_eval_samples": args.num_eval_samples,
            "num_steps": args.probe_steps,
            "lr": args.probe_lr,
            "weight_decay": args.probe_weight_decay,
            "batch_size": args.probe_batch_size,
        },
        "class_variance": {
            "num_xt": args.num_xt,
            "num_classes_sample": args.num_classes_sample,
        },
        "per_block": per_block,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = (
        f"depth_probe_{_safe_ckpt_name(ckpt_path)}_{args.weights}"
        f"_t{int(round(args.t_value * 100)):03d}_{timestamp}"
    )
    json_path = os.path.join(args.output_dir, f"{base}.json")
    png_path = os.path.join(args.output_dir, f"{base}.png")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {json_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["block_idx"] for r in per_block]
        train_acc = [r["probe_train_acc"] for r in per_block]
        eval_acc = [r["probe_eval_acc"] for r in per_block]
        cvf = [r["class_variance_fraction"] for r in per_block]

        fig, ax_left = plt.subplots(figsize=(8, 5))
        l_eval, = ax_left.plot(xs, eval_acc, "o-", color="tab:blue", label="probe eval acc")
        l_train, = ax_left.plot(
            xs, train_acc, "o--", color="tab:cyan",
            label="probe train acc", alpha=0.6,
        )
        l_chance = ax_left.axhline(
            chance_acc, linestyle=":", color="grey",
            label=f"chance ({chance_acc:.3f})",
        )
        ax_left.set_ylabel("probe accuracy")
        ax_left.set_xlabel("block index (0=first, depth-1=last)")
        ax_left.set_ylim(0, 1.0)

        ax_right = ax_left.twinx()
        l_cvf, = ax_right.plot(
            xs, cvf, "s-", color="tab:red",
            label="class_variance_fraction",
        )
        ax_right.set_ylabel("class_variance_fraction (log)")
        ax_right.set_yscale("log")

        ax_left.legend(
            handles=[l_eval, l_train, l_chance, l_cvf], loc="best"
        )
        plt.title(
            f"Depth-wise class probe — {ckpt_path.name}  step {step}  "
            f"({args.weights}, t={args.t_value:.2f})"
        )
        plt.tight_layout()
        plt.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"Wrote {png_path}")
    except Exception as e:
        print(f"Plot failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
