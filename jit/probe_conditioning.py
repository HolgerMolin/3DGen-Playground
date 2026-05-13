"""Offline class-conditioning probe for JiT checkpoints.

Loads a saved checkpoint, reconstructs the JiT model from the embedded ``args``,
and runs ``_measure_conditioning_signal`` over many classes / pairs to give a
distribution-aware view of how class-aware the model is at a given step.

Typical use is via ``jit/run_diagnostics.sh`` which orchestrates this together
with ``jit/inspect_weights.py`` and ``jit/diagnose_class_diversity.py``. Direct
invocation:

    source .3dgen/bin/activate
    python jit/probe_conditioning.py \\
        --checkpoint output/<run>/0260000.pt \\
        --per_class_dump_dir output/diagnostics/<run_id>/probe

Per-class JSONs land at ``cond_probe_t{NNN}_per_class.json`` for each t value.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
GS_ROOT = os.path.join(REPO_ROOT, "submodules", "gaussian-splatting")
if GS_ROOT not in sys.path:
    sys.path.insert(0, GS_ROOT)

from dataloaders.class_3dgen_loader import DC_ONLY_FEATURE_INDICES, FULL_3DGS_FEATURE_DIM
from jit.diffusion import create_diffusion
from jit.models import JiT_3DGS_models
from jit._diag_paths import per_class_dump_path, probe_dir
from jit.train_gsplat import _measure_conditioning_signal


def _latest_checkpoint(directory: Path) -> Path:
    cands = sorted(directory.glob("*.pt"))
    if not cands:
        raise FileNotFoundError(f"No .pt checkpoints in {directory}")
    return cands[-1]


def _build_model(args_dict: dict, num_classes: int, in_channels: int):
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


def _infer_num_classes_from_state(state_dict: dict, has_dropout: bool) -> int:
    w = state_dict["y_embedder.embedding_table.weight"]
    rows = w.shape[0]
    return rows - 1 if has_dropout else rows


def _run_probe(
    label: str,
    model: torch.nn.Module,
    *,
    num_classes: int,
    in_channels: int,
    diffusion_num_timesteps: int,
    device: torch.device,
    t_values,
    num_class_probes: int,
    num_class_pairs: int,
    batch_size: int,
    seed: int,
):
    print(f"\n=== {label} ===")
    for t_val in t_values:
        sig = _measure_conditioning_signal(
            model=model,
            num_classes=num_classes,
            in_channels=in_channels,
            diffusion_num_timesteps=diffusion_num_timesteps,
            device=device,
            t_value=float(t_val),
            batch_size=batch_size,
            seed=seed,
            num_class_probes=num_class_probes,
            num_class_pairs=num_class_pairs,
            return_per_class=True,
        )
        per_class = sig["cfg_signal_per_class"]
        sorted_items = sorted(per_class.items(), key=lambda kv: kv[1])
        bottom = sorted_items[:5]
        top = sorted_items[-5:]
        print(
            f"t={t_val:.2f}  cfg=mean {sig['cfg_signal']:.4f}  "
            f"min {sig['cfg_signal_min']:.4f}  max {sig['cfg_signal_max']:.4f}  "
            f"(n_classes={sig['num_class_probes']})"
        )
        print(
            f"          class=mean {sig['class_signal']:.4f}  "
            f"min {sig['class_signal_min']:.4f}  max {sig['class_signal_max']:.4f}  "
            f"(n_pairs={sig['num_class_pairs']})  pred_rms={sig['pred_rms']:.4f}"
        )
        print(
            "          weakest 5 classes (cfg): "
            + ", ".join(f"{c}:{v:.3f}" for c, v in bottom)
        )
        print(
            "          strongest 5 classes (cfg): "
            + ", ".join(f"{c}:{v:.3f}" for c, v in top)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint. Required unless --ckpt_dir is given (latest .pt picked).",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory to scan for the latest .pt when --checkpoint is unset.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num_class_probes",
        type=int,
        default=120,
        help="Number of distinct classes sampled for cfg_signal (capped at num_classes).",
    )
    parser.add_argument(
        "--num_class_pairs",
        type=int,
        default=64,
        help="Number of random class pairs sampled for class_signal.",
    )
    parser.add_argument(
        "--t_values",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="t values (FM convention: 0=noise, 1=clean) at which to run the probe.",
    )
    parser.add_argument(
        "--probe_ema",
        action="store_true",
        default=True,
        help="Probe the EMA weights (used at sampling time). On by default.",
    )
    parser.add_argument(
        "--probe_live",
        action="store_true",
        default=True,
        help="Probe the live (non-EMA) weights too. On by default.",
    )
    parser.add_argument(
        "--per_class_dump_dir",
        type=str,
        default=None,
        help="Directory to write one per-class cfg_signal JSON per t value "
             "(file: cond_probe_t{NNN}_per_class.json). EMA weights are dumped.",
    )
    args = parser.parse_args()

    if not args.checkpoint and not args.ckpt_dir:
        parser.error("Either --checkpoint or --ckpt_dir must be provided.")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(Path(args.ckpt_dir))
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    step = ckpt.get("step", "?")
    print(f"  step={step}  model={saved_args.get('model')}  "
          f"sh_degree0_only={saved_args.get('sh_degree0_only')}  "
          f"class_dropout_prob={saved_args.get('class_dropout_prob')}")

    in_channels = (
        len(DC_ONLY_FEATURE_INDICES)
        if saved_args.get("sh_degree0_only", True)
        else FULL_3DGS_FEATURE_DIM
    )
    has_dropout = float(saved_args.get("class_dropout_prob", 0.1)) > 0.0
    num_classes = _infer_num_classes_from_state(ckpt["model"], has_dropout=has_dropout)
    print(f"  num_classes={num_classes}  in_channels={in_channels}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build a single skeleton, then load each set of weights into it in turn.
    model = _build_model(saved_args, num_classes=num_classes, in_channels=in_channels).to(device)

    diffusion = create_diffusion(
        timestep_respacing="",
        predict_xstart=saved_args.get("predict_xstart", True),
        noise_schedule=saved_args.get("noise_schedule", "squaredcos_cap_v2"),
    )

    common = dict(
        num_classes=num_classes,
        in_channels=in_channels,
        diffusion_num_timesteps=diffusion.num_timesteps,
        device=device,
        t_values=args.t_values,
        num_class_probes=args.num_class_probes,
        num_class_pairs=args.num_class_pairs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if args.probe_live:
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"  [live] non-strict load: missing={missing} unexpected={unexpected}")
        _run_probe("live (training) weights", model, **common)

    if args.probe_ema and "ema" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["ema"], strict=False)
        if missing or unexpected:
            print(f"  [ema] non-strict load: missing={missing} unexpected={unexpected}")
        _run_probe("EMA (sampling) weights", model, **common)

        if args.per_class_dump_dir:
            dump_root = Path(args.per_class_dump_dir)
            dump_root.mkdir(parents=True, exist_ok=True)
            for t_val in args.t_values:
                sig = _measure_conditioning_signal(
                    model=model,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    diffusion_num_timesteps=diffusion.num_timesteps,
                    device=device,
                    t_value=float(t_val),
                    batch_size=args.batch_size,
                    seed=args.seed,
                    num_class_probes=args.num_class_probes,
                    num_class_pairs=args.num_class_pairs,
                    return_per_class=True,
                )
                # Use the orchestrator's filename convention so other tools can find it.
                tag = f"t{int(round(float(t_val) * 100)):03d}"
                out_path = dump_root / f"cond_probe_{tag}_per_class.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "checkpoint": str(ckpt_path),
                            "step": step,
                            "t_value": float(t_val),
                            "weights": "ema",
                            "cfg_signal_per_class": sig["cfg_signal_per_class"],
                            "summary": {
                                "cfg_signal": sig["cfg_signal"],
                                "cfg_signal_min": sig["cfg_signal_min"],
                                "cfg_signal_max": sig["cfg_signal_max"],
                                "class_signal": sig["class_signal"],
                                "class_signal_min": sig["class_signal_min"],
                                "class_signal_max": sig["class_signal_max"],
                                "pred_rms": sig["pred_rms"],
                            },
                        },
                        f,
                        indent=2,
                    )
                print(f"  Per-class dump written: {out_path}")


if __name__ == "__main__":
    main()
