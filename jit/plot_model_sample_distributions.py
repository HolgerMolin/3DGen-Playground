"""Per-channel distribution diagnostic for *model-generated* normalized 3DGS samples.

Mirrors ``jit/plot_sample_distributions.py`` (histograms + QQ vs N(0,1) for the
14 DC channels) but feeds samples drawn from a trained JiT checkpoint instead
of the dataset. Useful for eyeballing how close the generative distribution is
to the per-channel N(0,1) the trainer normalizes data to.

Two modes:

* ``--mode one_step``    Draw pure noise (x_t at t=0 under the FM convention
                         x_t = t*x_0 + (1-t)*eps) and take the model's single
                         x_0 prediction. Cheap; isolates the noise->x_0 map
                         the model has learned at the t=0 boundary.

* ``--mode sampler``     Run the full sampler (heun/euler/dpm/ddim/ddpm) via
                         ``jit.sampling.sample_model``. Reflects the actual
                         generative distribution at deployment.

Run from repo root with the project venv activated:
    source .3dgen/bin/activate && python jit/plot_model_sample_distributions.py \\
        --checkpoint path/to/checkpoint.pt \\
        --class_map  object_labels/morphological_labels/object_to_class.json \\
        --mode one_step --num_samples 32
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.class_3dgen_loader import DC_ONLY_FEATURE_INDICES, FULL_3DGS_FEATURE_DIM
from dataloaders.standard_3dgen_loader import Standard3DGenDataset
from jit.models import JiT_3DGS_models
from jit.plot_sample_distributions import (
    CHANNEL_LABELS,
    DC_INDICES,
    _plot_channel,
    _summary_text,
)
from jit.sampling import (
    SAMPLER_CHOICES,
    TIMESTEP_SCHEDULE_CHOICES,
    resolve_sampling_shape,
    sample_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Checkpoint helpers (lifted from infer_gsplat) ────────────────────────────

def _load_checkpoint(path: str, device: torch.device) -> dict:
    logger.info("Loading checkpoint: %s", path)
    return torch.load(path, map_location=device, weights_only=False)


def _ckpt_arg(ckpt: dict, key: str):
    return (ckpt.get("args") or {}).get(key)


def _resolve_arg(cli_value, ckpt: dict, key: str, fallback):
    if cli_value is not None:
        return cli_value
    stored = _ckpt_arg(ckpt, key)
    if stored is not None:
        return stored
    return fallback


def _strip_compile_prefix(state_dict: dict) -> dict:
    """torch.compile wraps the module so every key gets an `_orig_mod.` prefix.

    Strip it so the bare model can load the state dict.
    """
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


# ── Sample generation ────────────────────────────────────────────────────────

@torch.no_grad()
def _generate_one_step_batch(
    *,
    model: torch.nn.Module,
    shape: tuple[int, int, int, int],
    class_labels: torch.Tensor,
    device: torch.device,
    diffusion_steps: int,
    noise_scale: float,
    t_value: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """One forward pass: pure noise at t=t_value -> model x_0 prediction.

    Under the JiT flow-matching convention, x_t = t*x_0 + (1 - t)*eps, so at
    t=0 the input is pure noise and the model is asked to predict x_0 in a
    single step. We pass t_value (default 0.0) directly and let the model
    embed it via the discrete-timestep grid.
    """
    model_dtype = next(model.parameters()).dtype
    noise = noise_scale * torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    t_discrete = int(round(t_value * (diffusion_steps - 1)))
    t_discrete = max(0, min(diffusion_steps - 1, t_discrete))
    t_batch = torch.full((shape[0],), t_discrete, device=device, dtype=torch.long)
    pred_x0 = model(noise.to(dtype=model_dtype), t_batch, class_labels).float()
    return pred_x0


@torch.no_grad()
def _generate_sampler_batch(
    *,
    model: torch.nn.Module,
    shape: tuple[int, int, int, int],
    class_labels: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    predict_xstart: bool,
    noise_schedule: str,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    return sample_model(
        sampler=args.sampler,
        model=model,
        shape=shape,
        class_labels=class_labels,
        num_inference_steps=args.num_steps,
        device=device,
        predict_xstart=predict_xstart,
        noise_schedule=noise_schedule,
        diffusion_steps=args.diffusion_steps,
        solver_order=args.dpm_solver_order,
        algorithm_type=args.dpm_algorithm_type,
        solver_type=args.dpm_solver_type,
        timestep_spacing=args.dpm_timestep_spacing,
        use_karras_sigmas=args.dpm_use_karras_sigmas,
        cfg_scale=args.cfg_scale,
        cfg_interval=tuple(args.cfg_interval),
        t_eps=args.t_eps,
        noise_scale=args.noise_scale,
        timestep_schedule=args.timestep_schedule,
        P_mean=args.P_mean,
        P_std=args.P_std,
        ddim_eta=args.ddim_eta,
        generator=generator,
    )


def _draw_class_labels(
    *,
    num_classes: int,
    batch_size: int,
    fixed_class: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    if fixed_class is not None:
        return torch.full((batch_size,), fixed_class, device=device, dtype=torch.long)
    return torch.randint(0, num_classes, (batch_size,), device=device, dtype=torch.long)


# ── Data sampling ────────────────────────────────────────────────────────────

def _load_data_samples(
    *,
    obj_list,
    gs_path: str,
    mean_file: Optional[str],
    std_file: Optional[str],
    sphere2plane_path: str,
    rank_transform_file: Optional[str],
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """Pull `num_samples` data samples; return (N, C, H, W) float32."""
    ds = Standard3DGenDataset(
        obj_list=obj_list,
        gs_path=gs_path,
        mean_file=mean_file,
        std_file=std_file,
        sphere2plane_path=sphere2plane_path,
        rank_transform_file=rank_transform_file,
    )
    rng = np.random.default_rng(seed)
    n = min(num_samples, len(ds))
    indices = rng.choice(len(ds), size=n, replace=False)
    out = []
    for i in indices:
        sample = ds[int(i)]
        pc = sample["point_cloud"]
        if isinstance(pc, torch.Tensor):
            pc = pc.detach().cpu().numpy()
        out.append(pc.astype(np.float32))
    return np.stack(out, axis=0)


# ── Plotting (3-panel: model hist | data hist | QQ vs N(0,1)) ────────────────

def _hist_panel(ax, values: np.ndarray, bins: int, title_prefix: str, color: str) -> tuple[float, float]:
    lo, hi = float(np.percentile(values, 0.05)), float(np.percentile(values, 99.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            hi = lo + 1.0
    ax.hist(
        values, bins=bins, range=(lo, hi), density=True,
        color=color, alpha=0.75, edgecolor="none",
    )
    xs = np.linspace(lo, hi, 512)
    ax.plot(xs, stats.norm.pdf(xs), color="#d9534f", lw=1.6, label="N(0, 1)")
    ax.axvline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.set_title(f"{title_prefix}  ({bins} bins, [{lo:.2f}, {hi:.2f}])\n{_summary_text(values)}",
                 fontsize=9, family="monospace")
    ax.legend(loc="upper right", fontsize=9)
    return lo, hi


def _plot_channel_with_data(
    *,
    model_values: np.ndarray,
    data_values: np.ndarray,
    ch_idx: int,
    label: str,
    out_path: Path,
    bins: int,
    qq_subsample: int,
    rng: np.random.Generator,
) -> None:
    """3-panel figure: [model hist | data hist | QQ of model vs N(0,1)]."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    fig.suptitle(
        f"channel {ch_idx} ({label})  N_model={model_values.size:,}  N_data={data_values.size:,}",
        fontsize=11, family="monospace",
    )

    # Model hist + data hist share x-range so they're directly comparable.
    lo_m, hi_m = float(np.percentile(model_values, 0.05)), float(np.percentile(model_values, 99.95))
    lo_d, hi_d = float(np.percentile(data_values, 0.05)),  float(np.percentile(data_values, 99.95))
    lo, hi = min(lo_m, lo_d), max(hi_m, hi_d)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -4.0, 4.0

    for ax, vals, title, color in (
        (axes[0], model_values, "MODEL hist", "#3a7bd5"),
        (axes[1], data_values,  "DATA hist",  "#2ecc71"),
    ):
        ax.hist(vals, bins=bins, range=(lo, hi), density=True,
                color=color, alpha=0.75, edgecolor="none")
        xs = np.linspace(lo, hi, 512)
        ax.plot(xs, stats.norm.pdf(xs), color="#d9534f", lw=1.6, label="N(0, 1)")
        ax.axvline(0, color="black", lw=0.5, alpha=0.5)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.set_title(f"{title}  ({bins} bins, [{lo:.2f}, {hi:.2f}])\n{_summary_text(vals)}",
                     fontsize=9, family="monospace")
        ax.legend(loc="upper right", fontsize=9)

    # QQ: overlay model + data theoretical-quantile vs sample-quantile clouds.
    ax_q = axes[2]
    for vals, color, name in (
        (model_values, "#3a7bd5", "model"),
        (data_values,  "#2ecc71", "data"),
    ):
        sample = rng.choice(vals, size=qq_subsample, replace=False) if vals.size > qq_subsample else vals
        theor, ordered = stats.probplot(sample, dist="norm", fit=False, plot=None)
        ax_q.scatter(theor, ordered, s=4, alpha=0.35, color=color, label=f"{name}  (n={sample.size:,})")
    bounds = np.array([
        ax_q.get_xlim()[0], ax_q.get_xlim()[1],
        ax_q.get_ylim()[0], ax_q.get_ylim()[1],
    ])
    qmin, qmax = float(bounds.min()), float(bounds.max())
    ax_q.plot([qmin, qmax], [qmin, qmax], color="#d9534f", lw=1.4, label="y = x")
    ax_q.set_xlabel("theoretical quantiles (N(0, 1))")
    ax_q.set_ylabel("sample quantiles")
    ax_q.set_title("QQ plot vs N(0, 1)")
    ax_q.legend(loc="upper left", fontsize=8)
    ax_q.grid(alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    ckpt = _load_checkpoint(args.checkpoint, device)

    model_name   = _resolve_arg(args.model,            ckpt, "model",            "JiT-B/8")
    predict_x0   = _resolve_arg(args.predict_xstart,   ckpt, "predict_xstart",   False)
    noise_sched  = _resolve_arg(args.noise_schedule,   ckpt, "noise_schedule",   "linear")
    sh_degree0   = _resolve_arg(args.sh_degree0_only,  ckpt, "sh_degree0_only",  False)
    cls_dropout  = _resolve_arg(args.class_dropout_prob, ckpt, "class_dropout_prob", 0.1)
    bottleneck   = _resolve_arg(args.bottleneck,       ckpt, "bottleneck",       True)

    logger.info(
        "Model: %s | predict_xstart=%s | noise_schedule=%s | sh_degree0_only=%s | bottleneck=%s",
        model_name, predict_x0, noise_sched, sh_degree0, bottleneck,
    )

    if sh_degree0:
        in_channels = len(DC_ONLY_FEATURE_INDICES)
        latent_to_full_ch = list(DC_ONLY_FEATURE_INDICES)  # 14 packed -> 59-grid positions
    else:
        in_channels = FULL_3DGS_FEATURE_DIM
        latent_to_full_ch = list(range(FULL_3DGS_FEATURE_DIM))

    with open(args.class_map, "r") as f:
        class_map = json.load(f)
    num_classes = max(v for v in class_map.values() if v >= 0) + 1
    logger.info("Number of classes: %d", num_classes)

    if args.class_label is not None and not (0 <= args.class_label < num_classes):
        raise ValueError(
            f"--class_label {args.class_label} out of range [0, {num_classes - 1}]"
        )

    logger.info("Building model: %s", model_name)
    model = JiT_3DGS_models[model_name](
        input_size=128,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=cls_dropout,
        learn_sigma=False,
        bottleneck=bottleneck,
    ).to(device)

    if args.use_ema and "ema" in ckpt:
        model.load_state_dict(_strip_compile_prefix(ckpt["ema"]))
        logger.warning("Loaded EMA weights (--use_ema). Note: EMA may be collapsed; base usually better.")
    else:
        model.load_state_dict(_strip_compile_prefix(ckpt["model"]))
        logger.info("Loaded base model weights")
    model.eval()

    if args.mode == "sampler" and args.sampler in {"ddim", "dpm", "ddpm"}:
        logger.warning(
            "Sampler %r assumes DDPM noising; on JiT flow-matching checkpoints "
            "samples will be degenerate. Prefer 'heun' or 'euler'.", args.sampler,
        )
    if args.mode == "sampler" and args.sampler in {"heun", "euler"} and not predict_x0:
        raise ValueError(
            "JiT heun/euler sampling requires predict_xstart=True; the loaded "
            "checkpoint does not predict x0. Pass --predict_xstart to override "
            "(if you really know what you're doing) or pick a different sampler."
        )

    step_tag = ckpt.get("step", "?")
    logger.info("Checkpoint step: %s", step_tag)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.bins < 100:
        print(f"[warn] --bins {args.bins} is below the requested 100+, continuing anyway.", file=sys.stderr)

    # ── Generation loop ──────────────────────────────────────────────────────
    sample_shape_one = resolve_sampling_shape(model=model, batch_size=1, in_channels=in_channels)
    _, _, H, W = sample_shape_one

    total = args.num_samples
    bs = max(1, min(args.batch_size, total))
    n_batches = math.ceil(total / bs)
    logger.info(
        "Generating %d samples in %d batches (bs=%d) | mode=%s%s",
        total, n_batches, bs, args.mode,
        f" | sampler={args.sampler}, num_steps={args.num_steps}" if args.mode == "sampler" else "",
    )

    chunks: list[torch.Tensor] = []
    drawn = 0
    for b in range(n_batches):
        cur_bs = min(bs, total - drawn)
        shape = (cur_bs, in_channels, H, W)
        class_labels = _draw_class_labels(
            num_classes=num_classes,
            batch_size=cur_bs,
            fixed_class=args.class_label,
            device=device,
        )
        generator = None
        if args.seed is not None:
            generator = torch.Generator(device=device).manual_seed(args.seed + b)

        if args.mode == "one_step":
            sample = _generate_one_step_batch(
                model=model,
                shape=shape,
                class_labels=class_labels,
                device=device,
                diffusion_steps=args.diffusion_steps,
                noise_scale=args.noise_scale,
                t_value=args.one_step_t,
                generator=generator,
            )
        else:
            sample = _generate_sampler_batch(
                model=model,
                shape=shape,
                class_labels=class_labels,
                device=device,
                args=args,
                predict_xstart=predict_x0,
                noise_schedule=noise_sched,
                generator=generator,
            )

        chunks.append(sample.detach().to(dtype=torch.float32, device="cpu"))
        drawn += cur_bs
        logger.info("  batch %d/%d done (%d / %d samples)", b + 1, n_batches, drawn, total)

    batch = torch.cat(chunks, dim=0).numpy()  # (N, C, H, W)
    if batch.ndim != 4:
        raise RuntimeError(f"Unexpected batch shape {batch.shape}; expected (N, C, H, W)")
    n, c, _, _ = batch.shape
    logger.info("Generated batch: shape=%s dtype=%s", batch.shape, batch.dtype)

    flat = batch.transpose(1, 0, 2, 3).reshape(c, -1)  # (C, N*H*W)
    rng = np.random.default_rng(args.seed if args.seed is not None else 0)

    # ── Optional data overlay ────────────────────────────────────────────────
    data_flat: Optional[np.ndarray] = None
    if not args.no_data_overlay:
        data_obj_list = _resolve_arg(args.obj_list,            ckpt, "obj_list",            None)
        data_gs_path  = _resolve_arg(args.gs_path,             ckpt, "gs_path",             None)
        data_mean     = _resolve_arg(args.mean_file,           ckpt, "mean_file",           None)
        data_std      = _resolve_arg(args.std_file,            ckpt, "std_file",            None)
        data_s2p      = _resolve_arg(args.sphere2plane_path,   ckpt, "sphere2plane_path",   None)
        data_rank     = _resolve_arg(args.rank_transform_file, ckpt, "rank_transform_file", None)
        if data_obj_list and data_gs_path and data_s2p:
            data_n = args.data_num_samples if args.data_num_samples is not None else args.num_samples
            obj_list_arg = data_obj_list if isinstance(data_obj_list, (list, tuple)) else [data_obj_list]
            logger.info(
                "Loading %d data samples for overlay | obj_list=%s | gs_path=%s | rank_transform=%s",
                data_n, obj_list_arg, data_gs_path, data_rank,
            )
            data_batch = _load_data_samples(
                obj_list=obj_list_arg,
                gs_path=data_gs_path,
                mean_file=data_mean,
                std_file=data_std,
                sphere2plane_path=data_s2p,
                rank_transform_file=data_rank,
                num_samples=data_n,
                seed=args.seed if args.seed is not None else 0,
            )
            if data_batch.ndim != 4:
                raise RuntimeError(f"Unexpected data batch shape {data_batch.shape}; expected (N, C, H, W)")
            d_n, d_c, _, _ = data_batch.shape
            logger.info("Loaded data batch: shape=%s dtype=%s", data_batch.shape, data_batch.dtype)
            data_flat = data_batch.transpose(1, 0, 2, 3).reshape(d_c, -1)  # (C_full, N*H*W)
        else:
            logger.warning(
                "Skipping data overlay: missing one of obj_list/gs_path/sphere2plane_path "
                "(neither CLI nor checkpoint args provided them). Pass --no_data_overlay "
                "to silence, or supply the paths explicitly."
            )

    logger.info("Writing per-channel plots to %s", out_dir.resolve())
    # Iterate over latent positions; map back to the 59-grid index for the label.
    plotted = 0
    for latent_pos, full_ch_idx in enumerate(latent_to_full_ch):
        if full_ch_idx not in CHANNEL_LABELS:
            # Not one of the 14 DC channels; skip higher-order SH coefficients.
            continue
        if latent_pos >= c:
            print(f"  [skip] latent_pos {latent_pos} not present (C={c})")
            continue
        label = CHANNEL_LABELS[full_ch_idx]
        model_values = flat[latent_pos].astype(np.float64, copy=False)
        out_path = out_dir / f"ch{full_ch_idx:02d}_{label}.png"

        if data_flat is not None and full_ch_idx < data_flat.shape[0]:
            # Data is loaded full-59-channels; index by the 59-grid position.
            data_values = data_flat[full_ch_idx].astype(np.float64, copy=False)
            _plot_channel_with_data(
                model_values=model_values,
                data_values=data_values,
                ch_idx=full_ch_idx,
                label=label,
                out_path=out_path,
                bins=args.bins,
                qq_subsample=args.qq_subsample,
                rng=rng,
            )
        else:
            _plot_channel(
                values=model_values,
                ch_idx=full_ch_idx,
                label=label,
                out_path=out_path,
                bins=args.bins,
                qq_subsample=args.qq_subsample,
                rng=rng,
            )
        plotted += 1
        print(
            f"  [{plotted:>2d}/{len(DC_INDICES)}] ch{full_ch_idx:02d} ({label:>9s}) -> {out_path.name}"
        )

    logger.info("Done. %d channels plotted in %s", plotted, out_dir.resolve())
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    p.add_argument("--class_map",  required=True, help="Path to object_to_class.json (sets num_classes)")

    # Mode
    p.add_argument("--mode", choices=["one_step", "sampler"], default="one_step",
                   help="one_step: pure noise -> single model x0 prediction. "
                        "sampler: full sampler run via sample_model().")
    p.add_argument("--one_step_t", type=float, default=0.0,
                   help="Continuous t in [0, 1] for one-step mode (FM convention: "
                        "t=0 is pure noise, t=1 is clean). Default 0 mirrors the "
                        "noise->x_0 boundary the trainer's logit-normal density "
                        "tails into.")

    # Generation
    p.add_argument("--num_samples", type=int, default=32,
                   help="Total number of latent samples to draw and pool together per channel.")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Per-forward-pass batch size; set lower if you OOM at large model variants.")
    p.add_argument("--class_label", type=int, default=None,
                   help="Fixed class index (omit for random per-sample). "
                        "Use the same one if you want to inspect class-conditional marginals.")

    # Model overrides (auto-read from checkpoint if omitted)
    g_model = p.add_argument_group("Model (auto-detected from checkpoint if omitted)")
    g_model.add_argument("--model", type=str, default=None, choices=list(JiT_3DGS_models.keys()))
    g_model.add_argument("--predict_xstart", action=argparse.BooleanOptionalAction, default=None)
    g_model.add_argument("--noise_schedule", type=str, default=None,
                         choices=["linear", "squaredcos_cap_v2"])
    g_model.add_argument("--sh_degree0_only", action=argparse.BooleanOptionalAction, default=None)
    g_model.add_argument("--class_dropout_prob", type=float, default=None)
    g_model.add_argument("--bottleneck", action=argparse.BooleanOptionalAction, default=None)
    g_model.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=False)

    # Sampler args (only used when --mode sampler)
    g_sample = p.add_argument_group("Sampler (only used when --mode sampler)")
    g_sample.add_argument("--sampler", type=str, default="heun", choices=SAMPLER_CHOICES)
    g_sample.add_argument("--num_steps", type=int, default=50)
    g_sample.add_argument("--diffusion_steps", type=int, default=1000)
    g_sample.add_argument("--cfg_scale", type=float, default=1.0)
    g_sample.add_argument("--cfg_interval", type=float, nargs=2, default=[0.0, 1.0],
                          metavar=("LOW", "HIGH"))
    g_sample.add_argument("--t_eps", type=float, default=5e-2)
    g_sample.add_argument("--noise_scale", type=float, default=1.0,
                          help="Scale on the initial noise (used by both modes).")
    g_sample.add_argument("--timestep_schedule", choices=TIMESTEP_SCHEDULE_CHOICES,
                          default="logit_normal")
    g_sample.add_argument("--P_mean", type=float, default=0.0)
    g_sample.add_argument("--P_std", type=float, default=1.0)
    g_sample.add_argument("--ddim_eta", type=float, default=0.0)

    # Data overlay (auto-resolved from checkpoint if omitted)
    g_data = p.add_argument_group("Data overlay (auto-resolved from checkpoint args if omitted)")
    g_data.add_argument("--obj_list",            nargs="+", default=None,
                        help="Object-list JSON file(s) for the dataset.")
    g_data.add_argument("--gs_path",             type=str,  default=None)
    g_data.add_argument("--mean_file",           type=str,  default=None)
    g_data.add_argument("--std_file",            type=str,  default=None)
    g_data.add_argument("--sphere2plane_path",   type=str,  default=None)
    g_data.add_argument("--rank_transform_file", type=str,  default=None,
                        help="Gaussian rank-transform tables; auto-resolved from checkpoint args.")
    g_data.add_argument("--data_num_samples",    type=int,  default=None,
                        help="Number of data samples to draw for the overlay panel (default: --num_samples).")
    g_data.add_argument("--no_data_overlay",     action="store_true",
                        help="Skip the data-distribution panel; produce 2-panel plots only.")

    g_dpm = p.add_argument_group("DPM-Solver (only used when --sampler dpm)")
    g_dpm.add_argument("--dpm_solver_order", type=int, default=2, choices=[1, 2, 3])
    g_dpm.add_argument("--dpm_algorithm_type", type=str, default="dpmsolver++",
                       choices=["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"])
    g_dpm.add_argument("--dpm_solver_type", type=str, default="midpoint",
                       choices=["midpoint", "heun"])
    g_dpm.add_argument("--dpm_timestep_spacing", type=str, default="trailing",
                       choices=["linspace", "leading", "trailing"])
    g_dpm.add_argument("--dpm_use_karras_sigmas", action=argparse.BooleanOptionalAction, default=False)

    # Plotting
    p.add_argument("--output_dir", default="output/model_distribution_plots",
                   help="Directory to write per-channel PNGs.")
    p.add_argument("--bins", type=int, default=128, help="Histogram bins (>=100 recommended).")
    p.add_argument("--qq_subsample", type=int, default=50000,
                   help="Subsample size for the QQ plot.")
    p.add_argument("--seed", type=int, default=0)

    return p


if __name__ == "__main__":
    parser = _build_parser()
    raise SystemExit(main(parser.parse_args()))
