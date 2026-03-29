#!/usr/bin/env python3
"""
Scan the class-conditioned GaussianVerse training set for high diffusion MSE outliers.

The script rebuilds the dataset/model from a training checkpoint, runs the same
per-sample diffusion loss path used during training under `torch.no_grad()`, and
reports the worst objects. The full scan runs one deterministic training-like
trial per object. A second stage re-evaluates the worst candidates with several
additional trials to reduce RNG noise before ranking the final outliers.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GS_ROOT = os.path.join(REPO_ROOT, "submodules", "gaussian-splatting")
if GS_ROOT not in sys.path:
    sys.path.insert(0, GS_ROOT)

from dataloaders.class_3dgen_loader import (  # noqa: E402
    Class3DGenDataset,
    DC_ONLY_FEATURE_INDICES,
    FULL_3DGS_FEATURE_DIM,
)
from dataloaders.standard_3dgen_loader import Standard3DGenDataset  # noqa: E402
from dit.diffusion import create_diffusion  # noqa: E402
from unet.models import build_gaussianverse_unet  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("scan_gsplat_training_loss")


def _normalize_compile_key(key: str) -> str:
    return key.removeprefix("_orig_mod.").replace("._orig_mod.", ".")


def _normalize_compile_wrapped_keys(state: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        normalized_key = _normalize_compile_key(key)
        if normalized_key in normalized and normalized[normalized_key] is not value:
            raise KeyError(
                "Compile-key normalization produced duplicate parameter/state names: "
                f"{normalized_key!r} from {key!r}"
            )
        normalized[normalized_key] = value
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan GaussianVerse training objects for diffusion MSE outliers.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a UNet training checkpoint.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Output directory for partial CSVs and summaries. Defaults next to the checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Per-process batch size for the scan. Defaults to the checkpoint training batch size.",
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=None,
        help="Optional cap on the number of filtered dataset objects to scan. Useful for smoke tests.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Per-process DataLoader workers.")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor when workers > 0.")
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base RNG seed for deterministic stage-1 and stage-2 loss sampling.",
    )
    parser.add_argument(
        "--refine_top_k",
        type=int,
        default=256,
        help="Re-run the top-K stage-1 outliers with more trials.",
    )
    parser.add_argument(
        "--refine_trials",
        type=int,
        default=8,
        help="Extra deterministic training-like trials for stage-2 refinement.",
    )
    parser.add_argument(
        "--autocast",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "none"],
        help="CUDA autocast dtype. Defaults to bf16 for L4 throughput.",
    )
    parser.add_argument("--log_every", type=int, default=50, help="Log every N batches per process.")
    return parser.parse_args()


def _rank_prefix(rank: int, world_size: int) -> str:
    return f"[rank {rank}/{world_size}]"


def _init_distributed() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size, local_rank


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _load_checkpoint(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        raise TypeError(f"Checkpoint args must be a dict, got: {type(args)!r}")
    return checkpoint, args


def _checkpoint_arg(ckpt_args: dict[str, Any], name: str, default: Any) -> Any:
    return ckpt_args[name] if name in ckpt_args else default


def _resolve_results_dir(checkpoint_path: str, results_dir: Optional[str]) -> Path:
    if results_dir is not None:
        return Path(results_dir).resolve()
    ckpt = Path(checkpoint_path).resolve()
    return ckpt.parent / f"{ckpt.stem}_loss_scan"


def _resolve_autocast_dtype(spec: str) -> Optional[torch.dtype]:
    if spec == "bf16":
        return torch.bfloat16
    if spec == "fp16":
        return torch.float16
    return None


def _build_dataset(ckpt_args: dict[str, Any]) -> tuple[Class3DGenDataset, dict[str, int]]:
    with open(ckpt_args["class_map"], "r", encoding="utf-8") as handle:
        class_map = json.load(handle)

    base_dataset = Standard3DGenDataset(
        obj_list=[ckpt_args["obj_list"]],
        gs_path=ckpt_args["gs_path"],
        caption_path=None,
        mean_file=ckpt_args.get("mean_file"),
        std_file=ckpt_args.get("std_file"),
        sphere2plane_path=ckpt_args["sphere2plane_path"],
    )

    feature_indices = None
    if bool(_checkpoint_arg(ckpt_args, "sh_degree0_only", False)):
        feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)

    dataset = Class3DGenDataset(
        base_dataset,
        class_map,
        feature_indices=feature_indices,
        return_full_for_render=False,
        preload_to_cpu=False,
        lazy_cache_to_cpu=False,
    )
    return dataset, class_map


def _build_model(
    ckpt_args: dict[str, Any],
    class_map: dict[str, int],
    checkpoint: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    num_classes = max(value for value in class_map.values() if value >= 0) + 1
    in_channels = len(DC_ONLY_FEATURE_INDICES) if bool(_checkpoint_arg(ckpt_args, "sh_degree0_only", False)) else FULL_3DGS_FEATURE_DIM
    model = build_gaussianverse_unet(
        ckpt_args["model"],
        sample_size=128,
        in_channels=in_channels,
        out_channels=in_channels,
        num_classes=num_classes,
        class_embedding_dim=int(_checkpoint_arg(ckpt_args, "class_embed_dim", 768)),
        norm_num_groups=int(_checkpoint_arg(ckpt_args, "norm_num_groups", 32)),
        dropout=float(_checkpoint_arg(ckpt_args, "dropout", 0.0)),
        spatial_fold_factor=int(_checkpoint_arg(ckpt_args, "spatial_fold_factor", 1)),
        gradient_checkpointing=False,
    )
    state_dict = _normalize_compile_wrapped_keys(checkpoint["model"])
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected state-dict mismatch when loading checkpoint. "
            f"missing={incompatible.missing_keys[:8]} unexpected={incompatible.unexpected_keys[:8]}"
        )
    model.eval()
    model.to(device)
    return model


def _build_diffusion(ckpt_args: dict[str, Any]):
    return create_diffusion(
        timestep_respacing="",
        noise_schedule=str(_checkpoint_arg(ckpt_args, "noise_schedule", "squaredcos_cap_v2")),
        learn_sigma=False,
        predict_xstart=bool(_checkpoint_arg(ckpt_args, "predict_xstart", False)),
    )


class IndexedDataset(Dataset):
    def __init__(self, dataset: Class3DGenDataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        dataset_idx = self.indices[idx]
        sample = self.dataset[dataset_idx]
        if len(sample) == 4:
            x, y, _, hash_key = sample
        else:
            x, y, hash_key = sample
        return dataset_idx, x, int(y), str(hash_key)


def _build_loader(dataset: Dataset, batch_size: int, num_workers: int, prefetch_factor: int) -> DataLoader:
    kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        if prefetch_factor > 0:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def _use_min_snr_weighting(ckpt_args: dict[str, Any]) -> bool:
    snr_gamma = ckpt_args.get("snr_gamma")
    predict_xstart = bool(_checkpoint_arg(ckpt_args, "predict_xstart", False))
    allow_x0 = bool(_checkpoint_arg(ckpt_args, "allow_x0_min_snr_weighting", False))
    return snr_gamma is not None and (not predict_xstart or allow_x0)


def _compute_weighted_sample_losses(
    *,
    sample_losses: torch.Tensor,
    t: torch.Tensor,
    diffusion,
    ckpt_args: dict[str, Any],
    alphas_cumprod: Optional[torch.Tensor],
) -> torch.Tensor:
    if not _use_min_snr_weighting(ckpt_args):
        return sample_losses.float()

    assert alphas_cumprod is not None
    snr_gamma = float(ckpt_args["snr_gamma"])
    snr = alphas_cumprod[t] / (1.0 - alphas_cumprod[t])
    snr_weight = torch.minimum(snr, torch.full_like(snr, snr_gamma))
    if not bool(_checkpoint_arg(ckpt_args, "predict_xstart", False)):
        snr_weight = snr_weight / snr
    weighted = sample_losses.float() * snr_weight
    if not bool(_checkpoint_arg(ckpt_args, "predict_xstart", False)):
        weighted = weighted * 4.0
    return weighted


def _autocast_context(device: torch.device, autocast_dtype: Optional[torch.dtype]):
    if device.type != "cuda" or autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=autocast_dtype)


def _log_progress(
    *,
    prefix: str,
    batch_idx: int,
    total_batches: int,
    batch_size: int,
    processed: int,
    total: int,
    start_time: float,
) -> None:
    elapsed = max(1e-6, time.time() - start_time)
    rate = processed / elapsed
    remaining = max(0, total - processed)
    eta_seconds = remaining / max(rate, 1e-6)
    eta_minutes = eta_seconds / 60.0
    logger.info(
        "%s batch %d/%d processed=%d/%d rate=%.2f samples/s eta=%.1f min batch_size=%d",
        prefix,
        batch_idx,
        total_batches,
        processed,
        total,
        rate,
        eta_minutes,
        batch_size,
    )


def _write_stage1_partial(
    *,
    dataset: Class3DGenDataset,
    indices: list[int],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    model: torch.nn.Module,
    diffusion,
    ckpt_args: dict[str, Any],
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    base_seed: int,
    output_path: Path,
    rank: int,
    world_size: int,
    log_every: int,
) -> None:
    scan_dataset = IndexedDataset(dataset, indices)
    loader = _build_loader(scan_dataset, batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor)
    alphas_cumprod = None
    if _use_min_snr_weighting(ckpt_args):
        alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=device, dtype=torch.float32)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(base_seed + rank * 1_000_003))
    total = len(scan_dataset)
    total_batches = math.ceil(total / batch_size)
    processed = 0
    prefix = _rank_prefix(rank, world_size)
    start_time = time.time()

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_idx",
                "hash_key",
                "label",
                "trial_seed",
                "timestep",
                "loss",
                "input_nonfinite",
                "input_max_abs",
                "input_mean",
                "input_std",
            ]
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                dataset_idx, x, y, hash_keys = batch
                x = x.to(device=device, non_blocking=True)
                y = y.to(device=device, dtype=torch.long, non_blocking=True)

                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device, generator=generator)
                noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=generator)

                with _autocast_context(device, autocast_dtype):
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs=dict(y=y), noise=noise)
                sample_losses = _compute_weighted_sample_losses(
                    sample_losses=loss_dict["loss"],
                    t=t,
                    diffusion=diffusion,
                    ckpt_args=ckpt_args,
                    alphas_cumprod=alphas_cumprod,
                )

                flat_x = x.float().flatten(1)
                nonfinite = (~torch.isfinite(x)).flatten(1).sum(dim=1)
                max_abs = flat_x.abs().max(dim=1).values
                mean = flat_x.mean(dim=1)
                std = flat_x.std(dim=1, unbiased=False)

                loss_cpu = sample_losses.detach().cpu().tolist()
                t_cpu = t.detach().cpu().tolist()
                nonfinite_cpu = nonfinite.detach().cpu().tolist()
                max_abs_cpu = max_abs.detach().cpu().tolist()
                mean_cpu = mean.detach().cpu().tolist()
                std_cpu = std.detach().cpu().tolist()
                dataset_idx_cpu = dataset_idx.tolist()
                labels_cpu = y.detach().cpu().tolist()

                for row_idx, hash_key in enumerate(hash_keys):
                    writer.writerow(
                        [
                            int(dataset_idx_cpu[row_idx]),
                            str(hash_key),
                            int(labels_cpu[row_idx]),
                            int(base_seed + rank * 1_000_003),
                            int(t_cpu[row_idx]),
                            float(loss_cpu[row_idx]),
                            int(nonfinite_cpu[row_idx]),
                            float(max_abs_cpu[row_idx]),
                            float(mean_cpu[row_idx]),
                            float(std_cpu[row_idx]),
                        ]
                    )

                processed += x.shape[0]
                if log_every > 0 and (batch_idx % log_every == 0 or batch_idx == total_batches):
                    _log_progress(
                        prefix=prefix,
                        batch_idx=batch_idx,
                        total_batches=total_batches,
                        batch_size=x.shape[0],
                        processed=processed,
                        total=total,
                        start_time=start_time,
                    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _stage1_merge_and_select(
    *,
    dataset: Class3DGenDataset,
    results_dir: Path,
    refine_top_k: int,
    world_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged_path = results_dir / "stage1_all.csv"
    rows: list[dict[str, Any]] = []
    losses: list[float] = []

    with merged_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_idx",
                "hash_key",
                "tar_gz_path",
                "label",
                "trial_seed",
                "timestep",
                "loss",
                "input_nonfinite",
                "input_max_abs",
                "input_mean",
                "input_std",
            ]
        )
        for rank in range(world_size):
            partial_path = results_dir / f"stage1_rank{rank:02d}.csv"
            for row in _read_csv_rows(partial_path):
                typed = {
                    "dataset_idx": int(row["dataset_idx"]),
                    "hash_key": row["hash_key"],
                    "tar_gz_path": dataset.base_dataset.obj_data[row["hash_key"]],
                    "label": int(row["label"]),
                    "trial_seed": int(row["trial_seed"]),
                    "timestep": int(row["timestep"]),
                    "loss": float(row["loss"]),
                    "input_nonfinite": int(row["input_nonfinite"]),
                    "input_max_abs": float(row["input_max_abs"]),
                    "input_mean": float(row["input_mean"]),
                    "input_std": float(row["input_std"]),
                }
                writer.writerow(
                    [
                        typed["dataset_idx"],
                        typed["hash_key"],
                        typed["tar_gz_path"],
                        typed["label"],
                        typed["trial_seed"],
                        typed["timestep"],
                        typed["loss"],
                        typed["input_nonfinite"],
                        typed["input_max_abs"],
                        typed["input_mean"],
                        typed["input_std"],
                    ]
                )
                rows.append(typed)
                losses.append(typed["loss"])

    if not rows:
        raise RuntimeError("Stage-1 merge found no rows.")

    rows.sort(key=lambda item: item["loss"], reverse=True)
    losses_np = np.asarray(losses, dtype=np.float64)
    median = float(np.median(losses_np))
    mad = float(np.median(np.abs(losses_np - median)))
    stage1_summary = {
        "num_objects": int(len(rows)),
        "loss_mean": float(losses_np.mean()),
        "loss_std": float(losses_np.std()),
        "loss_median": median,
        "loss_mad": mad,
        "loss_p95": float(np.percentile(losses_np, 95)),
        "loss_p99": float(np.percentile(losses_np, 99)),
        "loss_p999": float(np.percentile(losses_np, 99.9)),
        "top_20": rows[:20],
    }
    selected = rows[: max(1, min(refine_top_k, len(rows)))]
    with (results_dir / "stage1_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(stage1_summary, handle, indent=2)
    with (results_dir / "stage1_topk.json").open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)
    return selected, stage1_summary


def _write_stage2_partial(
    *,
    dataset: Class3DGenDataset,
    selected_rows: list[dict[str, Any]],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    refine_trials: int,
    model: torch.nn.Module,
    diffusion,
    ckpt_args: dict[str, Any],
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    base_seed: int,
    output_path: Path,
    rank: int,
    world_size: int,
    log_every: int,
) -> None:
    selected_indices = [int(row["dataset_idx"]) for row in selected_rows]
    local_indices = selected_indices[rank::world_size]
    scan_dataset = IndexedDataset(dataset, local_indices)
    loader = _build_loader(scan_dataset, batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor)
    alphas_cumprod = None
    if _use_min_snr_weighting(ckpt_args):
        alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=device, dtype=torch.float32)

    stats = {
        int(row["dataset_idx"]): {
            "dataset_idx": int(row["dataset_idx"]),
            "hash_key": row["hash_key"],
            "label": int(row["label"]),
            "tar_gz_path": row["tar_gz_path"],
            "stage1_loss": float(row["loss"]),
            "input_nonfinite": int(row["input_nonfinite"]),
            "input_max_abs": float(row["input_max_abs"]),
            "input_mean": float(row["input_mean"]),
            "input_std": float(row["input_std"]),
            "loss_sum": 0.0,
            "loss_sumsq": 0.0,
            "loss_max": float("-inf"),
            "num_trials": 0,
        }
        for row in selected_rows[rank::world_size]
    }

    prefix = _rank_prefix(rank, world_size)
    total_batches = max(1, math.ceil(len(scan_dataset) / batch_size))
    start_time = time.time()

    with torch.no_grad():
        for trial in range(refine_trials):
            generator = torch.Generator(device=device)
            trial_seed = int(base_seed + 10_000_000 + trial * 1_009 + rank * 1_000_003)
            generator.manual_seed(trial_seed)
            processed = 0
            for batch_idx, batch in enumerate(loader, start=1):
                dataset_idx, x, y, _ = batch
                x = x.to(device=device, non_blocking=True)
                y = y.to(device=device, dtype=torch.long, non_blocking=True)

                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device, generator=generator)
                noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=generator)

                with _autocast_context(device, autocast_dtype):
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs=dict(y=y), noise=noise)
                sample_losses = _compute_weighted_sample_losses(
                    sample_losses=loss_dict["loss"],
                    t=t,
                    diffusion=diffusion,
                    ckpt_args=ckpt_args,
                    alphas_cumprod=alphas_cumprod,
                )

                for idx_value, loss_value in zip(dataset_idx.tolist(), sample_losses.detach().cpu().tolist()):
                    record = stats[int(idx_value)]
                    loss_float = float(loss_value)
                    record["loss_sum"] += loss_float
                    record["loss_sumsq"] += loss_float * loss_float
                    record["loss_max"] = max(record["loss_max"], loss_float)
                    record["num_trials"] += 1

                processed += x.shape[0]
                if log_every > 0 and (batch_idx % log_every == 0 or batch_idx == total_batches):
                    _log_progress(
                        prefix=f"{prefix} trial {trial + 1}/{refine_trials}",
                        batch_idx=batch_idx,
                        total_batches=total_batches,
                        batch_size=x.shape[0],
                        processed=processed,
                        total=len(scan_dataset),
                        start_time=start_time,
                    )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_idx",
                "hash_key",
                "tar_gz_path",
                "label",
                "stage1_loss",
                "refine_mean_loss",
                "refine_std_loss",
                "refine_max_loss",
                "num_trials",
                "input_nonfinite",
                "input_max_abs",
                "input_mean",
                "input_std",
            ]
        )
        for dataset_idx in sorted(stats):
            record = stats[dataset_idx]
            num_trials = max(1, int(record["num_trials"]))
            mean_loss = record["loss_sum"] / num_trials
            variance = max(0.0, record["loss_sumsq"] / num_trials - mean_loss * mean_loss)
            std_loss = math.sqrt(variance)
            writer.writerow(
                [
                    record["dataset_idx"],
                    record["hash_key"],
                    record["tar_gz_path"],
                    record["label"],
                    record["stage1_loss"],
                    mean_loss,
                    std_loss,
                    record["loss_max"],
                    num_trials,
                    record["input_nonfinite"],
                    record["input_max_abs"],
                    record["input_mean"],
                    record["input_std"],
                ]
            )


def _merge_stage2(results_dir: Path, world_size: int) -> list[dict[str, Any]]:
    merged_rows: list[dict[str, Any]] = []
    merged_path = results_dir / "stage2_refined.csv"
    with merged_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_idx",
                "hash_key",
                "tar_gz_path",
                "label",
                "stage1_loss",
                "refine_mean_loss",
                "refine_std_loss",
                "refine_max_loss",
                "num_trials",
                "input_nonfinite",
                "input_max_abs",
                "input_mean",
                "input_std",
            ]
        )
        for rank in range(world_size):
            partial_path = results_dir / f"stage2_rank{rank:02d}.csv"
            for row in _read_csv_rows(partial_path):
                typed = {
                    "dataset_idx": int(row["dataset_idx"]),
                    "hash_key": row["hash_key"],
                    "tar_gz_path": row["tar_gz_path"],
                    "label": int(row["label"]),
                    "stage1_loss": float(row["stage1_loss"]),
                    "refine_mean_loss": float(row["refine_mean_loss"]),
                    "refine_std_loss": float(row["refine_std_loss"]),
                    "refine_max_loss": float(row["refine_max_loss"]),
                    "num_trials": int(row["num_trials"]),
                    "input_nonfinite": int(row["input_nonfinite"]),
                    "input_max_abs": float(row["input_max_abs"]),
                    "input_mean": float(row["input_mean"]),
                    "input_std": float(row["input_std"]),
                }
                writer.writerow(
                    [
                        typed["dataset_idx"],
                        typed["hash_key"],
                        typed["tar_gz_path"],
                        typed["label"],
                        typed["stage1_loss"],
                        typed["refine_mean_loss"],
                        typed["refine_std_loss"],
                        typed["refine_max_loss"],
                        typed["num_trials"],
                        typed["input_nonfinite"],
                        typed["input_max_abs"],
                        typed["input_mean"],
                        typed["input_std"],
                    ]
                )
                merged_rows.append(typed)

    merged_rows.sort(key=lambda item: item["refine_mean_loss"], reverse=True)
    with (results_dir / "stage2_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "top_20": merged_rows[:20],
                "num_refined": len(merged_rows),
            },
            handle,
            indent=2,
        )
    return merged_rows


def main() -> None:
    args = _parse_args()
    rank, world_size, local_rank = _init_distributed()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this scan.")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoint, ckpt_args = _load_checkpoint(args.checkpoint)
    results_dir = _resolve_results_dir(args.checkpoint, args.results_dir)
    if rank == 0:
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Writing results to %s", results_dir)
    _barrier()

    batch_size = int(args.batch_size or _checkpoint_arg(ckpt_args, "batch_size", 64))
    dataset, class_map = _build_dataset(ckpt_args)
    model = _build_model(ckpt_args, class_map, checkpoint, device)
    diffusion = _build_diffusion(ckpt_args)
    autocast_dtype = _resolve_autocast_dtype(args.autocast)

    total_dataset = len(dataset)
    if args.max_objects is not None:
        if args.max_objects <= 0:
            raise ValueError("--max_objects must be positive when provided.")
        total_dataset = min(total_dataset, int(args.max_objects))
    rank_indices = list(range(rank, total_dataset, world_size))
    logger.info(
        "%s dataset_size=%d local_size=%d batch_size=%d device=%s model=%s predict_xstart=%s",
        _rank_prefix(rank, world_size),
        total_dataset,
        len(rank_indices),
        batch_size,
        device,
        ckpt_args["model"],
        bool(_checkpoint_arg(ckpt_args, "predict_xstart", False)),
    )

    stage1_partial = results_dir / f"stage1_rank{rank:02d}.csv"
    _write_stage1_partial(
        dataset=dataset,
        indices=rank_indices,
        batch_size=batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        model=model,
        diffusion=diffusion,
        ckpt_args=ckpt_args,
        device=device,
        autocast_dtype=autocast_dtype,
        base_seed=args.seed,
        output_path=stage1_partial,
        rank=rank,
        world_size=world_size,
        log_every=args.log_every,
    )
    _barrier()

    selected_rows: list[dict[str, Any]] = []
    stage1_summary: dict[str, Any] = {}
    if rank == 0:
        selected_rows, stage1_summary = _stage1_merge_and_select(
            dataset=dataset,
            results_dir=results_dir,
            refine_top_k=args.refine_top_k,
            world_size=world_size,
        )
        logger.info(
            "Stage 1 complete: median=%.6f p99=%.6f top_loss=%.6f top_hash=%s",
            stage1_summary["loss_median"],
            stage1_summary["loss_p99"],
            selected_rows[0]["loss"],
            selected_rows[0]["hash_key"],
        )
    _barrier()
    if rank != 0:
        with (results_dir / "stage1_topk.json").open("r", encoding="utf-8") as handle:
            selected_rows = json.load(handle)

    stage2_partial = results_dir / f"stage2_rank{rank:02d}.csv"
    _write_stage2_partial(
        dataset=dataset,
        selected_rows=selected_rows,
        batch_size=batch_size,
        num_workers=min(args.num_workers, 4),
        prefetch_factor=args.prefetch_factor,
        refine_trials=args.refine_trials,
        model=model,
        diffusion=diffusion,
        ckpt_args=ckpt_args,
        device=device,
        autocast_dtype=autocast_dtype,
        base_seed=args.seed,
        output_path=stage2_partial,
        rank=rank,
        world_size=world_size,
        log_every=args.log_every,
    )
    _barrier()

    if rank == 0:
        refined_rows = _merge_stage2(results_dir, world_size)
        if not refined_rows:
            raise RuntimeError("Stage-2 refinement produced no rows.")
        top = refined_rows[0]
        next_best = refined_rows[1] if len(refined_rows) > 1 else None
        final_summary = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "results_dir": str(results_dir.resolve()),
            "dataset_size": len(dataset),
            "scanned_dataset_size": total_dataset,
            "world_size": world_size,
            "stage1_summary": stage1_summary,
            "top_object": top,
            "runner_up": next_best,
            "top_20_refined": refined_rows[:20],
        }
        with (results_dir / "final_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(final_summary, handle, indent=2)
        logger.info(
            "Top refined outlier: dataset_idx=%d hash=%s tar=%s refine_mean=%.6f refine_std=%.6f stage1=%.6f",
            top["dataset_idx"],
            top["hash_key"],
            top["tar_gz_path"],
            top["refine_mean_loss"],
            top["refine_std_loss"],
            top["stage1_loss"],
        )
        if next_best is not None:
            logger.info(
                "Runner-up: dataset_idx=%d hash=%s tar=%s refine_mean=%.6f",
                next_best["dataset_idx"],
                next_best["hash_key"],
                next_best["tar_gz_path"],
                next_best["refine_mean_loss"],
            )

    _cleanup_distributed()


if __name__ == "__main__":
    main()
