"""Shared path helpers for the diagnostic pipeline (probe / weight inspection / diversity).

Centralises the convention so all three CLI tools and the orchestrator agree on
where to write per-checkpoint outputs and where to share expensive caches.

Layout produced:
    output/diagnostics/
        _cache/
            dinov2_features/
                dinov2_n<N>_seed<S>.npz        # shared across checkpoints
                dinov2_n<N>_seed<S>_keys.json
        <run_id>/                              # derived from checkpoint path
            probe/
                cond_probe_t030_per_class.json
                cond_probe_t050_per_class.json
                ...
            weights/                           # outputs of inspect_weights.py
                ...
            diversity/                         # outputs of diagnose_class_diversity.py
                ...
            SUMMARY.md                         # written by run_diagnostics.sh

`run_id` is `<parent_dir_name>_<ckpt_stem>` — e.g. `0260000.pt` inside
`output/jit_JiT-B_8_full_improved_renderprobe/` becomes
`jit_JiT-B_8_full_improved_renderprobe_0260000`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

DIAGNOSTICS_ROOT = Path("output/diagnostics")
SHARED_CACHE = DIAGNOSTICS_ROOT / "_cache"
DINOV2_FEATURES_CACHE = SHARED_CACHE / "dinov2_features"


def derive_run_id(ckpt_path: Union[str, Path]) -> str:
    """`<parent_dir_name>_<ckpt_stem>` — stable across reruns of the same checkpoint."""
    p = Path(ckpt_path).resolve()
    return f"{p.parent.name}_{p.stem}"


def run_dir(ckpt_path: Union[str, Path]) -> Path:
    return DIAGNOSTICS_ROOT / derive_run_id(ckpt_path)


def probe_dir(ckpt_path: Union[str, Path]) -> Path:
    return run_dir(ckpt_path) / "probe"


def weights_dir(ckpt_path: Union[str, Path]) -> Path:
    return run_dir(ckpt_path) / "weights"


def diversity_dir(ckpt_path: Union[str, Path]) -> Path:
    return run_dir(ckpt_path) / "diversity"


def per_class_dump_path(ckpt_path: Union[str, Path], t_value: float) -> Path:
    """Per-class cfg_signal JSON for a specific t value."""
    tag = f"t{int(round(float(t_value) * 100)):03d}"
    return probe_dir(ckpt_path) / f"cond_probe_{tag}_per_class.json"


def dinov2_cache_paths(n_per_class: int, seed: int) -> tuple[Path, Path]:
    """Stable feature/key cache paths shared across checkpoint reruns."""
    DINOV2_FEATURES_CACHE.mkdir(parents=True, exist_ok=True)
    feats = DINOV2_FEATURES_CACHE / f"dinov2_n{n_per_class}_seed{seed}.npz"
    keys = DINOV2_FEATURES_CACHE / f"dinov2_n{n_per_class}_seed{seed}_keys.json"
    return feats, keys


__all__ = [
    "DIAGNOSTICS_ROOT",
    "SHARED_CACHE",
    "DINOV2_FEATURES_CACHE",
    "derive_run_id",
    "run_dir",
    "probe_dir",
    "weights_dir",
    "diversity_dir",
    "per_class_dump_path",
    "dinov2_cache_paths",
]
