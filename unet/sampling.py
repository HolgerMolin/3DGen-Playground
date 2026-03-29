from __future__ import annotations

from typing import Optional

import torch
from diffusers import DPMSolverMultistepScheduler

from dit.diffusion.gaussian_diffusion import get_named_beta_schedule


def resolve_sampling_shape(
    *,
    model: torch.nn.Module,
    batch_size: int,
    in_channels: int,
) -> tuple[int, int, int, int]:
    sample_size = int(getattr(model, "sample_size", 128))
    model_in_channels = int(getattr(model, "in_channels", in_channels))
    spatial_fold_factor = int(getattr(model, "spatial_fold_factor", 1))
    folded_sample_size = int(getattr(model, "folded_sample_size", sample_size))
    folded_in_channels = int(getattr(model, "folded_in_channels", model_in_channels))

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if model_in_channels != in_channels:
        raise ValueError(f"Model in_channels={model_in_channels} does not match expected input channels={in_channels}")
    if spatial_fold_factor < 1:
        raise ValueError(f"spatial_fold_factor must be >= 1, got {spatial_fold_factor}")
    if sample_size % spatial_fold_factor != 0:
        raise ValueError(
            f"Model sample_size={sample_size} is not divisible by spatial_fold_factor={spatial_fold_factor}"
        )

    expected_folded_sample_size = sample_size // spatial_fold_factor
    expected_folded_in_channels = in_channels * (spatial_fold_factor**2)
    if folded_sample_size != expected_folded_sample_size:
        raise ValueError(
            f"Model folded_sample_size={folded_sample_size} does not match expected value={expected_folded_sample_size}"
        )
    if folded_in_channels != expected_folded_in_channels:
        raise ValueError(
            f"Model folded_in_channels={folded_in_channels} does not match expected value={expected_folded_in_channels}"
        )

    return (batch_size, in_channels, sample_size, sample_size)


def _validate_sampling_shape(model: torch.nn.Module, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(shape) != 4:
        raise ValueError(f"Expected sampling shape (N, C, H, W), got {shape}")

    batch_size, in_channels, height, width = (int(dim) for dim in shape)
    if height != width:
        raise ValueError(f"Sampling shape must be square, got H={height}, W={width}")

    expected_shape = resolve_sampling_shape(
        model=model,
        batch_size=batch_size,
        in_channels=in_channels,
    )
    if expected_shape != (batch_size, in_channels, height, width):
        raise ValueError(
            f"Sampling shape {shape} does not match model expectation {expected_shape}. "
            "The unfolded shape must match the model sample size and input channel count."
        )
    return expected_shape


def build_dpm_scheduler(
    *,
    predict_xstart: bool,
    noise_schedule: str = "linear",
    diffusion_steps: int = 1000,
    solver_order: int = 2,
    algorithm_type: str = "dpmsolver++",
    solver_type: str = "midpoint",
    timestep_spacing: str = "trailing",
    use_karras_sigmas: bool = False,
) -> DPMSolverMultistepScheduler:
    prediction_type = "sample" if predict_xstart else "epsilon"
    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    return DPMSolverMultistepScheduler(
        num_train_timesteps=diffusion_steps,
        trained_betas=betas,
        solver_order=solver_order,
        prediction_type=prediction_type,
        algorithm_type=algorithm_type,
        solver_type=solver_type,
        lower_order_final=True,
        use_karras_sigmas=use_karras_sigmas,
        timestep_spacing=timestep_spacing,
    )


@torch.no_grad()
def sample_with_dpm(
    *,
    model: torch.nn.Module,
    shape: tuple[int, ...],
    class_labels: torch.Tensor,
    num_inference_steps: int,
    device: torch.device,
    predict_xstart: bool,
    noise_schedule: str = "linear",
    diffusion_steps: int = 1000,
    solver_order: int = 2,
    algorithm_type: str = "dpmsolver++",
    solver_type: str = "midpoint",
    timestep_spacing: str = "trailing",
    use_karras_sigmas: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    shape = _validate_sampling_shape(model, shape)
    scheduler = build_dpm_scheduler(
        predict_xstart=predict_xstart,
        noise_schedule=noise_schedule,
        diffusion_steps=diffusion_steps,
        solver_order=solver_order,
        algorithm_type=algorithm_type,
        solver_type=solver_type,
        timestep_spacing=timestep_spacing,
        use_karras_sigmas=use_karras_sigmas,
    )
    scheduler.set_timesteps(num_inference_steps, device=device)

    sample_dtype = next(model.parameters()).dtype
    sample = torch.randn(shape, device=device, dtype=sample_dtype, generator=generator)
    sample = sample * scheduler.init_noise_sigma


    was_training = model.training
    model.eval()
    for timestep in scheduler.timesteps:
        timestep_batch = torch.full(
            (shape[0],),
            int(timestep.item()),
            device=device,
            dtype=torch.long,
        )
        model_input = scheduler.scale_model_input(sample, timestep)
        model_output = model(model_input, timestep_batch, class_labels)
        sample = scheduler.step(
            model_output,
            timestep,
            sample,
            generator=generator,
            return_dict=False,
        )[0]
    if was_training:
        model.train()

    return sample
