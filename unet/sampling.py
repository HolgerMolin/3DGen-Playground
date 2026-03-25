from __future__ import annotations

from typing import Optional

import torch
from diffusers import DPMSolverMultistepScheduler

from dit.diffusion.gaussian_diffusion import get_named_beta_schedule


def build_dpm_scheduler(
    *,
    predict_xstart: bool,
    diffusion_steps: int = 1000,
    solver_order: int = 2,
    algorithm_type: str = "dpmsolver++",
    solver_type: str = "midpoint",
    timestep_spacing: str = "trailing",
    use_karras_sigmas: bool = False,
) -> DPMSolverMultistepScheduler:
    prediction_type = "sample" if predict_xstart else "epsilon"
    betas = get_named_beta_schedule("linear", diffusion_steps)
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
    solver_order: int = 2,
    algorithm_type: str = "dpmsolver++",
    solver_type: str = "midpoint",
    timestep_spacing: str = "trailing",
    use_karras_sigmas: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    scheduler = build_dpm_scheduler(
        predict_xstart=predict_xstart,
        diffusion_steps=1000,
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
        model_output = model(sample, timestep_batch, class_labels)
        sample = scheduler.step(model_output, timestep, sample, return_dict=False)[0]
    if was_training:
        model.train()

    return sample
