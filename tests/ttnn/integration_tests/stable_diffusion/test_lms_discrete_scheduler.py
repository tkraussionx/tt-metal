# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from scipy import integrate

from diffusers import (
    UNet2DConditionModel,
    LMSDiscreteScheduler,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion.sd_helper_funcs import TtLMSDiscreteScheduler
from models.utility_functions import (
    skip_for_grayskull,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)


def get_lms_coefficient(order, t, current_order, sigmas):
    def lms_derivative(tau):
        prod = 1.0
        for k in range(order):
            if current_order == k:
                continue
            prod *= (tau - sigmas[t - k]) / (sigmas[t - current_order] - sigmas[t - k])
        return prod

    integrated_coeff = integrate.quad(lms_derivative, sigmas[t], sigmas[t + 1], epsrel=1e-4)[0]

    return integrated_coeff


def constant_prop_time_embeddings(timesteps, sample, time_proj):
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = time_proj(timesteps)
    return t_emb


def tt_latent_expansion(latents, scheduler, sigma, device):
    latent_model_input = ttnn.concat([latents, latents], dim=0)
    latent_model_input = scheduler.scale_model_input(latent_model_input, sigma, device)
    return latent_model_input


def tt_guide(noise_pred, guidance_scale):  # will return latents
    noise_pred_uncond, noise_pred_text = ttnn.split(noise_pred, noise_pred.shape[0] // 2, dim=0)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def guide(noise_pred, guidance_scale, t):  # will return latents
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


@skip_for_grayskull()
@pytest.mark.parametrize(
    "beta_start, beta_end, beta_schedule, num_train_timesteps",
    [
        (0.00085, 0.012, "scaled_linear", 1000),
    ],
)
def test_lms_discrete_scheduler(device, beta_start, beta_end, beta_schedule, num_train_timesteps):
    latents = torch.randn((1, 4, 64, 64))
    num_inference_steps = 2
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    ### Pytorch (diffuser) scheduler
    torch_scheduler = LMSDiscreteScheduler(
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        num_train_timesteps=num_train_timesteps,
    )
    torch_scheduler.set_timesteps(num_inference_steps)
    torch_latents = latents * torch_scheduler.init_noise_sigma
    torch_latents = torch.tensor(torch_latents)

    # for t in ttnn_scheduler.timesteps:
    #    torch_latents = ttnn_scheduler.step(noise_pred, t, torch_latents).prev_sample

    ### TTNN scheduler
    ttnn_scheduler = TtLMSDiscreteScheduler(
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        num_train_timesteps=num_train_timesteps,
    )
    ttnn_scheduler.set_timesteps(num_inference_steps)
    ttnn_latents = latents * ttnn_scheduler.init_noise_sigma
    ttnn_latents = torch.tensor(ttnn_latents)
    ttnn_latents = ttnn.from_torch(ttnn_latents, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    time_step_list = []
    ttnn_sigma = []
    ttnn_step_index = []
    timesteps_bkp = ttnn.to_torch(ttnn_scheduler.timesteps)
    sigma_tensor = ttnn.to_torch(ttnn_scheduler.sigmas)[0]
    step_index = (timesteps_bkp[0] == timesteps_bkp[0][0]).nonzero().item()
    ttnn_latent_model_input = tt_latent_expansion(ttnn_latents, ttnn_scheduler, float(sigma_tensor[step_index]), device)

    for t in timesteps_bkp[0]:
        _t = constant_prop_time_embeddings(t, ttnn_latent_model_input, unet.time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        time_step_list.append(_t)
        step_index = (timesteps_bkp[0] == t).nonzero().item()
        ttnn_step_index.append(step_index)
        ttnn_sigma.append(sigma_tensor[step_index])

    orders = 4
    order_list = []
    ttnn_lms_coeff = []
    lms_coeff = []
    for step_index in ttnn_step_index:
        order = min(step_index + 1, orders)
        order_list.append(order)
        lms_coeffs = [get_lms_coefficient(order, step_index, curr_order, sigma_tensor) for curr_order in range(order)]
        lms_coeff.append(lms_coeffs)

    for lms in lms_coeff:
        ttnn_lms_tensor = None
        for value in lms:
            lms_tensor = ttnn.full(
                (1, 4, 64, 64), fill_value=value, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )
            if ttnn_lms_tensor is not None:
                ttnn_lms_tensor = ttnn.concat([ttnn_lms_tensor, lms_tensor], dim=0)
            else:
                ttnn_lms_tensor = lms_tensor

        ttnn_lms_coeff.append(ttnn_lms_tensor)

    # run the TTNN/PyTorch scheduler module
    guidance_scale = 7.5  # Scale for classifier-free guidance
    for i, t in zip(range(len(time_step_list)), torch_scheduler.timesteps):
        # define "noise_pred" here
        noise_pred = torch.rand((2, 4, 64, 64))
        ttnn_noise_pred = ttnn.from_torch(noise_pred, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # PyTorch run
        noise_pred = guide(noise_pred, guidance_scale, t)
        torch_latents = torch_scheduler.step(noise_pred, t, torch_latents).prev_sample

        # TTNN run
        ttnn_noise_pred = tt_guide(ttnn_noise_pred, guidance_scale)
        ttnn_latents = ttnn_scheduler.step(
            model_output=ttnn_noise_pred,
            sample=ttnn_latents,
            sigma=float(ttnn_sigma[i]),
            lms_coeffs=ttnn_lms_coeff[i],
            device=device,
            order=order_list[i],
        ).prev_sample

        _ttnn_latents = ttnn.to_torch(ttnn_latents)
        assert_with_pcc(torch_latents, _ttnn_latents, 0.99)
        enable_persistent_kernel_cache()
