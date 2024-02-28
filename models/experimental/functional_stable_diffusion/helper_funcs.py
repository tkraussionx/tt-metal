# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import numpy as np
from scipy import integrate


def ttnn_latent_expansion(latents, scheduler, t, device):
    latents = ttnn.from_torch(latents, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    latent_model_input = ttnn.concat([latents] * 2, dim=0)

    beta_start = 0.00085
    beta_end = 0.012
    timesteps = t
    num_train_timesteps = 1000
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
    sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
    sigmas = torch.from_numpy(sigmas)

    timestep = torch.from_numpy(np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy())
    step_index = (timesteps == timestep).nonzero().item()
    sigma = sigmas[step_index]

    latent_model_input = ttnn.to_torch(latent_model_input)
    latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
    return latent_model_input


def get_lms_coefficient(order, t, current_order):
    """
    Compute a linear multistep coefficient.

    Args:
        order (TODO):
        t (TODO):
        current_order (TODO):
    """

    beta_start = 0.00085
    beta_end = 0.012
    num_train_timesteps = 1000
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
    sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
    sigmas = torch.from_numpy(sigmas)

    def lms_derivative(tau):
        prod = 1.0
        for k in range(order):
            if current_order == k:
                continue
            prod *= (tau - sigmas[t - k]) / (sigmas[t - current_order] - sigmas[t - k])
        return prod

    integrated_coeff = integrate.quad(lms_derivative, sigmas[t], sigmas[t + 1], epsrel=1e-4)[0]

    return integrated_coeff


def ttnn_step(model_output, time_steps, sample, config, order=4, return_dict=True):
    beta_start = 0.00085
    beta_end = 0.012
    timesteps = time_steps
    num_train_timesteps = 1000
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
    sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
    sigmas = torch.from_numpy(sigmas)

    timestep = torch.from_numpy(np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy())
    step_index = (timesteps == timestep).nonzero().item()
    sigma = sigmas[step_index]

    # if config.prediction_type == "epsilon":
    pred_original_sample = sample - sigma * model_output
    # elif config.prediction_type == "v_prediction":
    #     # * c_out + input * c_skip
    #     pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))

    derivatives = []
    derivative = (sample - pred_original_sample) / sigma
    derivatives.append(derivative)
    if len(derivatives) > order:
        derivatives.pop(0)

    order = min(step_index + 1, order)
    lms_coeffs = [get_lms_coefficient(order, step_index, curr_order) for curr_order in range(order)]

    # 4. Compute previous sample based on the derivatives path
    prev_sample = sample + sum(coeff * derivative for coeff, derivative in zip(lms_coeffs, reversed(derivatives)))

    if not return_dict:
        return (prev_sample,)

    return prev_sample


def constant_prop_time_embeddings(timesteps, sample, time_proj):
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = time_proj(timesteps)
    return t_emb


def guide(noise_pred, guidance_scale, t):  # will return latents
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def latent_expansion(latents, scheduler, t):
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
    return latent_model_input
