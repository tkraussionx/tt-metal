# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import json
import torch
import pytest
import numpy as np
from PIL import Image
from loguru import logger
from tqdm.auto import tqdm
from datasets import load_dataset

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    LMSDiscreteScheduler,
)
from models.experimental.functional_stable_diffusion.sd_helper_funcs import TtLMSDiscreteScheduler
from models.utility_functions import (
    comp_allclose_and_pcc,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_stable_diffusion.custom_preprocessing import custom_preprocessor
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_unet_2d_condition_model import (
    UNet2DConditionModel as ttnn_unet_condition,
)

from torchvision.transforms import ToTensor
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance
from scipy import integrate


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


def load_inputs(input_path):
    with open(input_path) as f:
        input_data = json.load(f)
        assert input_data, "Input data is empty."
        prompt = [item["prompt"] for item in input_data]
        return prompt


def constant_prop_time_embeddings(timesteps, sample, time_proj):
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = time_proj(timesteps)
    return t_emb


def save_image_and_latents(latents, iter, vae, pre_fix="", pre_fix2=""):
    pre_fix = "" if pre_fix == "" else f"{pre_fix}_"
    pre_fix2 = "" if pre_fix2 == "" else f"{pre_fix2}_"
    _latents = 1 / 0.18215 * latents

    with torch.no_grad():
        image = vae.decode(_latents).sample
    # Image post-processing
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images][0]
    pil_images.save(f"{pre_fix}{pre_fix2}image_iter_{iter}.png")


def guide(noise_pred, guidance_scale, t):  # will return latents
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def tt_guide(noise_pred, guidance_scale):  # will return latents
    noise_pred_uncond, noise_pred_text = ttnn.split(noise_pred, noise_pred.shape[0] // 2, dim=0)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def latent_expansion(latents, scheduler, t):
    latent_model_input = torch.cat([latents] * 2, dim=0)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
    return latent_model_input


def tt_latent_expansion(latents, scheduler, sigma, device):
    latent_model_input = ttnn.concat([latents, latents], dim=0)
    latent_model_input = scheduler.scale_model_input(latent_model_input, sigma, device)
    return latent_model_input


def calculate_fid_score(imgs_path1, imgs_path2):
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(imgs_path1, real=False)
    fid.update(imgs_path2, real=True)
    return fid.compute()


def preprocess_images(image_paths):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize((299, 299))
        image = ToTensor()(image)
        images.append(image)
    return torch.stack(images)


def run_demo_inference(device, reset_seeds, input_path, num_prompts, num_inference_steps):
    disable_persistent_kernel_cache()

    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # 4. load the K-LMS scheduler with some fitting parameters.
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    ttnn_scheduler = TtLMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, device=device
    )

    torch_device = "cpu"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(174)  # 10233 Seed generator to create the inital latent noise

    inputs = load_inputs(input_path)
    input_prompts = inputs[:num_prompts]

    for input_data in input_prompts:
        experiment_name = f"input_data_{input_prompts.index(input_data)}"
        input_prompt = [input_data]
        logger.info(f"input prompt : {input_prompt}")
        batch_size = len(input_prompt)

        ## First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
        # Tokenizer and Text Encoder
        text_input = tokenizer(
            input_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        # For classifier-free guidance, we need to do two forward passes: one with the conditioned input (text_embeddings),
        # and another with the unconditional embeddings (uncond_embeddings).
        # In practice, we can concatenate both into a single batch to avoid doing two forward passes.
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Initial random noise
        latents = torch.randn(
            (batch_size, unet.config.in_channels, height // 8, width // 8),
            generator=generator,
        )

        latents = latents.to(torch_device)

        scheduler.set_timesteps(num_inference_steps)
        ttnn_scheduler.set_timesteps(num_inference_steps, device)

        latents = latents * scheduler.init_noise_sigma
        ttnn_latents = ttnn.from_torch(latents, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        latents_dict = {}
        pcc_res = {}
        iter = 0

        # torch Denoising loop
        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = latent_expansion(latents, scheduler, t)
            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred = guide(noise_pred, guidance_scale, t)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            latents_dict[iter] = latents
            save_image_and_latents(latents, iter, vae, pre_fix=f"{experiment_name}_torch", pre_fix2="")
            iter += 1

        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        # Image post-processing
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images][0]
        torch_output_path = f"{experiment_name}_torch.png"
        pil_images.save(torch_output_path)

        iter = 0
        last_latents = None
        config = unet.config

        # # Denoising loop
        time_step_list = []
        ttnn_sigma = []
        ttnn_step_index = []
        timesteps_bkp = ttnn.to_torch(ttnn_scheduler.timesteps)
        sigma_tensor = ttnn.to_torch(ttnn_scheduler.sigmas)[0]
        step_index = (timesteps_bkp[0] == timesteps_bkp[0][0]).nonzero().item()
        ttnn_latent_model_input = tt_latent_expansion(
            ttnn_latents, ttnn_scheduler, float(sigma_tensor[step_index]), device
        )

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
        for step_index in ttnn_step_index:
            print("Step index :", step_index, " ", orders)
            order = min(step_index + 1, orders)
            order_list.append(order)
            lms_coeffs = [
                get_lms_coefficient(order, step_index, curr_order, sigma_tensor) for curr_order in range(order)
            ]

            ttnn_lms_coeff.append(lms_coeffs)

        for i in range(len(time_step_list)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            # ttnn_latent_model_input = tt_latent_expansion(ttnn_latents, ttnn_scheduler, t)
            ttnn_latent_model_input = tt_latent_expansion(ttnn_latents, ttnn_scheduler, float(ttnn_sigma[i]), device)
            ttnn_text_embeddings = ttnn.from_torch(
                text_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

            # predict the noise residual
            with torch.no_grad():
                parameters = preprocess_model_parameters(
                    initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
                )
                ttnn_noise_pred = ttnn_unet_condition(
                    sample=ttnn_latent_model_input,
                    timestep=time_step_list[i],
                    encoder_hidden_states=ttnn_text_embeddings,
                    parameters=parameters,
                    device=device,
                    config=config,
                )

            # perform guidance
            noise_pred = tt_guide(ttnn_noise_pred, guidance_scale)

            ttnn_latents = ttnn_scheduler.step(
                model_output=noise_pred,
                sample=ttnn_latents,
                sigma=float(ttnn_sigma[i]),
                lms_coeffs=ttnn_lms_coeff[i],
                device=device,
                order=order_list[i],
            ).prev_sample

            last_latents = ttnn_latents
            enable_persistent_kernel_cache()

        latents = last_latents
        latents = ttnn.to_torch(latents).to(torch.float32)
        for key, val in pcc_res.items():
            logger.info(f"{key}, {val}")

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        # Image post-processing
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images][0]
        ttnn_output_path = f"{experiment_name}_ttnn.png"
        pil_images.save(ttnn_output_path)


@pytest.mark.parametrize(
    "num_prompts",
    ((1),),
)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((3),),
)
def test_demo(device, reset_seeds, input_path, num_prompts, num_inference_steps):
    return run_demo_inference(device, reset_seeds, input_path, num_prompts, num_inference_steps)
