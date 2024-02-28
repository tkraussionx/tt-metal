# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import torch
import ttnn
from tqdm.auto import tqdm
from loguru import logger
import pytest

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler

from models.utility_functions import (
    skip_for_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report

from models.experimental.functional_stable_diffusion.tt.ttnn_functional_unet_2d_condition_model import (
    UNet2DConditionModel as ttnn_UNet2DConditionModel,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_stable_diffusion.custom_preprocessing import custom_preprocessor
from models.experimental.functional_stable_diffusion.helper_funcs import (
    ttnn_latent_expansion,
    ttnn_step,
    constant_prop_time_embeddings,
    guide,
)

NUM_INFERENCE_STEPS = 1  # Number of denoising steps
BATCH_SIZE = 1


@skip_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            140,
            140,
        ),
    ),
)
def test_perf(device, expected_inference_time, expected_compile_time, model_name, reset_seeds):
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

    # 4. load the K-LMS scheduler with some fitting parameters.
    tt_scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    disable_persistent_kernel_cache()
    torch_device = "cpu"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    config = unet.config

    prompt = [
        "oil painting frame of Breathtaking mountain range with a clear river running through it, surrounded by tall trees and misty clouds, serene, peaceful, mountain landscape, high detail",
    ]  # guidance 7.5

    # height and width much be divisible by 32, and can be as little as 64x64
    # 64x64 images are not coherent; but useful for a quick pcc test.

    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(174)  # 10233 Seed generator to create the inital latent noise
    batch_size = len(prompt)

    ## First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
    # Tokenizer and Text Encoder
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    # For classifier-free guidance, we need to do two forward passes: one with the conditioned input (text_embeddings),
    # and another with the unconditional embeddings (uncond_embeddings).
    # In practice, we can concatenate both into a single batch to avoid doing two forward passes.
    # in this demo, each forward pass will be done independently.
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Initial random noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)

    tt_scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    latents = latents * tt_scheduler.init_noise_sigma
    tt_latents = torch.tensor(latents)
    model = unet

    # # Denoising loop
    g = {}
    durations = []
    for _ in range(2):
        for t in tqdm(tt_scheduler.timesteps):
            parameters = preprocess_model_parameters(
                initialize_model=lambda: model, custom_preprocessor=custom_preprocessor, device=device
            )

            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            tt_latent_model_input = ttnn_latent_expansion(tt_latents, tt_scheduler, t, device)
            # tt_latent_model_input = latent_expansion(tt_latents, tt_scheduler, t)

            _t = constant_prop_time_embeddings(t, tt_latent_model_input, unet.time_proj)

            _t = ttnn.from_torch(
                _t.unsqueeze(0).unsqueeze(0), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )
            tt_latent_model_input = ttnn.from_torch(
                tt_latent_model_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )
            tt_text_embeddings = ttnn.from_torch(
                text_embeddings, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )

            # predict the noise residual
            start = time.time()
            with torch.no_grad():
                tt_noise_pred = ttnn_UNet2DConditionModel(
                    tt_latent_model_input,
                    _t,
                    encoder_hidden_states=tt_text_embeddings,
                    parameters=parameters,
                    device=device,
                    config=config,
                    reader_patterns_cache=reader_patterns_cache,
                )
                noise_pred = ttnn.to_torch(tt_noise_pred)
            end = time.time()

            # perform guidance
            noise_pred = guide(noise_pred, guidance_scale, t)
            # compute the previous noisy sample x_t -> x_t-1

            # tt_latents = tt_scheduler.step(noise_pred, t, tt_latents).prev_sample
            tt_latents = ttnn_step(noise_pred, t, tt_latents, config=config)

            durations.append(end - start)
            enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    comments = f"image size: {height}x{width} - v1.4"

    prep_perf_report(
        model_name="batched_stable_diffusions",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=0.0,
    )
    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")
