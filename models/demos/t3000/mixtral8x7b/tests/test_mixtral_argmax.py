# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger

# Set Mixtral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["MIXTRAL_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"

import ttnn
import tt_lib
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import sample


def test_mixtral_argmax(t3k_device_mesh, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat16
    batch_size = 32
    seq_len = 1

    tt_out_11BH = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_device_mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_device_mesh),
    )

    # Reference code on host
    logger.info(f"Running on host")
    ref_out_1B = (
        (
            ttnn.to_torch(tt_out_11BH, mesh_composer=ttnn.ConcatMeshToTensor(t3k_device_mesh, dim=0))[0]
            .squeeze(1)
            .view(batch_size, seq_len, -1)
            .float()
        )
        .squeeze()
        .argmax(axis=-1)
    )
    logger.info(f"ref_token_batch: {ref_out_1B.shape}")

    # # Update the users that are still in prefill and the ones generating new tokens
    # if iteration < max_prompt_len:
    #     tt_token_batch = torch.where(
    #         input_mask_pt[:, iteration], input_tokens_pt[:, iteration], tt_token_batch[:, 0]
    #     ).unsqueeze(1)

    # TODO Update argmax to ttnn when OP becomes available
    logger.info(f"Running on device")
    tt_out_B11B = ttnn.experimental.tensor.argmax(tt_out_11BH, dim=-1)
    tt_out_1B = ttnn.reshape(tt_out_B11B[:1, :, :, :], ttnn.Shape([1, batch_size]))  # [1, 32] Bfloat16
    logger.info(f"tt_out_1B shape: {tt_out_1B.shape}")
    # # Update the users that are still in prefill and the ones generating new tokens
    # if iteration < max_prompt_len:
    #     logger.info(f"Calling where with input shape {input_mask[iteration].shape}, input_tokens_tt[iteration]: {input_tokens_tt[iteration].shape}, tt_out_1B: {tt_out_1B.shape}")
    #     decode_input_1B = ttnn.where(input_mask[iteration], input_tokens_tt[iteration], tt_out_1B)
    # else:
    #     decode_input_1B = tt_out_1B
    tt_out_1B = ttnn.to_torch(tt_out_1B, mesh_composer=ttnn.ConcatMeshToTensor(t3k_device_mesh, dim=0))[0]
    passing = comp_allclose(ref_out_1B, tt_out_1B)

    if passing:
        logger.info("Mixtral_argmax Passed!")
    else:
        logger.warning("Mixtral_argmax Failed!")

    assert passing, f"Mixtral_argmax output does not match."
