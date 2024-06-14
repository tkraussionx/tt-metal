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


def test_mixtral_topk(t3k_device_mesh, use_program_cache, reset_seeds):
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
    ref_out_topk = (
        (
            ttnn.to_torch(tt_out_11BH, mesh_composer=ttnn.ConcatMeshToTensor(t3k_device_mesh, dim=0))[0]
            .squeeze(1)
            .view(batch_size, seq_len, -1)
            .float()
        )
        .squeeze()
        .topk(k=32)
    )
    logger.info(f"ref_out_topk: indices={ref_out_topk.indices.shape}, values={ref_out_topk.values.shape}")

    # # Update the users that are still in prefill and the ones generating new tokens
    # if iteration < max_prompt_len:
    #     tt_token_batch = torch.where(
    #         input_mask_pt[:, iteration], input_tokens_pt[:, iteration], tt_token_batch[:, 0]
    #     ).unsqueeze(1)

    # TODO Update argmax to ttnn when OP becomes available
    logger.info(f"Running on device")
    tt_values_11BK, tt_indices_11BK = ttnn.experimental.operations.primary.topk(tt_out_11BH, 32)
    # # Update the users that are still in prefill and the ones generating new tokens
    # if iteration < max_prompt_len:
    #     logger.info(f"Calling where with input shape {input_mask[iteration].shape}, input_tokens_tt[iteration]: {input_tokens_tt[iteration].shape}, tt_out_1B: {tt_out_1B.shape}")
    #     decode_input_1B = ttnn.where(input_mask[iteration], input_tokens_tt[iteration], tt_out_1B)
    # else:
    #     decode_input_1B = tt_out_1B

    tt_values_BK = ttnn.to_torch(tt_values_11BK, mesh_composer=ttnn.ConcatMeshToTensor(t3k_device_mesh, dim=0))[
        0
    ].squeeze()
    tt_indices_BK = ttnn.to_torch(tt_values_11BK, mesh_composer=ttnn.ConcatMeshToTensor(t3k_device_mesh, dim=0))[
        0
    ].squeeze()
    passing = comp_allclose(ref_out_topk.values, tt_values_BK) and comp_allclose(ref_out_topk.indices, tt_indices_BK)

    if passing:
        logger.info("Mixtral_argmax Passed!")
    else:
        logger.warning("Mixtral_argmax Failed!")

    assert passing, f"Mixtral_argmax output does not match."


def test_mixtral_top1(t3k_device_mesh, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b
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
    ref_out_topk = (
        (
            ttnn.to_torch(tt_out_11BH, mesh_composer=ttnn.ConcatMeshToTensor(t3k_device_mesh, dim=0))[0]
            .squeeze(1)
            .view(batch_size, seq_len, -1)
            .float()
        )
        .squeeze()
        .topk(k=1)
    )
    logger.info(f"ref_out_topk: indices={ref_out_topk.indices.shape}, values={ref_out_topk.values.shape}")

    logger.info(f"Running on device")
    tt_values_11BK, tt_indices_11BK = ttnn.experimental.operations.primary.topk(tt_out_11BH, 32)
    tt_out_111B = tt_indices_11BK[:, :, :, :1]
    # # Update the users that are still in prefill and the ones generating new tokens
    # if iteration < max_prompt_len:
    #     logger.info(f"Calling where with input shape {input_mask[iteration].shape}, input_tokens_tt[iteration]: {input_tokens_tt[iteration].shape}, tt_out_1B: {tt_out_1B.shape}")
    #     decode_input_1B = ttnn.where(input_mask[iteration], input_tokens_tt[iteration], tt_out_1B)
    # else:
    #     decode_input_1B = tt_out_1B

    tt_indices_B1 = ttnn.to_torch(tt_out_111B, mesh_composer=ttnn.ConcatMeshToTensor(t3k_device_mesh, dim=0))[
        0
    ].squeeze()
    passing = comp_allclose(ref_out_topk.indices, tt_indices_B1)

    if passing:
        logger.info("Mixtral_argmax Passed!")
    else:
        logger.warning("Mixtral_argmax Failed!")

    assert passing, f"Mixtral_argmax output does not match."


def test_where(t3k_device_mesh, use_program_cache, reset_seeds):
    max_prompt_len = 12
    input_mask = torch.randint(0, 2, (32, max_prompt_len), dtype=torch.bool)
    input_tokens = torch.randint(0, 32000, (32, max_prompt_len), dtype=torch.long)
    input_tokens_tt = [
        ttnn.from_torch(
            input_tokens[:, i].unsqueeze(1).unsqueeze(0).unsqueeze(0),
            device=t3k_device_mesh,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_device_mesh),
        )
        for i in range(max_prompt_len)
    ]
    input_mask_tt = [
        ttnn.from_torch(
            input_mask[:, i].unsqueeze(1).unsqueeze(0).unsqueeze(0),
            device=t3k_device_mesh,
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_device_mesh),
        )
        for i in range(max_prompt_len)
    ]

    tt_out_pt = torch.randint(0, 32000, (32, 1), dtype=torch.long)
    tt_out_11B1 = ttnn.from_torch(
        tt_out_pt.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_device_mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_device_mesh),
    )

    for iteration in range(max_prompt_len):
        decode_input_11B1 = ttnn.where(input_mask_tt[iteration], input_tokens_tt[iteration], tt_out_11B1)
        print(decode_input_11B1)
        decode_input_B_pt = torch.where(input_mask[:, iteration], input_tokens[:, iteration], tt_out_pt.view(32))
        tt_to_torch = ttnn.to_torch(decode_input_11B1, mesh_composer=ttnn.ConcatMeshToTensor(t3k_device_mesh, dim=0))[
            0
        ].view(32)
        passing, pcc = comp_pcc(decode_input_B_pt, tt_to_torch, 0.99)
        print(passing, pcc)


import torch
import ttnn
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


def test_typecasting(device):
    torch_input = torch.randint(0, 32000, (1, 1, 32, 32), dtype=torch.long)
    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_input = ttnn.experimental.tensor.typecast(tt_input, dtype=ttnn.bfloat16)
    tt_input_to_torch = ttnn.to_torch(tt_input)
    passing, pcc = comp_pcc(tt_input_to_torch, torch_input, 0.99)
    print(passing, pcc)

    tt_input = ttnn.experimental.tensor.typecast(tt_input, dtype=ttnn.uint32)
    tt_input_to_torch = ttnn.to_torch(tt_input)
    passing, pcc = comp_pcc(tt_input_to_torch, torch_input, 0.99)
    print(passing, pcc)
    assert passing, f"Typecasting failed"
