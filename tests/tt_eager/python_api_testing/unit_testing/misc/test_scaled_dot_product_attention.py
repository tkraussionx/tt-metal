# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import tt_lib
import ttnn
from loguru import logger
import pytest
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0
from tqdm import tqdm


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def run_test_sdpa_tt_ND(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    program_config = tt_lib.operations.primary.transformers.SDPAMultiCoreProgramConfig(
        # compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        compute_with_storage_grid_size=[1, 1],
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )

    Q = torch.eye(s, d).expand(b, nh, s, d) * 5
    K = torch.eye(s, d).expand(b, nkv, s, d) * 7
    V = torch.eye(s, d).expand(b, nkv, s, d) * 9
    E = torch.eye(s, d).expand(b, nkv, s, d) * 316

    attn_mask = torch.full((s, s), torch.finfo(torch.float32).min)
    attn_mask = torch.triu(attn_mask, diagonal=1).expand(b, 1, -1, -1)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")
    logger.debug(f"attn_mask: {attn_mask.shape}")

    dram_memcfg = ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
    )

    # tt_Q = tt_lib.tensor.Tensor(Q, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    tt_Q = ttnn.as_tensor(Q, dtype=dtype, device=device, layout=tt_lib.tensor.Layout.TILE, memory_config=dram_memcfg)
    # tt_K = tt_lib.tensor.Tensor(K, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    tt_K = ttnn.as_tensor(K, dtype=dtype, device=device, layout=tt_lib.tensor.Layout.TILE, memory_config=dram_memcfg)
    # tt_V = tt_lib.tensor.Tensor(V, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    tt_V = ttnn.as_tensor(V, dtype=dtype, device=device, layout=tt_lib.tensor.Layout.TILE, memory_config=dram_memcfg)
    # tt_attn_mask = tt_lib.tensor.Tensor(attn_mask, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    tt_attn_mask = ttnn.as_tensor(
        attn_mask, dtype=dtype, device=device, layout=tt_lib.tensor.Layout.TILE, memory_config=dram_memcfg
    )

    torch.set_printoptions(profile="full")
    for idx in tqdm(range(30)):
        tt_back = tt_lib.operations.primary.transformers.scaled_dot_product_attention(
            tt_Q, tt_K, tt_V, tt_attn_mask, is_causal=True, program_config=program_config
        )
        tt_back = tt_back.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
        if torch.all(E.eq(tt_back)):
            print("Matched Expected")
        else:
            print("Error")


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize("dtype", [tt_lib.tensor.DataType.BFLOAT16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [64], ids=["q128"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k128"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 1, 1, 128, 128],
        # [1, 5, 1, 1024, 128],
        # [1, 8, 1, 2048, 128],  # Llama2-70B
        # [1, 16, 1, 2048, 64],  # Falcon-40B
        # [1, 71, 1, 2048, 64],  # Falcon-7B
    ),
)
def test_sdpa_nd(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    # torch.set_printoptions(profile="full")
    # buggy_tensor_1 = torch.load("buggy_tensor_1.pt")
    # buggy_tensor_2 = torch.load("buggy_tensor_2.pt")
    # buggy_tensor_3 = torch.load("buggy_tensor_3.pt")
    # print(buggy_tensor_1)
    # print(buggy_tensor_2)
    # print(buggy_tensor_3)

    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")

    # NOP 27-32 gives buggy tensor 1-3, 5 produces mixed errors.
    for idx in tqdm(range(100)):
        os.environ["TT_NOP_INSERT"] = str(idx)
        tt_lib.device.DisablePersistentKernelCache()
        run_test_sdpa_tt_ND(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)
        # assert device.num_program_cache_entries() == 1
