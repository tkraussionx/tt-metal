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


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    torch.manual_seed(1234)

    program_config = tt_lib.operations.primary.transformers.SDPAMultiCoreProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )

    compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
        math_fidelity=tt_lib.tensor.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    Q = torch.randn(b, nh, s, d)
    K = torch.randn(b, nkv, s, d)
    V = torch.randn(b, nkv, s, d)
    attn_mask = torch.full((s, s), torch.finfo(torch.float32).min)
    attn_mask = torch.triu(attn_mask, diagonal=1).expand(b, 1, -1, -1)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")
    logger.debug(f"attn_mask: {attn_mask.shape}")

    tt_Q = tt_lib.tensor.Tensor(Q, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    tt_K = tt_lib.tensor.Tensor(K, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    tt_V = tt_lib.tensor.Tensor(V, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    tt_attn_mask = tt_lib.tensor.Tensor(attn_mask, dtype).to(tt_lib.tensor.Layout.TILE).to(device)

    tt_back = tt_lib.operations.primary.transformers.scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, tt_attn_mask, is_causal=True, program_config=program_config
    )
    tt_back = tt_back.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask, is_causal=False)

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.debug(f"python vs pytorch: {out_pcc}")
    assert out_pass


# @pytest.mark.skip(reason="ND PCC issues")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype", [tt_lib.tensor.DataType.BFLOAT8_B, tt_lib.tensor.DataType.BFLOAT16], ids=["bfp8", "bf16"]
)
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 8, 1, 2048, 128],  # Llama2-70B
        [1, 16, 1, 2048, 64],  # Falcon-40B
        [1, 71, 1, 2048, 64],  # Falcon-7B
        [8, 8, 1, 2048, 128],  # Llama2-70B large batch
        [1, 8, 1, 8192, 128],  # Llama2-70B large sequence
    ),
)
def test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")
    tt_lib.device.DisablePersistentKernelCache()
    run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)


@pytest.mark.skip(reason="ND PCC issues")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype", [tt_lib.tensor.DataType.BFLOAT8_B, tt_lib.tensor.DataType.BFLOAT16], ids=["bfp8", "bf16"]
)
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 8, 1, 2048, 128],  # Llama2-70B
        [1, 16, 1, 2048, 64],  # Falcon-40B
        [1, 71, 1, 2048, 64],  # Falcon-7B
    ),
)
def test_sdpa_tt_with_program_cache(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, use_program_cache):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")

    for _ in range(2):
        run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)

    assert device.num_program_cache_entries() == 1


def nearest_32(x):
    return ((x + 31) // 32) * 32


def run_test_sdpa_decode(device, b, nh, nkv, s, d, dtype):
    assert b == 16, "Haven't generalized yet"
    padded_num_heads = nearest_32(nh)
    torch.manual_seed(1234)

    program_config = tt_lib.operations.primary.transformers.SDPAMultiCoreProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=32,
    )

    compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
        math_fidelity=tt_lib.tensor.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.types.MemoryConfig(ttnn.types.TensorMemoryLayout.INTERLEAVED, ttnn.types.BufferType.DRAM)
    shard_grid = ttnn.experimental.tensor.CoreRangeSet(
        {
            ttnn.experimental.tensor.CoreRange(
                ttnn.experimental.tensor.CoreCoord(0, 0),
                ttnn.experimental.tensor.CoreCoord(7, 1),
            )
        }
    )
    shard_spec = ttnn.experimental.tensor.ShardSpec(
        shard_grid, (padded_num_heads, d), ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR, False
    )

    height_sharded_memcfg = ttnn.types.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    K = torch.randn(nkv, b, s, d)
    V = torch.randn(nkv, b, s, d)
    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    start_idx = 10
    while start_idx < s:
        # Test various sequence lengths
        padded_layer_len = nearest_32(start_idx)

        attn_mask = torch.zeros((b, padded_num_heads, padded_layer_len))
        # Assume all users are at same position
        attn_mask[:, :, start_idx:] = float("-inf")

        Q = torch.randn(1, b, padded_num_heads, d)

        tt_Q = ttnn.as_tensor(
            Q, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=height_sharded_memcfg
        )
        print(f"Q memcfg: {tt_Q.memory_config()}")

        tt_attn_mask = ttnn.as_tensor(
            attn_mask, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg
        )
        breakpoint()

        tt_back = tt_lib.operations.primary.transformers.scaled_dot_product_attention(
            tt_Q,
            tt_K,
            tt_V,
            tt_attn_mask,
            is_causal=False,
            program_config=program_config,
            valid_seq_len=padded_layer_len,
            compute_kernel_config=compute_kernel_config,
        )

        Q_slice = Q[:, :, :nh, :]
        K_slice = K[:, :, :padded_layer_len, :]
        V_slice = V[:, :, :padded_layer_len, :]
        attn_mask_slice = attn_mask[:, :nh, :]
        expect = torch.nn.functional.scaled_dot_product_attention(
            Q_slice, K_slice, V_slice, attn_mask_slice, is_causal=False
        )

        start_idx += 201
    attn_mask = torch.triu(attn_mask, diagonal=1).expand(b, 1, -1, -1)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")
    logger.debug(f"attn_mask: {attn_mask.shape}")

    tt_Q = tt_lib.tensor.Tensor(Q, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    tt_K = tt_lib.tensor.Tensor(K, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    tt_V = tt_lib.tensor.Tensor(V, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    tt_attn_mask = tt_lib.tensor.Tensor(attn_mask, dtype).to(tt_lib.tensor.Layout.TILE).to(device)

    tt_back = tt_lib.operations.primary.transformers.scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, tt_attn_mask, is_causal=True, program_config=program_config
    )
    tt_back = tt_back.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask, is_causal=False)

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.debug(f"python vs pytorch: {out_pcc}")
    assert out_pass


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype", [tt_lib.tensor.DataType.BFLOAT8_B, tt_lib.tensor.DataType.BFLOAT16], ids=["bfp8", "bf16"]
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [16, 8, 1, 8192, 128],  # Llama2-70B
        # [1, 8, 1, 2048, 128],  # Llama2-70B
        # [1, 16, 1, 2048, 64],  # Falcon-40B
        # [1, 71, 1, 2048, 64],  # Falcon-7B
        # [8, 8, 1, 2048, 128],  # Llama2-70B large batch
        # [1, 8, 1, 8192, 128],  # Llama2-70B large sequence
    ),
)
def test_sdpa_decode(device, b, nh, nkv, s, d, dtype):
    tt_lib.device.DisablePersistentKernelCache()
    run_test_sdpa_decode(device, b, nh, nkv, s, d, dtype)
