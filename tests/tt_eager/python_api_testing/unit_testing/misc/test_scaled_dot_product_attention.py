# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
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
        compute_with_storage_grid_size=[8, 7],
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )

    compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
        math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    import torch.nn.init as init

    # Q = torch.randn(b, nh, s, d)
    # K = torch.randn(b, nkv, s, d)
    # V = torch.randn(b, nkv, s, d)
    Q = torch.empty(b, nh, s, d)
    K = torch.empty(b, nkv, s, d)
    V = torch.empty(b, nkv, s, d)
    init.xavier_uniform_(Q)
    init.xavier_uniform_(K)
    init.xavier_uniform_(V)

    Q += torch.randn_like(Q) * 0.02
    K += torch.randn_like(K) * 0.02
    V += torch.randn_like(V) * 0.02
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
        tt_Q,
        tt_K,
        tt_V,
        tt_attn_mask,
        is_causal=True,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    tt_back = tt_back.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask, is_causal=False)

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.debug(f"python vs pytorch: {out_pcc}")

    # The following code prints more statistics about the difference between gt and tt_back. What is the MAE, what is the MSE, is there a bias to the error?
    logger.debug(f"MAE: {torch.mean(torch.abs(gt - tt_back))}")
    logger.debug(f"MSE: {torch.mean((gt - tt_back) ** 2)}")
    logger.debug(f"bias: {torch.mean(gt - tt_back)}")
    logger.debug(f"std: {torch.std(gt - tt_back)}")
    logger.debug(f"GT mean: {torch.mean(gt)}")
    logger.debug(f"TT mean: {torch.mean(tt_back)}")
    # breakpoint()
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
        # [1, 8, 1, 2048, 128],  # Llama2-70B
        [1, 16, 1, 128, 64],  # Falcon-40B
        # [1, 71, 1, 2048, 64],  # Falcon-7B
        # [8, 8, 1, 2048, 128],  # Llama2-70B large batch
        # [1, 8, 1, 8192, 128],  # Llama2-70B large sequence
    ),
)
def test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, use_program_cache):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")
    tt_lib.device.DisablePersistentKernelCache()
    run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)


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


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power


def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.experimental.tensor.CoreRange(
        ttnn.experimental.tensor.CoreCoord(0, 0),
        ttnn.experimental.tensor.CoreCoord(num_x - 1, num_y - 1),
    )


def get_chunk_size(s):
    # Not sure if optimal
    if s <= 32:
        return 32
    if s <= 64:
        return 64
    if s <= 128:
        return 128
    if s <= 256:
        return 256
    if s <= 2048:
        return 512
    return 1024


def run_test_sdpa_decode(device, b, nh, nkv, s, d, dtype):
    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
        math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.types.MemoryConfig(ttnn.types.TensorMemoryLayout.INTERLEAVED, ttnn.types.BufferType.DRAM)
    shard_grid = ttnn.experimental.tensor.CoreRangeSet({num_to_corerange(b)})
    shard_spec = ttnn.experimental.tensor.ShardSpec(
        shard_grid, (padded_num_heads, d), ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR, False
    )

    height_sharded_memcfg = ttnn.types.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    # K = torch.zeros(nkv, b, s, d)
    # K = torch.ones(nkv, b, s, d)
    K = torch.eye(s, d).expand(nkv, b, s, d)
    # V = torch.ones(nkv, b, s, d)
    V = torch.eye(s, d).expand(nkv, b, s, d)

    tt_K = ttnn.as_tensor(
        K, device=device, dtype=tt_lib.tensor.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg
    )
    tt_V = ttnn.as_tensor(
        V, device=device, dtype=tt_lib.tensor.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg
    )

    start_idx = 31

    while start_idx + 1 < s:
        scale = d**-0.5
        kv_len = start_idx + 1
        k_chunk_size = get_chunk_size(kv_len)
        program_config = tt_lib.operations.primary.transformers.SDPAMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=padded_num_heads,
            k_chunk_size=k_chunk_size,
        )

        padded_layer_len = nearest_n(kv_len, n=k_chunk_size)

        # Test various sequence lengths
        logger.debug(f"Testing with sequence length: {kv_len}")
        logger.debug(f"Using chunk size: {k_chunk_size}")
        logger.debug(f"Using padded layer length: {padded_layer_len}")
        logger.debug(f"Using padded num heads: {padded_num_heads}")

        attn_mask = torch.zeros((1, b, padded_num_heads, padded_layer_len))
        # attn_mask = torch.ones((1, b, padded_num_heads, padded_layer_len))
        # Assume all users are at same position
        # DEBUG: Found that all zeros attn_mask add still leads to ND
        # attn_mask[:, :, :, kv_len:] = torch.finfo(torch.float32).min

        torch.manual_seed(0)
        # Q = torch.zeros(1, b, padded_num_heads, d)
        # Q = torch.ones(1, b, padded_num_heads, d)
        Q = torch.eye(padded_num_heads, d).expand(1, b, padded_num_heads, d)

        Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
        K_slice = K[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
        V_slice = V[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
        attn_mask_slice = attn_mask[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, S
        expect = torch.nn.functional.scaled_dot_product_attention(
            Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
        )  # b, nh, 1, d
        expect = expect.squeeze().unsqueeze(0)

        tt_Q = ttnn.as_tensor(
            Q,
            device=device,
            dtype=tt_lib.tensor.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=height_sharded_memcfg,
        )

        tt_attn_mask = ttnn.as_tensor(
            attn_mask,
            device=device,
            dtype=tt_lib.tensor.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dram_memcfg,
        )
        for i in range(1000):
            logger.info(f"Iteration: {i}")

            tt_back = tt_lib.operations.primary.transformers.scaled_dot_product_attention(
                tt_Q,
                tt_K,
                tt_V,
                tt_attn_mask,
                is_causal=False,
                scale=scale,
                program_config=program_config,
                valid_seq_len=padded_layer_len,
                # compute_kernel_config=compute_kernel_config,
                output_mem_config=height_sharded_memcfg,
            )

            tt_back = ttnn.experimental.tensor.sharded_to_interleaved(tt_back)
            tt_back = ttnn.to_torch(tt_back)
            tt_back = tt_back[:, :, :nh, :]
            out_pass, out_pcc = comp_pcc(expect, tt_back, 0.99)

            # _, top_16_pcc = comp_pcc(expect[:,:16,...], tt_back[:,:16,...], 0.99)
            # _, bottom_16_pcc = comp_pcc(expect[:,16:,...], tt_back[:,16:,...], 0.99)
            if i == 0:
                tt_expect = tt_back
            else:
                if not (tt_back == tt_expect).all():
                    logger.info(f"Mismatch: {i}")
                    bad = []
                    for batch_idx in range(b):
                        if not (
                            tt_back[:, batch_idx : batch_idx + 1, ...] == tt_expect[:, batch_idx : batch_idx + 1, ...]
                        ).all():
                            bad.append(batch_idx)
                            print(f"Batch: {batch_idx}")
                            print(f"Expect: {tt_expect[:,batch_idx:batch_idx+1,...]}")
                            print(f"Got: {tt_back[:,batch_idx:batch_idx+1,...]}")
                    breakpoint()
                    # worst = 1
                    # worst_user = None
                    # for batch_id in range(b):
                    #     _, bad_pcc = comp_pcc(expect[:,batch_id:batch_id+1,...], tt_back[:,batch_id:batch_id+1,...], 0)
                    #     bad_pcc = float(bad_pcc.split("PCC:")[1].split(" ,")[0].strip())
                    #     logger.info("{bad_pcc}", bad_pcc="<red>"+f"{bad_pcc}"+"</red>"if bad_pcc < .99 else f"{bad_pcc}")
                    #     if bad_pcc < worst:
                    #         worst = bad_pcc
                    #         worst_user = batch_id
                    # logger.info(f"Worst user: {worst_user}, pcc={worst}")
                    # logger.info(f"Top 16: {top_16_pcc}, \nBottom 16: {bottom_16_pcc}")

        # logger.debug(f"python vs pytorch: {out_pcc}")
        if not out_pass:
            logger.debug(f"REPEAT PREVIOUS ITERATION FOR ND CHECK")
            start_idx -= 1
        assert out_pass

        start_idx += 1
        break


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype",
    [tt_lib.tensor.DataType.BFLOAT8_B, tt_lib.tensor.DataType.BFLOAT16],
    ids=["bfp8", "bf16"],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [32, 32, 1, 4096, 64],  # Llama2-70B
        # [16, 8, 1, 8192, 128],  # Llama2-70B
        # [32, 16, 1, 2048, 64],  # Falcon-40B
        # [32, 4, 1, 8192, 128],  # Mixtral
    ),
)
def test_sdpa_decode(device, b, nh, nkv, s, d, dtype):
    if dtype == tt_lib.tensor.DataType.BFLOAT8_B:
        pytest.skip()
    tt_lib.device.DisablePersistentKernelCache()
    run_test_sdpa_decode(device, b, nh, nkv, s, d, dtype)


def run_test_sdpa_decode_single_iter(device, b, nh, nkv, s, d, dtype):
    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
        math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.types.MemoryConfig(ttnn.types.TensorMemoryLayout.INTERLEAVED, ttnn.types.BufferType.DRAM)
    shard_grid = ttnn.experimental.tensor.CoreRangeSet({num_to_corerange(b)})
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

    # start_idx = s // 2
    start_idx = 138
    scale = d**-0.5

    k_chunk_size = get_chunk_size(start_idx)
    program_config = tt_lib.operations.primary.transformers.SDPAMultiCoreProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=padded_num_heads,
        k_chunk_size=k_chunk_size,
    )

    padded_layer_len = nearest_n(start_idx, n=k_chunk_size)

    # Test various sequence lengths
    logger.debug(f"Testing with sequence length: {start_idx}")
    logger.debug(f"Using chunk size: {k_chunk_size}")
    logger.debug(f"Using padded layer length: {padded_layer_len}")
    logger.debug(f"Using padded num heads: {padded_num_heads}")

    attn_mask = torch.zeros((1, b, padded_num_heads, padded_layer_len))
    # Assume all users are at same position
    attn_mask[:, :, :, start_idx:] = torch.finfo(torch.float32).min

    Q = torch.randn(1, b, padded_num_heads, d)

    tt_Q = ttnn.as_tensor(Q, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=height_sharded_memcfg)

    tt_attn_mask = ttnn.as_tensor(
        attn_mask, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg
    )

    tt_back = tt_lib.operations.primary.transformers.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        tt_attn_mask,
        is_causal=False,
        scale=scale,
        program_config=program_config,
        valid_seq_len=padded_layer_len,
        compute_kernel_config=compute_kernel_config,
        output_mem_config=height_sharded_memcfg,
    )

    tt_back = ttnn.to_torch(tt_back)
    tt_back = tt_back[:, :, :nh, :]

    Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
    K_slice = K[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
    V_slice = V[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
    attn_mask_slice = attn_mask[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, S
    expect = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )  # b, nh, 1, d
    expect = expect.squeeze().unsqueeze(0)

    out_pass, out_pcc = comp_pcc(expect, tt_back, 0.99)

    logger.debug(f"python vs pytorch: {out_pcc}")
    assert out_pass


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype",
    [tt_lib.tensor.DataType.BFLOAT8_B, tt_lib.tensor.DataType.BFLOAT16],
    ids=["bfp8", "bf16"],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    # ([16, 8, 1, 8192, 128],),  # Llama2-70B
    ([32, 8, 1, 2048, 128],),  # Llama2-70B
)
def test_sdpa_decode_program_cache(device, b, nh, nkv, s, d, dtype, use_program_cache):
    tt_lib.device.DisablePersistentKernelCache()

    dummy_tensors = []
    for _ in range(1000):
        dummy_tensors.append(
            ttnn.as_tensor(
                torch.zeros(32, 32),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.types.MemoryConfig(
                    ttnn.types.TensorMemoryLayout.INTERLEAVED, ttnn.types.BufferType.DRAM
                ),
            )
        )
        dummy_tensors.append(
            ttnn.as_tensor(
                torch.zeros(1, 1, 32, 32 * 32),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.types.MemoryConfig(
                    ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.types.BufferType.L1,
                    ttnn.experimental.tensor.ShardSpec(
                        ttnn.experimental.tensor.CoreRangeSet({num_to_corerange(32)}),
                        (32, 32),
                        ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
                        False,
                    ),
                ),
            )
        )
        run_test_sdpa_decode_single_iter(device, b, nh, nkv, s, d, dtype)

    assert device.num_program_cache_entries() == 1
