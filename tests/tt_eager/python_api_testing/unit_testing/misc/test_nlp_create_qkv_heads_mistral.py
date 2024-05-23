# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import tt2torch_tensor, comp_pcc
from models.utility_functions import is_grayskull
import torch

"""
Mistral 7B shapes + functionality
"""


def run_nlp_create_qkv_heads_mistral_test(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    dtype,
    in_mem_config,
    out_mem_config,
    device,
):
    torch.manual_seed(1234)

    in0_shape = [seq_len, 1, batch, (num_q_heads + 2 * num_kv_heads) * head_dim]
    A = torch.randn(in0_shape)
    in0_t = ttl.tensor.Tensor(A, dtype).to(ttl.tensor.Layout.TILE).to(device, in_mem_config)

    q, k, v = ttl.tensor.nlp_create_qkv_heads_mistral(
        in0_t,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        output_mem_config=out_mem_config,
    )

    # Check memory of inputs and outputs
    assert in0_t.memory_config().buffer_type == in_mem_config.buffer_type
    assert q.memory_config().buffer_type == out_mem_config.buffer_type
    assert k.memory_config().buffer_type == out_mem_config.buffer_type
    assert v.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0: {in0_t.memory_config().buffer_type} and {in0_t.get_dtype()}")
    logger.debug(f"q: {q.memory_config().buffer_type} and {q.get_dtype()}")
    logger.debug(f"k: {k.memory_config().buffer_type} and {k.get_dtype()}")
    logger.debug(f"v: {v.memory_config().buffer_type} and {v.get_dtype()}")

    assert list(q.get_legacy_shape()) == [batch, seq_len, num_q_heads, head_dim]
    assert list(k.get_legacy_shape()) == [batch, seq_len, 32, head_dim * num_kv_heads]
    assert list(v.get_legacy_shape()) == [batch, seq_len, 32, head_dim * num_kv_heads]

    pyt_got_back_rm_q = tt2torch_tensor(q).transpose(0, 2)
    pyt_got_back_rm_k = tt2torch_tensor(k)#[:, :, :1, :]
    print("K torch", pyt_got_back_rm_k)
    pyt_got_back_rm_k = pyt_got_back_rm_k.transpose(0, 2)[:, :, :1, :]
    pyt_got_back_rm_v = tt2torch_tensor(v).transpose(0, 2)[:, :, :1, :]

    (ref_q, ref_k, ref_v) = torch.split(
        A, [num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
    )

    # Additional shuffling for Q, K, V heads
    ref_q = torch.reshape(ref_q, [batch, seq_len, num_q_heads, head_dim])  # .transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch, seq_len, 1, num_kv_heads * head_dim])  # .transpose(-3, -2)
    print("ref k", ref_k)
    ref_v = torch.reshape(ref_v, [batch, seq_len, 1, num_kv_heads * head_dim])  # .transpose(-3, -2)

    if dtype == ttl.tensor.DataType.BFLOAT8_B:
        pcc = 0.99
    elif dtype == ttl.tensor.DataType.FLOAT32:  # conversion from fp32 to tf32 will decrease pcc
        pcc = 0.9999999
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, pcc)
    logger.debug(f"Q passing={passing_pcc_q}")
    logger.debug(f"Q output pcc={output_pcc_q}")

    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, pcc)
    logger.debug(f"K passing={passing_pcc_k}")
    logger.debug(f"K output pcc={output_pcc_k}")

    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, pcc)
    logger.debug(f"V passing={passing_pcc_v}")
    logger.debug(f"V output pcc={output_pcc_v}")
    assert passing_pcc_q
    assert passing_pcc_k
    assert passing_pcc_v


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_L1"],
)
@pytest.mark.parametrize(
    "in_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["in_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B,),
    ids=["BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, num_q_heads, num_kv_heads",
    ((32, 1, 128, 32, 8),),
)
def test_nlp_create_qkv_heads_mistral_test(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    dtype,
    in_mem_config,
    out_mem_config,
    device,
):
    if is_grayskull() and dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")
    else:
        run_nlp_create_qkv_heads_mistral_test(
            batch,
            seq_len,
            head_dim,
            num_q_heads,
            num_kv_heads,
            dtype,
            in_mem_config,
            out_mem_config,
            device,
        )


# def test_nlp_create_qkv_heads_with_program_cache(device, use_program_cache):
#     dtype = ttl.tensor.DataType.BFLOAT8_B
#     mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
#     for _ in range(2):
#         run_nlp_create_qkv_heads_mistral_test(32, 1, 128, 32, 8, dtype, mem_config, mem_config, device)
#         run_nlp_create_qkv_heads_mistral_test(32, 1, 128, 32, 8, dtype, mem_config, mem_config, device)
#         dummy_shape = [1, 1, 32, 32]
#         py_dummy_tensor = torch.randn(dummy_shape)
#         tt_dummy_tensor = ttl.tensor.Tensor(py_dummy_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config)

#     assert device.num_program_cache_entries() == 2
