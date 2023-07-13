from pathlib import Path
import sys
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../..")

import numpy as np

import tt_lib as ttl
from python_api_testing.models.utility_functions import (
    comp_pcc,
)
import torch
import pytest


def run_bert_large_fused_qkv_matmul_test(
    in0_dtype,
    in1_dtype,
    bias_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    bias_mem_config,
    out_mem_config,
):
    torch.manual_seed(1234)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 1)
    ttl.device.InitializeDevice(device, ttl.device.MemoryAllocator.L1_BANKING)
    host = ttl.device.GetHost()
    a_shape = [9, 1, 384, 1024]
    b_shape = [1, 1, 1024, 3072]
    bias_shape = [1, 1, 1, 3072]
    bias_pad_shape = [1, 1, 32, 3072]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95
    BIAS = torch.randint(-20, 20, bias_shape, dtype=torch.float)

    a_t = (
        ttl.tensor.Tensor(
            A.flatten().tolist(),
            a_shape,
            in0_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, in0_mem_config)
    )
    b_t = (
        ttl.tensor.Tensor(
            B.flatten().tolist(),
            b_shape,
            in1_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, in1_mem_config)
    )
    if bias_mem_config is not None:
        bias_t = (
            ttl.tensor.Tensor(
                BIAS.flatten().tolist(),
                bias_shape,
                bias_dtype,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .pad(bias_pad_shape, [0, 0, 0, 0], 0)
            .to(ttl.tensor.Layout.TILE)
            .to(device, bias_mem_config)
        )
    else:
        bias_t = None

    t2 = ttl.tensor.bert_large_fused_qkv_matmul(a_t, b_t, bias_t, out_mem_config, out_dtype)

    # Check memory and dtype of inputs and outputs
    assert a_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert a_t.dtype() == in0_dtype
    assert b_t.memory_config().buffer_type == in1_mem_config.buffer_type
    assert b_t.dtype() == in1_dtype
    if bias_mem_config is not None:
        assert bias_t.memory_config().buffer_type == bias_mem_config.buffer_type
        assert bias_t.dtype() == bias_dtype
    assert t2.memory_config().buffer_type == out_mem_config.buffer_type
    assert t2.dtype() == out_dtype
    logger.debug(f"in0: {a_t.memory_config().buffer_type} and {a_t.dtype()}")
    logger.debug(f"in1: {b_t.memory_config().buffer_type} and {b_t.dtype()}")
    if bias_mem_config is not None:
        logger.debug(f"bias: {bias_t.memory_config().buffer_type} and {bias_t.dtype()}")
    logger.debug(f"out: {t2.memory_config().buffer_type} and {t2.dtype()}")

    assert t2.shape() == [9, 1, 384, 3072]
    tt_host_rm = t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm = torch.Tensor(tt_host_rm.data()).reshape(tt_host_rm.shape())

    ref_bmm = torch.matmul(A, B)
    if bias_mem_config is not None:
        ref_bmm = ref_bmm + BIAS
    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, 0.99)
    logger.info(f"Passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")
    ttl.device.CloseDevice(device)
    assert passing_pcc


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "bias_mem_config",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
        None,
    ),
    ids=["bias_DRAM", "bias_L1", "bias_None"],
)
@pytest.mark.parametrize(
    "in1_mem_config",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["in1_DRAM", "in1_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
def test_bert_large_fused_qkv_matmul_test(
    dtype, in0_mem_config, in1_mem_config, bias_mem_config, out_mem_config, request
):
    ttl.profiler.set_profiler_flag(False)
    ttl.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/BERT_large_fused_qkv_matmul_{request.node.callspec.id}"
    )
    run_bert_large_fused_qkv_matmul_test(
        dtype,
        dtype,
        dtype,
        dtype,
        in0_mem_config,
        in1_mem_config,
        bias_mem_config,
        out_mem_config,
    )


@pytest.mark.parametrize(
    "in0_mem_config, in1_mem_config, bias_mem_config, out_mem_config",
    (
        (
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
        ),
    ),
    ids=["BERT_mem_config"],
)
@pytest.mark.parametrize(
    "out_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["out_BFLOAT8_B", "out_BFLOAT16"],
)
@pytest.mark.parametrize(
    "bias_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["bias_BFLOAT8_B", "bias_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in1_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["in1_BFLOAT8_B", "in1_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in0_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["in0_BFLOAT8_B", "in0_BFLOAT16"],
)
def test_bert_large_fused_qkv_matmul_test_mixed_precision(
    in0_dtype,
    in1_dtype,
    bias_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    bias_mem_config,
    out_mem_config,
    request,
):
    ttl.profiler.set_profiler_flag(False)
    ttl.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/BERT_large_fused_qkv_matmul_mixed_precision_{request.node.callspec.id}"
    )
    run_bert_large_fused_qkv_matmul_test(
        in0_dtype,
        in1_dtype,
        bias_dtype,
        out_dtype,
        in0_mem_config,
        in1_mem_config,
        bias_mem_config,
        out_mem_config,
    )


def test_bert_large_fused_qkv_matmul_with_program_cache(use_program_cache):
    dtype = ttl.tensor.DataType.BFLOAT8_B
    dram_mem_config = ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM)
    for _ in range(2):
        run_bert_large_fused_qkv_matmul_test(
            dtype,
            dtype,
            dtype,
            dtype,
            dram_mem_config,
            dram_mem_config,
            dram_mem_config,
            dram_mem_config,
        )

    dram_mem_config = ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1)
    for _ in range(2):
        run_bert_large_fused_qkv_matmul_test(
            dtype,
            dtype,
            dtype,
            dtype,
            dram_mem_config,
            dram_mem_config,
            dram_mem_config,
            dram_mem_config,
        )

    assert ttl.program_cache.num_entries() == 2
