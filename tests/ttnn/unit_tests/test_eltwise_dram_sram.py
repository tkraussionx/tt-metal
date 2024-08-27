# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_pcc
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


@pytest.mark.parametrize(
    "test_func_name, torch_func_name",
    [(ttnn.add, torch.add), (ttnn.mul, torch.mul)],
)
def test_run_elt_binary_dram_interleaved(test_func_name, torch_func_name, device):
    shape = [2, 56, 256, 256]

    torch.manual_seed(10)

    # TODO: fill in MemoryConfig(...) to make a configuration for interleaved
    # DRAM tensor
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    in0 = torch.randn(shape).bfloat16()
    in1 = torch.randn(shape).bfloat16()
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=mem_config)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=mem_config)

    out_t = test_func_name(in0_t, in1_t)
    out = tt2torch_tensor(out_t)

    passing, output = comp_pcc(out, torch_func_name(in0, in1), 0.9999)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "test_func_name, torch_func_name",
    [(ttnn.add, torch.add), (ttnn.mul, torch.mul)],
)
def test_run_elt_binary_sram_sharded(test_func_name, torch_func_name, device):
    shape = [2, 56, 256, 256]

    torch.manual_seed(10)

    # TODO: fill in create_sharded_memory_config(...) to make a configuration
    # for sharded SRAM tensor (Use 56 Tensix cores, BLOCK_SHARDED, ROW_MAJOR)
    mem_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=8, y=7),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    in0 = torch.randn(shape).bfloat16()
    in1 = torch.randn(shape).bfloat16()
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=mem_config)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=mem_config)

    out_t = test_func_name(in0_t, in1_t)
    out = tt2torch_tensor(out_t)

    passing, output = comp_pcc(out, torch_func_name(in0, in1), 0.9999)
    logger.info(output)
    assert passing
