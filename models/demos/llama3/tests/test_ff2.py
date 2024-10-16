from typing import Tuple
import torch
import json
from time import time
from datetime import datetime
from loguru import logger
import os
import ttnn
import math
import pytest
from models.demos.llama3.tt.llama_common import (
    get_single_rot_mat,
    get_prefill_rot_mat,
    prepare_inputs_ttnn_prefill,
    get_rot_transformation_mat,
    HostEmbedding,
    encode_prompt_llama_instruct,
)
from models.demos.llama3.tt.llama_model import TtTransformer
from models.demos.llama3.tt.llama_embedding import TtLlamaEmbedding
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.demos.utils.llm_demo_utils import create_benchmark_data, verify_perf


def create_dram_sharded_mem_config(k, n, device):
    dram_weight_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
            )
        }
    )
    """Create DRAM-sharded memory config for width-sharded tensors"""
    dram_cores = 12
    tile_size = 32
    padded_size = math.ceil(n / (tile_size * dram_cores)) * (tile_size * dram_cores)
    shard_spec = ttnn.ShardSpec(
        dram_weight_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR, False
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def dram_matmul_config(
    m: int, k: int, n: int, grid_size: Tuple[int, int]
) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
    # in0_block_w must evenly divide k and be no larger than tile_size * num_cores
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=1,  # math.ceil(k / (32 * grid_size[0] * grid_size[1])),
        per_core_M=math.ceil(m / 32),
        per_core_N=math.ceil(n / (32 * grid_size[0] * grid_size[1])),
        fused_activation=None,
    )


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_llama_demo(mesh_device, use_program_cache):
    memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 1),
            )
        }
    )
    shard_spec = ttnn.ShardSpec(core_range_set, (32, 256), ttnn.ShardOrientation.ROW_MAJOR, False)
    inputs_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )

    inputs = torch.ones(1, 1, 32, 3584)
    tt_inputs = ttnn.as_tensor(
        inputs,
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=inputs_memory_config,
    )
    w2_memory_config = create_dram_sharded_mem_config(3584, 8192, mesh_device)
    weights = torch.randn(1, 1, 8 * 3584, 8192)
    tt_weights = ttnn.as_tensor(
        weights,
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        memory_config=w2_memory_config,
    )

    compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,  # full precision for bfp8 @ bfp8
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    dtype = ttnn.bfloat16
    program_config = dram_matmul_config(m=32, k=3584, n=8192, grid_size=(4, 8))

    x = ttnn.linear(
        tt_inputs,
        tt_weights,
        compute_kernel_config=compute_kernel_config_hifi2,
        program_config=program_config,
        memory_config=memory_config,
        dtype=dtype,
    )

    print(f"{x.shape=}")
    print(f"{x=}")
