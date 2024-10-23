# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch

from tests.didt.op_test_base import OpTestBase, get_blackhole_grid_size
import ttnn
from models.utility_functions import skip_for_blackhole, is_blackhole
import math
from tqdm import tqdm


class LLamaDramShardedTest(OpTestBase):
    def __init__(
        self,
        mesh_device,
        in0_shape,
        in1_shape,
        in0_mem_config,
        in1_mem_config,
        out_mem_config,
        in0_dtype,
        in1_dtype,
        out_dtype,
        in0_layout,
        in1_layout,
        program_config,
        compute_config,
        loop_count=1000,
        determinism_check_enabled=False,
        determinism_check_iterations=False,
    ):
        super().__init__(
            mesh_device,
            in0_shape,
            in1_shape,
            in0_mem_config,
            in1_mem_config,
            out_mem_config,
            in0_dtype,
            in1_dtype,
            out_dtype,
            in0_layout,
            in1_layout,
            program_config,
            compute_config,
            loop_count,
            determinism_check_enabled,
            determinism_check_iterations,
        )

    def generate_tt_weights_from_torch(self, torch_tensor):
        return ttnn.from_torch(
            # torch_tensor.permute(1, 0),
            torch_tensor,
            dtype=self.in1_dtype,
            layout=self.in1_layout,
            memory_config=self.in1_mem_config,
            device=self.mesh_device,
            mesh_mapper=self.from_torch_mesh_mapper,
        )


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
        pytest.param((8, 4), id="galaxy"),
    ],
    indirect=["mesh_device"],
)
def test_llama_dram_sharded_matmul(
    mesh_device, iterations, determinism_check_iterations, use_program_cache, simulate_bh_harvesting
):
    if is_blackhole() == True:
        pytest.skip("Blackhole arch is not supported for this test")

    hidden_dim = 4096
    vocab_size = 64128
    seq_len = 32
    in0_shape = [1, 1, seq_len, hidden_dim]
    in1_shape = [1, 1, hidden_dim, vocab_size]
    in0_mem_config = ttnn.create_sharded_memory_config(
        (seq_len, hidden_dim // 64),  # Shard shape: [32, 64] -> 1 shard per core
        ttnn.CoreGrid(y=8, x=8),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    in1_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(mesh_device.dram_grid_size().x - 1, mesh_device.dram_grid_size().y - 1),
                )
            }
        ),
        (hidden_dim, vocab_size // 12),
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    out_mem_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    # Initialize matmul configurations
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=1,
        per_core_M=1,
        per_core_N=32,  # vocab_size / tile_size / core_count
        fused_activation=None,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    llama_dram_sharded_matmul_test = LLamaDramShardedTest(
        mesh_device=mesh_device,
        in0_shape=in0_shape,
        in1_shape=in1_shape,
        in0_mem_config=in0_mem_config,
        in1_mem_config=in1_mem_config,
        out_mem_config=out_mem_config,
        in0_dtype=ttnn.DataType.BFLOAT16,
        in1_dtype=ttnn.DataType.BFLOAT8_B,
        out_dtype=ttnn.DataType.BFLOAT8_B,
        in0_layout=ttnn.TILE_LAYOUT,
        in1_layout=ttnn.TILE_LAYOUT,
        program_config=program_config,
        compute_config=compute_kernel_config,
        loop_count=iterations,
        determinism_check_enabled=True if determinism_check_iterations > 0 else False,
        determinism_check_iterations=determinism_check_iterations,
    )

    # Run test
    llama_dram_sharded_matmul_test.run_op_test()


@skip_for_blackhole("Multi-chip Blackhole has not been tested")
@pytest.mark.parametrize("logical_chip_id", range(32), ids=[f"logical_chip_{i}_" for i in range(32)])
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
        pytest.param((8, 4), id="galaxy"),
    ],
    indirect=["mesh_device"],
)
def test_specific_chip_llama_dram_sharded_matmul(
    mesh_device, logical_chip_id, iterations, determinism_check_iterations, use_program_cache
):
    assert len(mesh_device.get_device_ids()) > logical_chip_id, "Not enough devices!"

    test_llama_dram_sharded_matmul(
        mesh_device.get_device(logical_chip_id), iterations, determinism_check_iterations, use_program_cache, False
    )


@skip_for_blackhole("Multi-board Blackhole has not been tested")
@pytest.mark.parametrize(
    "board_mesh_device",
    range(4),
    ids=[f"board_id_{i}" for i in range(4)],
    indirect=["board_mesh_device"],
)
def test_specific_board_llama_dram_sharded_matmul(
    board_mesh_device, iterations, determinism_check_iterations, use_program_cache
):
    test_llama_dram_sharded_matmul(
        board_mesh_device, iterations, determinism_check_iterations, use_program_cache, False
    )
