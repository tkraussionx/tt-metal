import ttnn
import torch
import pytest


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_linear(device):
    core_grid = ttnn.CoreGrid(y=8, x=4)
    program_configs = {
        "linear_configs": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=1,
            out_subblock_h=4,
            out_subblock_w=1,
            per_core_M=16,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
    }
    input_a = torch.randn(1, 16384, 32, dtype=torch.bfloat16)
    weight = torch.randn(32, 32, dtype=torch.bfloat16)
    bias = torch.randn(1, 32, dtype=torch.bfloat16)

    tt_input_a = ttnn.from_torch(input_a, layout=ttnn.TILE_LAYOUT, device=device)
    tt_weight = ttnn.from_torch(weight, layout=ttnn.TILE_LAYOUT, device=device)
    tt_bias = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, device=device)

    tt_input_a = ttnn.to_memory_config(
        tt_input_a,
        memory_config=ttnn.create_sharded_memory_config(
            [1, 16384, 32],
            core_grid=ttnn.CoreGrid(y=8, x=4),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )

    output = ttnn.linear(
        tt_input_a,
        tt_weight,
        bias=tt_bias,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=program_configs["linear_configs"],
    )
