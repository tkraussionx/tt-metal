import ttnn
import torch
import pytest


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_mish(device):
    input_a = torch.randn((1, 1, 102400, 32), dtype=torch.bfloat16)
    input_a = ttnn.from_torch(input_a, device=device, layout=ttnn.TILE_LAYOUT)

    input_a = ttnn.to_memory_config(
        input_a,
        memory_config=ttnn.create_sharded_memory_config(
            input_a.shape,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        dtype=ttnn.bfloat16,
    )

    input_a = ttnn.mish(input_a)
