import torch
import ttnn
import pytest


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        [361, 49, 96],
        [100, 49, 192],
        [25, 49, 384],
        [9, 49, 768],
    ],
)
def test_shard_MM_input_for_shifted_attention_config(device, input_shape):
    input_a = torch.randn(input_shape)
    if input_shape[-1] == 768:
        strategy_MM = ttnn.ShardStrategy.BLOCK
    else:
        strategy_MM = ttnn.ShardStrategy.HEIGHT
    ttnn_input_a = ttnn.from_torch(input_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print("ttn", ttnn_input_a.shape)
    ttnn_input_a = ttnn.to_memory_config(
        ttnn_input_a,
        memory_config=ttnn.create_sharded_memory_config(
            ttnn_input_a.shape,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            strategy=strategy_MM,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )
