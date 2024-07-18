import ttnn
import torch
import pytest


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test(device):
    a = torch.randn((1, 1, 100, 256), dtype=torch.float16)
    ttnn_input_tensor = ttnn.from_torch(
        a,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    output = ttnn.mish(ttnn_input_tensor)
