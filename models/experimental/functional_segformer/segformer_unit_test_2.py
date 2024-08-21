import torch
import ttnn
import pytest


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_reshape(device):
    input_a = torch.randn(1, 256, 256)
    hidden_states = ttnn.from_torch(input_a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    # hidden_states = ttnn.from_device(hidden_states)
    hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
    hidden_states = ttnn.reshape(hidden_states, (1, 256, 8, 32))
