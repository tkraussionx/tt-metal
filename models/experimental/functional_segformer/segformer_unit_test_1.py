import ttnn
import pytest
import torch


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer(device):
    input_a = torch.randn(1, 16384, 32)
    hidden_states = ttnn.from_torch(input_a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    hidden_states = ttnn.linear(
        hidden_states,
        ttnn.from_torch(torch.randn(32, 32), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device),
        bias=ttnn.from_torch(torch.randn(1, 32), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device),
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
        ),
    )
    new_shape = (1, 16384, 1, 32)
    device = hidden_states.device()
    # hidden_states = ttnn.from_device(hidden_states)
    hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
    hidden_states = ttnn.reshape(hidden_states, new_shape)
    hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)
    # hidden_states = ttnn.to_device(hidden_states, device)

    output = ttnn.permute(hidden_states, (0, 2, 1, 3))
