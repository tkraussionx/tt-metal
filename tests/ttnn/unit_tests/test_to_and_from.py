import pytest

import torch

import ttnn


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_to_and_from(device, h, w):
    torch_activations = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    tt_output = ttnn.from_torch(torch_activations)
    torch_output = ttnn.to_torch(tt_output)
    assert torch.allclose(torch_output, torch_activations, atol=1e-1, rtol=1e-2)
