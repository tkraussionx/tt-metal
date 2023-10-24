import pytest

import torch

import ttnn


# TODO(arakhmati): delete this test?
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_transpose(device, h, w):
    torch_activations = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output = torch_activations.transpose(2, 3)

    activations = ttnn.from_torch(torch_activations)
    tt_output = ttnn.permute(activations, (0, 1, 3, 2))
    tt_output = ttnn.to_torch(tt_output).clone()

    assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)
