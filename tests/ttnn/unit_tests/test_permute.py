import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_permute(device, h, w):
    torch_activations = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output = torch.permute(torch_activations, (3, 2, 1, 0))

    activations = ttnn.from_torch(torch_activations)
    activations = ttnn.copy_to_device(activations, device)
    tt_output = ttnn.permute(activations, (3, 2, 1, 0))
    tt_output = ttnn.copy_from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output).clone() # TODO: remove clone?

    assert_with_pcc(torch_output, tt_output, 0.9999)
    # assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_transpose(device, h, w):
    torch_activations = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output = torch_activations.transpose(2, 3)

    activations = ttnn.from_torch(torch_activations)
    activations = ttnn.copy_to_device(activations, device)
    tt_output = ttnn.permute(activations, (0, 1, 3, 2))
    tt_output = ttnn.copy_from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output).clone() # TODO: remove clone?

    assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)
