import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test(device, h, w):
    torch.manual_seed(0)

    torch_activations = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output = torch.exp(torch_activations)

    activations = ttnn.from_torch(torch_activations)
    tt_output = ttnn.experimental.exp(activations)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9998)
    # assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)
