import pytest

import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_softmax(device, h, w):
    torch_activations = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output = F.softmax(torch_activations, dim=1)

    activations = ttnn.from_torch(torch_activations)
    tt_output = ttnn.softmax(activations, dim=1)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)
    # assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)
