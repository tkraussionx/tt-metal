import ttnn
import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("k", [2 * 32])
def test_permute(device, m, k):
    activations = ttnn.random(shape=(1, 1, m, k))
    torch_activations = ttnn.to_torch(activations)
    torch_output = torch_activations.permute(3, 2, 1, 0)
    tt_output = ttnn.permute(activations, (3, 2, 1, 0))
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)
    # assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)
