import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
import torch.nn.functional as F


@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("k", [2 * 32])
def test_softmax(device, m, k):
    activations = ttnn.random(shape=(1, 1, m, k))
    torch_activations = ttnn.to_torch(activations)
    torch_output = F.softmax(torch_activations, dim=1)
    tt_output = ttnn.softmax(activations, dim=1)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)
    # assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)
