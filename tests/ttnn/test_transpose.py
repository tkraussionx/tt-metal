import ttnn
import torch
import pytest
import tt_lib as ttl


@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("k", [2 * 32])
def test_transpose(device, m, k):
    activations = ttnn.random(shape=(1, 1, m, k))
    torch_activations = ttnn.to_torch(activations)
    torch_output = torch_activations.transpose(2, 3)
    tt_output = ttnn.permute(activations, (0, 1, 3, 2))
    tt_output = ttnn.to_torch(tt_output)
    assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)
