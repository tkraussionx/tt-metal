import ttnn
import torch
import pytest


@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("k", [2 * 32])
@pytest.mark.parametrize("n", [4 * 32])
def test_matmul(device, m, k, n):
    activations = ttnn.random(shape=(1, 1, m, k))
    weights = ttnn.random(shape=(1, 1, k, n))

    torch_activations = ttnn.to_torch(activations)
    torch_weights = ttnn.to_torch(weights)
    torch_output = torch.matmul(torch_activations, torch_weights)

    tt_output = ttnn.matmul(activations, weights)
    tt_output = ttnn.to_torch(tt_output)

    print("From torch")
    print(torch_output[0:5, 0:5])
    print("From tt")
    print(tt_output[0:5, 0:5])
    print("Does that match?")
    assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)
