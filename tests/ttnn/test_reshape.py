import ttnn
import torch
import pytest


@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("k", [2 * 32])
def test_reshape(device, m, k):
    activations = ttnn.random(shape=(1, 1, m, k))

    torch_activations = ttnn.to_torch(activations)
    torch_output = torch_activations.reshape(1, 1, k, m)
    tt_output = activations.reshape(1, 1, k, m)
    tt_output = ttnn.to_torch(tt_output)

    print("From torch")
    print(torch_output[0:5, 0:5])
    print("From tt")
    print(tt_output[0:5, 0:5])
    print("Does that match?")
    assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("k", [2 * 32])
def test_reshape_negative_1(device, m, k):
    activations = ttnn.random(shape=(1, 1, m, k))

    torch_activations = ttnn.to_torch(activations)
    torch_output = torch_activations.reshape(-1)
    tt_output = activations.reshape(-1)
    tt_output = ttnn.to_torch(tt_output)

    print("From torch")
    print(torch_output[0:5, 0:5])
    print("From tt")
    print(tt_output[0:5, 0:5])
    print("Does that match?")
    assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)
