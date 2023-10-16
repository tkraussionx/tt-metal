import ttnn
import torch
import pytest


def torch_model(activations, weights):
    output = torch.matmul(activations, weights)
    return output


def tt_model(activations, weights):
    output = ttnn.matmul(activations, weights)
    return output


@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("k", [2 * 32])
@pytest.mark.parametrize("n", [4 * 32])
def test_ttnn(device, m, k, n):
    activations = ttnn.random(shape=(1, 1, m, k))
    weights = ttnn.random(shape=(1, 1, k, n))

    torch_activations = ttnn.to_torch(activations)
    torch_weights = ttnn.to_torch(weights)
    torch_output = torch_model(torch_activations, torch_weights)

    #    torch_output = torch_model(ttnn.to_torch(activations), ttnn.to_torch(weights))

    tt_output = tt_model(activations, weights)
    tt_output = ttnn.to_torch(tt_output)

    print("From torch")
    print(torch_output[0:5, 0:5])
    print("From tt")
    print(tt_output[0:5, 0:5])
    print("Does that match?")
    assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)


# @pytest.mark.parametrize("m", [7, 11])
# @pytest.mark.parametrize("k", [23, 71])
# @pytest.mark.parametrize("n", [11, 54])
# def test_ttnn(device, m, k, n):
#     activations = ttnn.random(shape=(1, 1, m, k))
#     weights = ttnn.random(shape=(1, 1, k, n))

#     torch_output = torch_model(ttnn.to_torch(activations), ttnn.to_torch(weights))

#     tt_output = tt_model(activations, weights)
#     tt_output = ttnn.to_torch(tt_output)

#     assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)
