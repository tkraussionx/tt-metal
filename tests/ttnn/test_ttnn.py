import ttnn
import torch


def torch_model(activations, weights, bias):
    output = activations @ weights
    output = output + bias
    output = torch.permute(output, (0, 2, 1, 3))
    output = torch.reshape(output, (1, 1, 7, 11))
    output = torch.softmax(output, dim=-1)
    return output


def tt_model(activations, weights, bias):
    output = activations @ weights
    output = output + bias
    output = ttnn.permute(output, (0, 2, 1, 3))
    output = ttnn.reshape(output, (1, 1, 7, 11))
    output = ttnn.softmax(output, dim=-1)
    return output


def test_ttnn(device):
    activations = ttnn.random(shape=(1, 1, 7, 23))
    weights = ttnn.random(shape=(1, 1, 23, 11))
    bias = ttnn.random(shape=(1, 1, 7, 11))

    torch_output = torch_model(ttnn.to_torch(activations), ttnn.to_torch(weights), ttnn.to_torch(bias))

    tt_output = tt_model(activations, weights, bias)
    tt_output = ttnn.to_torch(tt_output)

    assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)
