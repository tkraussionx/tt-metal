import ttnn
import torch
import pytest
import tt_lib as ttl


@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("k", [2 * 32])
def test_to_and_from(device, m, k):
    activations = ttnn.random(shape=(1, 1, m, k))

    torch_activations = ttnn.to_torch(activations)
    tt_output = ttnn.from_torch(torch_activations, ttl.tensor.DataType.BFLOAT16)
    torch_output = ttnn.to_torch(tt_output)

    print("From torch")
    print(torch_output[0:5, 0:5])
    print("From tt")
    print(tt_output[0:5, 0:5])
    print("Does that match?")
    assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)
