import ttnn
import torch
import pytest
import tt_lib as ttl


@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("k", [2 * 32])
def test_to_and_from(device, m, k):
    # start with tt tensor
    activations = ttnn.random(shape=(1, 1, m, k))
    torch_activations = ttnn.to_torch(activations)
    tt_output = ttnn.from_torch(torch_activations, ttl.tensor.DataType.BFLOAT16)
    torch_output = ttnn.to_torch(tt_output)
    assert torch.allclose(torch_output, torch_activations, atol=1e-1, rtol=1e-2)
    # start with pytorch tensor
    torch_tensor2 = torch.rand(1, 2, 3, dtype=torch.bfloat16)
    tt_tensor2 = ttnn.from_torch(torch_tensor2, ttl.tensor.DataType.BFLOAT16)
    tt_tensor2_to_torch_again = ttnn.to_torch(tt_tensor2)
    assert torch.allclose(torch_tensor2, tt_tensor2_to_torch_again, atol=1e-1, rtol=1e-2)
