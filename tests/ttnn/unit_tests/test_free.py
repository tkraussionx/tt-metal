import pytest

import torch

import ttnn


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_free(device, h, w):
    torch_activations = torch.rand((1, 1, h, w), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_activations)

    with pytest.raises(RuntimeError) as exception:
        ttnn.free(input_tensor)
    assert "Cannot deallocate tensor with borrowed storage!" in str(exception.value)

    output_tensor = input_tensor + input_tensor
    output_tensor_reference = ttnn.reshape(output_tensor, (1, 1, h, w))

    ttnn.free(output_tensor)
    with pytest.raises(RuntimeError) as exception:
        output_tensor_reference[:,:,:1]
    assert "Buffer must be allocated on device!" in str(exception.value)
