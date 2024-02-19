import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
def test_my_op(device, height, width):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(1, 1, height, width, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(1, 1, height, width, dtype=torch.bfloat16)
    torch_output_tensor = -1 * torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.ttl.tensor.my_op(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)

    # torch_input_tensor = torch.rand(1, 1, height, width)
    # torch_output_tensor = torch.exp(torch_input_tensor)

    # input_tensor = ttnn.from_torch(torch_input_tensor, device=device)
    # output_tensor = ttnn.ttl.tensor.<new_operation>(input_tensor)

    # output_tensor = ttnn.to_torch(output_tensor)

    # assert_with_pcc(torch_output_tensor, output_tensor)
