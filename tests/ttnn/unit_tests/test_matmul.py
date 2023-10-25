import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


# fmt: off
@pytest.mark.parametrize("h,w", [
    (1, 3),
    (3, 1),
    (1, 3),
    (3, 1),
    (1, 3),
    (3, 1),
    ])
# fmt: on
def test_matmul_with_matched_width_height(device, h, w):
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((w, h), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    tt_output = ttnn.matmul(input_tensor_a, input_tensor_b)
    tt_output = ttnn.to_torch(tt_output)

    assert len(tt_output.shape) == len(torch_output.shape)
    assert tt_output.shape == torch_output.shape
    assert_with_pcc(torch_output, tt_output, 0.99)


# fmt: off
@pytest.mark.parametrize("h,w", [
    (1, 3),
    (3, 1),
    (1, 3),
    (3, 1),
    (1, 3),
    (3, 1),
    (3, 3),
    ])
# fmt: on
def test_matmul_with_matched_width_height_from_1D(device, h, w):
    torch_input_tensor_a = torch.rand((w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((w, h), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    tt_output = ttnn.matmul(input_tensor_a, input_tensor_b)
    tt_output = ttnn.to_torch(tt_output)

    assert len(tt_output.shape) == len(torch_output.shape)
    assert tt_output.shape == torch_output.shape
    assert_with_pcc(torch_output, tt_output, 0.99)


@pytest.mark.parametrize("w", [(3), (1)])
def test_matmul_does_dot_product(device, w):
    torch_input_tensor_a = torch.rand((w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((w), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    tt_output = ttnn.matmul(input_tensor_a, input_tensor_b)
    tt_output = ttnn.to_torch(tt_output)

    assert torch_output.shape == []
    assert tt_output.shape == []
    assert torch_output[0] == tt_output[0]


# fmt: off
@pytest.mark.parametrize("n,c,h,w", [
    (1, 1, 1, 3),
    (1, 1, 3, 1),
    (3, 3, 1, 3),
    (3, 3, 3, 1),
    (1, 3, 1, 3),
    (3, 1, 3, 1),
    ])
# fmt: on
def test_matmul_with_matched_width_height_4D(device, n, c, h, w):
    torch_input_tensor_a = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n, c, w, h), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    tt_output = ttnn.matmul(input_tensor_a, input_tensor_b)
    tt_output = ttnn.to_torch(tt_output)

    assert len(tt_output.shape) == len(torch_output.shape)
    assert tt_output.shape == torch_output.shape
    assert_with_pcc(torch_output, tt_output, 0.99)


# fmt: off
@pytest.mark.parametrize("n,c,h,w", [
    (1, 1, 1, 1),
    (1, 1, 3, 3),
    (3, 3, 3, 3),
    (3, 1, 3, 3),
    (1, 3, 3, 3)
    ])
# fmt: on
def test_matmul_same_shape_and_valid(device, n, c, h, w):
    torch_input_tensor_a = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    tt_output = ttnn.matmul(input_tensor_a, input_tensor_b)
    tt_output = ttnn.to_torch(tt_output)

    assert len(tt_output.shape) == len(torch_output.shape)
    assert tt_output.shape == torch_output.shape
    assert_with_pcc(torch_output, tt_output, 0.99)


# fmt: off
@pytest.mark.parametrize("input_a,input_b", [
        ([1.0,2.0,3.0],[3.0,4.0,5.0])
    ])
# fmt: on
def test_matmul_same_shape_but_invalid(device, input_a, input_b):
    # pad the lists with zeros to make it 32 so that it fits nicely on the device.
    input_a += [0.0] * (32 - len(input_a))
    input_b += [0.0] * (32 - len(input_b))

    torch_input_tensor_a = torch.as_tensor(input_a, dtype=torch.bfloat16).reshape((1, 1, 1, len(input_a)))
    torch_input_tensor_b = torch.as_tensor(input_b, dtype=torch.bfloat16).reshape((1, 1, 1, len(input_b)))

    with pytest.raises(RuntimeError) as exception:
        torch.matmul(torch_input_tensor_a, torch_input_tensor_b)
    assert "Expected size for first two dimensions of batch2 tensor to be: [1, 32] but got: [1, 1]." in str(
        exception.value
    )

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)

    with pytest.raises(RuntimeError) as exception:
        ttnn.matmul(input_tensor_a, input_tensor_b)
    assert "The width of the first tensor must be equal to the height of the second tensor" in str(exception.value)
