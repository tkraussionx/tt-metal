import ttnn
import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
import tt_lib as ttl


# fmt: off
@pytest.mark.parametrize("d1,d2,d3,d4", [
    (1, 1, 1, 3),
    (1, 1, 3, 1),
    (3, 3, 1, 3),
    (3, 3, 3, 1),
    (1, 3, 1, 3),
    (3, 1, 3, 1),
    ])
# fmt: on
def test_matmul_with_matched_width_height(device, d1, d2, d3, d4):
    first_tensor = ttnn.random(shape=(d1, d2, d3, d4))
    second_tensor = ttnn.random(shape=(d1, d2, d4, d3))
    torch_first_tensor = ttnn.to_torch(first_tensor)
    torch_second_tensor = ttnn.to_torch(second_tensor)
    torch_output = torch.matmul(torch_first_tensor, torch_second_tensor)
    tt_output = ttnn.matmul(first_tensor, second_tensor)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.99)


# fmt: off
@pytest.mark.parametrize("d1,d2,d3,d4", [
    (1, 1, 1, 1),
    (1, 1, 3, 3),
    (3, 3, 3, 3),
    (3, 1, 3, 3),
    (1, 3, 3, 3)
    ])
# fmt: on
def test_matmul_same_shape_and_valid(device, d1, d2, d3, d4):
    first_tensor = ttnn.random(shape=(d1, d2, d3, d4))
    second_tensor = ttnn.random(shape=(d1, d2, d3, d4))
    torch_first_tensor = ttnn.to_torch(first_tensor)
    torch_second_tensor = ttnn.to_torch(second_tensor)
    torch_output = torch.matmul(torch_first_tensor, torch_second_tensor)
    tt_output = ttnn.matmul(first_tensor, second_tensor)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.99)


# fmt: off
@pytest.mark.parametrize("first,second", [
        ([1.0,2.0,3.0],[3.0,4.0,5.0])
    ])
# fmt: on
def test_matmul_same_shape_but_invalid(device, first, second):
    # pad the lists with zeros to make it 32 so that it fits nicely on the device.
    first += [0.0] * (32 - len(first))
    second += [0.0] * (32 - len(second))
    first_tensor = ttnn.Tensor(ttl.tensor.Tensor(
        first, [1, 1, 1, len(first)], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, device
    ))
    second_tensor = ttnn.Tensor(ttl.tensor.Tensor(
        second, [1, 1, 1, len(second)], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, device
    ))
    torch_first_tensor = ttnn.to_torch(first_tensor)
    torch_second_tensor = ttnn.to_torch(second_tensor)
    with pytest.raises(RuntimeError) as ex:
        ttnn.matmul(first_tensor, second_tensor)
    assert "The width of the first tensor must be equal to the height of the second tensor" in str(ex.value)
    with pytest.raises(RuntimeError) as ex2:
        torch.matmul(torch_first_tensor, torch_second_tensor)
    assert "Expected size for first two dimensions of batch2 tensor to be: [1, 32] but got: [1, 1]." in str(ex2.value)
