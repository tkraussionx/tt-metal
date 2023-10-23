import ttnn
import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc, update_process_id
import tt_lib as ttl


# fmt: off
@pytest.mark.parametrize("first,scalar", [
        ([1.0,2.0,3.0],3.0)
    ])
# fmt: on
def test_multiply_with_scalar(device, first, scalar):
    # pad the lists with zeros to make it 32 so that it fits nicely on the device.
    first += [0.0] * (32 - len(first))
    first_tensor = ttnn.Tensor(ttl.tensor.Tensor(
        first, [1, 1, 1, len(first)], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, device
    ))
    torch_first_tensor = ttnn.to_torch(first_tensor)
    torch_output = torch.mul(torch_first_tensor, scalar)
    tt_output = ttnn.mul(first_tensor, scalar)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.99)
