import ttnn
import torch
import pytest
import math
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "act_shape",
    (
        ## mnist shapes
        [1, 1, 28, 28],
    ),
)
def test(reset_seeds, device, act_shape):
    x = torch.randn(act_shape, dtype=torch.bfloat16)
    tt = ttnn.from_torch(x, dtype=ttnn.bfloat16)

    tt_x1 = ttnn.reshape(tt, (tt.shape[0], 1, 1, 784))

    tt = ttnn.to_device(tt, device=device)
    tt_x2 = ttnn.reshape(tt, (tt.shape[0], 1, 1, 784))

    tt_output = ttnn.to_torch(tt_x1)
    tt_output_device = ttnn.to_torch(tt_x2)

    assert_with_pcc(tt_output, tt_output_device, 1)
