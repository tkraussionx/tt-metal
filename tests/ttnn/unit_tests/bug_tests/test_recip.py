# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_pt_tt, compare_results
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops


def data_gen_pt_tt(input_shapes, device, required_grad=False, val=1):
    pt_tensor = (torch.ones(input_shapes, requires_grad=required_grad) * val).bfloat16()
    tt_tensor = ttnn.Tensor(pt_tensor, ttnn.DataType.BFLOAT16).to(ttnn.Layout.TILE).to(device)
    return pt_tensor, tt_tensor


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test1(input_shapes, device):
    print("==============================")
    print("recip")
    val = 0
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True, val=val)

    print("input_tensor", input_tensor)

    golden_tensor = pytorch_ops.recip(in_data)
    tt_output_tensor_on_device = ttnn.reciprocal(input_tensor)

    print("tt_output_tensor_on_device", tt_output_tensor_on_device)
    print("golden_tensor", golden_tensor)

    print("==============================")
    print("div_unary")
    val = 1
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True, val=val)
    print("input_tensor", input_tensor)

    # golden_tensor = pytorch_ops.div_unary(in_data , scalar=0)
    golden_function = ttnn.get_golden_function(ttnn.div)
    golden_tensor = golden_function(in_data, 0, round_mode=None)
    tt_output_tensor_on_device = ttnn.div(input_tensor, 0)
    print("tt_output_tensor_on_device", tt_output_tensor_on_device)
    print("golden_tensor", golden_tensor)

    status = compare_results([tt_output_tensor_on_device], [golden_tensor], pcc=0.99)
    assert status
