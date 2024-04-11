# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_with_range, compare_pcc
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_abs(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    golden_tensor = torch.floor(in_data)

    tt_output_tensor_on_device = tt_lib.tensor.recip(input_tensor)
    tt_out_tensor = tt_output_tensor_on_device.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    comp_pass, comp_out = comparison_funcs.comp_pcc(golden_tensor, tt_out_tensor)
    comp_all, _ = comparison_funcs.comp_allclose(golden_tensor, tt_out_tensor, atol=4, rtol=1e-1)
    logger.debug(comp_pass)
    logger.debug(comp_all)
    logger.debug(comp_out)
    print(input_tensor)
    print(tt_out_tensor)
    print(golden_tensor)
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
