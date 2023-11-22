# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import tt_lib
from models.utility_functions import print_diff_argmax, comp_pcc
from models.utility_functions import skip_for_wormhole_b0


@skip_for_wormhole_b0()
def test_addalpha(device):
    torch.manual_seed(0)

    input_shapes = [(3, 64, 128, 96)]

    for input_shape in input_shapes:
        input_tensor = torch.randn(input_shape).bfloat16()

        input_1 = (
            tt_lib.tensor.Tensor(input_tensor, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
        )

        other = (
            tt_lib.tensor.Tensor(input_tensor, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
        )

        tt_output_tensor_on_device = tt_lib.tensor.bw_addalpha(input_1, other, alpha=1)
        tt_output_tensor = tt_output_tensor_on_device.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

        golden_output_tensor = torch.add(input_tensor, input_tensor, alpha=1)
        print_diff_argmax(tt_output_tensor, golden_output_tensor)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        assert allclose, f"FAILED: {output}"
