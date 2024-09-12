# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import math
from models.utility_functions import skip_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn


@pytest.mark.parametrize(
    "input_shape",
    ([1, 512, 7, 7],),
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
def test_run_average_pool2d(
    input_shape,
    dtype,
    device,
):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(input_shape)
    AvgPool2d = torch.nn.AdaptiveAvgPool2d((7, 7))
    torch_output_tensor = AvgPool2d(torch_input_tensor)  # torch_output_tensor.shape = [1, 512, 7, 7]

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))  #
    input_tensor = ttnn.from_torch(input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.global_avg_pool2d(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))  # output_tensor.shape = [1, 512, 1, 1]

    assert_with_pcc(torch_output_tensor, output_tensor)
