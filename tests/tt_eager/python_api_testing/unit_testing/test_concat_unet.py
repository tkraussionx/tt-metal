# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys

import torch

import tt_lib as ttl
from models.utility_functions import print_diff_argmax
import pytest
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
    comp_equal,
)
from models.utility_functions import skip_for_wormhole_b0


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "shapes, dim",
    (
        (((8, 132, 20, 32), (8, 132, 20, 64)), 3),
        (((8, 264, 40, 32), (8, 264, 40, 32)), 3),
        (((8, 528, 80, 16), (8, 528, 80, 32)), 3),
        (((8, 1056, 160, 16), (8, 1056, 160, 16)), 3),
    ),
)
def test_multi_input_concat(shapes, dim, device, function_level_defaults):
    inputs = []
    tt_inputs = []
    for i in range(len(shapes)):
        shape = torch.Size(shapes[i])
        inputs.append(i + torch.arange(0, shape.numel()).reshape(shape).to(torch.bfloat16))
        tt_inputs.append(
            ttl.tensor.Tensor(
                inputs[i],
                ttl.tensor.DataType.BFLOAT16,
            ).to(device)
        )

    tt_cpu = torch.concat(inputs, dim)

    tt = ttl.tensor.concat(tt_inputs, dim)

    tt_dev = tt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)

    passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing
