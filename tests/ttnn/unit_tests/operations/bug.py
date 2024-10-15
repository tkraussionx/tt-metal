# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import pytest
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger
from models.utility_functions import is_wormhole_b0

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_npu,
    to_cpu,
)


@pytest.mark.parametrize(
    "shape_dim",
    [
        [[3, 4], 1],  # single tile
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_softmax_for_dim_hw(shape_dim, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    x = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16)

    dev_x = to_npu(x, device)

    tt_cpu = torch.softmax(x, dim)
    tt_npu = ttnn.operations.moreh.softmax(dev_x, dim, compute_kernel_config=compute_kernel_config)

    tt_dev = to_cpu(tt_npu, shape)

    print("x", x)
    print("tt_dev", tt_dev)
