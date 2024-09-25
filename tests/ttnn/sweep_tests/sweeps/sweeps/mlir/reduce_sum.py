# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn
import os

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random

from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import reduce_sum_h as tt_reduce_sum_h

parameters = {
    "dim": [0, 1, 2, 3],
    "batch_sizes": [(2,), (2, 2)],
    "height": [64],
    "width": [64],
    "input_dtype": [ttnn.bfloat16],
    "input_layout": [ttnn.TILE_LAYOUT],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
}


def skip(*, dim, batch_sizes, **_) -> Tuple[bool, Optional[str]]:
    print(":", dim, batch_sizes, len(batch_sizes))

    # if dim == 0:
    # return True, "batch sum (dim=0) is not supported"
    if dim >= len(batch_sizes) + 2:  # seg fault without this check
        return True, "dim is greater than the number of dimensions in the input tensor"
    return False, None


def run(
    dim, batch_sizes, height, width, input_dtype, input_layout, input_memory_config, output_memory_config, *, device
) -> Tuple[bool, Optional[str]]:
    input_shape = (*batch_sizes, height, width)

    torch_input_tensor = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=input_layout, device=device, memory_config=input_memory_config
    )

    # inspired by tests/tt_eager/python_api_testing/sweep_tests/tt_lib_ops.py
    output_tensor = ttnn.sum(input_tensor, dim, memory_config=input_memory_config)

    if "TT_METAL_MOCKUP_EN" in os.environ:
        return True, None
    else:
        torch_output_tensor = torch.sum(torch_input_tensor, dim=dim)
        output_tensor = ttnn.to_torch(output_tensor)
        squeezed_tensor = torch.squeeze(output_tensor, dim=dim)
        return check_with_pcc(torch_output_tensor, squeezed_tensor, 0.999)
