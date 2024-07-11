# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_unit_add(device, reset_seeds):
    tensor_a = torch.rand(1, 16, 3, 3)
    tensor_b = torch.rand(1, 1, 3, 3)

    torch_output = tensor_a + tensor_b

    tensor_a = ttnn.from_torch(tensor_a, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tensor_b = ttnn.from_torch(tensor_b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    ttnn_output = ttnn.add(tensor_a, tensor_b)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 1)  # PCC = 0.4731119553648677
