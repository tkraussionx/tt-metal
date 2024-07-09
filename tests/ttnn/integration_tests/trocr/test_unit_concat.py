# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_unit_permute(device, reset_seeds):
    tensor_a = torch.rand(1, 1, 768)
    tensor_b = torch.rand(1, 576, 768)
    torch_output = torch.cat((tensor_a, tensor_b), dim=1)

    tensor_a_rm = ttnn.from_torch(tensor_a, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tensor_b_rm = ttnn.from_torch(tensor_b, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output = ttnn.concat((tensor_a_rm, tensor_b_rm), dim=1)
    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)  # PCC = 0.9999956481098589

    tensor_a_tile = ttnn.from_torch(tensor_a, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tensor_b_tile = ttnn.from_torch(tensor_b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_output = ttnn.concat((tensor_a_tile, tensor_b_tile), dim=1)
    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 1)  # PCC = 0.00072778873165853
