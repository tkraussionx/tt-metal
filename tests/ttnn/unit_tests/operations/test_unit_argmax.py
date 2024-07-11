# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def test_unit_argmax(device, reset_seeds):
    tensor_a = torch.rand(1, 50265)
    tensor_a = ttnn.from_torch(tensor_a, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_output = ttnn.argmax(tensor_a, dim=-1)
