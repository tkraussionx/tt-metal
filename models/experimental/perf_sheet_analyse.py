# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from torch import nn
import tt_lib


class Sample:
    def __init__(self):
        super().__init__()

        self.p1_torch_maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.p2_tt_lib_fallack = tt_lib.fallback_ops.MaxPool2d(
            kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False
        )

    def __call__(self, tensor_ttnn: ttnn.Tensor, torch_input_tensor):
        output = ttnn.add(tensor_ttnn, tensor_ttnn)

        output_maxpool_tt_lib_fallback = self.p2_tt_lib_fallack(tensor_ttnn)
        output_fallback_silu = tt_lib.fallback_ops.silu(tensor_ttnn)

        output_torch = torch.add(torch_input_tensor, torch_input_tensor)
        output_torch = self.p1_torch_maxpool(output_torch)

        return output, output_torch, output_maxpool_tt_lib_fallback, output_fallback_silu


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_perf(device):
    ttnn_model = Sample()

    torch_input_tensor = torch.randn(1, 512, 10, 10)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output = ttnn_model(ttnn_input_tensor, torch_input_tensor)
