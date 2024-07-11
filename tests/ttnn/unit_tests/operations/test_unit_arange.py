# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def test_unit_arange(device, reset_seeds):
    torch_output = torch.arange(0, 5, 1)
    ttnn_output = ttnn.arange(0, 5, 1, device)

    ttnn_output = ttnn.to_torch(ttnn_output).to(torch.int32).squeeze(0).squeeze(0).squeeze(0)
    print("torch output", torch_output, torch_output.shape)  # [0, 1, 2, 3, 4] torch.Size([5])
    print("ttnn output", ttnn_output, ttnn_output.shape)  # [0, 1, 2, 3, 4, 0] torch.Size([6])
