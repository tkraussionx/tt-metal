# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.utility_functions import torch_random


@pytest.mark.parametrize(
    "m_size, k_size, n_size, n2_size",
    [
        # down-sampling
        (4096, 64, 96, 64),
        (4096, 64, 4096, 64),
        (1024, 96, 96, 96),
        (1024, 96, 1024, 96),
        (256, 160, 96, 160),
        (256, 160, 256, 160),
        (96, 160, 96, 160),
        (64, 160, 64, 160),
    ],
)
def test_sd_matmul_softmax(device, m_size, k_size, n_size, n2_size):
    num_heads = {
        64: 40,
        96: 80,
        160: 160,
    }

    # set configs
    torch.manual_seed(0)
    core_grid = ttnn.CoreGrid(x=8, y=8)
    batch_size = 2
    in_channels = 1
    dtype = ttnn.bfloat8_b
    attention_mask_shape = (2, 1, 1, n_size)
    head_size = num_heads[k_size]

    # instantiate input torch tensors
    torch_input_tensor_a = torch.randn((batch_size, in_channels, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((batch_size, in_channels, k_size, n_size), dtype=torch.bfloat16)
    torch_input_tensor_c = torch.randn((batch_size, in_channels, n_size, n2_size), dtype=torch.bfloat16)
    torch_attention_mask = torch.zeros(attention_mask_shape)

    # calculate golden result
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b
    torch_output_tensor = ttnn.transformer._torch_attention_softmax(
        torch_output_tensor, head_size=head_size, attention_mask=torch_attention_mask
    )
    torch_output_tensor = torch_output_tensor @ torch_input_tensor_c

    # convert input to ttnn tensors
    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_c = ttnn.from_torch(torch_input_tensor_c, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    attention_mask = ttnn.from_torch(torch_attention_mask, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    # apply the same operations in ttnn
    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        # use_1d_systolic_array=True,
        core_grid=core_grid,
    )
    output_tensor = ttnn.transformer.attention_softmax_(
        output_tensor, head_size=head_size, attention_mask=attention_mask
    )
    output_tensor = ttnn.matmul(
        output_tensor,
        input_tensor_c,
        # use_1d_systolic_array=True,
        core_grid=core_grid,
    )

    # compare the results
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.98)
