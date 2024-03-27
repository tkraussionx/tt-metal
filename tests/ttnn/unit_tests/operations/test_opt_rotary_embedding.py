# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def pos_neg_mask(tensor_shape):
    mask = torch.full(tensor_shape, 1, dtype=torch.bfloat16)
    for w_idx, w_val in enumerate(mask):
        for z_idx, z_val in enumerate(w_val):
            for y_idx, y_val in enumerate(z_val):
                if y_idx % 2:
                    mask[w_idx][z_idx][y_idx] = -1 * y_val

    return mask


def gen_sequential_data(torch_shape):
    result = torch.arange(
        0, (torch_shape[0] * torch_shape[1] * torch_shape[2] * torch_shape[3]) / 2, step=0.5, dtype=torch.bfloat16
    )
    result = torch.reshape(torch_input_tensor_a, torch_shape)
    return result


def gen_row_sequential_data(torch_shape):
    result = torch.zeros(torch_shape, dtype=torch.bfloat16)
    counter = 0
    for w_idx, w_val in enumerate(result):
        for z_idx, z_val in enumerate(w_val):
            for y_idx, y_val in enumerate(z_val):
                result[w_idx][z_idx][y_idx] = counter
                counter += 1
    return result


def rotary_embed_shuffle_data(tensor):
    print(tensor.shape)
    result = torch.zeros(tensor.shape, dtype=torch.bfloat16)
    for w_idx, w_val in enumerate(tensor):
        for z_idx, z_val in enumerate(w_val):
            for y_idx in range(0, len(z_val)):
                y_val = z_val[y_idx]
                if y_idx % 2:
                    result[w_idx][z_idx][y_idx - 1] = -1 * y_val
                else:
                    result[w_idx][z_idx][y_idx + 1] = y_val

    return result


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
def test_opt_rotary_embedding(device, h, w):
    torch.set_printoptions(sci_mode=False, profile="full")
    torch_shape = (1, 2, h, w)
    torch_input_tensor_a = gen_row_sequential_data(torch_shape)
    torch_input_tensor_b = pos_neg_mask(torch_shape)
    torch_output_tensor = rotary_embed_shuffle_data(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.mul(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)
    print(f"expected output = {torch_output_tensor}")
    print(f"device output = {torch_output_tensor}")

    assert_with_pcc(torch_output_tensor, output, 0.9999)
