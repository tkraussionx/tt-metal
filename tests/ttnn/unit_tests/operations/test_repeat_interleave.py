# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_repeat_interleave(device):
    torch_input_tensor = torch.tensor([[1, 2], [3, 4]])
    torch_result = torch.repeat_interleave(torch_input_tensor, 2, dim=0)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.repeat_interleave(input_tensor, 2, dim=0)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_result, output, 0.9999)


def test_repeat_interleave_with_repeat_tensor(device):
    torch_input_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    torch_repeats = torch.tensor([1, 2])
    torch_result = torch.repeat_interleave(torch_input_tensor, torch_repeats, dim=1)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    repeats = ttnn.from_torch(torch_repeats)
    output = ttnn.repeat_interleave(input_tensor, repeats, dim=1)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_result, output, 0.9999)


def test_f2(device):
    t = torch.randn((1, 1, 32, 5120), dtype=torch.bfloat16)

    ## golden
    torch_result = torch.repeat_interleave(t, (32), dim=3)

    delta_orig = ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    delta_orig = ttnn.permute(delta_orig, (3, 0, 1, 2))
    repeat_interleaved_output = ttnn.repeat_interleave(delta_orig, 32, dim=0)
    repeat_interleaved_output = ttnn.permute(repeat_interleaved_output, (1, 2, 3, 0))
    repeat_interleaved_output = ttnn.to_torch(repeat_interleaved_output)
    assert torch.allclose(torch_result, repeat_interleaved_output)

    delta_orig = ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    delta_orig = ttnn.permute(delta_orig, (0, 3, 1, 2))
    repeat_interleaved_output = ttnn.repeat_interleave(delta_orig, 32, dim=1)
    repeat_interleaved_output = ttnn.permute(repeat_interleaved_output, (0, 2, 3, 1))
    repeat_interleaved_output = ttnn.to_torch(repeat_interleaved_output)
    assert torch.allclose(torch_result, repeat_interleaved_output)

    ## fallback
    delta_orig = ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    delta_repeat = ttnn.repeat_interleave(delta_orig, 32, dim=3)
    delta_repeat_out = ttnn.to_torch(delta_repeat)
    print(delta_repeat_out.shape)
    assert torch.allclose(torch_result, delta_repeat_out)
