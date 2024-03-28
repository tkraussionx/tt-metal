# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import tt_lib
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000


def get_rot_transformation_mat(dhead):
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


class TtLlamaRotary:
    def __init__(self, device, head_dim):
        self.device = device
        self.transformation_mat = torch2tt_tensor(get_rot_transformation_mat(head_dim), device)

    def apply_rotary(self, x, cos, sin):
        batch, n_heads, _, _ = x.shape

        cos = ttnn.repeat(cos, ttnn.Shape([batch, n_heads, 1, 1]))
        sin = ttnn.repeat(sin, ttnn.Shape([batch, n_heads, 1, 1]))

        x_transformed = ttnn.matmul(x, self.transformation_mat)

        x_cos = ttnn.mul(cos, x)
        x_sin = ttnn.mul(sin, x_transformed)
        return ttnn.add(x_cos, x_sin)

    def __call__(self, xq, xk, cos, sin):
        xq = self.apply_rotary(xq, cos, sin)
        xk = self.apply_rotary(xk, cos, sin)
        return xq, xk


def gather_cos_sin(dhead, end, position_ids):
    cos, sin = precompute_freqs(dhead, end)
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


class PytorchLlamaRotaryMultiplyAddModel(torch.nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.transformation_mat = get_rot_transformation_mat(head_dim)

    def apply_rotary(self, x, cos, sin):
        return x * cos + x @ self.transformation_mat * sin

    def forward(self, xq, xk, cos, sin):
        # xq is shape of [batch, n_head, seq_len, head_dim]
        xq = self.apply_rotary(xq, cos, sin)
        xk = self.apply_rotary(xk, cos, sin)
        return xq, xk


def run_test_rotary(
    devices,
):
    device = devices[0]

    # Prepare input
    torch.manual_seed(0)
    batch = 1
    seq_len = 2048
    head_dim = 128
    inp = [
        (torch.rand(batch, 8, seq_len, head_dim) * 2) - 1,
        (torch.rand(batch, 1, seq_len, head_dim) * 2) - 1,
        (torch.randn(batch, 1, seq_len, head_dim) * 2) - 1,
        (torch.randn(batch, 1, seq_len, head_dim) * 2) - 1,
    ]

    # TT hardware -------------------------------------------------------------
    tt_model = TtLlamaRotary(device, head_dim)

    tt_inp = [torch2tt_tensor(i, device) for i in inp]

    tt_out = tt_model(*tt_inp)
    tt_out = [tt2torch_tensor(tt_out_tensor) for tt_out_tensor in tt_out]

    # PyTorch impl -------------------------------------------------------------
    pt_model = PytorchLlamaRotaryMultiplyAddModel(head_dim)
    pytorch_out = pt_model(*inp)

    # check outputs ----------------------------------------------------------------------

    does_pass = True
    for i in range(2):
        out_pass, output_pcc = comp_pcc(pytorch_out[i], tt_out[i], 0.9999)
        # Check each shape matches
        assert pytorch_out[i].shape == tt_out[i].shape
        logger.info(f"PCC value: {output_pcc}")
        does_pass = does_pass and out_pass

        mae = torch.mean(torch.abs(pytorch_out[i] - tt_out[i]))
        logger.info(f"MAE: {mae}")

        max_incorrect = torch.max(torch.abs(pytorch_out[i] - tt_out[i]))
        logger.info(f"Max incorrect: {max_incorrect}")

        max_gt = torch.max(torch.abs(pytorch_out[i]))
        logger.info(f"Max ground truth: {max_gt}")

    if does_pass:
        logger.info("Llama QKV output Passed!")
    else:
        logger.warning("Llama QKV output Failed!")
        assert does_pass, f"PCC value is lower than {output_pcc}"


def test_rotary(
    device,
):
    devices = [device]
    run_test_rotary(
        devices,
    )
