# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from models.utility_functions import torch2tt_tensor, profiler
import pytest


def run_test_bfp8_to_device(
    batch,
    seq_len,
    device,
):
    attn_mask_shape = (batch, seq_len, 32, 512)

    attn_mask = torch.zeros(*attn_mask_shape)
    BFLOAT16_DTYPE = tt_lib.tensor.DataType.BFLOAT16
    BFP8_DTYPE = tt_lib.tensor.DataType.BFLOAT8_B

    profiler.start("pushing_attn_mask_to_DRAM_bf16")
    attn_mask_bf16 = torch2tt_tensor(
        attn_mask.clone(),
        device,
        tt_dtype=BFLOAT16_DTYPE,
    )
    profiler.end("pushing_attn_mask_to_DRAM_bf16")

    profiler.start("pushing_attn_mask_to_DRAM_bf8")
    attn_mask_bf8 = torch2tt_tensor(
        attn_mask.clone(),
        device,
        tt_dtype=BFP8_DTYPE,
    )
    profiler.end("pushing_attn_mask_to_DRAM_bf8")
    profiler.print()


@pytest.mark.parametrize(
    "batch, seq_len",
    ((32, 1),),
)
def test_bfp8_to_device(
    batch,
    seq_len,
    pcie_devices,
):
    run_test_bfp8_to_device(
        batch,
        seq_len,
        pcie_devices[0],
    )
