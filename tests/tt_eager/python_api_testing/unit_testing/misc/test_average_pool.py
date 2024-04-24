# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest


import torch

import tt_lib as ttl

from tt_lib.utils import _nearest_32
from models.utility_functions import comp_pcc

TILE_HEIGHT = TILE_WIDTH = 32


def shape_padded(shape):
    return [shape[0], shape[1], _nearest_32(shape[2]), _nearest_32(shape[3])]


@pytest.mark.parametrize(
    "act_shape",
    (([1, 7, 7, 2048], ([1, 1, 32, 64]))),
    ids=["resnet50_unpadded", "tile_divisible"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=[
        "BFLOAT16",
    ],
)
def test_run_average_pool(act_shape, dtype, device):
    batch_size, _, _, channels = act_shape

    import time

    torch.manual_seed(0)

    act = torch.randn(act_shape, dtype=torch.bfloat16).float()
    ttact = ttl.tensor.Tensor(act, ttl.tensor.DataType.BFLOAT16)
    act_shape_padded = shape_padded(act_shape)
    if act_shape != act_shape_padded:
        ttact = ttact.pad_to_tile(0.0)
    ttact = ttact.to(device)

<<<<<<< HEAD
    out = ttl.tensor.average_pool_2d(ttact)
=======
    logger.info(f"TMZ -- running average pool")

    if not trace_captured:
        ttl.device.BeginTraceCapture(device)
        # time.sleep(2)
        logger.info(f"TMZ -- trace capture begin")
        out = ttl.tensor.average_pool_2d(ttact)
        ttl.device.EndTraceCapture(device)
        # time.sleep(2)
        logger.info(f"TMZ -- trace capture end")
        trace_captured = True

    # logger.info(f"TMZ -- eager start")
    # ttl.device.Tag(device, "eager")
    # out = ttl.tensor.average_pool_2d(ttact)
    # ttl.device.Untag(device)
    # logger.info(f"TMZ -- eager end")

    # Check if environment variable DO_TRACE is defined:
    # import required library
    import os
    if os.environ.get("DO_TRACE"):
        logger.info(f"TMZ -- trace enqueued")
        ttl.device.ExecuteLastTrace(device, True) # Hang here.
        logger.info(f"TMZ -- trace executed")
    else:
        logger.info(f"TMZ -- running legacy path")
        out = ttl.tensor.average_pool_2d(ttact)
        logger.info(f"TMZ -- done running legacy path")
>>>>>>> b4cd96f6b... #0: Debug stuff. DO_TRACE=1 TT_METAL_DPRINT_FILE=dprint_trace_prefetcher_hang_verbose.log TT_METAL_DPRINT_CORES=1,8 pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_average_pool.py::test_run_average_pool[BFLOAT16-resnet50_unpadded] |& tee test_hang.log

    out = out.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    out_shape = [batch_size, 1, 1, channels]
    out_shape_padded = shape_padded(out_shape)
    if out_shape != out_shape_padded:
        out = out.unpad_from_tile(out_shape)

    out_pytorch = out.to_torch()

    ## reference
    act_channels_first = torch.permute(act, (0, 3, 1, 2))  # Torch operates on channels-first tensors
    golden_pytorch = torch.nn.AdaptiveAvgPool2d((1, 1))(act_channels_first)

    ## test for equivalance
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch)
    print(f"Passing PCC = {passing_pcc}")
    print(f"Output PCC = {output_pcc}")

    assert passing_pcc
