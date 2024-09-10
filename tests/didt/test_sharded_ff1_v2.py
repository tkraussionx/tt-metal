# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest

import ttnn

import tt_lib as ttl
from models.utility_functions import comp_pcc, torch2tt_tensor, tt2torch_tensor, get_devices_for_t3000
import torch

# from tests.didt.test_didt import test_reproduce_matmul_hang_common
from tests.didt.test_ff1 import Ff1Test

FF1_HANG_PARAMETRIZATION = (1024, 4608, 18432, 4, 72, 3, 1, 8, 100000)

CHIP_ID_TO_COORDINATES_T3K = [None] * 8
CHIP_ID_TO_COORDINATES_T3K[0] = (1, 0)
CHIP_ID_TO_COORDINATES_T3K[1] = (1, 1)
CHIP_ID_TO_COORDINATES_T3K[2] = (2, 1)
CHIP_ID_TO_COORDINATES_T3K[3] = (2, 0)
CHIP_ID_TO_COORDINATES_T3K[4] = (0, 0)
CHIP_ID_TO_COORDINATES_T3K[5] = (0, 1)
CHIP_ID_TO_COORDINATES_T3K[6] = (3, 1)
CHIP_ID_TO_COORDINATES_T3K[7] = (3, 0)


@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    (FF1_HANG_PARAMETRIZATION,),
    ids=["ff1-hang"],
)
@pytest.mark.parametrize("num_devices", [1, 2, 8], ids=["1chips", "2chips", "8chips"])
def test_reproduce_matmul_2d_hang(
    num_devices,
    all_devices,
    seq_len,
    inner_dim,
    weights_n,
    per_core_M,
    per_core_N,
    in_block_w,
    out_subblock_h,
    out_subblock_w,
    loop_count,
    use_program_cache,
    determinism_check_enabled=False,
    determinism_check_iterations=1,
):
    ff1_test = Ff1Test(
        num_devices,
        all_devices,
        seq_len,
        inner_dim,
        weights_n,
        per_core_M,
        per_core_N,
        in_block_w,
        out_subblock_h,
        out_subblock_w,
        loop_count,
        determinism_check_enabled,
        determinism_check_iterations,
    )
    ff1_test.test_didt()
