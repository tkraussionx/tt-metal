import sys
import pytest
import itertools

from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np
import torch

from libs import tt_lib as ttl
from python_api_testing.models.utility_functions import (
    tilize_to_list,
    comp_pcc,
)

TILE_HEIGHT = TILE_WIDTH = 32

## parameters
# matrix sizes as number of blocks along h and w:
a_height_nblocks = [1]
a_width_nblocks = [1]   # == b_height_nblocks
b_width_nblocks = [1]
# block sizes as number of tiles along h and w:
a_block_height_ntiles = [1]
a_block_width_ntiles = [1]  # == b_block_height_ntiles
b_block_width_ntiles = [1]
# output sublobcking per block:
out_subblock_height_ntiles = [1]
out_subblock_width_ntiles = [1]


@pytest.mark.parametrize(
    'a_height_nblocks, a_width_nblocks, b_width_nblocks,\
     a_block_height_ntiles, a_block_width_ntiles, b_block_width_ntiles,\
     out_subblock_height_ntiles, out_subblock_width_ntiles',
    itertools.product(a_height_nblocks, a_width_nblocks, b_width_nblocks,
                      a_block_height_ntiles, a_block_width_ntiles, b_block_width_ntiles,
                      out_subblock_height_ntiles, out_subblock_width_ntiles)
)
def test_run_bmm_single_core_tilize_untilize(a_height_nblocks,
                                             a_width_nblocks,
                                             b_width_nblocks,
                                             a_block_height_ntiles,
                                             a_block_width_ntiles,
                                             b_block_width_ntiles,
                                             out_subblock_height_ntiles,
                                             out_subblock_width_ntiles):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    a_batch = b_batch = 1
    a_channel = b_channel = 1
    a_height = a_height_nblocks * a_block_height_ntiles * TILE_HEIGHT
    a_width = a_width_nblocks * a_block_width_ntiles * TILE_WIDTH   # == b_height
    b_width = b_width_nblocks * b_block_width_ntiles * TILE_WIDTH
    a_shape = [a_batch, a_channel, a_height, a_width]
    b_shape = [b_batch, b_channel, a_width, b_width]
    out_shape = [a_batch, a_channel, a_height, b_width]

    torch.manual_seed(0)
    a = torch.randn(a_shape, dtype=torch.bfloat16).float()
    b = torch.randn(b_shape, dtype=torch.bfloat16).float()

    ## a in row-major
    tta = ttl.tensor.Tensor(
        a.flatten().tolist(),
        a_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        device)
    ## b in tile major
    ttb = ttl.tensor.Tensor(
        tilize_to_list(b),
        b_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device)

    ## compute out
    out = ttl.tensor.bmm_tilize_untilize(tta, ttb,
                                         a_height_nblocks, a_width_nblocks, b_width_nblocks,
                                         a_block_height_ntiles, a_block_width_ntiles, b_block_width_ntiles,
                                         out_subblock_height_ntiles, out_subblock_width_ntiles)
    out = out.to(host)

    out_pytorch = torch.tensor(out.data()).reshape(out_shape)
    ttl.device.CloseDevice(device)

    ## reference
    golden_pytorch = torch.matmul(a, b)

    ## test for equivalance
    assert(out_pytorch.shape == golden_pytorch.shape)
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch)
    print(f'Passing PCC = {passing_pcc}')
    print(f'Output PCC = {output_pcc}')

    assert(passing_pcc)
