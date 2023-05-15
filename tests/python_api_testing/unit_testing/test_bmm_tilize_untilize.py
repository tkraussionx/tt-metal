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
    pad_activation,
    pad_weight,
    tilize,
    untilize,
    tilize_to_list,
    print_diff_argmax,
    pad_weight,
    is_close,
    comp_pcc,
)

TILE_HEIGHT = TILE_WIDTH = 32

## parameters
## input matrix size as number of tiles
a_height_ntiles = [1]   #[1, 2]
a_width_ntiles = [1]    #[9, 32]
b_width_ntiles = [1]    #[1, 9]
## blocking and subblocking parameters as number of tiles
num_blocks = [1]        #[1, 2]
out_subblock_h = [1]    #[2, 4]
out_subblock_w = [1]    #[2, 4]

@pytest.mark.parametrize(
    'a_height_ntiles, a_width_ntiles, b_width_ntiles, num_blocks, out_subblock_h, out_subblock_w',
    itertools.product(a_height_ntiles, a_width_ntiles, b_width_ntiles, num_blocks, out_subblock_h, out_subblock_w)
)
def test_run_bmm_single_core_tilize_untilize(a_height_ntiles,
                                             a_width_ntiles,
                                             b_width_ntiles,
                                             num_blocks,
                                             out_subblock_h,
                                             out_subblock_w):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    a_height = a_height_ntiles * TILE_HEIGHT
    a_width = a_width_ntiles * TILE_WIDTH
    b_width = b_width_ntiles * TILE_WIDTH
    a_shape = [1, 1, a_height, a_width]
    b_shape = [1, 1, a_width, b_width]
    out_shape = [1, 1, a_height, b_width]

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
    out = ttl.tensor.bmm_single_core_tilize_untilize(tta, ttb, num_blocks, out_subblock_h, out_subblock_w)
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
