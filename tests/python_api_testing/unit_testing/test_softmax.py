from pathlib import Path
import sys
import pytest
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../../..")
sys.path.append(f"{f}/../..")

import torch

import tt_lib as ttl

from python_api_testing.models.utility_functions import (
    tilize_to_list,
    tilize,
    untilize,
    comp_pcc,
)

def make_mask(mask_shape, tensor_shape):
    mask = torch.zeros_like(tensor_shape)
    ## put 1 in all 'mask_shape' locations
    ## ...
    return mask

def run_softmax_test(device, host, dtype, shape, scale=None, mask_shape=None):
    assert(dtype == ttl.tensor.DataType.BFLOAT16)
    data = torch.randn(shape, dtype=torch.bfloat16).float()
    input = ttl.tensor.Tensor(tilize_to_list(data),shape, dtype, ttl.tensor.Layout.TILE, device)
    if not scale:
        scale = 0.0
    if mask_shape:
        mask = make_mask(mask_shape, shape)
        output = ttl.tensor.scale_mask_softmax_in_place(scale, mask, input).to(host)
    else:
        output = ttl.tensor.softmax_in_place(input).to(host)

    ## untilize and reshape the output
    output = untilize(torch.tensor(output.data()).reshape(shape))

    ## reference
    ref_output = torch.softmax(data, dim=-1)

    ## comapre output and ref_output
    assert(ref_output.shape == output.shape)
    passing_pcc, output_pcc = comp_pcc(ref_output, output)
    print(f'Passing PCC = {passing_pcc}')
    print(f'Output PCC = {output_pcc}')


@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
def test_softmax(dtype):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    shape1 = [1, 1, 32, 32]
    shape2 = [1, 1, 2048, 1024]

    run_softmax_test(device, host, dtype, shape1)
    # run_softmax_test(device, host, dtype, shape2)

    ttl.device.CloseDevice(device)
