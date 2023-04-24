import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

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
import torch

result = []
A_i = []
B_i = []

# @pytest.mark.parametrize(
#     "Hat, Wat, Wbt, tilize_act, untilize_out",
#     (
#         #(7, 72, 8),
#         (2, 9, 9, False, True),
#         (2, 9, 9, False, True),
#     ),
# )
def test_run_large_matmul_test(Hat, Wat, Wbt, tilize_act, untilize_out):
    global result
    global A_i
    global B_i
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    TILE_HEIGHT = TILE_WIDTH = 32

    Ha = Hat * TILE_HEIGHT
    Wa = Wat * TILE_WIDTH
    Wb = Wbt * TILE_WIDTH
    host = ttl.device.GetHost()
    a_shape = [1, 1, Ha, Wa]
    b_shape = [1, 1, Wa, Wb]
    torch.manual_seed(0)
    a = torch.randn(a_shape, dtype=torch.bfloat16).float()
    if (A_i == []):
        A_i = torch.flatten(a).tolist()
    elif(A_i != torch.flatten(a).tolist()):
        assert False
    b = torch.randn(b_shape, dtype=torch.bfloat16).float()
    if (B_i == []):
        B_i = torch.flatten(b).tolist()
    elif(B_i != torch.flatten(b).tolist()):
        assert False

    layout_a = ttl.tensor.Layout.ROW_MAJOR if tilize_act else ttl.tensor.Layout.TILE
    def tt_a():
        if layout_a == ttl.tensor.Layout.ROW_MAJOR:
            return a.flatten().tolist()
        else:
            return tilize_to_list(a)


    tta = ttl.tensor.Tensor(
        tt_a(),
        a_shape,
        ttl.tensor.DataType.BFLOAT16,
        layout_a,
        device)

    ttb = ttl.tensor.Tensor(
        tilize_to_list(b),
        b_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device)

    out = ttl.tensor.large_bmm(tta, ttb, tilize_act, untilize_out)
    out_shape = [1,1,Ha,Wb]
    out = out.to(host)
    if not untilize_out:
        # untilize
        out = out.to(ttl.tensor.Layout.ROW_MAJOR)
    out_pytorch = torch.tensor(out.data()).reshape(out_shape)
    ttl.device.CloseDevice(device)
    if (result == []):
        result = torch.flatten(out_pytorch).tolist()
    elif(result != torch.flatten(out_pytorch).tolist()):
        assert False
    golden_pytorch = torch.matmul(a,b)
    assert(out_pytorch.shape == golden_pytorch.shape)
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc

if __name__ == "__main__":
    #test_run_large_matmul_test(2, 9, 9, False, True)
    #test_run_large_matmul_test(2, 9, 9, False, True)
    # Incorrect output
    #test_run_conv_as_large_matmul(256, 128, 28, 28, 3, 3, 2, 2, 1, 1)
    #test_run_large_matmul_test(7, 36, 8, True, True)
    # Hang
    # test_run_conv_as_large_matmul(32, 1024, 5, 5, 1, 1, 1, 1, 0, 0)
    #test_run_large_matmul_test(1, 32, 1, True, False)
    # Crash
    #test_run_conv_as_large_matmul(256, 256, 14, 14, 3, 3, 1, 1, 1, 1)
    test_run_large_matmul_test(7, 72, 8, True, True)
