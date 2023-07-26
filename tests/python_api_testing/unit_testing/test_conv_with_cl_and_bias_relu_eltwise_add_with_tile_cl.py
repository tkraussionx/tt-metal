import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

import tt_lib as ttl
from tt_lib.utils import tilize_to_list, tilize, untilize, channels_last, _nearest_32, _nearest_y, convert_weights_2d_matrix
from python_api_testing.models.utility_functions import print_diff_argmax, is_close, comp_pcc
from tests.python_api_testing.conv.conv_unit_test_utils import create_conv_act_tensor, create_conv_weight_tensor
import torch

@pytest.mark.parametrize(
    "K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w",
    (
        (32, 32, 32, 32, 1, 1, 1, 1, 0, 0),
        # w/ conv padding
        # (32, 32, 5, 5, 1, 1, 1, 1, 1, 1),
        # # Hat = 1, Wat = 1, Wbt = 1
        # (32, 32, 5, 5, 1, 1, 1, 1, 0, 0),
        # # Hat = 2, Wat = 1, Wbt = 1
        # (32, 32, 8, 8, 1, 1, 1, 1, 0, 0),
        # # Hat = 1, Wat = 2, Wbt = 1
        # (32, 64, 5, 5, 1, 1, 1, 1, 0, 0),
        # # Hat = 2, Wat = 2, Wbt = 1
        # (32, 64, 8, 8, 1, 1, 1, 1, 0, 0),
        # # Hat = 1, Wat = 1, Wbt = 2
        # (64, 32, 5, 5, 1, 1, 1, 1, 0, 0),
        # # Hat = 1, Wat = 2, Wbt = 2
        # (64, 64, 5, 5, 1, 1, 1, 1, 0, 0),
        # # Hat = 2, Wat = 1, Wbt = 2
        # (64, 32, 8, 8, 1, 1, 1, 1, 0, 0),
        # # Hat = 2, Wat = 2, Wbt = 2
        # (64, 64, 8, 8, 1, 1, 1, 1, 0, 0),
        # # Hat = 8, Wat = 8, Wbt = 8
        # (8*32, 8*32, 16, 16, 1, 1, 1, 1, 0, 0),
    ),
)
def test_run_conv_as_large_matmul(use_program_cache, K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w):

    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    #torch.set_printoptions(threshold=10000)
    torch.manual_seed(0)
    a_activation_shape = [1,C,H,W]
    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    b_weights_shape = [K,C,R,S]
    B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()
    c_bias_shape = [1,K,1,1]
    C_pyt = torch.randn(c_bias_shape, dtype=torch.bfloat16).float()

    # Parameters to define block dims
    act_block_h = 4
    act_block_w = 4
    weight_block_h = act_block_w
    weight_block_w = 4
    out_subblock_h = 4
    out_subblock_w = 2

    OH = ((int) ((H - R + 2 * pad_h) / stride_h)) + 1
    OW = ((int) ((W - S + 2 * pad_w) / stride_w)) + 1
    conv_output_shape = [1,K, OH,OW]

    # Prepare activations
    A_cl_host = create_conv_act_tensor(A_pyt, 1, C, H, W)
    A_tile_cl = A_cl_host.to(ttl.tensor.Layout.TILE_CL).to(device)
    A = A_cl_host.to(device)

    # Prepare weights
    B_tiled_host = create_conv_weight_tensor(B_pyt, K, C, R, S, weight_block_h, weight_block_w)
    B_tiled = B_tiled_host.to(device)

    #Prepare bias
    C_tiled_cl = ttl.tensor.Tensor(
        torch.flatten(C_pyt).tolist(),
        c_bias_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.CHANNELS_LAST).pad([1,_nearest_32(K),1,32], (0,0,0,0), 0.0).to(ttl.tensor.Layout.TILE_CL).to(device)

    # Calculate conv result with golden result. Run Pytorch conv
    out_golden = torch.nn.functional.conv2d(A_pyt, B_pyt, bias=C_pyt.reshape(-1), stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    out_golden = torch.relu(out_golden)
    out_golden = out_golden + A_pyt

    # Run TT metal OP
    out = ttl.tensor.conv(A, B_tiled, [R,S,stride_h,stride_w,pad_h,pad_w], act_block_h, act_block_w, weight_block_w, out_subblock_h, out_subblock_w, K)
    assert(out.shape() == conv_output_shape)
    assert(out.layout() == ttl.tensor.Layout.CHANNELS_LAST)
    # Run conv bias
    out = ttl.tensor.bcast(out, C_tiled_cl, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.H)
    # Run relu
    out = ttl.tensor.relu(out)
    assert(out.shape() == conv_output_shape)
    assert(out.layout() == ttl.tensor.Layout.TILE_CL)
    out = ttl.tensor.add(out, A_tile_cl)
    out = out.to(host).to(ttl.tensor.Layout.CHANNELS_LAST)
    # Copy output to host and convert tt tensor to pytorch tensor
    conv_output_shape_cl = [1,OH,OW,K]
    out_result = torch.tensor(out.data()).reshape(conv_output_shape_cl)
    out_result = torch.transpose(out_result, 2, 3)
    out_result = torch.transpose(out_result, 1, 2)

    # Compare against golden
    assert(out_result.shape == out_golden.shape)
    passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc
    ttl.device.CloseDevice(device)
