import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from libs import tt_lib as ttl
from libs.tt_lib.utils import tilize_to_list, tilize, untilize, channels_last, _nearest_32, convert_weights_2d_matrix
from python_api_testing.models.utility_functions import print_diff_argmax, is_close, comp_pcc
from python_api_testing.conv.pytorch_conv_tb import TestLevel, generate_conv_tb_with_pytorch_golden, generate_conv_tb

import torch

@pytest.mark.parametrize(
    "K, C, H, W, R, S",
    (
        (32, 32, 8, 4, 1, 1),
        # (32, 32, 10, 10, 3, 3),
        # (64, 64, 32, 16, 1, 1),
        # (64, 64, 10, 10, 1, 1),
    ),
)
def test_run_conv_as_large_matmul(K, C, H, W, R, S):
    a_activation_shape = [1,C,H,W]
    b_weights_shape = [K,C,R,S]
    OH = H - R + 1
    OW = W - S + 1
    mm_output_shape = [1,1,_nearest_32(OH*OW),K]

    #torch.manual_seed(0)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    a_activation_shape = [1,C,H,W]
    b_weights_shape = [K,C,R,S]
    mm_input_shape = [1, 1, _nearest_32(OH*OW), C*R*S]
    mm_weight_shape = [1, 1, C*R*S, K]
    mm_output_shape = [1,1,_nearest_32(OH*OW),K]

    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    A_ = ttl.tensor.Tensor(
        torch.flatten(A_pyt).tolist(),
        a_activation_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR)
    A_cl = A_.to(ttl.tensor.Layout.CHANNELS_LAST)

    # Prepare weights
    B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()
    B_ = ttl.tensor.Tensor(
        torch.flatten(B_pyt).tolist(),
        b_weights_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR
    )
    A_cl_data = A_cl.data()
    # Call DTX pass to transform A
    A_transformed_data = ttl.dtx.evaluate(A_cl_data, ttl.dtx.conv_transform([C,H,W], [R,S,1,1,0,0], ([-1],[-1])))
    A_transformed_pytorch_tensor = torch.tensor(A_transformed_data).reshape(mm_input_shape)

    B_tiled_ = ttl.tensor.convert_conv_weight_tensor_to_tiled_layout(B_)
    B_rm = B_tiled_.to(ttl.tensor.Layout.ROW_MAJOR)
    assert(B_rm.shape() == [1, 1, C*R*S, K])
    B_data = B_rm.data()
    B_pytorch_tensor = torch.tensor(B_data).reshape(mm_weight_shape)


    # Run pytorch matmul
    print("matmul input shape - " + str(A_transformed_pytorch_tensor.shape))
    print("matmul weight shape - " + str(B_pytorch_tensor.shape))
    out_pytorch = torch.matmul(A_transformed_pytorch_tensor, B_pytorch_tensor)
    assert(list(out_pytorch.shape) == mm_output_shape)
    # remove padding
    out_pytorch = out_pytorch[:, :, 0 : (OH * OW), :]

    # Convert matmul output layout to conv output layout
    out_tr = torch.transpose(out_pytorch, 2, 3)
    assert(list(out_tr.shape) == [1,1,K,(OH*OW)])
    out_result = out_tr.reshape([1,K,OH,OW])

    # Compare against pytorch golden result
    out_golden = torch.nn.functional.conv2d(A_pyt, B_pyt)
    assert(out_result.shape == out_golden.shape)
    passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert(passing_pcc)
