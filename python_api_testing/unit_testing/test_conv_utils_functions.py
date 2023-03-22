from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import numpy as np

from pymetal import ttmetal as ttm
from pymetal.ttmetal.utils import tilize_to_list, tilize, untilize, channels_last, _nearest_32, convert_act_2d_matrix, convert_weights_2d_matrix
from python_api_testing.models.utility_functions import print_diff_argmax, is_close
import torch

# This test validates the following util functions - 1. convert_act_2d_matrix, 2. convert_weights_2d_matrix.
# The util functions are used for preparing conv activation and weights for TT metal OP.
# This test does not run any TT metal OPs. It runs pytorch matmul and compares the results with pytorch convolution.
def run_golden_conv (N, C, H, W, K, R, S, stride_y, stride_x, pad_y, pad_x):
    activation_shape = [N,C,H,W]
    weights_shape = [K,C,R,S]
    print("Running test with - activation shape= " + str(activation_shape) + " weight shape=" + str(weights_shape)
    + " stride(y,x)=(" + str(stride_y) + "," + str(stride_x) + ") padding(y,x)=(" + str(pad_y) + "," + str(pad_x) + ")")
    conv_output_height = ((H - R + 2*pad_y) // stride_y) + 1
    conv_output_width = ((W - S + 2*pad_x) // stride_x) + 1
    conv_matrix_rows = conv_output_height*conv_output_width
    conv_matrix_cols = C*R*S
    A = torch.randn(activation_shape)
    B = torch.randn(weights_shape)
    A_matrix = convert_act_2d_matrix(A, R, S, stride_y, stride_x, pad_y, pad_x)
    assert list(A_matrix.shape) == [1, N, conv_matrix_rows, conv_matrix_cols]
    B_matrix = convert_weights_2d_matrix(B, weights_shape)
    assert list(B_matrix.shape) == [1, 1, conv_matrix_cols, K]
    # Run pytorch matmul on the inputs prepared by util functions
    C_matrix = A_matrix.matmul(B_matrix)
    assert list(C_matrix.shape) == [1, N, conv_matrix_rows, K]
    C_matrix_t = C_matrix.transpose(2, 3)
    out = C_matrix_t.view(N, K, conv_output_height, conv_output_width)
    # Compare against Pytorch convolution output
    C_conv = torch.nn.functional.conv2d(A, B, stride=(stride_y, stride_x), padding=(pad_y, pad_x))
    assert (C_conv - out).abs().max() < 0.005

if __name__ == "__main__":
    # Kernel Sizes
    # 3x3 stride 1 pad 0
    run_golden_conv(1, 32, 5, 5, 32, 3, 3, 1, 1, 0, 0)
    # 1x1 stride 1 pad 0
    run_golden_conv(1, 32, 5, 5, 32, 1, 1, 1, 1, 0, 0)
    # 7x7 stride 1 pad 0
    run_golden_conv(1, 32, 9, 9, 32, 7, 7, 1, 1, 0, 0)

    # +Stride
    # 3x3 stride 2 pad 0
    run_golden_conv(1, 32, 5, 5, 32, 3, 3, 2, 2, 0, 0)
    # 1x1 stride 2 pad 0
    run_golden_conv(1, 32, 5, 5, 32, 1, 1, 2, 2, 0, 0)
    # 7x7 stride 2 pad 0
    run_golden_conv(1, 32, 9, 9, 32, 7, 7, 2, 2, 0, 0)

    # +Padding
    # 3x3 stride 2 pad 1
    run_golden_conv(1, 32, 5, 5, 32, 3, 3, 2, 2, 1, 1)
    # 1x1 stride 2 pad 1
    run_golden_conv(1, 32, 5, 5, 32, 1, 1, 2, 2, 1, 1)
    # 7x7 stride 2 pad 1
    run_golden_conv(1, 32, 9, 9, 32, 7, 7, 2, 2, 1, 1)

    print("ALL PASSED!")
