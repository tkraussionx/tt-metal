from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import numpy as np

from libs import tt_lib as ttl
from libs.tt_lib.utils import (
    tilize_to_list,
    tilize,
    untilize,
    channels_last,
    _nearest_32,
    convert_weights_2d_matrix,
    convert_act_2d_matrix,
)

import torch


def run_tilize_conv_act_test(K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w):
    assert R == S
    # check if params are valid
    assert (H - R + 2 * pad_h) >= 1 and (W - S + 2 * pad_w) >= 1
    OH = ((int) ((H - R + 2 * pad_h) / stride_h)) + 1
    OW = ((int) ((W - S + 2 * pad_w) / stride_w)) + 1
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    a_activation_shape = [1, C, H, W]

    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    A_cl = channels_last(A_pyt)
    A = ttl.tensor.Tensor(
        torch.flatten(A_cl).tolist(),
        a_activation_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.CHANNELS_LAST,
        device,
        ttl.tensor.MemoryConfig(False, 0),
    )
    # Tilize conv activation on device
    A_t = ttl.tensor.tilize_conv_activation(A, [R,S,stride_h,stride_w,pad_h,pad_w], K)
    output_shape = [1, 1, _nearest_32(OH * OW), _nearest_32(C * R * S)]
    pyt_got_back = np.array(A_t.to(host).data(), dtype=float).reshape(output_shape)
    print("Pytorch tensor got back shape - " + str(pyt_got_back.shape))
    # remove padding
    A_ut = pyt_got_back[:, :, 0 : (OH * OW), :]
    A_golden = convert_act_2d_matrix(A_pyt, R, S, stride_h, stride_w, pad_h, pad_w)
    ttl.device.CloseDevice(device)
    assert A_ut.shape == A_golden.shape
    print(abs(A_golden - A_ut).max())
    assert (
        abs(A_golden - A_ut) < 0.02
    ).all(), "Max abs difference for tilize can be 0.02 due to bfloat conversions"


def test_run_tilize_convs1_act_test():
    # incorrect output
    run_tilize_conv_act_test(256, 128, 28, 28, 3, 3, 2, 2, 1, 1)
    # Hang
    #run_tilize_conv_act_test(32, 1024, 5, 5, 1, 1, 1, 1, 0, 0)

    # Crash
    #run_tilize_conv_act_test(256, 256, 14, 14, 3, 3, 1, 1, 1, 1)
    #Works
    #run_tilize_conv_act_test(32, 32, 5, 5, 3, 3, 1, 1, 1, 1)
    #run_tilize_conv_act_test(32, 5, 5, 3, 3, 1, 1, 0, 0)
    #run_tilize_conv_act_test(32, 5, 5, 1, 1, 1, 1, 0, 0)
