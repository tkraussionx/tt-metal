from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from pymetal import ttlib as ttl
from pymetal.ttlib.utils import tilize_to_list, channels_last, convert_weights_2d_matrix, untilize
from python_api_testing.models.utility_functions import is_close
import torch

def run_conv_test(K, C, H, W, untilize_out):
    #torch.manual_seed(0)
    a_activation_shape = [1,C,H,W]
    b_weights_shape = [K,C,3,3]

    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    A_cl = channels_last(A_pyt)
    A = ttl.tensor.Tensor(
        torch.flatten(A_cl).tolist(),
        a_activation_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.CHANNELS_LAST,
        device,
        ttl.tensor.MemoryConfig(False, 0)
    )

    # Prepare weights
    B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()
    B_matrix = convert_weights_2d_matrix(B_pyt, b_weights_shape)
    # Conv as large matmul only works for certain values
    # hard code for now
    TILE_HEIGHT = TILE_WIDTH = 32
    Ha = 8 * TILE_HEIGHT
    Wa = 9 * TILE_WIDTH
    Wb = 4 * TILE_WIDTH
    assert(B_matrix.shape[2] == Wa and B_matrix.shape[3] == Wb)
    B_t = ttl.tensor.Tensor(
        tilize_to_list(B_matrix),
        B_matrix.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
        ttl.tensor.MemoryConfig(False, 1)
        )

    # Run TT metal OP
    out = ttl.tensor.conv_as_large_bmm_single_block_single_core(A, B_t, untilize_out)
    out_shape = [1,1,Ha,Wb]
    assert(out.shape() == out_shape)
    out_pytorch = torch.tensor(out.to(host).data()).reshape(out_shape)
    if not untilize_out:
        out_pytorch = untilize(out_pytorch)
    OH = H-2
    OW = W-2
    # Convert matmul output layout to conv output layout
    out_tr = torch.transpose(out_pytorch, 2, 3)
    assert(list(out_tr.shape) == [1,1,K,(OH*OW)])
    out_result = out_tr.reshape([1,K,OH,OW])

    # Calculate conv result with golden result. Run Pytorch conv
    out_golden = torch.nn.functional.conv2d(A_pyt, B_pyt)
    assert(out_result.shape == out_golden.shape)
    maxmag = out_golden.abs().max().item() # % of max magnitude since that determines cancellations
    match = is_close(out_result, out_golden, 0.07, 0.07, maxmag, 0.01)
    print("Match=", match.item())

if __name__ == "__main__":
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    # Conv activation shape
    # Conv as large matmul only works for certain values
    # hard code for now
    H=18
    W=18
    C=32
    K=128
    run_conv_test(K,C,H,W,False)
    ttl.device.CloseDevice(device)
