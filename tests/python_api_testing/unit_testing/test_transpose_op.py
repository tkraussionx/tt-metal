import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../")
import torch
from libs import tt_lib as ttl

# Initialize the device
device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
ttl.device.InitializeDevice(device)
host = ttl.device.GetHost()

# [#755] Transpose hangs in MNIST
if __name__ == '__main__':
    N = 1
    C = 1
    H = 120
    W = 784
    #Creating PyTorch Tensor
    tensor_pt = torch.randn((N,C,H,W))
    #Creating tensor in TT accelerator
    tensor_tt = ttl.tensor.Tensor(tensor_pt.view(-1).tolist(), 
                                    [N, C, H, W],
                                    ttl.tensor.DataType.BFLOAT16, 
                                    ttl.tensor.Layout.ROW_MAJOR, device)

    tt_res = ttl.tensor.transpose(tensor_tt)
    assert(tt_res.shape() == [N,C,W,H])
    