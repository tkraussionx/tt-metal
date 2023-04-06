
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from libs import tt_lib
import torch

#TODO: Imports break when importing batchnorm, fix this
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, pad_weight
from python_api_testing.models.revnet.block import TT_residual

def test_block():
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    H = 32
    W = 32
    C = 1
    epsf = 1e-4
    betaf = [0.333, 0.444]
    gammaf = [0.111, 0.222]
    mean_runf = [0.777, 0.888]
    var_runf = [0.555, 0.666]
    torch.manual_seed(123)
    x = torch.randn((1,C,H,W))
    ## TEMP MATMUL SETUP UNTIL CONV IS WRITTEN

    ident = torch.eye(H,W)
    r = torch.tensor([x for x in range(H)])
    c = torch.tensor([x-1 for x in range(W,0,-1)])
    permute=ident[r, c[:, None]]
    matmul = torch.reshape(torch.tensor(permute), (1,1,H, W))

    ### END of TEMP section

    gamma    = [pad_weight(torch.full((1,C,32,32), x))        for x in gammaf]
    beta     = [pad_weight(torch.full((1,C,32,32), x))        for x in betaf]
    mean_run = [pad_weight(torch.full((1,C,32,32), x))        for x in mean_runf]
    var_run  = [pad_weight(torch.full((1,C,32,32), x + epsf)) for x in var_runf]

    t0 = tt_lib.tensor.Tensor(tilize_to_list(x), [1, C, H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
    matmul0 = tt_lib.tensor.Tensor(tilize_to_list(matmul), [1, C, H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)

    ttgamma    = [tilize_to_list(x) for x in gamma]
    ttbeta     = [tilize_to_list(x) for x in beta]
    ttmean_run = [tilize_to_list(x) for x in mean_run]
    ttvar_run  = [tilize_to_list(x) for x in var_run]

    TT_residual(t0, matmul0, ttmean_run, ttvar_run, ttgamma, ttbeta, C, device)

    tt_lib.device.CloseDevice(device)
