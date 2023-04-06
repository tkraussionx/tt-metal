
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"

sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")


import torch
from libs import tt_lib
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, pad_weight
#from python_api_testing.fused_ops.batchnorm import Batchnorm
#from libs.tt_lib.fused_ops.batchnorm import Batchnorm
from python_api_testing.fused_ops.batchnorm import *


def test_batchnorm():
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    H = 32
    W = 32
    C = 1
    epsf = 1e-4
    betaf = 0.345
    gammaf = 0.123
    mean_runf = 0.789
    var_runf = 0.567
    torch.manual_seed(123)
    x = torch.randn((1,C,H,W))

    gamma = pad_weight(torch.full((1,C,32,32), gammaf))
    beta = pad_weight(torch.full((1,C,32,32), betaf))
    mean_run = pad_weight(torch.full((1,C,32,32), mean_runf))
    var_run = pad_weight(torch.full((1,C,32,32), var_runf + epsf))

    t0 = tt_lib.tensor.Tensor(tilize_to_list(x), [1, C, H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
    ttgamma = tilize_to_list(gamma)
    ttbeta = tilize_to_list(beta)
    ttmean_run = tilize_to_list(mean_run)
    ttvar_run = tilize_to_list(var_run)

    func = Batchnorm(ttmean_run, ttvar_run, ttgamma, ttbeta, C, device)

    t1 = func(t0)
    tt_lib.device.CloseDevice(device)
