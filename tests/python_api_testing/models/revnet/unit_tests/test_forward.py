
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")


#TODO: fix import statements
import torch
from torch import nn
from libs import tt_lib

from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, pad_weight
torch.set_printoptions(threshold=10_000)

from tests.python_api_testing.models.revnet.forward import TT_forward

def test_forward():

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    host = tt_lib.device.GetHost()

    H = 32
    W = 32
    C = 2
    epsf = 1e-4
    f_betaf = [0.333, 0.444]
    f_gammaf = [0.111, 0.222]
    f_mean_runf = [0.777, 0.888]
    f_var_runf = [0.555, 0.666]
    g_betaf = [0.333, 0.444]
    g_gammaf = [0.111, 0.222]
    g_mean_runf = [0.777, 0.888]
    g_var_runf = [0.555, 0.666]

    torch.manual_seed(123)
    x = torch.randn((1,C,H,W))
    in_channels = 2
    out_channels = 2
    stride = 1
    padding = 1
    dilation = 1

    ## TEMP MATMUL SETUP UNTIL CONV IS WRITTEN
    ident = torch.eye(H,W)
    r = torch.tensor([x for x in range(H)])
    c = torch.tensor([x-1 for x in range(W,0,-1)])
    permute=ident[r, c[:, None]]
    matmul = torch.reshape(torch.tensor(permute), (1,1,H, W))
    ### END of TEMP section

    # TODO: Fix channel computation when we generalize
    f_gamma    = [pad_weight(torch.full((1,1,32,32), x))        for x in f_gammaf]
    f_beta     = [pad_weight(torch.full((1,1,32,32), x))        for x in f_betaf]
    f_mean_run = [pad_weight(torch.full((1,1,32,32), x))        for x in f_mean_runf]
    f_var_run  = [pad_weight(torch.full((1,1,32,32), x + epsf)) for x in f_var_runf]

    g_gamma    = [pad_weight(torch.full((1,1,32,32), x))        for x in g_gammaf]
    g_beta     = [pad_weight(torch.full((1,1,32,32), x))        for x in g_betaf]
    g_mean_run = [pad_weight(torch.full((1,1,32,32), x))        for x in g_mean_runf]
    g_var_run  = [pad_weight(torch.full((1,1,32,32), x + epsf)) for x in g_var_runf]


    ## REFERENCE function
    # TODO: Finish reference
    ref_fwd = REF_forward(x, in_channels, out_channels,
                         stride, padding, dilation,
                         f_mean_runf, f_var_runf, f_gammaf, f_betaf,
                         g_mean_runf, g_var_runf, g_gammaf, g_betaf,
                         matmul,  epsf)

    # Set up input data for device
    t0 = tt_lib.tensor.Tensor(tilize_to_list(x), [1, C, H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
    matmul0 = tt_lib.tensor.Tensor(tilize_to_list(matmul), [1, 1, H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
    # TODO: fix C in matmul? Might not need when multi channel ^^ is working

    f_ttgamma    = [tilize_to_list(x) for x in f_gamma]
    f_ttbeta     = [tilize_to_list(x) for x in f_beta]
    f_ttmean_run = [tilize_to_list(x) for x in f_mean_run]
    f_ttvar_run  = [tilize_to_list(x) for x in f_var_run]

    g_ttgamma    = [tilize_to_list(x) for x in g_gamma]
    g_ttbeta     = [tilize_to_list(x) for x in g_beta]
    g_ttmean_run = [tilize_to_list(x) for x in g_mean_run]
    g_ttvar_run  = [tilize_to_list(x) for x in g_var_run]

    # Run TT_residual
    t1 = TT_forward(t0, in_channels, out_channels,  stride, padding, dilation,
        f_ttmean_run, f_ttvar_run, f_ttgamma, f_ttbeta,
        g_ttmean_run, g_ttvar_run, g_ttgamma, g_ttbeta,
        matmul0, device, host, C,H,W)

    tt_lib.device.CloseDevice(device)
