###################################################################
######################## TT Implementation ########################
################################################################### REV BLOCK

# INPUT
    # X: [32x32]

# For conv
    # kernel: [3x3]
    # stride: [1]
    # padding: [1]
    # dilation: True False
    #

#For substitute matmul
    # M: 32x32, rotated itentity matrix or random input

#For Bnorm
    # mean_run [2]
    # var_run  [2]
    # epsilon  [1]
    # Channels [1]

# For ReLu
    #none

#def TT_residual(x, params, buffers, training, stride=1, padding=1, dilation=1, no_activation=False):

import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from torch import nn
#from pymetal import ttmetal
#from pymetal import ttlib as ttl
from libs import tt_lib

import torch.nn.functional as F

from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, pad_weight
torch.set_printoptions(threshold=10_000)

from tests.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_allclose_and_pcc
from python_api_testing.fused_ops.batchnorm import Batchnorm


def TT_residual(x, matmul, mean_run, var_run, gamma, beta, C, device):

    bnorm1 = Batchnorm(mean_run[0], var_run[0], gamma[0], beta[0], C, device)
    bnorm2 = Batchnorm(mean_run[1], var_run[1], gamma[1], beta[1], C, device)

    out = bnorm1(x)
    out = tt_lib.tensor.relu(out)
    out = tt_lib.tensor.matmul(out, matmul)
    out = bnorm2(out)
    out = tt_lib.tensor.relu(out)
    out = tt_lib.tensor.matmul(out, matmul)

    return out


def REF_residual(x, matmul, mean_run, var_run, gamma, beta, eps, training=False):

    out = F.batch_norm(x, torch.tensor([mean_run[0]]), torch.tensor([var_run[0]]), torch.tensor(gamma[0]), torch.tensor(beta[0]), training, eps=eps )
    out = F.relu(out)
    out = torch.matmul(out,matmul)
    out = F.batch_norm(out, torch.tensor([mean_run[1]]), torch.tensor([var_run[1]]), torch.tensor(gamma[1]), torch.tensor(beta[1]), training, eps=eps )
    out = F.relu(out)
    out = torch.matmul(out,matmul)

    return out

if __name__ == "__main__":
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    host = tt_lib.device.GetHost()

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

    ## REFERENCE
    ref_revblock = REF_residual(x, matmul, mean_runf, var_runf, gammaf, betaf, epsf, training=False)


    t0 = tt_lib.tensor.Tensor(tilize_to_list(x), [1, C, H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
    matmul0 = tt_lib.tensor.Tensor(tilize_to_list(matmul), [1, C, H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)

    ttgamma    = [tilize_to_list(x) for x in gamma]
    ttbeta     = [tilize_to_list(x) for x in beta]
    ttmean_run = [tilize_to_list(x) for x in mean_run]
    ttvar_run  = [tilize_to_list(x) for x in var_run]


    t1 = TT_residual(t0, matmul0, ttmean_run, ttvar_run, ttgamma, ttbeta, C, device)
    t2_data = t1.to(host).data()

    tt_got_back = torch.Tensor(t2_data).reshape((1,C,H,W))
    tt_got_back = untilize(tt_got_back)


    print ('=========COMPLETE=============')
    print ("GOLDEN PCC TEST")
    #print (comp_pcc(ref_revblock, tt_got_back, pcc=.99))
    print (comp_allclose_and_pcc(ref_revblock, tt_got_back))

    tt_lib.device.CloseDevice(device)


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
