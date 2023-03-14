
import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from gpai import gpai
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight
torch.set_printoptions(threshold=10_000)

#v1_ assume input is always 32x32
def Batchnorm(mean_run, var_run, gamma, beta, C, device): #
    # gamma, beta, epsilon should be vectors of size C
    mean_run = gpai.tensor.Tensor(
        mean_run, 
        [1, C, 32, 32],  # will this auto-broadcast?
        gpai.tensor.DataType.BFLOAT16,
        gpai.tensor.Layout.TILE,
        device
    )

    var_run = gpai.tensor.Tensor(
        var_run ,
        [1, C, 32, 32],  # will this auto-broadcast?
        gpai.tensor.DataType.BFLOAT16,
        gpai.tensor.Layout.TILE,
        device
    )

    gamma = gpai.tensor.Tensor(
        gamma,
        [1, C, 32, 32],  # will this auto-broadcast?
        gpai.tensor.DataType.BFLOAT16,
        gpai.tensor.Layout.TILE,
        device
    )

    beta = gpai.tensor.Tensor(
        beta,
        [1, C, 32, 32],
        gpai.tensor.DataType.BFLOAT16,
        gpai.tensor.Layout.TILE,
        device
    )

    BCHW = gpai.tensor.BcastOpDim.HW
    BCMUL = gpai.tensor.BcastOpMath.MUL
    BCSUB = gpai.tensor.BcastOpMath.SUB
    BCADD = gpai.tensor.BcastOpMath.ADD

    def batchnorm_(x):
        # first subtract running mean
        x_minus_mean = gpai.tensor.bcast(x, mean_run, BCSUB, BCHW)
        print ('xxxxxxxxxx')
        #x
        print (x_minus_mean)

        hostx = gpai.device.GetHost()
        y=x_minus_mean.to(hostx).data()
        z=torch.Tensor(y).reshape((1,C,H,W))
        print(z[0,0,:,:])

        print ('xxxxxxxxxx')
        breakpoint()

        # take sqrt of running_var+eps
        var_sqrt = gpai.tensor.sqrt(var_run)
        # reciprocal
        inv_sqrt = gpai.tensor.recip(var_sqrt)
        x_div_sqrt = gpai.tensor.bcast(x_minus_mean, inv_sqrt, BCMUL, BCHW)
        x_gamma = gpai.tensor.bcast(x_div_sqrt, gamma, BCMUL, BCHW)
        x_result = gpai.tensor.bcast(x_gamma, beta, BCADD, BCHW)

        return x_result

    return batchnorm_


def ref_batchnorm(x, eps, gamma, beta, mean_run, var_run):
    bnorm = torch.nn.BatchNorm2d(x.shape[1], eps=eps, momentum=1)
    bnorm.weight       = torch.nn.Parameter(torch.tensor([gamma] * x.shape[1]))
    bnorm.bias         = torch.nn.Parameter(torch.tensor([beta] * x.shape[1]))
    bnorm.running_mean = torch.nn.Parameter(torch.tensor([mean_run] * x.shape[1]))
    bnorm.running_var  = torch.nn.Parameter(torch.tensor([var_run] * x.shape[1]))
    
    with torch.no_grad():
        result = bnorm(x)
    return result

if __name__ == "__main__":
    # Initialize the device
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, 0)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    H = 32
    W = 32
    C = 1
    epsf = 1e-4
    betaf = 0.345
    gammaf = 0.123
    mean_runf = 0. # 0.789
    var_runf = 0.567 
    torch.manual_seed(123)
    x = torch.randn((1,C,H,W))

    print ('THIS IS XXXXX')
    print (x)

    #breakpoint()

    ref_bnorm = ref_batchnorm(x, epsf, gammaf, betaf, mean_runf, var_runf)

    gamma = pad_weight(torch.full((1,C,H,W), gammaf))
    beta = pad_weight(torch.full((1,C,H,W), betaf))
    mean_run = pad_weight(torch.full((1,C,H,W), mean_runf))
    var_run = pad_weight(torch.full((1,C,H,W), var_runf + epsf))
    
    t0 = gpai.tensor.Tensor(tilize_to_list(x), [1, C, H, W], gpai.tensor.DataType.BFLOAT16, gpai.tensor.Layout.TILE, device)
    ttgamma = tilize_to_list(gamma)
    ttbeta = tilize_to_list(beta)
    ttmean_run = tilize_to_list(mean_run)
    ttvar_run = tilize_to_list(var_run)


    func = Batchnorm(ttmean_run, ttvar_run, ttgamma, ttbeta, C, device)

    t1 = func(t0)
    t2_data = t1.to(host).data()

    tt_got_back = torch.Tensor(t2_data).reshape((1,C,H,W))
    tt_got_back = untilize(tt_got_back)

    print ('======================')
    print (tt_got_back)
    print ( ref_bnorm[0,0,:,:])
    #print (list(torch.Tensor(ref_bnorm).reshape((H*W))))
    print ('-------++++_------------')
    
    print("Batchnorm max absdiff=")
    print_diff_argmax(tt_got_back, ref_bnorm)

    gpai.device.CloseDevice(device)
