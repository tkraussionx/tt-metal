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
from torch.autograd import Function, Variable

from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, pad_weight
torch.set_printoptions(threshold=10_000)

from tests.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_allclose_and_pcc
from tests.python_api_testing.models.revnet.block import TT_residual, REF_residual


## input is a TT tensor that is changed to torch tensor
def TT_forward(tt_x, in_channels, out_channels,
            stride, padding, dilation,
            f_mean_run, f_var_run, f_gamma, f_beta,
            g_mean_run, g_var_run, g_gamma, g_beta,
            matmul,
            device,
            host,
            C,H,W
            ):
    print ('_forward')

    x = tt_x.to(host).data()
    x = torch.Tensor(x).reshape((1,C,H,W))
    x = untilize(x)

    x1, x2 = torch.chunk(x, 2, dim=1)

    with torch.no_grad():
        x1 = Variable(x1.contiguous())
        x2 = Variable(x2.contiguous())

        x1_ = possible_downsample(x1, in_channels, out_channels, stride,
                                padding, dilation)
        x2_ = possible_downsample(x2, in_channels, out_channels, stride,
                                padding, dilation)

        #tt_x1 = tt_lib.tensor.Tensor(tilize_to_list(x1),  [1, x1.size()[1], H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
        tt_x2  = tt_lib.tensor.Tensor(tilize_to_list(x2),  [1, x2.size()[1], H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
        tt_x1_ = tt_lib.tensor.Tensor(tilize_to_list(x1_), [1, x1_.size()[1], H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
        tt_x2_ = tt_lib.tensor.Tensor(tilize_to_list(x2_), [1, x2_.size()[1], H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)

        tt_f_x2 = TT_residual(
            tt_x2,
            matmul,
            f_mean_run,
            f_var_run,
            f_gamma,
            f_beta,
            C = x2.size()[1],
            device=device
        )

        tt_y1 = tt_lib.tensor.add(tt_f_x2, tt_x1_)

        tt_g_y1 = TT_residual(
            tt_y1,
            matmul,
            g_mean_run,
            g_var_run,
            g_gamma,
            g_beta,
            C = x1.size()[1],
            device=device
        )

        tt_y2 = tt_lib.tensor.add(tt_g_y1, tt_x2_)

        y1 = tt_y1.to(host).data()
        y1 = torch.Tensor(y1).reshape((1,x1_.size()[1],H,W))
        y1 = untilize(y1)

        y2 = tt_y2.to(host).data()
        y2 = torch.Tensor(y2).reshape((1,x2_.size()[1],H,W))
        y2 = untilize(y2)

        y = torch.cat([y1, y2], dim=1)

        ## Returns a torch.tensor, NOT TT TENSOR

        del y1, y2
        del x1, x2

    return y

####### REFERENCE FUNCTION ###########

## input is a TT tensor that is changed to torch tensor
def REF_forward(x, in_channels, out_channels,
            stride, padding, dilation,
            f_mean_run, f_var_run, f_gamma, f_beta,
            g_mean_run, g_var_run, g_gamma, g_beta,
            matmul,
            eps
            ):
    print ('_forward')

    x1, x2 = torch.chunk(x, 2, dim=1)

    with torch.no_grad():
        x1 = Variable(x1.contiguous())
        x2 = Variable(x2.contiguous())

        x1_ = possible_downsample(x1, in_channels, out_channels, stride,
                                padding, dilation)
        x2_ = possible_downsample(x2, in_channels, out_channels, stride,
                                padding, dilation)

        f_x2 = REF_residual(
            x2,
            matmul,
            f_mean_run,
            f_var_run,
            f_gamma,
            f_beta,
            eps
        )

        y1 = f_x2 + x1_

        g_y1 = REF_residual(
            y1,
            matmul,
            g_mean_run,
            g_var_run,
            g_gamma,
            g_beta,
            eps
        )

        y2 = g_y1 + x2_

        y = torch.cat([y1, y2], dim=1)

        ## Returns a torch.tensor, NOT TT TENSOR

        del y1, y2
        del x1, x2

    return y


#######################
## Support Functions ##
#######################

def possible_downsample(x, in_channels, out_channels, stride=1, padding=1,
                        dilation=1):
    _, _, H_in, W_in = x.size()

    _, _, H_out, W_out = size_after_residual(x.size(), out_channels, 3, stride, padding, dilation)


    ## Shrink size of skipped input to match F(x) or G(x) size.
    ## TODO: Does this happen on device?
    # Downsample image
    if H_in > H_out or W_in > W_out:
        out = F.avg_pool2d(x, 2*dilation+1, stride, padding)


    ## Pad skipped input with zero tensors to match F(x) or G(x).
    ## TODO: Does this happen on device?
    # Pad with empty channels
    if in_channels < out_channels:

        try: out
        except: out = x

        pad = Variable(torch.zeros(
            out.size(0),
            (out_channels - in_channels) // 2,
            out.size(2), out.size(3)
        ), requires_grad=True)

        temp = torch.cat([pad, out], dim=1)
        out = torch.cat([temp, pad], dim=1)

    # If we did nothing, add zero tensor, so the output of this function
    # depends on the input in the graph
    ## TODO: Does this happen on device?
    try: out
    except:
        injection = Variable(torch.zeros_like(x.data), requires_grad=True)

#        if CUDA:
#            injection.cuda()

        out = x + injection

    return out


# Example call:
# _, _, H_out, W_out = size_after_residual(x.size(), out_channels, 3, stride, padding, dilation)

def size_after_residual(size, out_channels, kernel_size, stride, padding, dilation):
    """Calculate the size of the output of the residual function
    """
    N, C_in, H_in, W_in = size

    H_out = math.floor(
        (H_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1
    )
    W_out = math.floor(
        (W_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1
    )
    return N, out_channels, H_out, W_out

####################################################################

if __name__ == "__main__":
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

    # TT_forward returns a torch tensor for now.
    #t2_data = t1.to(host).data()
    #tt_got_back = torch.Tensor(t2_data).reshape((1,C,H,W))
    #tt_got_back = untilize(tt_got_back)


    print ('=========COMPLETE=============')
    print ("GOLDEN PCC TEST")
    #print (comp_pcc(ref_fwd, t1, pcc=.99))
    print (comp_allclose_and_pcc(ref_fwd, t1))

    tt_lib.device.CloseDevice(device)

######### REFERENCES

'''
x,
in_channels,
out_channels,
training      bool
stride,       conv
padding,      conv
dilation,     conv
f_params,     BN1
f_buffs,      BN1
g_params,     BN2
g_buffs,      BN2
no_activation=False

F.batch_norm(out,   buffers[0],   buffers[1],  params[0],   params[1],  training)
F.batch_norm(out,   buffers[-2],  buffers[-1], params[-4],  params[-3], training)
f.batch_norm(input, running_mean, running_var, weight=None, bias=None,  training=False, momentum=0.1, eps=1e-05)

F.conv2d(out,   params[-6], params[-5], stride,   padding=padding, dilation=dilation   )
F.conv2d(out,   params[-2], params[-1], stride=1, padding=1,       dilation=1          )
f.conv2d(input, weight,     bias=None,  stride=1, padding=0,       dilation=1, groups=1)

buffers[0], buffers[1],  BN1
buffers[-2],buffers[-1], BN2
params[0],  params[1],  BN1
params[-6], params[-5],  conv1
params[-4], params[-3],  BN2
params[-2], params[-1], conv2
'''
