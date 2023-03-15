import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from pymetal import ttmetal as ttm
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax
from python_api_testing.models.utility_functions import tt2torch, tt2torch_rm
from sweep_tests import comparison_funcs

def gelu(x, stable=False):
    H, W = 64, 96

    z = x

    k1 = torch.full((1,1,H, W), 0.5)
    k1 = tilize_to_list(k1)
    k1_dev = ttm.tensor.Tensor(k1, [1, 1, H, W], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    k2 = torch.full((1,1,H, W), 0.044715)
    k2 = tilize_to_list(k2)
    k2_dev = ttm.tensor.Tensor(k2, [1, 1, H, W], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    k3 = torch.full((1,1,H, W), 0.7978845608)
    k3 = tilize_to_list(k3)
    k3_dev = ttm.tensor.Tensor(k3, [1, 1, H, W], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)


    #0.5*x
    factor1 = ttm.tensor.mul(k1_dev, z) # exp(z)

    #x*x
    pow2 = ttm.tensor.mul(z, z)

    #(x + 0.044715 * torch.pow(x, 3)))
    #torch.pow(x, 3))
    pow3 = ttm.tensor.mul(pow2, z)
    factor3 = ttm.tensor.mul(k2_dev, pow3)
    #(x + 0.044715 * torch.pow(x, 3)))
    factor3 = ttm.tensor.add(factor3, z)

    sumtanh = ttm.tensor.mul(k3_dev, factor3)
    upper_first = ttm.tensor.exp(sumtanh)

    recip_first = ttm.tensor.recip(upper_first)
    upper_el = ttm.tensor.sub(upper_first, recip_first)
    lower_el = ttm.tensor.exp(sumtanh)
    lower_el = ttm.tensor.add(upper_first, recip_first)
    lower_el = ttm.tensor.recip(lower_el)

    tanh = ttm.tensor.mul(upper_el, lower_el)

    #sumtanh_data = sumtanh.to(host).data()
    #sumtanh_got_back = torch.Tensor(sumtanh_data).reshape((1,1,H,W))

    #tanh_host = torch.tanh(sumtanh_got_back)

    #tanh_list = tilize_to_list(tanh_host)
    #tanh = gpai.tensor.Tensor(tanh_list, [1, 1, H, W], gpai.tensor.DataType.FLOAT32, gpai.tensor.Layout.TILE, device)

    k4 = torch.full((1,1,H, W), 1)
    k4 = tilize_to_list(k4)
    k4_dev = ttm.tensor.Tensor(k4, [1, 1, H, W], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)


    total = ttm.tensor.add(k4_dev, tanh)

    output = ttm.tensor.mul(factor1, total)

    return output

def ref_stable_gelu(x):
    gelu = torch.nn.GELU()
    return gelu(x)

if __name__ == "__main__":
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    H, W = 64, 96
    torch.manual_seed(123)

    max = 1
    min = -1

    x = (max-min)*torch.rand((1,1,H,W))+min
    print(x)
    ref_gelu = ref_stable_gelu(x)

    x_t = tilize_to_list(x)
    t0 = ttm.tensor.Tensor(x_t, [1, 1, H, W], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
    func = gelu
    t1 = func(t0)
    t2_data = t1.to(host).data()

    tt_got_back = torch.Tensor(t2_data).reshape((1,1,H,W))
    tt_got_back = untilize(tt_got_back)

    print("Comparison to golden outputs")
    passing, output = comparison_funcs.comp_allclose_and_pcc(ref_gelu, tt_got_back)
    print(passing)
    print(output)
    ttm.device.CloseDevice(device)
