import torch

import tt_lib

import python_api_testing.models.nanogpt.nanogpt_utils as nanogpt_utils


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def tt_new_gelu(x, device):
    z = x

    k1 = torch.full(x.shape(), 0.5)
    tt_k1 = bloom_utils.torch2tt_tensor(k1, device)

    k2 = torch.full(x.shape(), 0.044715)
    tt_k2 = bloom_utils.torch2tt_tensor(k2, device)

    res = tt_lib.tensor.sqrt()

    k3 = torch.full(x.shape(), 0.79788456)
    tt_k3 = bloom_utils.torch2tt_tensor(k3, device)

    #0.5*x
    factor1 = tt_lib.tensor.mul(tt_k1, z) # exp(z)

    #x*x
    pow2 = tt_lib.tensor.mul(z, z)

    #(x + 0.044715 * torch.pow(x, 3)))
    #torch.pow(x, 3))
    pow3 = tt_lib.tensor.mul(pow2, z)
    factor3 = tt_lib.tensor.mul(tt_k2, pow3)

    #(x + 0.044715 * torch.pow(x, 3)))
    factor3 = tt_lib.tensor.add(factor3, z)

    sumtanh = tt_lib.tensor.mul(tt_k3, factor3)
    tanh = tt_lib.tensor.tanh(sumtanh)

    k4 = torch.full(x.shape(), 1.0)
    tt_k4 = bloom_utils.torch2tt_tensor(k4, device)

    total = tt_lib.tensor.add(tt_k4, tanh)
    output = tt_lib.tensor.mul(factor1, total)

    return output
