

import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from pymetal import ttmetal as ttm
from utility_functions import tilize_to_list, print_diff_argmax, untilize, tilize, tilize_to_list
from utils import move_to_cpu, move_to_device
from python_api_testing.fused_ops.linear import Linear as TtLinear
from fused_ops.softmax import softmax as TtSoftmax
from typing import List, Optional, Tuple, Union
# from torch_utils.py for randn_tensor


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


##### randn

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape, generator=generator, device=self.parameters.device, dtype=self.parameters.dtype
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    # def nll(self, sample, dims=[1, 2, 3]):
    #     if self.deterministic:
    #         return torch.Tensor([0.0])
    #     logtwopi = np.log(2.0 * np.pi)
    #     return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)
    # Note: not used for StableDiffusion

    def mode(self):
        return self.mean



class TtDiagonalGaussianDistribution(nn.Module):
    def __init__(self, parameters, device, deterministic=False):
        super().__init__()
        # parameters is torch tensor
        self.device = device
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.mean = move_to_device(self.mean, device)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.logvar = move_to_device(self.logvar, device)
        self.deterministic = deterministic

        # self.std = torch.exp(0.5 * self.logvar)

        half_tensor = torch.ones(self.logvar.shape()) * 0.5
        half_tensor = move_to_device(half_tensor, device)
        half_logvar = ttm.tensor.mul(half_tensor, self.logvar)
        self.std = ttm.tensor.exp(half_logvar)

        self.var = ttm.tensor.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )



    def sample(self):
        # make sure sample is on the same device as the parameters and has same dtype
        sample = torch.randn(self.mean.shape())
        sample = move_to_device(sample, self.device)

        sample = ttm.tensor.mul(sample, self.std)
        sample = ttm.tensor.add(sample, self.mean)
        return sample
        # I have not managed to run this code and look at the shapes however!


    def kl(self, other=None):

        if self.deterministic:
            _r = torch.ones((1, 1, 1, 1))
            return move_to_device(_r, self.device)

        x_mean = self.mean
        if other is not None:
            x_mean = ttm.tensor.sub(self.mean, other.mean)
        x_var = self.var
        if other is not None:
            r_other_var = ttm.tensor.recip(other.var)
            x_var = ttm.tensor.mul(x_var, r_other_var)

        ones = torch.ones(self.mean.shape())
        ones = move_to_device(ones)
        # torch.pow(self.mean [-other.mean], 2)
        x_mean_2 = ttm.tensor.mul(x_mean, x_mean)
        # + self.var
        mean_var = ttm.tensor.add(x_mean_2, x_var)
        # - 1
        mean_var1 = ttm.tensor.sub(mean_var, ones)

        mean_var_logvar = ttm.tensor.sub(mean_var1, self.logvar)
        if other is not None:
            mean_var_logvar = ttm.tensor.add(mean_var_logvar, other.logvar)

        # summing over dim 1, 2, 3
        # reduce op!

        # r1 = ttm.tensor.reduce(mean_var_logvar, ttm.tensor.ReduceOpMath.SUM, ttm.tensor.ReduceOpDim.HW)
        r1 = torch.sum(move_to_cpu(mean_var_logvar), dim=[1, 2, 3])
        return move_to_device(r1, self.device)

    def mode(self):
        return self.mean




def run_diagonal_gaussian_dist_inference(device):


    input_shape = [1, 2, 32, 32]
    input = torch.randn(input_shape)

    torch_dg = DiagonalGaussianDistribution(input)
    torch_out = torch_dg.sample()

    # tt_input = ttm.tensor.Tensor(tilize_to_list(input), input_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    tt_dg = TtDiagonalGaussianDistribution(input, device)

    tt_out = tt_dg.sample().data()
    tt_out.pretty_print()

    tt_untilized_output = untilize(torch.Tensor(tt_out).reshape(torch_out.shape))
    print_diff_argmax(tt_untilized_output, torch_out)
    assert np.allclose(pytorch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)





if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_diagonal_gaussian_dist_inference(device)
    ttm.device.CloseDevice(device)
