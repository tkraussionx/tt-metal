# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional

import sys

import ttnn

import ttnn._ttnn.deprecated as ttl

__all__ = []


def apply_activations(tensor, activations):
    import torch

    string_to_function = {
        "relu": torch.relu,
        "gelu": torch.nn.functional.gelu,
        "silu": torch.nn.functional.silu,
    }

    if activations is not None:
        for activation in activations:
            activation_function = string_to_function[activation]
            tensor = activation_function(tensor)
    return tensor


def _golden_function(input_tensor_a, input_tensor_b, *args, activations=None, **kwargs):
    output_tensor = input_tensor_a + input_tensor_b
    return apply_activations(output_tensor, activations)


ttnn.attach_golden_function(ttnn.add, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.add_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, activations=None, **kwargs):
    output_tensor = input_tensor_a - input_tensor_b
    return apply_activations(output_tensor, activations)


ttnn.attach_golden_function(ttnn.subtract, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.subtract_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, activations=None, **kwargs):
    output_tensor = input_tensor_a * input_tensor_b
    return apply_activations(output_tensor, activations)


ttnn.attach_golden_function(ttnn.multiply, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.multiply_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.eq(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.eq, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.ne(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.ne, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.gt(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.gt, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.ge(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.ge, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.lt(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.lt, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.le(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.le, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.logical_and(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.logical_and, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.logical_or(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.logical_or, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.ldexp(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.ldexp, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.logaddexp(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.logaddexp, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.logaddexp2(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.logaddexp2, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.divide(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.divide, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.nn.functional.gelu(torch.add(x, y))


ttnn.attach_golden_function(ttnn.bias_gelu, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.squared_difference(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.squared_difference, golden_function=_golden_function)


def _golden_function_addalpha(input_tensor_a, input_tensor_b, alpha, *args, **kwargs):
    import torch

    return torch.add(input_tensor_a, input_tensor_b, alpha=alpha)


ttnn.attach_golden_function(ttnn._ttnn.operations.binary.addalpha, golden_function=_golden_function_addalpha)


def _golden_function_subalpha(input_tensor_a, input_tensor_b, alpha, *args, **kwargs):
    import torch

    return torch.sub(input_tensor_a, input_tensor_b, alpha=alpha)


ttnn.attach_golden_function(ttnn._ttnn.operations.binary.subalpha, golden_function=_golden_function_subalpha)


def _golden_function_xlogy(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.xlogy(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn._ttnn.operations.binary.xlogy, golden_function=_golden_function_xlogy)


def _golden_function_hypot(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.hypot(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn._ttnn.operations.binary.hypot, golden_function=_golden_function_hypot)


def _golden_function_maximum(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.maximum(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn._ttnn.operations.binary.maximum, golden_function=_golden_function_maximum)


def _golden_function_minimum(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.minimum(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn._ttnn.operations.binary.minimum, golden_function=_golden_function_minimum)


def _golden_function_logical_xor(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.logical_xor(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn._ttnn.operations.binary.logical_xor, golden_function=_golden_function_logical_xor)


def _golden_function_atan2(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.atan2(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn._ttnn.operations.binary.atan2, golden_function=_golden_function_atan2)


def _golden_function_div(input_tensor_a, input_tensor_b, round_mode, *args, **kwargs):
    import torch

    if round_mode == "None":
        return torch.div(input_tensor_a, input_tensor_b, rounding_mode=None)
    return torch.div(input_tensor_a, input_tensor_b, rounding_mode=round_mode)


ttnn.attach_golden_function(ttnn._ttnn.operations.binary.div, golden_function=_golden_function_div)


def torch_squared_difference(x, y, *args, **kwargs):
    import torch

    return torch.square(torch.sub(x, y))


def register_ttl_elt_binary_function(name, ttl_elt_binary_function, op_name):
    doc = f"""{name}(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

            Performs eltwise-binary {op_name} operation on two tensors :attr:`input_a` and :attr:`input_b`.

            .. math::
                {name.replace('_',' ')}(\\mathrm{{input\\_tensor\\_a}}_i \\; , \\; \\mathrm{{input\\_tensor\\_b}}_i )

            Args:
                * :attr:`input_tensor_a`
                * :attr:`input_tensor_b`

            Example::
                >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]]), dtype=torch.bfloat16)), device)
                >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 1], [4, 4]]), dtype=torch.bfloat16)), device)
                >>> output = ttnn.{name}(tensor1, tensor2)
            """

    @ttnn.register_python_operation(name=f"ttnn.{name}", golden_function=_golden_function, doc=doc)
    def elt_binary_function(
        input_tensor_a: ttnn.Tensor,
        input_tensor_b: Union[ttnn.Tensor, int, float],
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        if not (input_tensor_a.shape == input_tensor_b.shape):
            raise RuntimeError("input_tensors must be of same size!")

        if not isinstance(input_tensor_a, ttnn.Tensor) or not isinstance(input_tensor_b, ttnn.Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if not ttnn.is_tensor_storage_on_device(input_tensor_a) or not ttnn.is_tensor_storage_on_device(input_tensor_b):
            raise RuntimeError("input_tensors must be on device!")

        original_shape = input_tensor_a.shape

        input_tensor_a = ttnn.unsqueeze_to_4D(input_tensor_a)
        input_tensor_b = ttnn.unsqueeze_to_4D(input_tensor_b)

        output_tensor = ttl_elt_binary_function(input_tensor_a, input_tensor_b, output_mem_config=memory_config)

        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor


def _golden_function(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, **_):
    import torch

    return torch.nextafter(input_tensor_a, input_tensor_b)


@ttnn.register_python_operation(
    name="ttnn.nextafter",
    golden_function=_golden_function,
)
def nextafter(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: Optional[ttnn.DataType] = None,
) -> ttnn.Tensor:
    r"""
    nextafter(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

    Returns the next floating-point value after input_a towards input_b of the input tensors input_a and input_b.

    .. math::
        \mathrm{{input\_tensor\_a}}_i , \mathrm{{input\_tensor\_b}}_i

    Args:
        * :attr:`input_tensor_a`
        * :attr:`input_tensor_b`

    Keyword args:
        :attr:`memory_config` (ttnn.MemoryConfig): memory config for the output tensor
        :attr:`dtype` (Optional[ttnn.DataType]): data type for the output tensor


    """

    output = ttnn.experimental.tensor.nextafter(
        input_tensor_a,
        input_tensor_b,
        output_mem_config=memory_config,
    )
    return output


def torch_polyval(input_tensor, coeff):
    curVal = 0
    for curValIndex in range(len(coeff) - 1):
        curVal = (curVal + coeff[curValIndex]) * input_tensor[0]
    return curVal + coeff[len(coeff) - 1]


def _golden_function(input_tensor: ttnn.Tensor, coeff: List[float], **_):
    return torch_polyval(input_tensor, coeff)


@ttnn.register_python_operation(
    name="ttnn.polyval",
    golden_function=_golden_function,
)
def polyval(
    input_tensor: ttnn.Tensor,
    coeff: List[float],
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: Optional[ttnn.DataType] = None,
) -> ttnn.Tensor:
    r"""
    polyval(input_tensor_a: ttnn.Tensor, coeff: List[float], *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

    Returns tensor with the polyval of all of elements of the input tensor input with coefficients coeffs.

    .. math::
        \mathrm{{input\_tensor\_a}}_i , \mathrm{{coeff}}_i

    Args:
        * :attr:`input_tensor_a`
        * :attr:`coeff`

    Keyword args:
        :attr:`memory_config`
        :attr:`dtype`


    """

    output = ttnn.experimental.tensor.polyval(
        input_tensor,
        coeff,
        output_mem_config=memory_config,
    )
    return output


def _golden_function(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    param1: float = 1e-05,
    param2: float = 1e-08,
    equal_nan: bool = False,
    **_,
):
    import torch

    return torch.isclose(input_tensor_a, input_tensor_b, rtol=param1, atol=param2, equal_nan=equal_nan)


@ttnn.register_python_operation(
    name=f"ttnn.isclose",
    golden_function=_golden_function,
)
def isclose(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """isclose(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Applies the isclose function to the elements of the input tensor :attr:`input_a` and :attr:`input_b`.

    isclose(input_a, input_b, rtol, atol) = ∣input_a−input_B∣ ≤ atol+rtol×∣input_b∣.

    .. math::
        ttnn.isclose(\\mathrm{{input\\_tensor\\_a}}_i \\; , \\; \\mathrm{{input\\_tensor\\_b}}_i  \\; , \\; \\mathrm{{atol}}\\; , \\; \\mathrm{{rtol}})

    Args:
        * :attr:`input_tensor_a`
        * :attr:`input_tensor_b`



    Example::
        >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]]), dtype=torch.bfloat16)), device)
        >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1 + 1e-10, 1], [4, 4 + 1e-10]]), dtype=torch.bfloat16)), device)
        >>> output = ttnn.isclose(tensor1, tensor2, rtol, atol)
    """
    return ttl.tensor.isclose(input_tensor_a, input_tensor_b, rtol, atol, equal_nan, output_mem_config=memory_config)


__all__ = []
