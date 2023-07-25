import math

import time
import torch
import numpy as np
from loguru import logger
from tests.python_api_testing.models.utility_functions_new import (
    is_close,
    get_oom_of_float,
    enable_compile_cache,
    disable_compile_cache,
    get_compile_cache_enabled,
    enable_compilation_reports,
    disable_compilation_reports,
    enable_memory_reports,
    disable_memory_reports,
    comp_allclose,
    comp_pcc,
    comp_allclose_and_pcc,
    Profiler,
    profiler,
    tt_to_torch_tensor,
    torch_to_tt_tensor,
    torch_to_tt_tensor_rm,
    unpad_from_zero,
    pad_by_zero,
    tt2torch_tensor,
    torch2tt_tensor
)

import tt_lib as ttl
from tt_lib.utils import (
    _nearest_32 as nearest_32,
    pad_activation,
    pad_weight,
    tilize,
    tilize_to_list,
    untilize,
    print_diff_argmax,
    tt2torch,
    tt2torch_rm,
    roundup,
    roundup32,
    float_to_bits,
    divup,
    channels_last,
    convert_weights_2d_matrix,
)





# def print_diff_tt_pyt(a, b, annotation=""):
#     # first convert a pytorch tensor argument b to tt
#     padded_b = pad_weight(b)
#     pyt_a = tt2torch(a)  # untilizes also
#     return print_diff_argmax(pyt_a, padded_b, annotation)




def ttP(x, count=4, offset=0, stride=1):
    if type(x) == torch.Tensor:
        t1 = x.reshape(-1)
    else:
        host = ttl.device.GetHost()
        shp = x.shape()
        tt_out = x.to(host)
        torch_out = untilize(torch.Tensor(tt_out.data()).reshape(shp))
        t1 = torch_out.reshape(-1)
    print("Tensor vals: (", end="")
    for j in range(offset, offset + count * stride, stride):
        print(t1[j].item(), " ", end="")
    print(")")


def enable_persistent_kernel_cache():
    """
    Enables persistent compiled kernel caching - disables recompiling the kernels for the duration of running process if built/kernels/.../hash directory with kernel binaries is present.
    """
    ttl.device.EnablePersistentKernelCache()


def disable_persistent_kernel_cache():
    """
    Disables persistent compiled kernel caching. This is the default state.
    """
    ttl.device.DisablePersistentKernelCache()

def enable_compilation_reports():
    """
    Enables generating reports of compilation statistics in .reports/tt_metal dir
    """
    return ttl.device.EnableCompilationReports()













def torch2tt_tensor(
    py_tensor: torch.Tensor,
    tt_device,
    tt_layout=ttl.tensor.Layout.TILE,
    tt_memory_config=ttl.tensor.MemoryConfig(True),
):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = (
        ttl.tensor.Tensor(
            py_tensor.reshape(-1).tolist(),
            size,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(tt_layout)
        .to(tt_device, tt_memory_config)
    )

    return tt_tensor


def tt2torch_tensor(tt_tensor, tt_host=None):
    if tt_host == None:
        host = ttl.device.GetHost()
    else:
        host = tt_host
    tt_output = tt_tensor.to(host)
    if tt_output.layout() != ttl.tensor.Layout.ROW_MAJOR:
        tt_output = tt_output.to(ttl.tensor.Layout.ROW_MAJOR)
    dtype = {
        ttl.tensor.DataType.FLOAT32:   torch.float,
        ttl.tensor.DataType.BFLOAT16:  torch.bfloat16,
        ttl.tensor.DataType.BFLOAT8_B: torch.float,
    }[tt_tensor.dtype()]

    py_output = torch.frombuffer(tt_output.data(), dtype=dtype).reshape(tt_output.shape())
    return py_output


def pad_by_zero(x: torch.Tensor, device):
    initial_shape = x.shape
    if initial_shape[3] % 32 != 0 or initial_shape[2] % 32 != 0:
        x = tt_lib.tensor.Tensor(x.contiguous().to(torch.bfloat16))
        x = x.pad(
            (initial_shape[0], initial_shape[1], nearest_32(initial_shape[2]), nearest_32(initial_shape[3])),
            (0, 0, 0, 0),
            0,
        )
        x = x.to(ttl.tensor.Layout.TILE).to(device)

    else:
        x = torch2tt_tensor(x, device)
    return x, initial_shape


def unpad_from_zero(x, desired_shape, host):
    if x.shape()[-1] == desired_shape[-1] and x.shape()[-2] == desired_shape[-2]:
        x = tt2torch_tensor(x)
    else:
        x = x.to(host)
        if x.layout() != ttl.tensor.Layout.ROW_MAJOR:
            x = x.to(ttl.tensor.Layout.ROW_MAJOR)
        x = x.unpad(
            (0, 0, 0, 0),
            (
                desired_shape[0] - 1,
                desired_shape[1] - 1,
                desired_shape[2] - 1,
                desired_shape[3] - 1,
            ),
        )
        dtype = {
            ttl.tensor.DataType.FLOAT32:   torch.float,
            ttl.tensor.DataType.BFLOAT16:  torch.bfloat16,
            ttl.tensor.DataType.BFLOAT8_B: torch.float,
        }[x.dtype()]

        x = torch.frombuffer(x.data(), dtype=dtype).reshape(x.shape())
    return x


def torch_to_tt_tensor_rm(py_tensor, device, shape=None, put_on_device=True):
    if shape is None:
        shape = list(py_tensor.size())
        while len(shape) < 4:
            shape.insert(0, 1)

    tt_tensor = ttl.tensor.Tensor(
        py_tensor.reshape(-1).tolist(),  # PyTorch tensor flatten into a list of floats
        shape,  # shape of TT Tensor that will be created
        ttl.tensor.DataType.BFLOAT16,  # data type that will be used in created TT Tensor
        ttl.tensor.Layout.ROW_MAJOR,  # memory layout that will be used in created TT Tensor
    )
    if put_on_device:
        tt_tensor = tt_tensor.to(device)
    return tt_tensor


def torch_to_tt_tensor(py_tensor, device):
    shape = list(py_tensor.size())
    while len(shape) < 4:
        shape.insert(0, 1)

    tt_tensor = (
        ttl.tensor.Tensor(
            py_tensor.reshape(
                -1
            ).tolist(),  # PyTorch tensor flatten into a list of floats
            shape,  # shape of TT Tensor that will be created
            ttl.tensor.DataType.BFLOAT16,  # data type that will be used in created TT Tensor
            ttl.tensor.Layout.ROW_MAJOR,  # memory layout that will be used in created TT Tensor
        )
        .to(
            ttl.tensor.Layout.TILE
        )  # change memory layout of TT Tensor to TILE (as operation that will use it expects TILE layout)
        .to(
            device
        )  # move TT Tensor from host to TT accelerator device (device is of type tt_lib.device.Device)
    )

    return tt_tensor
