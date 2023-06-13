import torch
import numpy as np
from libs import tt_lib as ttl
from utility_functions_new import pad_by_zero, unpad_from_zero, torch2tt_tensor
from python_api_testing.fused_ops.conv import conv as TtConv
from libs.tt_lib.utils import (
    _nearest_32 as nearest_32,
)

import conv_on_device_utils_new


def deprecated(func, *args, **kwargs):
    def helper():
        logger.warning("this file is deprecated and will be remove soon")
        return func(args, kwargs)
    return helper


@deprecated
def is_conv_supported_on_device(conv_params):
    return conv_on_device_utils_new.is_conv_supported_on_device(conv_params)

@deprecated
def can_run_conv_on_device(act_shape, conv_params):
    return conv_on_device_utils_new.can_run_conv_on_device(act_shape, conv_params)

@deprecated
def run_conv_on_tt_device(x: torch.Tensor, conv_on_tt, conv_params, device, host):
    return conv_on_device_utils_new.run_conv_on_device(x, conv_on_tt, conv_params, device, host)

@deprecated
def run_conv_on_device_wrapper(conv_weight, conv_params, device, host, conv_bias=None):
    return conv_on_device_utils_new.run_conv_on_device_wrapper(conv_weight, conv_params, device, host, conv_bias)
