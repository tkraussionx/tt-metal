import torch
from torch import nn
from loguru import logger
from typing import Optional, Union
from transformers import MobileNetV2Config
import tt_lib


def apply_tf_padding(features, conv_layer):
    """
    Apply TensorFlow-style "SAME" padding to a convolution layer. See the notes at:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    """
    host = tt_lib.device.GetHost()

    batch, ch, in_height, in_width = features.shape()
    c2, c1, k1, k2, s1, s2, p1, p2, d, g = conv_layer.conv_params
    stride_height, stride_width = s1, s2
    kernel_height, kernel_width = k1, k2
    dilation_height = dilation_width = d

    if in_height % stride_height == 0:
        pad_along_height = max(kernel_height - stride_height, 0)
    else:
        pad_along_height = max(kernel_height - (in_height % stride_height), 0)

    if in_width % stride_width == 0:
        pad_along_width = max(kernel_width - stride_width, 0)
    else:
        pad_along_width = max(kernel_width - (in_width % stride_width), 0)

    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top

    padding = [
        pad_left * dilation_width,
        pad_right * dilation_width,
        pad_top * dilation_height,
        pad_bottom * dilation_height,
    ]
    input_tensor_start = [
        0,
        0,
        padding[0],
        padding[2],
    ]  # top left inidices of padding is where tensor starts
    dim_y = sum(padding[:2])
    dim_x = sum(padding[2:])
    output_tensor_shape = [batch, ch, in_height + dim_y, in_width + dim_x]
    # Tensor must be on host for padding
    features = features.to(host)
    features = features.pad(output_tensor_shape, input_tensor_start, 0.0)

    return features


def make_divisible(
    value: int, divisor: int = 8, min_value: Optional[int] = None
) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def apply_depth_multiplier(config: MobileNetV2Config, channels: int) -> int:
    return make_divisible(
        int(round(channels * config.depth_multiplier)),
        config.depth_divisible_by,
        config.min_depth,
    )
