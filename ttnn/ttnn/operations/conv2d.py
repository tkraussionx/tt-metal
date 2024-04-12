# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Dict, Optional
import warnings
import math
import ttnn
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_utils import calculate_shard_grid, roundup
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_composite_conv import (
    TTPyCompositeConv,
    SlidingWindowOpParams,
    _nearest_32,
    find_closest_common_largest_divisor,
    find_closest_largest_divisor,
    find_closest_largest_divisor_with_num_padding,
)
import ttnn.experimental


class Conv2d:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        dtype: ttnn.DataType = None,
        *,
        device: ttnn.Device,
        use_1d_systolic_array: bool,
        batch_size: int,
        input_height: int,
        input_width: int,
        reader_patterns_cache: Optional[Dict],
        weight: ttnn.Tensor,
        bias: ttnn.Tensor = None,
        math_fidelity: ttnn.MathFidelity = None,
        weights_dtype: ttnn.DataType = None,
        activation: str = None,
        conv_blocking_and_parallelization_config_override: Dict = None,
        reallocate_halo_output: bool = False,
        using_parameters_cache: bool = False,
        move_weights_to_device: bool = True,
        use_shallow_conv_variant: bool = False,
        enable_auto_formatting: bool = False,
        deallocate_activation: bool = False,
        padded_input_channels: Optional[int] = None,
        compute_kernel_config: Union[ttnn.GrayskullComputeKernelConfig, ttnn.WormholeComputeKernelConfig] = None,
        use_dram_for_matmul: bool = False,
        output_layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    ):
        assert (
            padding_mode == "zeros"
        ), f"Only convs with padding_mode=zeroes supported. Found padding_mode set to {padding_mode}."
        if isinstance(kernel_size, int):
            window_h = kernel_size
            window_w = kernel_size
        else:
            window_h, window_w = kernel_size

        if isinstance(stride, int):
            stride_h = stride
            stride_w = stride
        else:
            stride_h, stride_w = stride

        if isinstance(padding, int):
            pad_h = padding
            pad_w = padding
        else:
            pad_h, pad_w = padding

        if isinstance(dilation, int):
            dilation_h = dilation
            dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        assert dilation_h == 1, f"Only convs with dilation == 1 supported. Found dilation_h={dilation_h}"
        assert dilation_w == 1, f"Only convs with dilation == 1 supported. Found dilation_w={dilation_w}"
        assert groups == 1, "Only convs with groups == 1 supported"
        sliding_window_op_params = SlidingWindowOpParams(
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
            window_h=window_h,
            window_w=window_w,
            batch_size=batch_size,
            input_h=input_height,
            input_w=input_width,
        )
        fuse_relu = False
        if activation is not None:
            activation = activation.lower()
            assert activation == "relu", f"Only support relu fusion with conv. Got activation={activation}."
            fuse_relu = True
        self.conv = TTPyCompositeConv(
            sliding_window_op_params,
            weight,
            out_channels,
            in_channels,
            device,
            use_1d_systolic_array,
            reader_patterns_cache,
            bias=bias,
            conv_blocking_and_parallelization_config_override=conv_blocking_and_parallelization_config_override,
            fuse_relu=fuse_relu,
            output_dtype=dtype,
            weights_dtype=weights_dtype,
            math_fidelity=math_fidelity,
            move_utwh_output=reallocate_halo_output,
            using_parameters_cache=using_parameters_cache,
            move_weights_to_device=move_weights_to_device,
            use_shallow_conv_variant=use_shallow_conv_variant,
            enable_auto_formatting=enable_auto_formatting,
            deallocate_activation=deallocate_activation,
            padded_input_channels=padded_input_channels,
            compute_kernel_config=compute_kernel_config,
            use_dram_for_matmul=use_dram_for_matmul,
            output_layout=output_layout,
        )
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = (input_height + (2 * pad_h) - dilation_h * (window_h - 1) - 1) // stride_h + 1
        self.output_width = (input_width + (2 * pad_w) - dilation_w * (window_w - 1) - 1) // stride_w + 1
        self.in_channels = in_channels
        self.out_channels = out_channels

    @ttnn.register_operation(
        name="ttnn.Conv2d.__call__", validate_input_tensors=lambda *args, **kwargs: None, is_method=True
    )
    def __call__(self, activation: ttnn.Tensor):
        return self.conv(activation)

    @ttnn.register_operation(
        name="ttnn.Conv2d.copy_input_to_device", validate_input_tensors=lambda *args, **kwargs: None, is_method=True
    )
    def copy_input_to_device(self, input: ttnn.Tensor):
        return self.conv.copy_input_to_device(input)

    @ttnn.register_operation(
        name="ttnn.Conv2d.copy_output_from_device", validate_input_tensors=lambda *args, **kwargs: None, is_method=True
    )
    def copy_output_from_device(self, output: ttnn.Tensor):
        return self.conv.copy_output_from_device(output)

    def get_parallel_config(self):
        return self.conv.get_parallel_config()


class ConvKernelBlockConfig:
    def __init__(self, act_block_h):
        self.act_block_h = act_block_h


class ParallelConfig:
    def __init__(
        self,
        num_cores_y: int,
        num_cores_x: int,
        num_cores_nhw: int,
        shard_scheme: ttnn.TensorMemoryLayout,
        shard_orientation: ttnn.ShardOrientation,
    ):
        # TODO: using core range set would be better
        self.grid_size = ttnn.experimental.tensor.CoreCoord(num_cores_x, num_cores_y)
        self.num_cores_nhw = num_cores_nhw
        self.shard_scheme = shard_scheme
        self.shard_orientation = shard_orientation


def determine_parallel_config(
    shard_scheme,
    batch_size,
    input_channels,
    output_height,
    output_width,
    output_channels,
    device,
    config_override=None,
    is_out_tiled=True,
):
    is_1d_systolic = shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    if config_override is None:
        config_override = {}
    for k in config_override.keys():
        assert k == "grid_size" or k == "num_cores_nhw"

    conv_out_2d_matrix_height = batch_size * output_height * output_width
    # pad height to 32
    conv_out_2d_matrix_height = _nearest_32(conv_out_2d_matrix_height)
    if is_out_tiled:
        conv_out_2d_matrix_height_ntiles = (int)(conv_out_2d_matrix_height / 32)
        conv_out_2d_matrix_width_ntiles = (int)(_nearest_32(output_channels) / 32)
    else:
        conv_out_2d_matrix_height_ntiles = conv_out_2d_matrix_height
        conv_out_2d_matrix_width_ntiles = output_channels

    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    device_grid_size = (compute_with_storage_grid_size.x, compute_with_storage_grid_size.y)
    max_num_cores = device_grid_size[0] * device_grid_size[1]

    def calculate_num_cores_nhw(override):
        num_cores_nhw = (
            find_closest_largest_divisor(conv_out_2d_matrix_height_ntiles, max_num_cores)
            if is_1d_systolic
            else find_closest_largest_divisor_with_num_padding(conv_out_2d_matrix_height_ntiles, device_grid_size[0])
        )
        if override is not None and num_cores_nhw != override:
            warnings.warn(f"Overriding config: num_cores_nhw from {num_cores_nhw} to user provided config={override}")
            num_cores_nhw = override
        return num_cores_nhw

    def calculate_grid_size(num_cores_nhw, override):
        if is_1d_systolic:
            grid_size = [
                device_grid_size[0] if num_cores_nhw >= device_grid_size[0] else num_cores_nhw,
                math.ceil(num_cores_nhw / device_grid_size[0]),
            ]  # for 1d systolic array, grid size is the tightest bound of num_cores_nhw as a rectangle (x,y)
            assert (
                num_cores_nhw <= grid_size[0] * grid_size[1]
            ), "Error: For 1d systolic conv, num_cores_nhw must be <= grid size"
        else:
            grid_size = [
                num_cores_nhw,
                find_closest_common_largest_divisor(
                    conv_out_2d_matrix_width_ntiles, _nearest_32(input_channels) // 32, device_grid_size[1]
                ),
            ]
            assert (
                num_cores_nhw == grid_size[0]
            ), "Error: For 2d systolic conv, num_cores_nhw must be == # of cols in grid size"

        if override is not None and grid_size != override:
            warnings.warn(f"Overriding config: grid_size from {grid_size} to user provided config={override}")
            grid_size = override
        return grid_size

    num_cores_nhw = calculate_num_cores_nhw(config_override.get("num_cores_nhw", None))
    grid_size = calculate_grid_size(num_cores_nhw, config_override.get("grid_size", None))
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR if is_1d_systolic else ttnn.ShardOrientation.COL_MAJOR
    return ParallelConfig(grid_size[1], grid_size[0], num_cores_nhw, shard_scheme, shard_orientation)


# todo: there are different versions of this function. Commonize.
def create_sharded_memory_config_from_parallel_config(tensor_shape, parallel_config, tile_size):
    # tensor_shape is [N, H, W, C]
    assert len(tensor_shape) == 4
    assert tensor_shape[0] == 1 and tensor_shape[1] == 1  # todo: add support for generic non-2d shapes
    channels = tensor_shape[3]
    num_cores_nhw = parallel_config.num_cores_nhw
    num_cores_x = parallel_config.grid_size.x
    num_cores_y = parallel_config.grid_size.y
    shard_scheme = parallel_config.shard_scheme
    shard_orientation = parallel_config.shard_orientation
    is_1d_systolic = shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    if is_1d_systolic:
        logical_grid_size = (num_cores_nhw, 1)
    else:
        logical_grid_size = (num_cores_x, num_cores_y)

    shard_grid, shard_layout = calculate_shard_grid((num_cores_x, num_cores_y), num_cores_nhw)
    assert shard_layout == shard_scheme
    nhw_shape = tensor_shape[0] * tensor_shape[1] * tensor_shape[2]
    nhw_padded = roundup(nhw_shape, num_cores_nhw * tile_size)
    nhw_shard = nhw_padded // num_cores_nhw
    assert channels % logical_grid_size[1] == 0
    shard_shape = [nhw_shard, channels // logical_grid_size[1]]
    shard_halo = False
    shard_spec = ttnn.experimental.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)
    return ttnn.MemoryConfig(shard_scheme, ttnn.BufferType.L1, shard_spec)


@ttnn.register_operation(name="ttnn.conv2d", is_cpp_function=True)
def conv2d(
    input_tensor: ttnn.Tensor,
    weight_tensor: ttnn.Tensor,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    input_height: int,
    input_width: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    dtype: ttnn.DataType = None,
    *,
    device: ttnn.Device,
    bias_tensor: Optional[ttnn.Tensor] = None,
    shard_scheme: Optional[ttnn.TensorMemoryLayout] = None,
    parallel_config: Optional[ParallelConfig] = None,
    math_fidelity: ttnn.MathFidelity = None,
    weights_dtype: ttnn.DataType = None,
    conv_kernel_block_config_override: Optional[ConvKernelBlockConfig],
    compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
    activation: str = None,
) -> ttnn.Tensor:
    output_height = ((int)((input_height - kernel_size[0] + 2 * padding[0]) / stride[0])) + 1
    output_width = ((int)((input_width - kernel_size[1] + 2 * padding[1]) / stride[1])) + 1
    if shard_scheme is not None:
        parallel_config = ttnn.determine_parallel_config(
            shard_scheme, batch_size, in_channels, output_height, output_width, out_channels, device
        )
        input_tensor_sharded_memory_config_new = create_sharded_memory_config_from_parallel_config(
            [batch_size, input_height, input_width, in_channels], parallel_config, tile_size=32
        )
        print("Running reshard or interleaved to sharded op on input tensor")
        input_tensor = input_tensor.to_memory_config(input_tensor_sharded_memory_config_new)
    # if parallel_config is None:
    # input_memory_config =
    block_and_parallel_config_override = {}
    if conv_kernel_block_config_override is not None:
        block_and_parallel_config_override["act_block_h"] = conv_kernel_block_config_override.act_block_h
    if parallel_config is not None:
        block_and_parallel_config_override["grid_size"] = [parallel_config.grid_size.x, parallel_config.grid_size.y]
        block_and_parallel_config_override["num_cores_nhw"] = parallel_config.num_cores_nhw

    # assert (
    #     shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    #     or shard_scheme == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    # )
    # if shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
    #     assert shard_orientation == ttnn.ShardOrientation.ROW_MAJOR
    # else:
    #     assert shard_orientation == ttnn.ShardOrientation.COL_MAJOR
    # if parallel_config is not None:
    #     # input and output should have same shard strategy and orientation
    #     # halo doesnt support reshard between different strategies or orientation
    #     assert shard_scheme == parallel_config.shard_scheme
    #     assert shard_orientation == parallel_config.shard_orientation

    # Build conv op object
    conv = ttnn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dtype=dtype,
        device=device,
        use_1d_systolic_array=shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        reader_patterns_cache={},
        weight=weight_tensor,
        bias=bias_tensor,
        math_fidelity=math_fidelity,
        weights_dtype=weights_dtype,
        conv_blocking_and_parallelization_config_override=block_and_parallel_config_override,
        compute_kernel_config=compute_kernel_config,
        activation=activation,
    )
    # Run conv
    print("Running halo op followed by conv op")
    return conv(input_tensor)


__all__ = []
