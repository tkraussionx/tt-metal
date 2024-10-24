# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import pytest
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger
import torch.nn.functional as F
from models.utility_functions import is_wormhole_b0

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_ttnn,
)


def get_torch_dtype(dtype):
    if dtype == ttnn.int32:
        return torch.int32
    elif dtype == ttnn.float32:
        return torch.float32
    else:
        return torch.bfloat16


def run_moreh_logsoftmax_test(
    shape,
    dim,
    ttnn_dtype,
    layout,
    device,
    rtol,
    atol,
    use_randint,
    multiple_of_32,
    optional_output_tensor=False,
    strategy=None,
    compute_kernel_options=None,
):
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip(f"bfloat8_b is not supported")
    torch_dtype = get_torch_dtype(ttnn_dtype)
    if use_randint == True:
        torch_input = torch.randint(low=0, high=4, size=shape).to(torch_dtype) + 100
    else:
        torch_input = torch.rand(size=shape, dtype=torch_dtype) + 100
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=layout, device=device)

    torch_output = F.log_softmax(torch_input, dim)

    if optional_output_tensor == True:
        optional_output = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=layout, device=device)
        ttnn_output = ttnn.operations.moreh.logsoftmax(ttnn_input, dim, output_tensor=optional_output)
    elif compute_kernel_options is not None:
        compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
        if strategy is None:
            ttnn_output = ttnn.operations.moreh.logsoftmax(ttnn_input, dim, compute_kernel_config=compute_kernel_config)
        else:
            ttnn_output = ttnn.operations.moreh.logsoftmax(
                ttnn_input, dim, compute_kernel_config=compute_kernel_config, strategy=strategy
            )
    else:
        ttnn_output = ttnn.operations.moreh.logsoftmax(ttnn_input, dim, strategy=strategy)

    if multiple_of_32 == False:
        ttnn_output = ttnn_output.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(shape)

    assert list(ttnn_output.shape.with_tile_padding()) == list(torch_output.shape)

    ttnn_output = ttnn.to_torch(ttnn_output)
    passing, out = comp_allclose_and_pcc(torch_output, ttnn_output, rtol=rtol, atol=atol)
    logger.debug(out)
    assert passing


def run_moreh_logsoftmax_backward_test(
    shape,
    dim,
    ttnn_dtype,
    layout,
    device,
    rtol,
    atol,
    use_randint,
    multiple_of_32,
    optional_output_tensor=False,
    strategy=None,
    compute_kernel_options=None,
):
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip(f"bfloat8_b is not supported")
    torch_dtype = get_torch_dtype(ttnn_dtype)
    if use_randint == True:
        torch_x = torch.randint(low=0, high=4, size=shape).to(torch_dtype).requires_grad_(True)
        torch_dy = torch.randint(low=0, high=4, size=shape).to(torch_dtype)
    else:
        torch_x = torch.rand(size=shape, dtype=torch_dtype).requires_grad_(True)
        torch_dy = torch.rand(size=shape, dtype=torch_dtype)

    torch_y = F.log_softmax(torch_x, dim)

    ttnn_y = ttnn.from_torch(torch_y, dtype=ttnn_dtype, layout=layout, device=device)
    ttnn_dy = ttnn.from_torch(torch_dy, dtype=ttnn_dtype, layout=layout, device=device)

    torch_y.backward(torch_dy)

    if optional_output_tensor == True:
        optional_output = ttnn.from_torch(torch_dy, dtype=ttnn_dtype, layout=layout, device=device)
        ttnn_output = ttnn.operations.moreh.logsoftmax_backward(ttnn_y, ttnn_dy, dim, input_grad_tensor=optional_output)
    elif compute_kernel_options is not None:
        compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
        if strategy is None:
            ttnn_output = ttnn.operations.moreh.logsoftmax_backward(
                ttnn_y, ttnn_dy, dim, compute_kernel_config=compute_kernel_config
            )
        else:
            ttnn_output = ttnn.operations.moreh.logsoftmax_backward(
                ttnn_y, ttnn_dy, dim, compute_kernel_config=compute_kernel_config, strategy=strategy
            )
    else:
        ttnn_output = ttnn.operations.moreh.logsoftmax_backward(ttnn_y, ttnn_dy, dim, strategy=strategy)

    if multiple_of_32 == False:
        ttnn_output = ttnn_output.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(shape)

    assert list(ttnn_output.shape.with_tile_padding()) == list(torch_x.grad.shape)

    ttnn_output = ttnn.to_torch(ttnn_output)
    passing, out = comp_allclose_and_pcc(torch_x.grad, ttnn_output, rtol=rtol, atol=atol)
    logger.debug(out)
    assert passing


@pytest.mark.parametrize(
    "shape_dim",
    (
        [[32, 32], 1],  # single tile
        [[3, 32, 32 * 5], 2],  # mutiple tile with dim W
        [[5, 6, 32, 32], 3],  # multiple cores
        [[10, 20, 32 * 3, 32 * 5], 3],  # multiple tiles per core
        [[32, 32], 0],  # single tile
        [[3, 32 * 5, 32], 1],  # mutiple tile with dim H
        [[5, 6, 32, 32], 2],  # multiple cores
        [[10, 20, 32 * 3, 32 * 5], 2],  # multiple tiles per core
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_for_dim_hw(shape_dim, dtype, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.1

    run_moreh_logsoftmax_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        True,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    (
        [[2, 3, 32 * 4, 32 * 5], 3],
        [[2, 3, 32 * 4, 32 * 5], 2],
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_large_algorithm_for_dim_hw(shape_dim, dtype, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.1

    run_moreh_logsoftmax_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        True,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    (
        [[1, 1, 10, 15], 3],  # single tile
        [[1, 1, 10, 32 * 2 + 10], 3],  # mutiple tile with dim
        [[1, 1, 15, 10], 2],  # single tile
        [[1, 1, 32 * 2 + 10, 32], 2],  # mutiple tile with dim
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_not_multiple_of_32_for_dim_hw(shape_dim, dtype, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.1

    run_moreh_logsoftmax_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        False,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    (
        [[1, 15, 32, 32], 1],  # single tile c
        [[1, 15, 32 * 7, 32 * 5], 1],  # mutiple cores
        [[109, 15, 32, 32], 1],  # mutiple tiles per cores
        [[15, 1, 32, 32], 0],  # single tile n
        [[15, 1, 32 * 7, 32 * 5], 0],  # mutiple cores
        [[15, 109, 32 * 2, 32 * 2], 0],  # mutiple tiles per cores
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_for_dim_nc(shape_dim, dtype, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)

    rtol = atol = 0.1
    run_moreh_logsoftmax_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        True,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    (
        [[32, 32], 1],  # single tile
        [[3, 32, 32 * 2], 2],  # mutiple tile with dim W
        [[5, 6, 32, 32], 3],  # multiple cores
        [[10, 20, 32 * 3, 32 * 5], 3],  # multiple tiles per core
        [[32, 32], 0],  # single tile
        [[3, 32 * 5, 32], 1],  # mutiple tile with dim H
        [[5, 6, 32, 32], 2],  # multiple cores
        [[10, 20, 32 * 5, 32], 2],  # multiple tiles per core
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_backward_for_dim_hw(shape_dim, dtype, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.5

    run_moreh_logsoftmax_backward_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        True,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    (
        [[2, 3, 32 * 4, 32 * 5], 3],
        [[2, 3, 32 * 4, 32 * 5], 2],
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_backward_large_algorithm_for_dim_hw(shape_dim, dtype, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.5

    run_moreh_logsoftmax_backward_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        True,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    (
        [[1, 1, 10, 15], 3],  # single tile
        [[1, 1, 10, 32 * 2 + 10], 3],  # mutiple tile with dim
        [[1, 1, 15, 10], 2],  # single tile
        [[1, 1, 32 * 2 + 10, 32], 2],  # mutiple tile with dim
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_backward_not_multiple_of_32_for_dim_hw(shape_dim, dtype, compute_kernel_options, device):
    device.enable_program_cache()
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.1

    run_moreh_logsoftmax_backward_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        False,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    (
        [[1, 15, 32, 32], 1],  # single tile c
        [[1, 15, 32 * 7, 32 * 5], 1],  # mutiple cores
        [[109, 15, 32, 32], 1],  # mutiple tiles per cores
        [[15, 1, 32, 32], 0],  # single tile n
        [[15, 1, 32 * 7, 32 * 5], 0],  # mutiple cores
        [[15, 109, 32 * 2, 32 * 2], 0],  # mutiple tiles per cores
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_backward_for_dim_nc(shape_dim, dtype, compute_kernel_options, device):
    device.enable_program_cache()
    shape, dim = shape_dim
    torch.manual_seed(0)

    rtol = atol = 0.5
    run_moreh_logsoftmax_backward_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        True,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    [
        [[32, 32], 1],
    ],  # single tile
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_logsoftmax_optional_output_tensor(shape_dim, dtype, device):
    device.enable_program_cache()

    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.05

    run_moreh_logsoftmax_test(
        shape, dim, dtype, ttnn.TILE_LAYOUT, device, rtol, atol, True, True, optional_output_tensor=True
    )


@pytest.mark.parametrize(
    "shape_dim",
    [
        [[32, 32], 1],
    ],  # single tile
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_logsoftmax_backward_optional_output_tensor(shape_dim, dtype, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.05

    run_moreh_logsoftmax_backward_test(
        shape, dim, dtype, ttnn.TILE_LAYOUT, device, rtol, atol, True, True, optional_output_tensor=True
    )


@pytest.mark.parametrize(
    "shape_dim_strategy",
    (
        [[32, 32], 1, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.SMALL_W],
        [[32, 32], 0, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.SMALL_H],
        [[2, 3, 32 * 4, 32 * 5], 3, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_W],
        [[2, 3, 32 * 4, 32 * 5], 2, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_H],
        [[1, 15, 32, 32], 1, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_C],
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_logsoftmax_callback(shape_dim_strategy, dtype, device, use_program_cache):
    shape, dim, strategy = shape_dim_strategy
    torch.manual_seed(0)
    rtol = atol = 0.1

    for i in range(2):
        run_moreh_logsoftmax_test(
            shape, dim, dtype, ttnn.TILE_LAYOUT, device, rtol, atol, True, True, strategy=strategy
        )
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)


@pytest.mark.parametrize(
    "shape_dim_strategy",
    (
        [[32, 32], 1, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.SMALL_W],
        [[32, 32], 0, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.SMALL_H],
        [[2, 3, 32 * 4, 32 * 5], 3, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.LARGE_W],
        [[2, 3, 32 * 4, 32 * 5], 2, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.LARGE_H],
        [[1, 15, 32, 32], 1, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.LARGE_C],
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_logsoftmax_backward_callback(shape_dim_strategy, dtype, device, use_program_cache):
    shape, dim, strategy = shape_dim_strategy
    torch.manual_seed(0)

    rtol = atol = 0.5

    for i in range(2):
        run_moreh_logsoftmax_backward_test(
            shape, dim, dtype, ttnn.TILE_LAYOUT, device, rtol, atol, True, True, strategy=strategy
        )
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)


# @pytest.mark.parametrize(
#     "shape_dim_strategy",
#     (
#         [[50, 32], 1, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.SMALL_W],
#         [[32, 32], 0, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.SMALL_H],
#         [[2, 3, 32 * 4, 32 * 5], 3, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_W],
#         [[2, 3, 32 * 4, 32 * 5], 2, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_H],
#         [[1, 15, 32, 32], 1, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_C],
#     ),
# )
# def test_logsoftmax_bfp8(shape_dim_strategy, device, use_program_cache):
#     # TODO @thanhnguyen-moreh: Support bfloat8_b in kernel
#     pytest.skip(f"bfloat8_b is not supported in the kernel")
#     shape, dim, strategy = shape_dim_strategy
#     torch.manual_seed(0)

#     x = torch.rand(size=shape, dtype=torch.bfloat16) + 100
#     tt_cpu = F.log_softmax(x, dim)
#     dev_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
#     tt_npu = ttnn.operations.moreh.logsoftmax(dev_x, dim, strategy=strategy)

#     tt_npu = ttnn.to_torch(tt_npu)
#     assert list(tt_dev.shape.with_tile_padding()) == list(tt_cpu.shape)

#     rtol = atol = 0.1
#     passing, out = comp_allclose_and_pcc(tt_cpu, tt_npu, rtol=rtol, atol=atol)
#     logger.debug(out)
#     assert passing


# @pytest.mark.parametrize(
#     "shape_dim_strategy",
#     (
#         [[32, 32], 1, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.SMALL_W],
#         [[32, 32], 0, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.SMALL_H],
#         [[2, 3, 32 * 4, 32 * 5], 3, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.LARGE_W],
#         [[2, 3, 32 * 4, 32 * 5], 2, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.LARGE_H],
#         [[1, 15, 32, 32], 1, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.LARGE_C],
#     ),
# )
# def test_logsoftmax_backward_bfp8(shape_dim_strategy, device):
#     # TODO @thanhnguyen-moreh: Support bfloat8_b in kernel
#     pytest.skip(f"bfloat8_b is not supported in the kernel")
#     device.enable_program_cache()
#     shape, dim, strategy = shape_dim_strategy
#     torch.manual_seed(0)

#     x = torch.rand(size=shape, dtype=bfloat16).requires_grad_(True)

#     y = F.log_softmax(x, dim)
#     dev_y = ttnn.from_torch(y, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

#     dy = torch.rand(size=shape, dtype=bfloat16)
#     dev_dy = ttnn.from_torch(dy, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

#     y.backward(dy)
#     tt_npu = ttnn.operations.moreh.logsoftmax_backward(dev_y, dev_dy, dim, strategy=strategy)

#     assert list(tt_npu.shape.with_tile_padding()) == list(x.grad.shape)
#     tt_npu = ttnn.to_torch(tt_npu)

#     rtol = atol = 0.5
#     passing, out = comp_allclose_and_pcc(x.grad, tt_npu, rtol=rtol, atol=atol)
#     logger.debug(out)
#     assert passing
