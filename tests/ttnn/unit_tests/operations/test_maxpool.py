# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import math

from models.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc

import ttnn
from ttnn.operations.conv2d import determine_parallel_config, create_sharded_memory_config_from_parallel_config


def run_max_pool(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    dtype,
):
    in_n, in_c, in_h, in_w = act_shape
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    if 2 * pad_h > kernel_h or 2 * pad_w > kernel_w:
        pytest.skip("Invalid case")

    if (kernel_h == 3 and pad_h != 1) or (kernel_h == 2 and pad_h != 0):
        pytest.skip("kernel size and padding combination not supported")

    out_h = math.floor((in_h + 2 * pad_h - (dilation_h * kernel_h - 1) - 1) / stride_h) + 1
    out_w = math.floor((in_w + 2 * pad_w - (dilation_w * kernel_w - 1) - 1) / stride_w) + 1
    # if in_c % 16 != 0:
    #     pytest.skip("Current maxpool writer needs nchannels to be multiple of 16!")

    if in_c == 16 and dtype == ttnn.bfloat8_b and in_n * in_h * in_w > 600000:
        pytest.skip("This case runs out of memory on Grayskull")

    if in_n > 16 and in_c > 64 and dtype == ttnn.bfloat8_b and is_wormhole_b0():
        pytest.skip("This case runs out of memory on Wormhole b0")

    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    ## construct the tensor in NCHW shape
    act = torch.randn(act_shape, dtype=torch.bfloat16)
    # act = torch.zeros(act_shape, dtype=torch.bfloat16)
    # act = torch.ones(act_shape, dtype=torch.bfloat16)
    # act = torch.arange(0, volume(act_shape), dtype=torch.bfloat16).reshape(act_shape)
    # for n in range(act_shape[0]):
    #     for c in range(act_shape[1]):
    #         for h in range(act_shape[2]):
    #             for w in range(act_shape[3]):
    #                 act[n, c, h, w] = 1 + n + h + w + c # + torch.rand(1) * 0.15
    # torch.save(act, "act.pt")
    # act = torch.load("act.pt")

    ## this op expects input tensor as { N, 1, H * W, C }, so rearrange and reshape tensor
    ## but before that, make sure in_c is multiple of tile width
    act_shape = (1, 1, in_n * in_h * in_w, in_c)
    act_permuted = torch.permute(act, (0, 2, 3, 1))
    act_reshaped = act_permuted.reshape(act_shape)
    print("the shape of the tesnor to the maxpool is", act_reshaped.shape)
    if dtype == ttnn.bfloat8_b:
        if (in_h * in_w) % 32 != 0:
            pytest.skip("For BFP8_B datatype, input height * width should be multiple of 32")
        ttact = ttnn.from_torch(act_reshaped, dtype, layout=ttnn.TILE_LAYOUT)
    else:
        ttact = ttnn.from_torch(act_reshaped, dtype)

    pre_shard = True
    # pre_shard = False

    ttact_device = ttnn.to_device(ttact, device)
    if pre_shard:
        parallel_config = determine_parallel_config(
            is_1d_systolic=True,
            batch_size=in_n,
            input_channels=in_c,
            output_height=out_h,
            output_width=out_w,
            output_channels=in_c,
            device=device,
            is_out_tiled=False,
        )
        sharded_memory_config = create_sharded_memory_config_from_parallel_config(
            tensor_shape=act_shape,
            parallel_config=parallel_config,
            tile_size=32 if dtype == ttnn.bfloat8_b else 1,
        )
        ttact_device = ttnn.to_memory_config(ttact_device, sharded_memory_config)
    output = ttnn.max_pool2d(
        input_tensor=ttact_device,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=[kernel_h, kernel_w],
        stride=[stride_h, stride_w],
        padding=[pad_h, pad_w],
        dilation=[dilation_h, dilation_w],
        device=device,
    )

    # interleaved_mem_config = ttnn.L1_MEMORY_CONFIG
    # output = ttnn.to_memory_config(output, interleaved_mem_config)
    output_host = output.cpu()
    output_pytorch_padded = torch.Tensor(ttnn.to_torch(output_host))
    output_pytorch = output_pytorch_padded[:, :, :, :in_c]

    ## reference
    golden_pytorch = torch.nn.MaxPool2d(
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=False,
        ceil_mode=False,
    )(act)

    ## test for equivalance
    golden_shape = golden_pytorch.shape
    output_pytorch = output_pytorch.reshape(golden_shape[0], golden_shape[2], golden_shape[3], golden_shape[1])

    # torch.save(output_pytorch, "output_pytorch.pt")
    # torch.save(golden_pytorch, "golden_pytorch.pt")

    output_pytorch = torch.permute(output_pytorch, (0, 3, 1, 2))  ## N, C, H, W
    passing, pcc = assert_with_pcc(output_pytorch, golden_pytorch)

    logger.debug(f"Passing: {passing}, PCC: {pcc}")

    ## do more rigorous comparision for each element
    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    if dtype == ttnn.bfloat8_b:
        atol = 0.35

    allclose = torch.allclose(output_pytorch, golden_pytorch, atol=atol)
    isclose = torch.all(torch.isclose(output_pytorch, golden_pytorch, atol=atol))
    isequal = torch.equal(output_pytorch, golden_pytorch)

    assert allclose
    assert isclose
    if dtype == ttnn.bfloat16:
        assert isequal


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
    (([1, 6, 28, 28],)),
)
@pytest.mark.parametrize(
    "kernel_size",
    ((2, 2),),
)
@pytest.mark.parametrize(
    "padding",
    ((0, 0),),
)
@pytest.mark.parametrize(
    "stride",
    ((2, 2),),
)
@pytest.mark.parametrize("dilation", ((1, 1),))  ## default
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_run_max_pool(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    dtype,
    use_program_cache,
):
    run_max_pool(act_shape, kernel_size, padding, stride, dilation, device, dtype)
