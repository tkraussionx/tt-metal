import sys
import pytest

from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import torch

import tt_lib as ttl

from python_api_testing.models.utility_functions import comp_pcc

## max-pool params:
## kernel_h, kernel_w
## stride_h, stride_w
## pad_h, pad_w
## dilation_h, dilation_w
@pytest.mark.parametrize(
    "act_shape",    ## NCHW
    ((  [1, 1, 32, 32],
        [1, 2, 32, 32],
        [1, 1, 112, 112],
        [1, 64, 112, 112],
        [1, 6, 96, 96],
        [1, 1, 128, 128],))
)
@pytest.mark.parametrize(
    "kernel_size",
    ((1, 1),
     (3, 3),),
)
@pytest.mark.parametrize(
    "padding",
    ((0, 0),    ## default
     (1, 1),),
)
@pytest.mark.parametrize(
    "stride",
    ((1, 1),    ## default
     (2, 2),)
)
@pytest.mark.parametrize(
    "dilation",
    ((1, 1),)    ## default
)
@pytest.mark.parametrize(
    "in_mem_config",
    (   ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),),
    ids=["in_DRAM", "in_L1"],
)
@pytest.mark.parametrize(
    "out_mem_config",
    (   ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),),
    ids=["out_DRAM", "out_L1"],
)
def test_run_max_pool(act_shape, kernel_size, padding, stride, dilation, in_mem_config, out_mem_config):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    torch.manual_seed(0)

    act = torch.randn(act_shape, dtype=torch.bfloat16).float()
    # act = torch.zeros(act_shape, dtype=torch.bfloat16).float()
    ttact = ttl.tensor.Tensor(act.flatten().tolist(),
                              act_shape,
                              ttl.tensor.DataType.BFLOAT16,
                              ttl.tensor.Layout.ROW_MAJOR)
    ttact = ttact.to(device, in_mem_config)
    # ttact_host = ttact.cpu()

    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    # print (f'kernel: {kernel_size}')
    # print (f'padding: {padding}')

    if 2 * pad_h > kernel_h or 2 * pad_w > kernel_w:
        print('Invalid/Unsupported case')
        pytest.skip()

    out = ttl.tensor.max_pool2d(ttact, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, out_mem_config)

    out = out.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    out_shape = out.shape()
    out_pytorch = torch.tensor(out.data()).reshape(out_shape)

    ttl.device.CloseDevice(device)

    ## reference
    golden_pytorch = torch.nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=1, return_indices=False, ceil_mode=False)(act)

    ## test for equivalance
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch)
    print(f'Passing PCC = {passing_pcc}')
    print(f'Output PCC = {output_pcc}')

    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000)
    # print(f"INPUT:\n{act}")
    # # ttact_host.pretty_print()
    # print(f"OUTPUT:\n{out_pytorch}")
    # print(f"GOLDEN:\n{golden_pytorch}")


    assert(passing_pcc)
