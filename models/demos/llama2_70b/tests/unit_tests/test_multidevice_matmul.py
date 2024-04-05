# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import tt_lib
import ttnn

from models.demos.llama2_70b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000


def tt_multidevice_matmul_decode(x, w1, devices, model_config):
    num_devices = len(devices)
    hidden_size = x.size(-1)

    # prepare weights
    w1_list = []
    w1_chunks = torch.chunk(w1, num_devices, dim=-1)
    for i in range(num_devices):
        w1_host = torch2tt_tensor(
            w1_chunks[i],
            None,
            tt_memory_config=model_config["DRAM_MEMCFG"],
            tt_dtype=model_config["BFP8_DTYPE"],
        )
        w1_list.append(w1_host.to(devices[i], model_config["DRAM_MEMCFG"]))

    # prepare input
    x_multichip = []
    batch, seq_len = 32, 1
    assert x.size() == (seq_len, 1, batch, hidden_size)
    for i in range(num_devices):
        x_multichip.append(
            torch2tt_tensor(
                x.clone(),
                devices[i],
                tt_dtype=model_config["LN_MLP_OUTPUT_DTYPE"],
                tt_memory_config=model_config["L1_MEMCFG"],
            )
        )
    for i in range(num_devices):
        x_multichip[i] = tt_lib.tensor.interleaved_to_sharded(
            x_multichip[i], sharded_mem_config=model_config["LN_MLP_OUTPUT_MEMCFG"]
        )

    # matmul
    w1_outs = []
    for i in range(num_devices):
        w1_outs.append(
            tt_lib.operations.primary.matmul_1d(
                x_multichip[i],
                w1_list[i],
                program_config=model_config["PADDED_FF1_MM_PROGCFG"],
                output_mem_config=model_config["WIDTH_SHARDED_MEMCFG"],
                output_dtype=model_config["PADDED_FF1_MM_OUTPUT_DTYPE"],
                compute_kernel_config=model_config["COMPUTE_KERNEL_CONFIG"],
            )
        )

    # all gather
    for i in range(len(w1_outs)):
        # Put w2_inputs in DRAM
        w1_outs[i] = tt_lib.tensor.sharded_to_interleaved(w1_outs[i], output_mem_config=model_config["L1_MEMCFG"])

    w1_outs = tt_lib.tensor.all_gather(
        w1_outs,
        dim=3,
        num_links=model_config["ALL_GATHER_NUM_LINKS"],
        output_mem_config=model_config["DRAM_MEMCFG"],
    )

    return tt2torch_tensor(w1_outs[0])


def tt_multidevice_matmul_prefill(x, w1, devices, model_config):
    num_devices = len(devices)
    hidden_size = x.size(-1)

    # prepare weights
    w1_list = []
    w1_chunks = torch.chunk(w1, num_devices, dim=-1)
    for i in range(num_devices):
        w1_host = torch2tt_tensor(
            w1_chunks[i],
            None,
            tt_memory_config=model_config["DRAM_MEMCFG"],
            tt_dtype=model_config["BFP8_DTYPE"],
        )
        w1_list.append(w1_host.to(devices[i], model_config["DRAM_MEMCFG"]))

    # prepare input
    x_multichip = []
    for i in range(num_devices):
        x_multichip.append(
            torch2tt_tensor(
                x.clone(),
                devices[i],
                tt_dtype=model_config["LN_MLP_OUTPUT_DTYPE"],
            )
        )

    # matmul
    w1_outs = []
    seq_tiles = x_multichip[0].shape[2] // 32
    model_config["PADDED_FF1_MM_PROGCFG"] = model_config["PADDED_FF1_MM_PROGCFG_LAMBDA"](seq_tiles)
    for i in range(num_devices):
        """
        x_multichip[i] is shape [1,32,128,8192]
        w1_list[i] is shape [1,1,8192,4096]
        """
        w1_outs.append(
            tt_lib.operations.primary.matmul(
                x_multichip[i],
                w1_list[i],
                program_config=model_config["PADDED_FF1_MM_PROGCFG"],
                compute_kernel_config=model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG"],
            )
        )

    # all gather
    w1_outs = tt_lib.tensor.all_gather(
        w1_outs,
        dim=3,
        num_links=model_config["ALL_GATHER_NUM_LINKS"],
    )

    return tt2torch_tensor(w1_outs[0])


def run_test_matmul(
    devices,
    batch,
    seq_len,
    pcc,
    model_config,
):
    for device in devices:
        tt_lib.device.Synchronize(device)

    # Prepare input tensors
    input_tensor = (
        torch.randn(1, batch, seq_len, 8192)
        if model_config["LLM_MODE"] == "prefill"
        else torch.randn(seq_len, 1, batch, 8192)
    )
    weights = torch.randn(1, 1, 8192, 8192 * 4)

    # torch output
    output_torch = torch.matmul(input_tensor, weights)
    output_torch = torch.nn.SiLU()(output_torch)

    # tt output on multidevice
    # x: (seq_len, batch, hidden_size), seq_len=1 and batch=32 is mode decode
    if model_config["LLM_MODE"] == "decode":
        output_tt = tt_multidevice_matmul_decode(input_tensor, weights, devices, model_config)
    elif model_config["LLM_MODE"] == "prefill":
        output_tt = tt_multidevice_matmul_prefill(input_tensor, weights, devices, model_config)
    else:
        raise ValueError(f"Unknown llm_mode: {model_config['LLM_MODE']}")

    # Compute PCC
    pcc_val = comp_pcc(output_torch, output_tt)
    logger.info(f"PCC: {pcc_val}")


@pytest.mark.parametrize(
    "batch, seq_len, pcc",
    ((32, 1, 0.9999), (1, 128, 0.9998), (1, 2048, 0.9998)),
    ids=("decode", "prefill_128", "prefill_2k"),
)
def test_multidevice_matmul(
    batch,
    seq_len,
    pcc,
    all_devices,
):
    n_devices = 8
    devices = get_devices_for_t3000(all_devices, num_devices=n_devices)
    model_config = get_model_config(model_config_str="BFLOAT16-DRAM", num_devices=n_devices, seq_len=seq_len)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    if len(all_devices) < n_devices:
        pytest.skip(f"Requires {n_devices} devices to run")

    run_test_matmul(
        devices,
        batch,
        seq_len,
        pcc,
        model_config,
    )
