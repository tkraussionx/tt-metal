# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import json
from pathlib import Path
import ttnn
from models.demos.mixtral8x7b.tt.mixtral_mlp_ttnn import TtMixtralMLP
from models.demos.mixtral8x7b.tt.mixtral_moe_ttnn import TtMoeLayer
from models.demos.mistral7b.tt.model_config_ttnn import TtModelArgs
from models.demos.mixtral8x7b.reference.moe import MoeArgs, MoeLayer
from models.demos.mixtral8x7b.reference.model import FeedForward
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import get_devices_for_t3000


@pytest.mark.parametrize(
    "iterations",
    ((1,)),
)
def test_mistral_moe_inference(all_devices, iterations):
    pcc = 0.99
    dtype = ttnn.bfloat8_b
    torch.manual_seed(0)

    devices = all_devices
    num_devices = len(devices)
    assert num_devices in (4, 8), "This test requires a T3000 (4 or 8 devices)"

    devices = get_devices_for_t3000(devices, num_devices)  # [ttnn.open_device(device_id=i) for i in range(8)]
    if num_devices == 4:
        devices += devices

    mistral_path = "/proj_sw/user_dev/hf_data/mistral/Mixtral-8x7B-v0.1/"
    state_dict = {}
    for i in range(1):
        state_dict_i = torch.load(mistral_path + f"consolidated.{str(i).zfill(2)}.pt")
        state_dict.update(state_dict_i)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k[9:]: v
        for k, v in state_dict.items()
        if (k.startswith("layers.0.") and "attention" not in k and "norm" not in k)
    }
    partial_state_dict["gate.weight"] = partial_state_dict["block_sparse_moe.gate.weight"]
    del partial_state_dict["block_sparse_moe.gate.weight"]

    w1 = partial_state_dict["block_sparse_moe.w1"].view(8, 14336, 4096)
    w2 = partial_state_dict["block_sparse_moe.w2"].view(8, 4096, 14336)
    w3 = partial_state_dict["block_sparse_moe.w3"].view(8, 14336, 4096)
    for i in range(8):
        partial_state_dict[f"experts.{i}.w1.weight"] = w1[i]
        partial_state_dict[f"experts.{i}.w2.weight"] = w2[i]
        partial_state_dict[f"experts.{i}.w3.weight"] = w3[i]
    partial_state_dict.pop("block_sparse_moe.w1")
    partial_state_dict.pop("block_sparse_moe.w2")
    partial_state_dict.pop("block_sparse_moe.w3")

    with open("/proj_sw/user_dev/hf_data/mistral/mistral-7B-v0.1/params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))
    model_args.max_batch_size = 32
    model_args.moe = True
    model_args.num_experts = 8
    model_args.num_experts_per_tok = 2

    state_dict = None
    # Initialize TT model
    tt_model = TtMoeLayer(
        experts=[
            TtMixtralMLP(
                device=devices[i],
                state_dict=partial_state_dict,
                args=model_args,
                layer_num=None,
                expert_num=i,
                dtype=dtype,
            )
            for i in range(len(devices))
        ],
        state_dict=partial_state_dict,
        # layer_num=layer_num,
        moe_args=model_args,
        devices=devices,
        num_devices=num_devices,
        dtype=dtype,
    )

    reference_model = MoeLayer(
        experts=[FeedForward(args=model_args) for _ in range(8)],
        gate=torch.nn.Linear(model_args.dim, 8, bias=False),
        moe_args=model_args,
    )
    reference_model.load_state_dict(partial_state_dict)
    all_tests_pass = True

    seqlen = 1
    batch = 32

    # TODO Update start_pos (check llama test for reference)
    for i in range(iterations):
        print(f"[Decoder] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = [
            ttnn.from_torch(
                pt_decode_input,
                device=device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            )
            for device in devices
        ]
        # Run TT model
        tt_out, selected_tt, res_tt = tt_model(tt_decode_input)
        print("Converting to torch")
        tt_output_torch = ttnn.to_torch(tt_out[0]).squeeze(2)  # [batch, seq, hidden_dim]

        print("Converted to torch")

        # Reference model
        ref_output, selected_torch, res_torch = reference_model(pt_decode_input)
        print("REF MODEL DONE", selected_tt, selected_torch)
        for head in range(8):
            print(f"head {head}")
            # print("imtermediate results:", comp_pcc(res_tt[head][0], res_torch[head][0])[1])
            print("imtermediate batches:", res_tt[head][2], res_torch[head][2])
            # print("imtermediate weights:", comp_pcc(res_tt[head][1], res_torch[head][1])[1])
            print("imtermediate heads:", res_tt[head][3], res_torch[head][3])
            print("all weights:", comp_pcc(res_tt[head][4], res_torch[head][4])[1])
        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("Mistral MOE Passed!")
        else:
            logger.warning("Mistral MOE Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All {iterations} Mistral MOE iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral MOE Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
