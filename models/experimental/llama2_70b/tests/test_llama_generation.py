# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import tt_lib
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor, ListMeshToTensor


import scipy
from sklearn.metrics import top_k_accuracy_score
import numpy as np

from models.experimental.llama2_70b.reference.llama.llama import Llama
from models.experimental.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration
from models.experimental.llama2_70b.tt.model_config import (
    get_model_config,
)

from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000
from models.experimental.llama2_70b.tt.llama_common import (
    get_llama_path,
    extract_pcc_from_log,
    MAX_SEQ_LEN,
    BASE_URL,
    UNIT_TEST_START_POS,
    UNIT_TEST_GENERATION_LENGTH,
    comp_pcc,
    should_skip_model_load,
    check_kv_cache,
)

from models.experimental.llama2_70b.demo.demo import (
    build_generator,
    load_prompts_file,
    intialize_inputs,
    prepare_next_input,
)


def run_test_generation(args):
    # Prepare paths and devices
    # t3k_device_mesh, ckpt_dir, tokenizer_path, cache_path = get_llama_path(
    #     t3k_device_mesh, model_config, n_devices, emulated
    # )
    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        skip_model_load=args.skip_model_load,
        n_layers=1 if args.implementation == "tt" else args.num_layers,
    )

    # state_dict = load_llama_state_dict(args.ckpt_dir, n_layers=args.num_layers)

    tt_model = TtLlamaModelForGeneration(
        configuration=generator.model.params,
        state_dict=generator.model.state_dict(),
        device_mesh=args.device_mesh,
        n_devices=args.n_devices,
        n_layers=args.num_layers,
        batch=args.max_batch_size,
        emulated=args.emulated,
        cache_path=args.cache_path,
    )

    # args.implementation = "meta"
    # pt_generator = build_generator(args)
    pt_model = generator.model

    # args.implementation = "tt"
    # tt_generator = build_generator(args)
    # tt_model = tt_generator.model

    tokenizer = generator.tokenizer
    prompt_tokens, prompts = load_prompts_file(args, tokenizer)

    all_tests_pass = True
    all_pccs, all_top1, all_top5 = [], [], []

    # decode arguments
    bsz = args.max_batch_size
    model_args = pt_model.params
    max_gen_len = args.num_tokens
    args.greedy = args.top_k == 1  # greedy decoding is top-k with k=1

    min_prompt_len = min(len(t) for t in prompt_tokens) if not args.decode_only else 1
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= model_args.max_seq_len
    total_len = min(model_args.max_seq_len, max_gen_len + max_prompt_len)
    assert total_len <= model_args.max_seq_len

    # prepare inputs
    tokens, input_text_mask, eos_reached = intialize_inputs(tokenizer, prompt_tokens, bsz, total_len)
    prev_pos = 0

    # some profiling and logging

    for cur_pos in range(min_prompt_len, total_len):
        input_tokens = tokens[:, prev_pos:cur_pos]
        # Print all relevant details
        logger.info(f"Input idx {cur_pos}: input_tokens shape: {input_tokens.shape}, prev_pos: {prev_pos}")
        tt_logits = tt_model.forward(input_tokens, prev_pos, decode_only=args.decode_only)
        pt_logits = pt_model.forward(input_tokens, prev_pos, decode_only=args.decode_only)
        # expects logits to be of shape (bsz, 1, vocab_size)

        # sample next token
        # TODO: Check pt against tt output tokens
        if args.greedy:
            next_token = torch.argmax(pt_logits[:, -1], dim=-1)
        else:
            next_token = top_pk_logits_efficient(
                pt_logits[:, -1], p=args.top_p, k=args.top_k, temperature=args.temperature
            )
        next_token = next_token.reshape(-1)

        tokens, eos_reached, prev_pos = prepare_next_input(tokenizer, tokens, input_text_mask, cur_pos, next_token)

        # check outputs ----------------------------------------------------------------------
        does_pass, output_pcc = comp_pcc(pt_logits, tt_logits, 0.999)
        logger.info(f"Output idx {cur_pos}: {output_pcc}")
        # all_pccs.append(extract_pcc_from_log(output_pcc))

        # kl_divs = scipy.stats.entropy(
        #     torch.nn.functional.softmax(pytorch_out, dim=-1), torch.nn.functional.softmax(tt_out, dim=-1), axis=-1
        # )
        # logger.info(f"Mean KL Divergence: {kl_divs.mean()}")

        # # Write the code to check top-5 and top-1 accuracy. It should show the
        # # percentage where the top-1 prediction in pytorch was in the top-5
        # # predictions in tt.
        # reference_top1 = np.argmax(pytorch_out, axis=-1)
        # top1_acc = top_k_accuracy_score(reference_top1, tt_out, k=1, labels=np.arange(tt_out.shape[-1]))
        # top5_acc = top_k_accuracy_score(reference_top1, tt_out, k=5, labels=np.arange(tt_out.shape[-1]))

        # all_top1.append(top1_acc)
        # all_top5.append(top5_acc)

        # logger.info(f"Mean Top-1: {top1_acc}")
        # logger.info(f"Mean Top-5: {top5_acc}")

        # if does_pass:
        #     logger.info(f"[start_pos={start_pos}] {model_name} Model output Passed!")
        # else:
        #     logger.warning(f"[start_pos={start_pos}] {model_name} Model output Failed! PCC value is lower than {pcc}")
        #     all_tests_pass = False

    # logger.info(f"Average PCC over {len(all_pccs)} tokens: {sum(all_pccs) / len(all_pccs)}")
    # logger.info(f"Average Top-1 over {len(all_top1)} tokens: {sum(all_top1) / len(all_top1)}")
    # logger.info(f"Average Top-5 over {len(all_top5)} tokens: {sum(all_top5) / len(all_top5)}")
    # # Check kv cache
    # # PyTorch output --------------------------------------------------------------------
    # pytorch_layer_present = [
    #     pytorch_model.model.layers[0]
    #     .attention.cache_k.clone()
    #     .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
    #     pytorch_model.model.layers[0]
    #     .attention.cache_v.clone()
    #     .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
    # ]

    # tt_layer_present_all = [ttnn.from_device(lp) for lp in tt_model.layers[0].attention.layer_past]
    # tt_layer_present_all = [
    #     ttnn.to_torch(lp, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0)).transpose(0, 1)
    #     for lp in tt_layer_present_all
    # ]

    # cache_test_pass = check_kv_cache(
    #     pytorch_layer_present,
    #     tt_layer_present_all,
    #     generation_start_pos,
    #     generation_length,
    #     seq_len,
    #     model_config["LLM_MODE"] == "prefill",
    #     pcc,
    # )
    # if all_tests_pass:
    #     logger.info(f"{model_name} output Passed!")
    # else:
    #     logger.warning(f"{model_name} output Failed!")
    #     assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"


class Args:
    def __init__(
        self,
        # model args
        implementation="meta",
        ckpt_dir="/home/llama-data-repacked-2/llama-2-70b/",
        tokenizer_path="/home/llama-data/tokenizer.model",
        skip_model_load=False,
        max_batch_size=32,
        num_layers=None,
        max_seq_len=4096,
        # Generation args
        num_tokens=128,
        prompts_file="models/demos/t3000/llama2_70b/demo/data/multi_prompt.json",
        output_at_end=True,
        top_p=1,
        top_k=1,
        temperature=1.0,
        # TT args
        device_mesh=None,
        n_devices=8,
        emulated=False,
        cache_path=None,
        decode_only=False,
    ):
        self.implementation = implementation
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.skip_model_load = skip_model_load
        self.max_batch_size = max_batch_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_tokens = num_tokens
        self.prompts_file = prompts_file
        self.output_at_end = output_at_end
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.device_mesh = device_mesh
        self.n_devices = n_devices
        self.emulated = emulated
        self.cache_path = cache_path
        self.decode_only = decode_only


def construct_arg(**kwargs):
    return Args(**kwargs)


@pytest.mark.timeout(240000)
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("decode_only", (True, False), ids=["decode_only", "prefill_decode"])
@pytest.mark.parametrize("num_layers", (1, 2, 10, 80), ids=["1L", "2L", "10L", "80L"])
@pytest.mark.parametrize(
    "implementation, skip_model_load, n_devices, emulated",
    [
        (
            "tt",
            False,
            8,
            False,
        ),
    ],
    ids=["tt-70b-T3000"],
)
@pytest.mark.parametrize(
    "num_tokens, prompts_file, output_at_end, top_p, top_k, temperature",
    [
        (128, "models/demos/t3000/llama2_70b/demo/data/multi_prompt.json", True, 1, 1, 1.0),
        (128, "models/demos/t3000/llama2_70b/demo/data/multi_prompt.json", True, 0.9, 10, 1.0),
    ],
    ids=["greedy", "sampling"],
)
def test_LlamaModel_inference(
    implementation,
    skip_model_load,
    num_layers,
    # Generation args
    num_tokens,
    prompts_file,
    output_at_end,
    top_p,
    top_k,
    temperature,
    # TT args
    # all_devices,
    t3k_device_mesh,
    n_devices,
    emulated,
    decode_only,
):
    model_config = get_model_config(model_config_str="BFLOAT16-DRAM", num_devices=n_devices)

    if t3k_device_mesh.get_num_devices() < n_devices and not emulated:
        pytest.skip(f"Requires at {n_devices} devices to run")

    compute_grid_size = t3k_device_mesh.get_device(0).compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    for i in t3k_device_mesh.get_device_ids():
        device = t3k_device_mesh.get_device(i)
        device.enable_program_cache()

    t3k_device_mesh, ckpt_dir, tokenizer_path, cache_path = get_llama_path(
        t3k_device_mesh, model_config, n_devices, emulated
    )

    args = construct_arg(
        implementation=implementation,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        skip_model_load=skip_model_load,
        num_layers=num_layers,
        num_tokens=num_tokens,
        prompts_file=prompts_file,
        output_at_end=output_at_end,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        device_mesh=t3k_device_mesh,
        n_devices=n_devices,
        emulated=emulated,
        cache_path=cache_path,
        decode_only=decode_only,
    )
    run_test_generation(args)
