# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import json
from time import time
from loguru import logger
import os

# Set Mistral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["MISTRAL_CKPT_DIR"] = "/mnt/MLPerf/ttnn/models/demos/mistral7b/"
    os.environ["MISTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/ttnn/models/demos/mistral7b/"
    os.environ["MISTRAL_CACHE_PATH"] = "/mnt/MLPerf/ttnn/models/demos/mistral7b/"
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

import ttnn
import pytest
from models.demos.wormhole.mistral7b.tt.mistral_common import (
    prepare_inputs_ttnn,
    sample,
    precompute_freqs,
    freqs_to_rotation_matrix,
    cache_attention,
    get_prefill_rot_mat,
    prepare_inputs_ttnn_prefill,
    get_rot_transformation_mat,
)
from models.demos.wormhole.mistral7b.tt.mistral_model import TtTransformer
from models.demos.wormhole.mistral7b.tt.mistral_embedding import TtMistralEmbedding
from models.demos.wormhole.mistral7b.tt.model_config import TtModelArgs
from models.demos.wormhole.mistral7b.reference.tokenizer import Tokenizer


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


# load from json, return as a list
def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    for i in range(batch):
        in_prompt.append(user_input[i]["prompt"])
    return in_prompt


def preprocess_inputs_prefill(input_prompts, tokenizer, model_args, dtype, embd, instruct, device):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    if instruct:
        # Pre append [INST] and post append [/INST] to the encoded prompts if instruct mode
        encoded_prompts = [tokenizer.encode("[INST] " + prompt + " [/INST]") for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]

    prompt_lens = [len(x) for x in encoded_prompts]

    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)
    assert (
        max_prompt_len <= model_args.max_seq_len
    ), f"Max prompt length {max_prompt_len} exceeds model max seq len {model_args.max_seq_len}"
    assert min_prompt_len > 0, "Minimum prompt length must be greater than 0"
    assert min_prompt_len <= max_prompt_len, f"Minimum prompt length {min_prompt_len} exceeds max len {max_prompt_len}"

    if min_prompt_len < 128:
        prefill_seq_len = 0  # For short prompts do decode-as-prefill instead
    else:
        prefill_seq_len = (
            1024 if min_prompt_len > 1024 else (512 if min_prompt_len > 512 else 128)
        )  # TODO Only supports prefill lengths of 128, 512, 1024
        # Initial prefill tensor full of pad tokens
        input_tokens_prefill = torch.full((len(input_prompts), prefill_seq_len), tokenizer.pad_id, dtype=torch.int32)

    # Initial decode tensor full of pad tokens
    input_tokens_decode = torch.full(
        (len(input_prompts), max_prompt_len - prefill_seq_len), tokenizer.pad_id, dtype=torch.long
    )

    for i, encoded in enumerate(encoded_prompts):
        if prefill_seq_len > 0:
            input_tokens_prefill[i] = torch.tensor(encoded[:prefill_seq_len]).to(input_tokens_prefill)
        input_tokens_decode[i, : len(encoded[prefill_seq_len:])] = torch.tensor(encoded[prefill_seq_len:]).to(
            input_tokens_decode
        )

    input_mask = (input_tokens_decode != tokenizer.pad_id).to(torch.bool)

    num_users = len(encoded_prompts)
    logger.info(f"# of users: {num_users}")

    # Select the first token from the prompts for initial decoding
    pt_tokenized_inputs_decode = torch.tensor(input_tokens_decode)
    pt_tokenized_inputs_prefill = torch.tensor(input_tokens_prefill)
    emb_inputs_decode = embd(pt_tokenized_inputs_decode[:, 0]).view(model_args.max_batch_size, 1, -1)
    if prefill_seq_len > 0:
        emb_prefill_inputs = [
            embd(pt_tokenized_inputs_prefill[b, :]).view(1, prefill_seq_len, -1)
            for b in range(model_args.max_batch_size)
        ]
    else:
        emb_prefill_inputs = None

    # Return the rotational embedding matrix on device
    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    rot_emb_matrix = freqs_to_rotation_matrix(cos, sin)

    rot_emb_matrix_list = []
    for i in range(rot_emb_matrix.shape[0]):
        rot_emb_matrix_list.append(
            ttnn.from_torch(
                rot_emb_matrix[i, :, :].unsqueeze(0).unsqueeze(0), device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT
            )
        )  # ttnn.bfloat16

    return (
        emb_inputs_decode,
        pt_tokenized_inputs_decode,
        emb_prefill_inputs,
        input_mask,
        rot_emb_matrix_list,
        prefill_seq_len,
        encoded_prompts,
    )


def run_mistral_demo(user_input, batch_size, device, instruct_mode, is_ci_env):
    assert batch_size == 32, "Batch size must be 32"

    embed_on_device = False
    dtype = ttnn.bfloat8_b

    logger.info(f"Reading inputs...")
    if len(user_input) == 1:
        input_prompts = user_input * 32  # Always process 32 users
    else:
        input_prompts = load_inputs(user_input, 32)

    # Load model args, weights, and tokenizer
    model_args = TtModelArgs(device, instruct=instruct_mode)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    # TODO: Currently only supporting 1k decode
    model_args.n_layers = 32
    model_args.max_batch_size = 32
    model_args.max_seq_len = 1024
    model_args.sliding_window = 1024
    model_args.kv_seq_len = 1024

    logger.info("Loading weights...")
    state_dict = torch.load(model_args.consolidated_weights_path)
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }
    logger.info("Loading weights finished!")

    # TODO Should we keep initial embedding on host?
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    # Preprocess initial prompt inputs
    (
        pt_encoded_input,
        tt_decode_input,
        pt_prefill_input,
        input_mask,
        rot_emb_matrix_list,
        prefill_seq_len,
        encoded_prompts,
    ) = preprocess_inputs_prefill(input_prompts, tokenizer, model_args, dtype, embd, instruct_mode, device)
    generation_start_pos = prefill_seq_len
    max_generated_tokens = 120
    users_decoding = True
    logger.info("Caching attention ops...")
    cache_attention(device, state_dict, model_args, rot_emb_matrix_list, dtype)

    if instruct_mode:
        tokenizer._model.pad_id = tokenizer._model.eos_id

    # Load TTNN mistral model
    logger.info("Loading weights to device...")
    tt_model = TtTransformer(
        args=model_args,
        device=device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layers=list(range(model_args.n_layers)),
        rot_mat=rot_emb_matrix_list,
        start_pos=generation_start_pos,
    )
    tt_embd = TtMistralEmbedding(
        device=device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )
    logger.info("Finished loading weights to device. Starting inference...")

    if prefill_seq_len > 0:
        logger.info(f"Starting prefill [{prefill_seq_len} tokens]...")
        rot_mats_prefill = get_prefill_rot_mat(
            model_args.head_dim, model_args.max_seq_len, device, seq_len=prefill_seq_len
        )
        head_dim = model_args.dim // model_args.n_heads
        transformation_mat_torch = get_rot_transformation_mat(head_dim)
        transformation_mats = ttnn.as_tensor(
            transformation_mat_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for batch_id in range(batch_size):
            prefill_input, attn_mask, _ = prepare_inputs_ttnn_prefill(
                pt_prefill_input[batch_id],
                device,
            )
            tt_out = tt_model(
                prefill_input,
                0,  # Current position
                attn_mask,
                rot_mats_prefill,
                transformation_mats,
                user_id=batch_id,
                mode="prefill",
            )

        logger.info(f"Prefill finished [{prefill_seq_len} tokens]!")

    logger.info("Starting decode...")

    # Keep track of generated outputs to print out every iteration
    all_outputs = [encoded_prompts[b][:prefill_seq_len] for b in range(batch_size)]
    user_done = [False] * batch_size  # Keeps track when a user reaches EoD token

    iteration = 0
    # Keep running inference as long as there is a user in the batch still decoding or max tokens per user are decoded
    while users_decoding:
        iteration_time_start = time()
        curr_pos = generation_start_pos + iteration

        # Prepare inputs for decode mode (rotary embeddings, attention mask, padding)
        # TODO Move the attn mask to device
        decode_input, current_pos = prepare_inputs_ttnn(
            pt_encoded_input,
            curr_pos,
            model_args.dim,
            model_args.sliding_window,
            tt_model.device,
        )

        # Run ttnn mistral model
        tt_out = tt_model(decode_input, current_pos)
        tt_output_torch = ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)  # [batch, seq, hidden_dim]

        # If temperature is 0, does greedy decoding (top-1)
        tt_out_tok = sample(tt_output_torch, temperature=0, top_p=0.8)

        # TODO argmax on device
        # tt_out = ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)
        # tt_out = ttnn.permute(tt_out, (2, 1, 0, 3))
        # tt_out = ttnn.reshape(tt_out, (tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]))  # Squeeze(1)
        # tt_out_argmax = ttnn.experimental.tensor.argmax(tt_out, dim=-1)
        # Typecast from bf16 to uint32 for embedding
        # tt_out_tok = ttnn.clone(tt_out_argmax, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.uint32)
        # tt_out_tok = ttnn.experimental.tensor.typecast(tt_out_tok, dtype=ttnn.uint32)

        if iteration < input_mask.shape[1]:  # If prefill
            # If token is pad token, start generating new token, otherwise, push the next prompt token to the model
            tt_out_tok = torch.where(
                input_mask[:, iteration], tt_decode_input[:, iteration], tt_out_tok[:, 0]
            ).unsqueeze(1)

        # Save output token to print out later
        for user in range(batch_size):
            user_tok = tt_out_tok[user].tolist()
            if user_tok[0] != 28803 and user_done[user] == False:  # Stop saving the ouput after hitting the EOS token
                all_outputs[user].append(user_tok[0])
            else:
                user_done[user] = True
                if (
                    iteration < input_mask.shape[1]
                ):  # Still in prefill, so ignore EOS token and save the generated token
                    all_outputs[user].append(user_tok[0])
                else:
                    logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                    if all(user_done):
                        users_decoding = False

        if embed_on_device:
            tt_out_tok = ttnn.from_torch(tt_out_tok, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            pt_encoded_input = tt_embd(tt_out_tok)
        else:
            pt_encoded_input = embd(tt_out_tok)

        # Print out generated outputs for each user at the end of every iteration
        iteration_time = time() - iteration_time_start
        tokens_per_second_per_user = 1 / iteration_time
        # Print out generated outputs for each user at the end of every iteration
        if not is_ci_env:
            if len(user_input) == 1:
                logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs[0]))))
            else:
                for user in range(batch_size):
                    text = "".join(tokenizer.decode(all_outputs[user]))
                    if len(text) > 100:
                        text = "..." + text[-97:]
                    text = text.replace("\n", " ")
                    logger.info("[User {}] {}".format(user, text))

        # Always print perf at every iteration
        logger.info(
            f"Iteration {iteration}: {1000*iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
        )

        iteration += 1

        # Upper limit of generated tokens for each user (to avoid infinite generation in case eos is not seen)
        if iteration >= max_generated_tokens:
            users_decoding = False

    # In CI only print the final generated output to avoid spamming the logs
    if is_ci_env:
        if len(user_input) == 1:
            logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs[0]))))
        else:
            for user in range(batch_size):
                text = "".join(tokenizer.decode(all_outputs[user]))
                logger.info("[User {}] {}".format(user, text))


# Avoid running this test when in CI
@pytest.mark.skipif(os.getenv("CI") == "true", reason="Non-CI tests")
@pytest.mark.parametrize(
    "input_prompts, instruct_weights",
    [
        ("models/demos/wormhole/mistral7b/demo/input_data_prefill_128.json", False),
        ("models/demos/wormhole/mistral7b/demo/input_data_questions_prefill_128.json", True),
    ],
    ids=["general_weights", "instruct_weights"],
)
def test_mistral7B_demo(device, use_program_cache, input_prompts, instruct_weights, is_ci_env):
    return run_mistral_demo(
        user_input=input_prompts, batch_size=32, device=device, instruct_mode=instruct_weights, is_ci_env=is_ci_env
    )


# CI only runs general-weights demo
@pytest.mark.skipif(not os.getenv("CI") == "true", reason="CI-only test")
@pytest.mark.parametrize(
    "input_prompts, instruct_weights",
    [
        ("models/demos/wormhole/mistral7b/demo/input_data_questions_prefill_128.json", False),
    ],
    ids=[
        "general_weights",
    ],
)
def test_mistral7B_demo_CI(device, use_program_cache, input_prompts, instruct_weights, is_ci_env):
    return run_mistral_demo(
        user_input=input_prompts, batch_size=32, device=device, instruct_mode=instruct_weights, is_ci_env=is_ci_env
    )
