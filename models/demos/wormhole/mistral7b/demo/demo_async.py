# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import json
from time import time
from loguru import logger
import os
import asyncio
import ttnn
import pytest
from models.demos.wormhole.mistral7b.tt.mistral_common import (
    prepare_inputs_ttnn,
    sample,
    get_single_rot_mat,
    cache_attention_rot,
)
from models.demos.wormhole.mistral7b.tt.mistral_model import TtTransformer
from models.demos.wormhole.mistral7b.tt.mistral_embedding import TtMistralEmbedding
from models.demos.wormhole.mistral7b.reference.tokenizer import Tokenizer


class HostEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


class Demo:
    def __init__(self, device, instruct_mode, is_ci_env):
        # Set Mistral flags for CI
        if (
            is_ci_env and instruct_mode
        ):  # Update paths for instruct mode, otherwise use default paths for general weights
            os.environ["MISTRAL_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/mistral-7B-v0.1/instruct/"
            os.environ["MISTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/mistral-7B-v0.1/instruct/"
            os.environ["MISTRAL_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/mistral-7B-v0.1/instruct/"

        # This module requires the env paths above for CI runs
        from models.demos.wormhole.mistral7b.tt.model_config import TtModelArgs

        self.device = device
        self.instruct_mode = instruct_mode
        self.dtype = ttnn.bfloat8_b
        self.embed_on_host = False
        self.seqlen = 1
        self.model_args = TtModelArgs(self.device, instruct=self.instruct_mode)
        self.batch_size = self.model_args.max_batch_size

        self.tokenizer = Tokenizer(self.model_args.tokenizer_path)
        self.state_dict = torch.load(self.model_args.consolidated_weights_path)

        # Embedding on host
        if self.embed_on_host:
            self.embd = HostEmbedding()
            self.embd.load_state_dict({"emb.weight": self.state_dict["tok_embeddings.weight"]})
        else:
            self.tt_embds = TtMistralEmbedding(
                device=device,
                args=self.model_args,
                weight_cache_path=self.model_args.weight_cache_path(self.dtype),
                state_dict=self.state_dict,
                dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
            )

        if instruct_mode:
            self.tokenizer._model.pad_id = self.tokenizer._model.eos_id
        self.model_args.n_layers = 1
        # Load TTNN mistral model
        logger.info("Loading weights to device...")
        self.tt_model = TtTransformer(
            args=self.model_args,
            device=device,
            dtype=self.dtype,
            state_dict=self.state_dict,
            weight_cache_path=self.model_args.weight_cache_path(self.dtype),
            layers=list(range(self.model_args.n_layers)),
            rot_mat=None,
            start_pos=0,
        )

    def read_inputs(self, user_input):
        if len(user_input) == 1:
            input_prompts = user_input * self.batch_size
        else:
            input_prompts = load_inputs(user_input, self.batch_size)
        return input_prompts

    def cache_model_attention(self, start_pos, max_generated_tokens):
        current_rot_mat, rot_matrix = get_single_rot_mat(
            self.model_args.head_dim,
            self.device,
        )
        cache_attention_rot(
            self.device,
            self.state_dict,
            self.model_args,
            current_rot_mat,
            self.dtype,
            max_generated_tokens,
        )

    async def model_process(self, initial_input, start_pos, iteration, current_rot_mat, rot_matrix):
        if self.embed_on_host:
            pt_decode_input = self.embd(initial_input).view(self.batch_size, 1, -1)
            decode_input_11BH, current_pos = prepare_inputs_ttnn(
                pt_decode_input,
                start_pos,
                self.model_args.dim,
                self.model_args.sliding_window,
                self.device,
            )
        else:
            print("creating input tensor", initial_input.shape)
            emd_input_B1 = ttnn.reshape(initial_input, ttnn.Shape([32, 32]))
            print("0", emd_input_B1.shape)
            emd_input_B1 = ttnn.typecast(emd_input_B1, dtype=ttnn.uint32)
            print("1", emd_input_B1.shape)
            emd_input_B1 = ttnn.to_layout(emd_input_B1, layout=ttnn.ROW_MAJOR_LAYOUT)
            print("2", emd_input_B1.shape)
            decode_input_1BH = self.tt_embds(emd_input_B1)[:, :1, :]
            print("3", decode_input_1BH.shape)
            decode_input_11BH = ttnn.reshape(decode_input_1BH, ttnn.Shape([1, 1, 32, self.model_args.dim]))
            print("3", decode_input_11BH.shape)
            decode_input_11BH = ttnn.to_layout(decode_input_11BH, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
            print("created input tensor")
        # Run ttnn mistral model
        print("starting model")
        tt_out_11BH = self.tt_model(decode_input_11BH, start_pos, rot_mat=current_rot_mat)
        print("model done", tt_out_11BH)
        if self.embed_on_host:
            # Convert ttnn tensor to torch tensor
            tt_output_torch = (
                ttnn.to_torch(tt_out_11BH)
                .squeeze(1)[:, : self.batch_size, :]
                .view(self.batch_size, 1, -1)
                .detach()
                .float()
            )
            print("tt_output_torch", tt_output_torch.shape)
            # Argmax on host to get the new generated tokens
            output_token = sample(tt_output_torch, temperature=0, top_p=0.8)
            # Update the users that are still in prefill and the ones generating new tokens
            if iteration < self.max_prompt_len:
                output_token = torch.where(
                    self.input_mask_pt[:, iteration], self.input_tokens_pt[:, iteration], output_token[:, 0]
                ).unsqueeze(1)
        else:
            tt_out_11BH_padded = ttnn.pad(
                ttnn.typecast(tt_out_11BH, dtype=ttnn.bfloat16),
                [1, 1, 32, 32768],  # closest power of 2 to vocab_size
                [0, 0, 0, 0],
                value=-99.99,
            )
            print("0", tt_out_11BH)
            tt_out_11BH.deallocate()
            tt_values_11BK, tt_indices_11BK = ttnn.topk(tt_out_11BH_padded, 32)
            # print("starting argmax")
            # tt_indices_11BK = ttnn.argmax(tt_out_11BH, dim=-1)
            print("1", tt_indices_11BK)
            tt_out_11B1 = tt_indices_11BK[:, :, :, :1]
            print("2", tt_out_11B1)
            if iteration < self.max_prompt_len:
                print(
                    "masking",
                    self.input_mask_tt[iteration].shape,
                    self.input_tokens_tt[iteration].shape,
                    tt_out_11B1.shape,
                )
                decode_input_11B1 = ttnn.where(
                    self.input_mask_tt[iteration], self.input_tokens_tt[iteration], tt_out_11B1
                )
            else:
                decode_input_11B1 = tt_out_11B1
            output_token = decode_input_11B1
            print("output token", output_token.shape)

        current_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)

        return output_token, current_rot_mat

    async def decode_process(self, tt_token_batch, iteration, iteration_time_start):
        if not self.embed_on_host:
            tt_token_batch = ttnn.to_torch(tt_token_batch)[:, :, : self.batch_size, :].view(self.batch_size)
        # Get the generated tokens for each user for printing in the log
        for user in range(self.batch_size):
            user_tok = int(tt_token_batch[user].item())
            if user_tok == self.tokenizer.eos_id:  # Stop saving the ouput after hitting the EOS token
                self.finished_generation[user] = True
            if self.finished_generation[user] == False:
                self.all_outputs[user].append(user_tok)

        iteration_time = time() - iteration_time_start
        tokens_per_second_per_user = 1 / iteration_time
        # Print out generated outputs for each user at the end of every iteration
        if os.getenv("CI") != "true":  # Avoid printing every iteration in CI
            for user in range(self.batch_size):
                logger.info("[User {}] {}".format(user, "".join(self.tokenizer.decode(list(self.all_outputs[user])))))

        # Always print iteration perf
        logger.info(
            f"Iteration {iteration}: {1000*iteration_time:.2f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({self.batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
        )

    async def run_demo(self, user_input):
        logger.info(f"Reading inputs...")
        input_prompts = self.read_inputs(user_input)

        # Preprocess initial prompt inputs
        (
            self.input_tokens_tt,
            self.max_prompt_len,
            self.input_mask_tt,
            self.input_tokens_pt,
            self.input_mask_pt,
        ) = preprocess_inputs(
            input_prompts, self.tokenizer, self.model_args, self.dtype, self.instruct_mode, self.device
        )

        generation_start_pos = 0
        max_generated_tokens = 20

        self.cache_model_attention(generation_start_pos, max_generated_tokens)
        current_rot_mat, rot_matrix = get_single_rot_mat(
            self.model_args.head_dim,
            self.device,
            start_pos=0,
        )
        initial_input = self.input_tokens_pt[:, 0] if self.embed_on_host else self.input_tokens_tt[0]
        decode_task = None

        # Keep track of generated outputs to print out every iteration
        self.all_outputs = [[] for _ in range(self.batch_size)]

        # Keep track of users that are done generating and stop printing their outputs
        self.finished_generation = [False] * self.batch_size

        for iteration in range(max_generated_tokens):
            # Check if all users have finished generating (reached EoS token). If so, stop decoding.
            if all(self.finished_generation):
                logger.info("All users have finished generating tokens")
                break

            iteration_time_start = time()
            start_pos = generation_start_pos + iteration

            initial_input, current_rot_mat = await asyncio.create_task(
                self.model_process(initial_input, start_pos, iteration, current_rot_mat, rot_matrix)
            )
            # if iteration > 0:
            #     await decode_task

            decode_task = asyncio.create_task(self.decode_process(initial_input, iteration, iteration_time_start))

            await decode_task

        if os.getenv("CI") == "true":
            self.CI_checks()

    def CI_checks(self):
        pass


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


def preprocess_inputs(input_prompts, tokenizer, model_args, dtype, instruct, device):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    if instruct:
        encoded_prompts = [tokenizer.encode("[INST] " + prompt + " [/INST]") for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]

    prompt_lens = [len(x) for x in encoded_prompts]

    # Pad the inputs to the max length prompt
    max_prompt_len = max(prompt_lens)
    input_tokens = torch.full((len(input_prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long)

    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)
    input_mask_bool = input_tokens != tokenizer.pad_id
    input_mask = input_mask_bool.int()  # from_torch doesn't support bool type

    # convert to ttnn tensor
    # Encoded input tokens need to be uint32 for embedding. Otherwise the dtype conversion to bfloat16 will change the tokenizer ID
    input_tokens_tt = [
        ttnn.from_torch(
            input_tokens[:, i].unsqueeze(1),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
        )
        for i in range(max_prompt_len)
    ]
    input_mask_tt = [
        ttnn.from_torch(
            input_mask[:, i].unsqueeze(1),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        for i in range(max_prompt_len)
    ]
    return input_tokens_tt, max_prompt_len, input_mask_tt, input_tokens, input_mask_bool


@pytest.mark.parametrize(
    "input_prompts, instruct_weights",
    [
        ("models/demos/wormhole/mistral7b/demo/input_data.json", False),
        ("models/demos/wormhole/mistral7b/demo/input_data_questions.json", True),
    ],
    ids=["general_weights", "instruct_weights"],
)
def test_mistral7b_demo(device, use_program_cache, input_prompts, instruct_weights, is_ci_env):
    if is_ci_env and instruct_weights == False:
        pytest.skip("CI demo test only runs instruct weights to reduce CI pipeline load (both are supported)")
    start_time = time()
    mistral_demo = Demo(device, instruct_weights, is_ci_env)
    asyncio.run(mistral_demo.run_demo(user_input=input_prompts))
    print(f"Total time: {time()-start_time}")
