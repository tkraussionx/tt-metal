# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import json
import pytest
from loguru import logger
from time import time
import multiprocessing
import asyncio

# Set Mixtral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["MIXTRAL_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    load_inputs,
    preprocess_inputs,
    prepare_inputs_ttnn,
    get_single_rot_mat,
    sample,
    cache_attention,
)
from models.demos.t3000.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.t3000.mixtral8x7b.tt.mixtral_embedding import TtMixtralEmbedding
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer
from models.utility_functions import nearest_32

from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


class Demo:
    def __init__(self, device_mesh, instruct_mode):
        self.device_mesh = device_mesh
        self.instruct_mode = instruct_mode
        self.dtype = ttnn.bfloat8_b
        self.embed_on_host = False
        self.seqlen = 1
        self.batch_size = 32
        self.model_args = TtModelArgs(self.device_mesh.get_device(0), instruct=self.instruct_mode)
        self.tokenizer = Tokenizer(self.model_args.tokenizer_path)
        self.state_dict = torch.load(self.model_args.state_dict_path)

        # Embedding on host
        if self.embed_on_host:
            self.embd = Emb()
            self.embd.load_state_dict({"emb.weight": self.state_dict["tok_embeddings.weight"]})
        else:
            self.tt_embds = TtMixtralEmbedding(
                device_mesh=device_mesh,
                args=self.model_args,
                weight_cache_path=self.model_args.weight_cache_path(self.dtype),
                state_dict=self.state_dict,
                dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
            )

        if instruct_mode:
            self.tokenizer._model.pad_id = self.tokenizer._model.eos_id

        # Load TTNN mixtral model
        logger.info("Loading weights to device...")
        self.tt_model = TtTransformer(
            device_mesh=device_mesh,
            state_dict=self.state_dict,
            args=self.model_args,
            layers=list(range(self.model_args.n_layers)),
            dtype=self.dtype,
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
            self.device_mesh,
        )
        cache_attention(
            self.device_mesh,
            self.state_dict,
            self.model_args,
            current_rot_mat,
            rot_matrix,
            start_pos,
            max_generated_tokens,
            self.dtype,
        )

    async def model_process(self, initial_input, start_pos, iteration):
        if self.embed_on_host:
            pt_decode_input = self.embd(initial_input).view(self.batch_size, 1, -1)
            decode_input_11BH, attn_mask = prepare_inputs_ttnn(
                pt_decode_input,
                self.model_args.dim,
                start_pos,
                self.model_args,
                self.device_mesh,
            )
        else:
            print("creating input tensor", initial_input.shape)
            emd_input_B1 = ttnn.reshape(initial_input, ttnn.Shape([self.batch_size, 32]))
            emd_input_B1 = ttnn.experimental.tensor.typecast(emd_input_B1, dtype=ttnn.uint32)
            emd_input_B1 = ttnn.to_layout(emd_input_B1, layout=ttnn.ROW_MAJOR_LAYOUT)
            decode_input_1BH = self.tt_embds(emd_input_B1)[:, :1, :]
            decode_input_11BH = ttnn.reshape(decode_input_1BH, ttnn.Shape([1, 1, self.batch_size, self.model_args.dim]))
            decode_input_11BH = ttnn.to_layout(decode_input_11BH, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
            print("created input tensor")
            # Attention mask
            padded_layer_past_len = nearest_32(start_pos + 1)
            attn_mask = torch.zeros(1, 32, 32, padded_layer_past_len)
            attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min

            attn_mask = ttnn.as_tensor(
                attn_mask,
                device=self.device_mesh,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.create_sharded_memory_config(
                    shape=(32, padded_layer_past_len),
                    core_grid=ttnn.CoreGrid(y=4, x=8),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                ),
                mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
                cache_file_name=self.model_args.weight_cache_path(ttnn.bfloat4_b) / (f"attention_mask.{start_pos}"),
            )

        # Run ttnn mixtral model
        print("starting model")
        tt_out_11BH = decode_input_11BH  # self.tt_model(decode_input_11BH, start_pos, start_pos, attn_mask)
        print("model done", tt_out_11BH)
        if self.embed_on_host:
            # Convert ttnn tensor to torch tensor
            tt_output_torch = (
                ttnn.to_torch(tt_out_11BH, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=0))[0]
                .squeeze(1)
                .view(self.batch_size, 1, -1)
                .detach()
                .float()
            )
            # Argmax on host to get the new generated tokens
            output_token = sample(tt_output_torch, temperature=0, top_p=0.8)
            # Update the users that are still in prefill and the ones generating new tokens
            if iteration < self.max_prompt_len:
                output_token = torch.where(
                    self.input_mask_pt[:, iteration], self.input_tokens_pt[:, iteration], output_token[:, 0]
                ).unsqueeze(1)
        else:
            tt_out_11BH = ttnn.experimental.tensor.pad(
                ttnn.experimental.tensor.typecast(tt_out_11BH, dtype=ttnn.bfloat16),
                [1, 1, 32, 32768],
                [0, 0, 0, 0],
                pad_value=-99.99,
            )
            print("0", tt_out_11BH)
            breakpoint()
            tt_values_11BK, tt_indices_11BK = ttnn.topk(tt_out_11BH, 32)
            print("1", tt_indices_11BK)
            tt_out_11B1 = tt_indices_11BK[:, :, :, :1]
            if iteration < self.max_prompt_len:
                decode_input_11B1 = ttnn.where(self.input_mask[iteration], self.input_tokens_tt[iteration], tt_out_11B1)
            else:
                decode_input_11B1 = tt_out_11B1
            output_token = decode_input_11B1
            print("output token", output_token.shape)

        # self.model_output_queue.put(output_token)
        return output_token

    async def decode_process(self, tt_token_batch, iteration, iteration_time_start):
        if not self.embed_on_host:
            tt_token_batch = ttnn.to_torch(tt_token_batch, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=0))[
                0
            ].view(self.batch_size)
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
            self.input_mask,
            self.input_tokens_pt,
            self.input_mask_pt,
        ) = preprocess_inputs(
            input_prompts, self.tokenizer, self.model_args, self.dtype, self.instruct_mode, self.device_mesh
        )

        generation_start_pos = 0
        max_generated_tokens = 20

        self.cache_model_attention(generation_start_pos, max_generated_tokens)

        initial_input = self.input_tokens_pt[:, 0] if self.embed_on_host else self.input_tokens_tt[0]
        decode_task = None

        # Create a Manager object to manage shared state
        # manager = multiprocessing.Manager()

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

            # self.model_output_queue = multiprocessing.Queue()
            # model_process_proc = multiprocessing.Process(target=self.model_process, args=(initial_input, start_pos, iteration))
            # model_process_proc.start()
            # model_process_proc.join()  # Wait for model processing to finish
            # initial_input = self.model_output_queue.get()

            # Start decoding in a separate process if there is an output token
            # if decode_process_proc:
            #     decode_process_proc.join()  # Wait for decoding to finish
            # decode_process_proc = multiprocessing.Process(target=self.decode_process, args=(initial_input, iteration, iteration_time_start))
            # decode_process_proc.start()

            initial_input = await asyncio.create_task(self.model_process(initial_input, start_pos, iteration))
        #     if iteration > 0:
        #         await decode_task

        #     decode_task = asyncio.create_task(self.decode_process(initial_input, iteration, iteration_time_start))

        # await decode_task

        if os.getenv("CI") == "true":
            self.CI_checks()

    def CI_checks(self):
        # In CI only print the final generated output to avoid spamming the logs
        for user in range(self.batch_size):
            logger.info("[User {}] {}".format(user, "".join(self.tokenizer.decode(self.all_outputs[user]))))

        # When running in CI, check the output against the expected output to avoid accuracy regressions
        expected_output = "models/demos/t3000/mixtral8x7b/demo/expected_outputs.json"
        with open(expected_output, "r") as f:
            expected_out = json.load(f)
        assert (
            len(expected_out) >= self.batch_size * 2
        ), f"expected_outputs.json should have 64 outputs: 32 for general weights and 32 for instruct weights!"

        for i in range(self.batch_size):
            user_output = "".join(self.tokenizer.decode(self.all_outputs[i]))
            user_expect = expected_out[i]["output_general"]

            assert user_output == user_expect, f"Output for user {i} does not match expected output!"
        logger.info("[CI-Only] Output token validation passed!")


# Avoid running this test when in CI
@pytest.mark.skipif(os.getenv("CI") == "true", reason="Non-CI tests")
@pytest.mark.parametrize(
    "input_prompts, instruct_weights",
    [
        ("models/demos/t3000/mixtral8x7b/demo/input_data.json", False),
        ("models/demos/t3000/mixtral8x7b/demo/input_data_questions.json", True),
    ],
    ids=["general_weights", "instruct_weights"],
)
def test_mixtral8x7b_demo(t3k_device_mesh, use_program_cache, input_prompts, instruct_weights):
    start_time = time()
    mixtral_demo = Demo(t3k_device_mesh, instruct_weights)
    asyncio.run(mixtral_demo.run_demo(user_input=input_prompts))
    print(f"Total time: {time()-start_time}")


# CI only runs general-weights demo
@pytest.mark.skipif(not os.getenv("CI") == "true", reason="CI-only test")
@pytest.mark.parametrize(
    "input_prompts, instruct_weights",
    [
        ("models/demos/t3000/mixtral8x7b/demo/input_data.json", False),
    ],
    ids=[
        "general_weights",
    ],
)
def test_mixtral8x7b_demo_CI(t3k_device_mesh, use_program_cache, input_prompts, instruct_weights):
    start_time = time()
    mixtral_demo = Demo(t3k_device_mesh, instruct_weights)
    asyncio.run(mixtral_demo.run_demo(user_input=input_prompts))
    print(f"Total time: {time()-start_time}")
