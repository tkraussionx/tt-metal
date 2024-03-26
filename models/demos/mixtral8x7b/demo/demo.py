# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger
import json
import ttnn
from models.demos.mixtral8x7b.tt.mixtral_common_ttnn import prepare_inputs_ttnn, sample
from models.demos.mixtral8x7b.tt.mixtral_model_ttnn import TtTransformer
from models.demos.mixtral8x7b.tt.model_config_ttnn import TtModelArgs
from models.demos.mixtral8x7b.reference.tokenizer import Tokenizer
from models.utility_functions import get_devices_for_t3000


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


def preprocess_inputs(input_prompts, tokenizer, model_args, dtype, instruct, devices):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    if instruct:
        # Pre append [INST] and post append [/INST] to the encoded prompts if instruct mode
        encoded_prompts = [tokenizer.encode("[INST] " + prompt + " [/INST]") for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]

    prompt_lens = [len(x) for x in encoded_prompts]

    # Pad the inputs to the max length prompt
    max_prompt_len = max(prompt_lens)
    input_tokens = torch.full((len(input_prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long)

    # TODO Change padding to be left padding instead of right padding
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)
    input_mask = input_tokens != tokenizer.pad_id

    num_users = len(encoded_prompts)
    logger.info(f"# of users: {num_users}")

    # Helper function supports multiple devices but we are only using one in this demo

    return input_tokens, max_prompt_len, input_mask


@torch.no_grad()
def run_mistral_demo(user_input, batch_size, devices):
    assert batch_size == 32, "Batch size must be 32"

    instruct_mode = False

    dtype = ttnn.bfloat8_b

    logger.info(f"Reading inputs...")
    if len(user_input) == 1:
        input_prompts = user_input * 32  # Always process 32 users
    else:
        input_prompts = load_inputs(user_input, 32)

    # Load model args, weights, and tokenizer
    # Specify model_base_path=<MISTRAL_WEIGHTS_PATH> below to use your own weights
    model_args = TtModelArgs()  # TtModelArgs(model_base_path=<weights_path>)

    tokenizer = Tokenizer(model_args.tokenizer_path)

    logger.info("Loading weights...")
    state_dict = torch.load(model_args.state_dict_path)

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    # Preprocess initial prompt inputs
    input_tokens, max_prompt_len, input_mask = preprocess_inputs(
        input_prompts, tokenizer, model_args, dtype, instruct_mode, devices
    )

    # TODO should we just change the pad after initial pad of the inputs?
    if instruct_mode:
        tokenizer._model.pad_id = tokenizer._model.eos_id

    # Load TTNN mistral model
    logger.info("Loading weights to device...")
    tt_model = TtTransformer(
        devices=devices,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
    )

    logger.info("Finished loading weights to device. Starting inference...")

    generation_start_pos = 0
    max_generated_tokens = 150
    users_decoding = True

    # Keep track of generated outputs to print out every iteration
    all_outputs = [[] for _ in range(batch_size)]
    user_done = [False] * batch_size  # Keeps track when a user reaches EoD token
    pt_decode_input = embd(input_tokens[:, 0]).view(batch_size, 1, -1)

    iteration = 0
    # Keep running inference as long as there is a user in the batch still decoding or max tokens per user are decoded
    while users_decoding:
        start_pos = generation_start_pos + iteration
        current_pos = start_pos % model_args.sliding_window

        # Prepare inputs for decode mode (rotary embeddings, attention mask, padding)
        decode_input, rot_mat = prepare_inputs_ttnn(
            pt_decode_input,
            model_args.dim,
            model_args.head_dim,
            model_args.max_seq_len,
            tt_model.devices,
        )

        # Run ttnn mistral model
        tt_out = tt_model(decode_input, start_pos, current_pos, rot_mat)
        # Convert ttnn tensor to torch tensor
        tt_output_torch = ttnn.to_torch(tt_out[0]).squeeze(1).view(32, 1, -1)  # [seq, batch, hidden_dim]

        # If temperature is 0, does greedy decoding (top-1)
        tt_out_tok = sample(tt_output_torch, temperature=0, top_p=0.8).view(32)
        if iteration < input_mask.shape[1]:  # If prefill
            # If token is pad token, start generating new token, otherwise, push the next prompt token to the model
            tt_out_tok = torch.where(input_mask[:, iteration], input_tokens[:, iteration], tt_out_tok[:])

        # Save output token to print out later
        for user in range(batch_size):
            user_tok = tt_out_tok[user].item()
            if user_tok != tokenizer.eos_id:  # Stop saving the ouput after hitting the EOS token
                all_outputs[user].append(user_tok)
            else:
                if (
                    iteration < input_mask.shape[1]
                ):  # Still in prefill, so ignore EOS token and save the generated token
                    all_outputs[user].append(user_tok)
                else:
                    logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                    user_done[user] = True
                    if all(user_done):
                        users_decoding = False

        pt_decode_input = embd(tt_out_tok).view(batch_size, 1, -1)

        # Print out generated outputs for each user at the end of every iteration
        if len(user_input) == 1:
            logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs[0]))))
        else:
            for user in range(batch_size):
                logger.info("[User {}] {}".format(user, "".join(tokenizer.decode(all_outputs[user]))))

        iteration += 1

        # Upper limit of generated tokens for each user (to avoid infinite generation in case eos is not seen)
        if iteration >= max_generated_tokens:
            users_decoding = False


def test_demo(all_devices):
    user_input = "models/demos/mixtral8x7b/reference/input_data.json"
    devices = get_devices_for_t3000(all_devices, 8)
    return run_mistral_demo(user_input=user_input, batch_size=32, devices=devices)
