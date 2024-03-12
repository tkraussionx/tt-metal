# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import json
from loguru import logger
import ttnn
from models.demos.mistral7b.tt.mistral_common_ttnn import (
    generate_cos_sin_cache_ttnn,
    prepare_inputs_ttnn,
)
from models.demos.mistral7b.tt.mistral_model_ttnn import TtTransformer
from models.demos.mistral7b.tt.model_config_ttnn import TtModelArgs
from models.demos.mistral7b.reference.tokenizer import Tokenizer


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


def preprocess_inputs(input_prompts, tokenizer, model_args, dtype, embd, instruct, device):
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
    tt_cos_cached, tt_sin_cached = generate_cos_sin_cache_ttnn(
        [device], model_args.head_dim, model_args.max_seq_len * 2, 10000, dtype
    )

    seqlen = 1  # Generating one token per user at a time
    # Select the first token from the prompts for initial decoding
    pt_tokenized_inputs = torch.tensor(input_tokens)
    emb_inputs = embd(pt_tokenized_inputs[:, 0]).view(model_args.max_batch_size, seqlen, -1)

    return emb_inputs, tt_cos_cached, tt_sin_cached, pt_tokenized_inputs, max_prompt_len, input_mask


def run_mistral_demo(user_input, batch_size, device):
    ttnn.enable_program_cache()

    assert batch_size == 32, "Batch size must be 32"

    instruct_mode = True

    dtype = ttnn.bfloat8_b

    logger.info(f"Reading inputs...")
    if len(user_input) == 1:
        input_prompts = user_input * 32  # Always process 32 users
    else:
        input_prompts = load_inputs(user_input, 32)

    # Load model args, weights, and tokenizer
    # Specify model_base_path=<MISTRAL_WEIGHTS_PATH> below to use your own weights
    model_args = TtModelArgs(instruct=instruct_mode)  # TtModelArgs(model_base_path=<weights_path>)

    tokenizer = Tokenizer(model_args.tokenizer_path)

    logger.info("Loading weights...")
    if instruct_mode:  # Instruct weights are divided into 3 checkpoints
        state_dict = {}
        for i in range(3):
            state_dict_i = torch.load(model_args.consolidated_weights_path(i + 1), map_location="cpu")
            state_dict.update(state_dict_i)
        # Update state_dict keys to match the generative keys (saves modyfing the mistal modules)
        state_dict = {
            model_args.key_mapping[key]: value for key, value in state_dict.items() if key in model_args.key_mapping
        }
        # Match pad token to eos token when using instruct weights
        # tokenizer._model.pad_id = tokenizer._model.eos_id
    else:  # Generative weights are consolidadted into a single checkpoint
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

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    # Preprocess initial prompt inputs
    tt_decode_input, tt_cos_cached, tt_sin_cached, pt_encoded_input, max_prompt_len, input_mask = preprocess_inputs(
        input_prompts, tokenizer, model_args, dtype, embd, instruct_mode, device
    )

    # TODO should we just change the pad after initial pad of the inputs?
    if instruct_mode:
        tokenizer._model.pad_id = tokenizer._model.eos_id

    # Load TTNN mistral model
    logger.info("Loading weights to device...")
    tt_model = TtTransformer(
        args=model_args,
        device=device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype, instruct=instruct_mode),
        layers=list(range(model_args.n_layers)),
        tt_cos_cached=tt_cos_cached,
        tt_sin_cached=tt_sin_cached,
    )
    logger.info("Finished loading weights to device. Starting inference...")

    generation_start_pos = 0
    max_generated_tokens = 100
    users_decoding = True

    # Keep track of generated outputs to print out every iteration
    all_outputs = [[] for _ in range(batch_size)]
    user_done = [False] * batch_size  # Keeps track when a user reaches EoD token

    iteration = 0
    # Keep running inference as long as there is a user in the batch still decoding or max tokens per user are decoded
    while users_decoding:
        start_pos = generation_start_pos + iteration

        # Prepare inputs for decode mode (rotary embeddings, attention mask, padding)
        decode_input, start_pos, attn_mask, current_pos, rot_mat = prepare_inputs_ttnn(
            tt_decode_input,
            start_pos,
            model_args.dim,
            model_args.head_dim,
            model_args.sliding_window,
            model_args.max_seq_len,
            tt_model.device,
        )

        # Run ttnn mistral model
        tt_out = tt_model(decode_input, start_pos, current_pos, attn_mask, rot_mat)
        # Convert ttnn tensor to torch tensor
        tt_output_torch = ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        tt_out_tok = torch.argmax(tt_output_torch, dim=-1)
        if iteration < input_mask.shape[1]:  # If prefill
            # If token is pad token, start generating new token, otherwise, push the next prompt token to the model
            tt_out_tok = torch.where(
                input_mask[:, iteration], pt_encoded_input[:, iteration], tt_out_tok[:, 0]
            ).unsqueeze(1)

        # Save output token to print out later
        for user in range(batch_size):
            user_tok = tt_out_tok[user].tolist()
            if user_tok[0] != tokenizer.eos_id:  # Stop saving the ouput after hitting the EOS token
                all_outputs[user].append(user_tok[0])
            else:
                if (
                    iteration < input_mask.shape[1]
                ):  # Still in prefill, so ignore EOS token and save the generated token
                    all_outputs[user].append(user_tok[0])
                else:
                    print(f"[User {user}] Finished decoding at iteration {iteration}")
                    user_done[user] = True
                    if all(user_done):
                        users_decoding = False

        tt_decode_input = embd(tt_out_tok)

        # Print out generated outputs for each user at the end of every iteration
        if len(user_input) == 1:
            logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs[0]))))
        else:
            for user in range(batch_size):
                logger.info("[User {}] {}".format(user, "".join(tokenizer.decode(all_outputs[user]))))
                # logger.info("[User {}] {}".format(user, ",".join([str(tok) for tok in all_outputs[user]])))

        iteration += 1

        # Upper limit of generated tokens for each user (to avoid infinite generation in case eos is not seen)
        if iteration >= max_generated_tokens:
            users_decoding = False


def test_demo(
    user_input,
    device,
):
    return run_mistral_demo(user_input=user_input, batch_size=32, device=device)
