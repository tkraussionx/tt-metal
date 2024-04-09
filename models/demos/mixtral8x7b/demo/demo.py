# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger
import json
import ttnn
from models.demos.mixtral8x7b.tt.mixtral_common import prepare_inputs_ttnn, sample
from models.demos.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.mixtral8x7b.tt.mixtral_embedding import TtMixtralEmbedding
from models.demos.mixtral8x7b.tt.model_config import TtModelArgs
from models.demos.mixtral8x7b.reference.tokenizer import Tokenizer
from models.utility_functions import get_devices_for_t3000
from models.demos.mixtral8x7b.reference.model import Transformer


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

    logger.info(f"# of users: {len(encoded_prompts)}")
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)

    input_mask = input_tokens != tokenizer.pad_id
    input_mask = input_mask.int()  # from_torch doesn't support bool type

    # convert to ttnn tensor
    input_tokens_tt = [
        [
            ttnn.from_torch(
                input_tokens[:, i].unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            for device in devices
        ]
        for i in range(max_prompt_len)
    ]
    input_mask_tt = [
        [
            ttnn.from_torch(
                input_mask[:, i].unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            for device in devices
        ]
        for i in range(max_prompt_len)
    ]
    return input_tokens_tt, max_prompt_len, input_mask_tt


@torch.no_grad()
def run_mixtral_demo(user_input, batch_size, devices):
    assert batch_size == 32, "Batch size must be 32"

    instruct_mode = False

    dtype = ttnn.bfloat8_b

    logger.info(f"Reading inputs...")
    if len(user_input) == 1:
        input_prompts = user_input * 32  # Always process 32 users
    else:
        input_prompts = load_inputs(user_input, 32)

    # Load model args, weights, and tokenizer
    # Specify model_base_path=<mixtral_WEIGHTS_PATH> below to use your own weights
    model_args = TtModelArgs(devices[0], instruct=instruct_mode)  # TtModelArgs(model_base_path=<weights_path>)

    model_args.n_layers = 1  # TODO Remove for full model
    tokenizer = Tokenizer(model_args.tokenizer_path)

    logger.info("Loading weights...")
    state_dict = torch.load(model_args.state_dict_path)
    # If not using the full model, remove the layers that are not used
    keys_dict = list(state_dict.keys())[:]
    remv = [f"layers.{i}" for i in range(model_args.n_layers, 32)]
    for k in keys_dict:
        if any([r in k for r in remv]):
            state_dict.pop(k)

    # # Embedding on host
    # embd = Emb()
    # embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    logger.info("Loading weights finished!")

    # Preprocess initial prompt inputs
    input_tokens, max_prompt_len, input_mask = preprocess_inputs(
        input_prompts, tokenizer, model_args, dtype, instruct_mode, devices
    )

    # TODO should we just change the pad after initial pad of the inputs?
    if instruct_mode:
        tokenizer._model.pad_id = tokenizer._model.eos_id

    # Load TTNN mixtral model
    logger.info("Loading weights to device...")
    tt_model = TtTransformer(
        devices=devices,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
    )

    tt_embds = [
        TtMixtralEmbedding(
            device=devices[i],
            args=model_args,
            weight_cache_path=model_args.weight_cache_path(dtype),
            state_dict=state_dict,
            dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
        )
        for i in range(len(devices))
    ]

    logger.info("Finished loading weights to device.")

    # Prepare the first token embedding for each user
    # Each device does its own embedding
    decode_input_11BH = [tt_embds[i](input_tokens[0][i]) for i in range(len(devices))]
    # Reshape and change row major to tile layout
    decode_input_11BH = [
        ttnn.reshape(decode_input_11BH[i], ttnn.Shape([1, 1, batch_size, model_args.dim])) for i in range(len(devices))
    ]
    decode_input_11BH = [ttnn.to_layout(decode_input_11BH[i], layout=ttnn.TILE_LAYOUT) for i in range(len(devices))]
    # decode_input_11BH = [ttnn.experimental.tensor.tilize(decode_input_11BH[i]) for i in range(len(devices))]
    # decode_input_11BH = [ttnn.experimental.tensor.tilize_with_val_padding(decode_input_11BH[i], ) for i in range(len(devices))]
    logger.info("Finished first token embedding. Starting inference...")

    generation_start_pos = 0
    max_generated_tokens = 15  # TODO change back to 120+

    # Keep track of generated outputs to print out every iteration
    all_outputs = [[] for _ in range(batch_size)]

    # Prepare inputs for decode mode (rotary embeddings, attention mask, padding)
    rot_mats = prepare_inputs_ttnn(
        None,
        model_args.dim,
        model_args.head_dim,
        model_args.max_seq_len,
        tt_model.devices,
    )

    # Keep running inference as long as there is a user in the batch still decoding or max tokens per user are decoded
    for iteration in range(max_generated_tokens):
        start_pos = generation_start_pos + iteration
        current_pos = start_pos % model_args.sliding_window

        # Run ttnn mixtral model
        tt_out_11BH = tt_model(decode_input_11BH, start_pos, current_pos, rot_mats)

        for i in range(len(devices)):
            # TODO Update argmax to ttnn when available
            tt_out_B11B = ttnn.experimental.tensor.argmax(tt_out_11BH[i], dim=-1)
            tt_out_1B = ttnn.reshape(tt_out_B11B[:1, :, :, :], ttnn.Shape([1, batch_size]))
            print(f"argmax_out_1B dtype = {tt_out_1B.dtype}")
            print(f"argmax_out_1B shape = {tt_out_1B.shape}")

            if iteration < max_prompt_len:
                print("ttnn where...")
                decode_input_1B = ttnn.where(input_mask[iteration][i], input_tokens[iteration][i], tt_out_1B)
                # Input mask shape: ttnn.Shape([1[32], 32])
                # Input tokens shape: Shape([1, 32]), , layout=Layout::ROW_MAJOR)
                # tt_out_1B shape: Shape([1, 32]), layout=Layout::ROW_MAJOR)
                decode_input_1B = tt_out_1B

            print(f"decode_input_1B to feed embedding shape = {decode_input_1B.shape}")
            # embed inputs
            decode_input_1BH = tt_embds[i](decode_input_1B)
            decode_input_11BH = ttnn.reshape(decode_input_1BH, ttnn.Shape([1, 1, batch_size, model_args.dim]))
            decode_input_11BH = ttnn.to_layout(decode_input_11BH, layout=ttnn.TILE_LAYOUT)

            print(f"New decode input_11BH = {decode_input_11BH}")

        # Convert ttnn tensor to torch tensor and print decoded output (from device 0)
        tt_output_torch = ttnn.to_torch(decode_input_1B[0])
        for user in range(batch_size):
            user_tok = tt_output_torch[user].item()
            if user_tok != tokenizer.eos_id:  # Stop saving the ouput after hitting the EOS token
                all_outputs[user].append(user_tok)

        # Print out generated outputs for each user at the end of every iteration
        if len(user_input) == 1:
            logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs[0]))))
        else:
            for user in range(batch_size):
                logger.info("[User {}] {}".format(user, "".join(tokenizer.decode(all_outputs[user]))))


def test_demo(all_devices, user_input="models/demos/mixtral8x7b/reference/input_data.json"):
    devices = get_devices_for_t3000(all_devices, 8)
    return run_mixtral_demo(user_input=user_input, batch_size=32, devices=devices)
