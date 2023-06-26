from pathlib import Path
import sys


import os
import pickle
import tiktoken

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib

from transformers import GPT2LMHeadModel

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger
import python_api_testing.models.nanogpt.nanogpt_utils as nanogpt_utils
import python_api_testing.models.nanogpt.nanogpt_block as nanogpt_block
import python_api_testing.models.nanogpt.nanogpt_attention as nanogpt_attention
import python_api_testing.models.nanogpt.nanogpt_model as nanogpt_model


def pad_input_2(tensor, value):
    len = tensor.shape[1]

    if len % 2 == 0:
        return tensor

    padded_len = ((len // 2) + 1) * 2

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([pad_tensor, tensor], dim=1)

    return tensor

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 20 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = None # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device_select = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------


def run_nanogpt_model_test(device):
    # Prepare input

    model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
    sd = model_hf.state_dict()
    model_hf.eval()
    torch.manual_seed(0)

    test_in = torch.randint(10,700, (1,16) )

    model_type = 'gpt2'

    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    }[model_type]

    config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    config_args['bias'] = True # always True for GPT model checkpoints

    config = nanogpt_attention.GPTConfig(**config_args)

    tt_test_in = nanogpt_utils.torch2tt_tensor(test_in, device)

    tt_model = nanogpt_model.TtGPT(config, sd, device)

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    text = 'One joke about Bill gates: '
    start_ids = encode(text)
    while (len(start_ids)%2==1):
        text = text + ' '
        start_ids = encode(text)

        print('Startiox')

        print(start_ids)


    #attention_mask = pad_input_2(tokenized.attention_mask, 0)

    #print(start_ids.shape)
    x = (torch.tensor(start_ids, dtype=torch.long, device='cpu')[None, ...])
    print(x.shape)
    #x = x.squeeze(0)

    print('BEGIN-----')
    print('X-SHAPE-------')
    print(x.shape)
        #x = pad_input_2(x, 0)
    y = tt_model.generate(x, device, max_new_tokens, temperature=temperature, top_k=top_k)
    print(decode(y[0].tolist()))


def test_nanogpt_model():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_nanogpt_model_test(device)
    tt_lib.device.CloseDevice(device)

if __name__ == "__main__":
    test_nanogpt_model()
