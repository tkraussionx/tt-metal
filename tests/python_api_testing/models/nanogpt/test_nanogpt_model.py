from pathlib import Path
import sys

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


def run_nanogpt_block_test(device):
    # Prepare input

    model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
    sd = model_hf.state_dict()
    model_hf.eval()

    torch.manual_seed(0)

    test_in = torch.randint(10,700, (1,16) )

    pt_attn = model_hf
    pt_out = model_hf.forward(test_in)

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


    tt_out = tt_model.forward(
        test_in,
        device
    )

    tt_out_converted = nanogpt_utils.tt2torch_tensor(tt_out[0])

    does_pass, pcc_message = comp_pcc(pt_out[0], tt_out_converted, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("nanogpt_block: Passed!")
    else:
        logger.warning("nanogpt_block: Failed!")

    assert does_pass


def test_nanogpt_block():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_nanogpt_block_test(device)
    tt_lib.device.CloseDevice(device)

if __name__ == "__main__":
    test_nanogpt_block()
