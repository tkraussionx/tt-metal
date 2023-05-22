from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib

from transformers import BloomForCausalLM
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger
import python_api_testing.models.bloom.bloom_causal_lm as bloom_causal_lm


def run_bloom_causal_lm_test(device):
    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    hugging_bloom_reference_model.eval()

    config = hugging_bloom_reference_model.config
    state_dict = hugging_bloom_reference_model.state_dict()

    tt_bloom_causal_lm = bloom_causal_lm.TtBloomForCausalLM(config, state_dict, device)
    pt_bloom_causal_lm = hugging_bloom_reference_model

    # Prepare input
    torch.manual_seed(0)
    input_ids = torch.randint(0, 100, (1, 64))

    pt_out = pt_bloom_causal_lm.forward(input_ids)
    print("PT finished")

    tt_out = tt_bloom_causal_lm.forward(device, input_ids)
    print("TT finished")

    pt_out = pt_out[0]
    tt_out = tt_out[0]
    tt_out = tt_out.squeeze(0)
    tt_out = tt_out.squeeze(0)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.50)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_causal_lm: Passed!")
    else:
        logger.warning("bloom_causal_lm: Failed!")

    assert does_pass


def test_bloom_causal_lm():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_bloom_causal_lm_test(device)
    tt_lib.device.CloseDevice(device)


if __name__ == "__main__":
    test_bloom_causal_lm()
