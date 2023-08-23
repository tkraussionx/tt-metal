import torch
import pytest
from loguru import logger

import tt_lib
from tests.python_api_testing.models.falcon.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from tests.python_api_testing.models.falcon.falcon_attention import TtFalconAttention
from tests.python_api_testing.models.falcon.model_config import (
    get_model_config,
    get_tt_cache_path,
)
from tests.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class PytorchFalconAttentionModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.attention = hf_reference_model.transformer.h[layer_num].self_attention

        # Disable dropout
        self.attention.eval()

    def forward(self, x, alibi, attention_mask):
        result = self.attention(
            hidden_states=x, alibi=alibi, attention_mask=attention_mask
        )[0]
        return result


def run_test_FalconAttention_inference(
    device,
    model_version,
    batch,
    seq_len,
    pcc,
    model_config,
    tt_cache_path,
    model_location_generator,
):
    model_name = model_location_generator(model_version, model_subdir="Falcon")

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)
    attention_input = (torch.rand(batch, seq_len, configuration.hidden_size) * 2) - 1
    layer_num = 0
    base_url = "transformer.h"
    max_position_embeddings = 2048

    # Generate attention_mask -----------------------------------------------------------
    # TODO: Generate attention_mask on device
    q_len, kv_seq_len = seq_len, seq_len
    attention_mask_bool = torch.ones(batch, 1, q_len, kv_seq_len, dtype=bool).triu(
        diagonal=1
    )
    tt_attention_mask = torch2tt_tensor(
        (attention_mask_bool * -100000).expand(-1, configuration.n_head, -1, -1), device
    )

    # PyTorch output --------------------------------------------------------------------
    pytorch_FalconAttention_model = PytorchFalconAttentionModel(
        hugging_face_reference_model, layer_num
    )
    pytorch_out = pytorch_FalconAttention_model(
        attention_input, alibi=None, attention_mask=attention_mask_bool
    )

    # TT hardware execution -------------------------------------------------------------
    tt_FalconAttention_model = TtFalconAttention(
        device,
        state_dict,
        base_url,
        layer_num,
        configuration.hidden_size,
        configuration.n_head,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    )

    tt_attention_input = attention_input.unsqueeze(1)
    tt_attention_input = torch2tt_tensor(tt_attention_input, device)

    tt_out, past_key_value = tt_FalconAttention_model(
        tt_attention_input, alibi=None, attention_mask=tt_attention_mask
    )
    tt_out = tt2torch_tensor(tt_out).squeeze(1)

    # check outputs ----------------------------------------------------------------------
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Falcon Attention output Passed!")
    else:
        logger.warning("Falcon Attention output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (
        (
            "tiiuae/falcon-7b-instruct",
            1,
            128,
            0.98,
        ),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test_FalconAttention_inference(
    model_version,
    batch,
    seq_len,
    pcc,
    model_config_str,
    model_location_generator,
):
    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    run_test_FalconAttention_inference(
        device,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
    tt_lib.device.CloseDevice(device)
