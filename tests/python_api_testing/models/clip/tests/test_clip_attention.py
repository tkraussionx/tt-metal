from pathlib import Path
import sys
from tests.python_api_testing.models.clip.tt.clip_attention import TtCLIPAttention
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")


from transformers import AutoProcessor
from transformers import CLIPModel as HF_CLIPModel
import torch
import pytest

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor
from tt.clip_attention import TtCLIPAttention
from clip_utils import setup_device_and_host, compare_and_display


# related to vision:
# hidden shape torch.Size([1, 50, 768]) None None False



# related to text
# hidden shape torch.Size([2, 7, 512]) attention mask: torch.Size([2, 1, 7, 7]) causal_attention_mask: torch.Size([2, 1, 7, 7]) output_attentions: False
@pytest.mark.parametrize(
    "input_shape",
    ((1, 1, 50, 768), ),
)
def test_clip_attention(input_shape, pcc=0.99):

    device, host = setup_device_and_host()


    hidden_state = torch.randn(input_shape)
    with torch.no_grad():
        HF_model = HF_CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        state_dict = HF_model.state_dict()

        reference = HF_model.vision_model.encoder.layers[0].self_attn
        # config = HF_model.config.text_config
        config = HF_model.config.vision_config
        HF_output = reference(hidden_state.squeeze(0))

        tt_hidden_state = torch_to_tt_tensor_rm(hidden_state, device, put_on_device=False)


        tt_layer = TtCLIPAttention(config, base_address="vision_model.encoder.layers.0.self_attn", state_dict=state_dict, device=device)

        tt_output = tt_layer(tt_hidden_state)
        tt_output = tt_to_torch_tensor(tt_output, host)


        compare_and_display(HF_output, tt_output, "clip_attention", pcc)
        tt_lib.device.CloseDevice(device)
