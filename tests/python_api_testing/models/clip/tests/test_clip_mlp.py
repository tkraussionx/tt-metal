from pathlib import Path
import sys
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
from tt.clip_mlp import TtCLIPMLP
from clip_utils import setup_device_and_host, compare_and_display


@pytest.mark.parametrize(
    "input_shape",
    ((1, 2, 7, 512), ),
)
def test_clip_mlp(input_shape, pcc=0.99):

    device, host = setup_device_and_host()


    hidden_state = torch.randn(input_shape)
    with torch.no_grad():
        HF_model = HF_CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        state_dict = HF_model.state_dict()

        reference = HF_model.text_model.encoder.layers[0].mlp
        config = HF_model.config.text_config
        HF_output = reference(hidden_state)

        tt_hidden_state = torch_to_tt_tensor_rm(hidden_state, device, put_on_device=False)


        tt_layer = TtCLIPMLP(config, base_address="text_model.encoder.layers.0.mlp", state_dict=state_dict, device=device)

        tt_output = tt_layer(tt_hidden_state)
        tt_output = tt_to_torch_tensor(tt_output, host)


        compare_and_display(HF_output, tt_output, "clip_mlp", pcc)
        tt_lib.device.CloseDevice(device)
