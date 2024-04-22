# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

from models.demos.falcon40b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.demos.falcon40b.tt.falcon_causallm import TtFalconCausalLM
from models.demos.falcon40b.tt.model_config import (
    get_model_config,
)


def test_download_weights(
    model_location_generator,
    get_tt_cache_path,
):
    model_name = model_location_generator("tiiuae/falcon-40b-instruct", model_subdir="Falcon")
    input_shape = [1, 32]
    num_devices = 8
    model_config = get_model_config("BFLOAT8_B-DRAM", "prefill", input_shape, num_devices)
    tt_cache_path = get_tt_cache_path(
        "tiiuae/falcon-40b-instruct", model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name)

    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    mock_device_array = [0, 1, 2, 3, 4, 5, 6, 7]
    base_url = ""
    num_layers = 60
    use_global_cos_sin_cache = True

    tt_FalconDecoder_model = TtFalconCausalLM(
        mock_device_array,
        state_dict,
        base_url,
        num_layers,
        configuration,
        2048,
        model_config,
        tt_cache_path,
        use_global_cos_sin_cache,
    )
