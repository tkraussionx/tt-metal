# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model
from models.experimental.functional_segformer.tt.ttnn_segformer_mlp import (
    TtSegformerMLP,
)

from models.experimental.functional_segformer.reference.segformer_mlp import (
    SegformerMLP,
)
from transformers import SegformerForSemanticSegmentation


@pytest.mark.parametrize(
    "input_dim, mlp_id, batch_size, height, width,",
    [
        (32, 0, 1, 128, 128),
        (64, 1, 1, 64, 64),
        (160, 2, 1, 32, 32),
        (256, 3, 1, 16, 16),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_mlp(device, input_dim, batch_size, height, width, mlp_id):
    torch_input_tensor = torch.randn(batch_size, input_dim, height, width)
    torch_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    config = torch_model.config

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    torch_model = torch_model.decode_head.linear_c[mlp_id]
    reference_model = SegformerMLP(config, input_dim)

    sd = torch_model.state_dict()
    reference_model.load_state_dict(sd)

    reference_model.eval()
    torch_output = reference_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: reference_model,
        run_model=lambda model: model(torch_input_tensor),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtSegformerMLP()
    ttnn_output = ttnn_model(ttnn_input_tensor, parameters=parameters)

    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
