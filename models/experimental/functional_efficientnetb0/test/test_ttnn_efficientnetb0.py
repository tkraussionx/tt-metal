# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_efficientnetb0.reference import efficientnetb0
from models.experimental.functional_efficientnetb0.tt.model_preprocessing import (
    create_efficientnetb0_input_tensors,
    create_efficientnetb0_model_parameters,
)
from models.experimental.functional_efficientnetb0.tt import ttnn_efficientnetb0


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize(
    "to_pass",
    [True, False],
)
def test_efficientnetb0_mbconv(device, use_program_cache, reset_seeds, to_pass):
    state_dict = torch.load("models/experimental/functional_efficientnetb0/test/efficientnet-b0-355c32eb.pth")
    ds_state_dict = {k: v for k, v in state_dict.items()}

    torch_model = efficientnetb0.Efficientnetb0()

    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    ref_model = torch_model._blocks2

    torch_input, ttnn_input = create_efficientnetb0_input_tensors(device, 1, 24, 56, 56)
    torch_output = ref_model(torch_input, to_pass=to_pass)

    parameters = create_efficientnetb0_model_parameters(ref_model, torch_input, device=device)

    ttnn_model = ttnn_efficientnetb0.MBConvBlock(device, parameters, batch=1, is_depthwise_first=False)

    ttnn_output = ttnn_model(ttnn_input, to_pass)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)

    assert_with_pcc(torch_output, ttnn_output, 0.99)
