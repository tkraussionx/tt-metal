# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn
from models.experimental.functional_unet.tests.common import check_pcc_conv, UNET_FULL_MODEL_PCC, verify_with_pcc


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_unet_model(batch, groups, device, use_program_cache, reset_seeds):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=True)
    model = unet_shallow_torch.UNet.from_random_weights(groups=1)
    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    for p in parameters:
        if "batch_size" in parameters[p]:
            parameters[p]["batch_size"] = 2

    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    torch_output_tensor = model(torch_input)
    output_tensor = ttnn_model.forward(ttnn_input)
    output_tensor = [
        ttnn.to_torch(x).reshape(2, 1056, 160, 1).permute(0, 3, 1, 2) for x in output_tensor
    ]  # 1, 1, BHW, C -> B, H, W, C
    output_tensor = torch.concat(output_tensor, dim=0)
    breakpoint()

    verify_with_pcc(torch_output_tensor, output_tensor, UNET_FULL_MODEL_PCC)
