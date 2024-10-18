# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_yolov11.reference import yolov11
from models.experimental.functional_yolov11.tt.model_preprocessing import (
    create_yolov11_input_tensors,
    create_yolov11_model_parameters,
)
from models.experimental.functional_yolov11.tt import ttnn_yolov11


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov11(device, use_program_cache, reset_seeds):
    torch_input, ttnn_input = create_yolov11_input_tensors(device)

    torch_model = yolov11.YoloV11()
    torch_model.eval()
    torch_output = torch_model(torch_input)

    parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)
    ttnn_model = ttnn_yolov11.YoloV11(device, parameters)
    ttnn_output = ttnn_model(ttnn_input)

    print(ttnn_output.shape, torch_output.shape)

    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99999)
