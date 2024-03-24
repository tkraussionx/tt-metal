# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import torch
import torchvision

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import disable_memory_reports, enable_memory_reports
from models.experimental.resnet.reference.torch_functional_resnet import ResNet50, custom_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters


class ResNet50TestInfra:
    def __init__(self, device, batch_size, act_dtype, weight_dtype, math_fidelity):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.math_fidelity = math_fidelity

        self.torch_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).eval()
        self.torch_model.to(torch.bfloat16)

        self.input_shape = (batch_size, 3, 224, 224)
        self.torch_input_tensor = torch.rand(self.input_shape, dtype=torch.float32)
        self.torch_input_tensor = self.torch_input_tensor.to(torch.bfloat16)
        self.torch_output_tensor = self.torch_model(self.torch_input_tensor)

        self.parameters = preprocess_model_parameters(
            # model_name=f"torch_functional_resnet",
            initialize_model=lambda: self.torch_model,
            convert_to_ttnn=lambda *_: False,
            custom_preprocessor=custom_preprocessor,
        )

        def model(x):
            return ResNet50(x, parameters=self.parameters)

        self.model = model

    def preprocess_torch_input(self, torch_input_tensor=None):
        self.input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        return self.input_tensor

    def run(self, torch_input_tensor=None):
        self.input_tensor = (
            self.torch_input_tensor if torch_input_tensor is None else self.preprocess_torch_input(torch_input_tensor)
        )
        self.output_tensor = self.model(self.input_tensor)
        return self.output_tensor

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        valid_pcc = 0.99
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

        logger.info(
            f"ResNet50 batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, math_fidelity={self.math_fidelity}, PCC={self.pcc_message}"
        )


def create_test_infra(device, batch_size, act_dtype, weight_dtype, math_fidelity):
    return ResNet50TestInfra(device, batch_size, act_dtype, weight_dtype, math_fidelity)


@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    (
        # (8, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.HiFi4),  ## pass
        # (8, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),  ## pass
        # (16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),  ## pass
        # (16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),  ## pass
        # (20, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),  ## pass
        (20, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),  ## pass
    ),
)
def test_resnet_50(device, batch_size, act_dtype, weight_dtype, math_fidelity):
    test_infra = create_test_infra(device, batch_size, act_dtype, weight_dtype, math_fidelity)
    test_infra.preprocess_torch_input()
    test_infra.run()
    test_infra.validate()


@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((20, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),  ## pass
)
def test_code_gen(device, batch_size, act_dtype, weight_dtype, math_fidelity):
    test_infra = create_test_infra(device, batch_size, act_dtype, weight_dtype, math_fidelity)
    with ttnn.tracer.trace():
        torch_input_tensor = test_infra.torch_input_tensor
        outputs = test_infra.torch_model(torch_input_tensor)
    ttnn.tracer.codegen(outputs)
