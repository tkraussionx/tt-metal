# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from typing import Optional
import ttnn
from models.demos.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.demos.mamba.tt.full_model import TtTensorLoader
from models.demos.mamba.tt.mamba_one_step_ssm import TtMambaSSM
from models.demos.mamba.tt.transforms import MambaSsmBlockTransformer
from models.demos.mamba.tt import model_config
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)


class PytorchMambaSSM(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.block = hf_reference_model.layers[layer_num].mixer
        self.block.eval()

    def forward(self, x):
        result = self.block.ssm(x)
        return result


@pytest.mark.parametrize(
    "model_version, batch, pcc, cache_dir",
    (
        (
            "state-spaces/mamba-2.8b",
            32,
            0.99,
            None,
        ),
    ),
)
def test_mamba_ssm_inference(
    device: ttnn.Device,
    use_program_cache,
    model_version: MambaPretrainedModelName,
    batch: int,
    pcc: float,
    cache_dir: Optional[str],
):
    torch.manual_seed(0)

    LAYER_NUM = 0

    reference_model = MambaDecode.from_pretrained(model_version)
    reference_model.args.batch_size = batch

    d_in = reference_model.args.d_model * reference_model.args.expand
    input = torch.rand(batch, 1, d_in)

    reference_output = PytorchMambaSSM(reference_model, LAYER_NUM)(input)

    residual_block = reference_model.layers[LAYER_NUM]
    assert not isinstance(residual_block, torch.Tensor), "Expected torch.Module"

    if cache_dir:
        cache_path = model_config.get_weights_cache_path(model_version, cache_dir)
    else:
        cache_path = None

    config = model_config.create_model_config(batch, reference_model.args.d_model)

    loader = TtTensorLoader(reference_model.state_dict(), device, tt_cache_path=cache_path)
    transformer = MambaSsmBlockTransformer(
        device, batch, reference_model.args.d_inner, reference_model.args.d_state * 2
    )

    model = TtMambaSSM(reference_model.args, device, config, loader.get_tensor_loader(LAYER_NUM), transformer)
    tt_input = input.view(1, 1, batch, d_in)
    tt_input = ttnn.to_device(
        ttnn.from_torch(tt_input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_output = model(tt_input)
    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.view(batch, 1, -1)
    logger.info(comp_allclose(reference_output, tt_output))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Mamba SSM output failed")
        assert does_pass, f"PCC value is lower than {pcc}"
