
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
import torchvision as tv
from datasets import load_dataset

from loguru import logger
import pytest

from libs import tt_lib as ttl
from utility_functions import comp_allclose_and_pcc, comp_pcc
from vit import vision_transformer

_batch_size = 1

@pytest.mark.parametrize("fuse_ops", [False, True], ids=['Not Fused', "Ops Fused"])
def test_vit_inference(fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input
    batch_size = _batch_size


    torch_model = tv.models.vit_b_16(tv.models.ViT_B_16_Weights.DEFAULT)


    torch_model.eval()


    # print(torch_model)
    state_dict = torch_model.state_dict()
    tt_model = vision_transformer(state_dict=state_dict)
    tt_model.eval()


    # if fuse_ops:
    #     modules_to_fuse = [["conv_stem.first_conv.convolution", "conv_stem.first_conv.normalization"]]
    #     modules_to_fuse.extend([["conv_stem.conv_3x3.convolution", "conv_stem.conv_3x3.normalization"]])
    #     modules_to_fuse.extend([["conv_stem.reduce_1x1.convolution", "conv_stem.reduce_1x1.normalization"]])

    #     for i in range(16):
    #         modules_to_fuse.extend([[f"layer.{i}.expand_1x1.convolution", f"layer.{i}.expand_1x1.normalization"]])
    #         modules_to_fuse.extend([[f"layer.{i}.conv_3x3.convolution", f"layer.{i}.conv_3x3.normalization"]])
    #         modules_to_fuse.extend([[f"layer.{i}.reduce_1x1.convolution", f"layer.{i}.reduce_1x1.normalization"]])

    #     modules_to_fuse.extend([[f"conv_1x1.convolution", f"conv_1x1.normalization"]])

    #     tt_model = torch.ao.quantization.fuse_modules(tt_model, modules_to_fuse)

    torch_output = torch_model(image)

    # tt_output = tt_model(image)[0]

    # passing = comp_pcc(torch_output, tt_output)
    assert passing[0], passing[1:]

    logger.info(f"PASSED {passing[1]}")



# @pytest.fixture
def imagenet_sample_input():
    from PIL import Image
    from torchvision import transforms
    path = "/mnt/MLPerf/tt_dnn-models/samples/ILSVRC2012_val_00048736.JPEG"
    # path = model_location_generator(sample_path)
    im = Image.open(path)
    im = im.resize((224, 224))
    return transforms.ToTensor()(im).unsqueeze(0)


test_vit_inference(False, imagenet_sample_input())
#
