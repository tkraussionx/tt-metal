import os
import sys
from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import tt_lib
import torch
from loguru import logger
from python_api_testing.models.squeezenet_1.tt.squeezenet_conv2d import (
    TtSqueezenetConv2D,
)
from python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from utility_functions_new import comp_pcc
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights


def download_images(path, imgsz):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    if imgsz is not None:
        image = image.resize(imgsz)

    image.save(path / "input_image.jpg")


def get_test_input(refence_module):
    stride = max(int(max(refence_module.stride)), 32)  # model stride
    imgsz = check_img_size((640, 640), s=stride)  # check image size

    download_images(Path(ROOT), None)  # imgsz)
    dataset = LoadImages(ROOT, img_size=imgsz, stride=stride, auto=True)

    for path, test_input, im0s, _, s in dataset:
        test_input = torch.from_numpy(test_input)
        test_input = test_input.float()
        test_input /= 255  # 0 - 255 to 0.0 - 1.0

        if len(test_input.shape) == 3:
            test_input = test_input[None]  # expand for batch dim

    logger.debug(f"Running inference on {path}")
    return test_input


def test_Squeezenet_Conv2D():
    hugging_face_reference_model = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()

    # get conv2d layer from the model
    reference_module = hugging_face_reference_model.features[0]

    in_channels = reference_module.in_channels
    out_channels = reference_module.out_channels
    kernel_size = reference_module.kernel_size[0]
    stride = reference_module.stride[0]
    padding = reference_module.padding[0]
    groups = reference_module.groups
    dilation = reference_module.dilation

    block = 0

    logger.debug(f"in_channels {in_channels}")
    logger.debug(f"out_channels {out_channels}")
    logger.debug(f"kernel_size {kernel_size}")
    logger.debug(f"stride {stride}")
    logger.debug(f"padding {padding}")
    logger.debug(f"groups {groups}")
    logger.debug(f"dilation {dilation}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 3, 64, 64)
    pt_out = reference_module(test_input)

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    tt_module = TtSqueezenetConv2D(
        state_dict=hugging_face_reference_model.state_dict(),
        base_address=f"features.{block}",
        device=device,
        c1=in_channels,
        c2=out_channels,
        k=kernel_size,
        s=stride,
        p=padding,
        g=groups,
        d=dilation[0],
    )

    # CHANNELS_LAST
    test_input = torch2tt_tensor(test_input, device)
    tt_out = tt_module(test_input)

    tt_out = tt_out.to(tt_lib.device.GetHost())
    tt_out = tt_out.to(tt_lib.tensor.Layout.ROW_MAJOR)

    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    logger.debug(f"pt_out shape {pt_out.shape}")
    logger.debug(f"tt_out shape {tt_out.shape}")

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_Squeezenet_Conv2D Passed!")
    else:
        logger.warning("test_Squeezenet_Conv2D Failed!")

    assert does_pass
