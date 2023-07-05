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
from python_api_testing.models.yolov7.tt.yolov7_model import yolov7_fused_model
from python_api_testing.models.yolov7.reference.utils.datasets import LoadImages
from python_api_testing.models.yolov7.reference.utils.general import check_img_size
from python_api_testing.models.yolov7.reference.models.load_torch_model import (
    get_yolov7_fused_cpu_model,
)
from utility_functions_new import (
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)


def test_yolov7_model(model_location_generator):
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    # Get data
    data_path = model_location_generator("tt_dnn-models/Yolo/data/")
    data_image_path = str(data_path / "images/horses.jpg")
    imgsz = 640

    # Load models
    reference_model = get_yolov7_fused_cpu_model(model_location_generator)
    tt_model = yolov7_fused_model(device, model_location_generator)

    with torch.no_grad():
        tt_model.eval()
        reference_model.eval()

        # Load data
        stride = max(int(max(reference_model.stride)), 32)  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        dataset = LoadImages(data_image_path, img_size=imgsz, stride=stride)

        path, im, im0s, _ = next(iter(dataset))
        im = torch.from_numpy(im)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference- fused torch
        pt_out = reference_model(im)

        # Inference- fused tt
        tt_im = torch2tt_tensor(im, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)
        tt_out = tt_model(tt_im)

    tt_lib.device.CloseDevice(device)

    # Check all outputs PCC
    does_all_pass = True

    does_pass, pcc_message = comp_pcc(pt_out[0], tt_out[0], 0.99)
    does_all_pass &= does_pass
    logger.info(f"Output prediction from the highest scale: {pcc_message}")

    for i in range(len(pt_out[1])):
        does_pass, pcc_message = comp_pcc(pt_out[1][i], tt_out[1][i], 0.99)
        logger.info(f"Object detection {i}: {pcc_message}")
        does_all_pass &= does_pass

    if does_all_pass:
        logger.info(f"YOLOv7 Full Detection Model Passed!")
    else:
        logger.warning(f"YOLOv7 Full Detection Model Failed!")

    assert does_all_pass
