import torch
from datasets import load_dataset
from loguru import logger
from pathlib import Path
import sys
from numpy import random
import tt_lib

file_path = f"{Path(__file__).parent}"
sys.path.append(f"{file_path}")
sys.path.append(f"{file_path}/..")
sys.path.append(f"{file_path}/../..")
sys.path.append(f"{file_path}/../../..")
sys.path.append(f"{file_path}/../../../..")
sys.path.append(f"{file_path}/../../../../..")

from python_api_testing.models.yolov7.reference.models.load_torch_model import (
    get_yolov7_fused_cpu_model,
)
from python_api_testing.models.yolov7.tt.yolov7_model import yolov7_fused_model
from python_api_testing.models.yolov7.reference.utils.datasets import LoadImages
from python_api_testing.models.yolov7.reference.utils.general import check_img_size
from utility_functions_new import (
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
    profiler,
    disable_compile_cache,
    enable_compile_cache,
    prep_report,
)

BATCH_SIZE = 1


def test_perf(model_location_generator):
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    disable_compile_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    # Get data
    data_path = model_location_generator("tt_dnn-models/Yolo/data/")
    data_image_path = str(data_path / "images/horses.jpg")
    imgsz = 640

    # Load models
    reference_model = get_yolov7_fused_cpu_model(model_location_generator)
    tt_model = yolov7_fused_model(device, model_location_generator)

    # Load data and setups
    stride = int(reference_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    dataset = LoadImages(data_image_path, img_size=imgsz, stride=stride)

    with torch.no_grad():
        tt_model.eval()

        path, im, im0s, _ = next(iter(dataset))
        im = torch.from_numpy(im)
        im = im.float()
        im /= 255
        if im.ndimension() == 3:
            im = im.unsqueeze(0)
        tt_im = torch2tt_tensor(im, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)

        profiler.start(cpu_key)
        pt_out = reference_model(im)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output_1 = tt_model(tt_im)
        profiler.end(first_key)

        enable_compile_cache()

        profiler.start(second_key)
        tt_output_2 = tt_model(tt_im)
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    prep_report(
        "yolov7",
        BATCH_SIZE,
        first_iter_time,
        second_iter_time,
        "yolov7-fused",
        cpu_time,
    )

    tt_lib.device.CloseDevice(device)
