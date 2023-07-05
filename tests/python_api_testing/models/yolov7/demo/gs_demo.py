import torch
from datasets import load_dataset
from loguru import logger
from pathlib import Path
import sys
import cv2
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
from python_api_testing.models.yolov7.reference.utils.datasets import LoadImages
from python_api_testing.models.yolov7.reference.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from python_api_testing.models.yolov7.reference.utils.plots import plot_one_box
from python_api_testing.models.yolov7.tt.yolov7_model import yolov7_fused_model
from utility_functions_new import (
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)


def test_gs_demo(model_location_generator):
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    # Get data
    data_path = model_location_generator("tt_dnn-models/Yolo/data/")
    data_image_path = str(data_path / "images/horses.jpg")
    imgsz = 640
    save_img = True  # save inference images

    # Load models
    reference_model = get_yolov7_fused_cpu_model(model_location_generator)
    tt_model = yolov7_fused_model(device, model_location_generator)

    # Load data and setups
    stride = int(reference_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    dataset = LoadImages(data_image_path, img_size=imgsz, stride=stride)
    names = reference_model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    with torch.no_grad():
        tt_model.eval()

        path, im, im0s, _ = next(iter(dataset))
        im = torch.from_numpy(im)
        im = im.float()
        im /= 255
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        tt_im = torch2tt_tensor(im, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)
        # Run tt inference
        pred = tt_model(tt_im)[0]

        # Apply NMS
        conf_thres, iou_thres = 0.25, 0.45
        classes = None  # filter by class
        agnostic_nms = False
        save_conf = True
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=classes, agnostic=False
        )

        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, s, im0, frame = (
                file_path,
                "",
                im0s.copy(),
                getattr(dataset, "frame", 0),
            )
            p = Path(p)  # to Path
            save_path_input = str(p / "yolov7_gs_input.jpg")
            save_path_output = str(p / "yolov7_gs_output.jpg")

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_img:  # Add bbox to image
                    label = f"{names[int(cls)]} {conf:.2f}"
                    plot_one_box(
                        xyxy,
                        im0,
                        label=label,
                        color=colors[int(cls)],
                        line_thickness=1,
                    )

            # Save input image
            cv2.imwrite(save_path_input, im0s)
            # Save result image (image with detections)
            cv2.imwrite(save_path_output, im0)

    logger.info(f"Input image saved as {save_path_input}")
    logger.info(f"Result image saved as {save_path_output}")

    tt_lib.device.CloseDevice(device)
