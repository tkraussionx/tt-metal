from typing import Any
import ttnn
import torch
import pytest

ttnn.enable_fast_runtime_mode = False
ttnn.enable_logging = True
ttnn.report_name = "yolo_fail"
ttnn.enable_graph_report = False
ttnn.enable_detailed_buffer_report = True
ttnn.enable_detailed_tensor_report = True
ttnn.enable_comparison_mode = False

from models.experimental.yolov4.reference.downsample1 import DownSample1
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.yolov4.ttnn.downsample1 import Down1


class YOLOv4:
    def __init__(self, path) -> None:
        self.torch_model = torch.load(path)
        self.torch_keys = self.torch_model.keys()
        self.down1 = Down1(self)
        self.downs = [self.down1]

    def __call__(self, device, input_tensor):
        output = self.down1(device, input_tensor)
        return output

    def __str__(self) -> str:
        this_str = ""
        for down in self.downs:
            this_str += str(down)
            this_str += " \n"
        return this_str


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolo(device):
    yolov4 = YOLOv4("/localdev/smanoj/models/yolov4.pth")
    print(yolov4)

    torch_input = torch.randn((1, 320, 320, 3), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    torch_input = torch_input.permute(0, 3, 1, 2).float()
    torch_model = DownSample1()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    ds_state_dict = {k: v for k, v in yolov4.torch_model.items() if (k.startswith("down1."))}

    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    print(keys)
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    result = ttnn.to_torch(yolov4(device, ttnn_input))
    ref = torch_model(torch_input)
    ref = ref.permute(0, 2, 3, 1)
    result = result.reshape(1, 160, 160, 64)
    assert_with_pcc(result, ref, 0.99)
