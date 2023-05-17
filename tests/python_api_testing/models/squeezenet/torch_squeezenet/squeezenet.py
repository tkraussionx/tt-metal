from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
from python_api_testing.fused_ops.conv import conv as TtConv
from python_api_testing.models.conv_on_device_utils import can_run_conv_on_device, run_conv_on_tt_device

class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int, state_dict=None, base_address="", device=None, host=None) -> None:
        super().__init__()
        self.device = device
        self.host = host
        self.inplanes = inplanes

        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze.weight = nn.Parameter(state_dict[f"{base_address}.squeeze.weight"])
        self.squeeze.bias = nn.Parameter(state_dict[f"{base_address}.squeeze.bias"])

        squeeze_weight = state_dict[f"{base_address}.squeeze.weight"]
        squeeze_bias = state_dict[f"{base_address}.squeeze.bias"].tolist()
        self.squeeze_params = [squeeze_planes, inplanes, 1, 1, 1, 1, 0, 0, 1, 1]
        self.squeeze_conv_on_tt = TtConv(squeeze_weight.reshape(-1).tolist(), self.squeeze_params, self.device, squeeze_bias)

        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1.weight = nn.Parameter(state_dict[f"{base_address}.expand1x1.weight"])
        self.expand1x1.bias = nn.Parameter(state_dict[f"{base_address}.expand1x1.bias"])

        expand1x1_weight = state_dict[f"{base_address}.expand1x1.weight"]
        expand1x1_bias = state_dict[f"{base_address}.expand1x1.bias"].tolist()
        self.expand1x1_params = [expand1x1_planes, squeeze_planes, 1, 1, 1, 1, 0, 0, 1, 1]
        self.expand1x1_conv_on_tt = TtConv(expand1x1_weight.reshape(-1).tolist(), self.expand1x1_params, self.device, expand1x1_bias)

        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3.weight = nn.Parameter(state_dict[f"{base_address}.expand3x3.weight"])
        self.expand3x3.bias = nn.Parameter(state_dict[f"{base_address}.expand3x3.bias"])

        expand3x3_weight = state_dict[f"{base_address}.expand3x3.weight"]
        expand3x3_bias = state_dict[f"{base_address}.expand3x3.bias"].tolist()
        self.expand3x3_params = [expand3x3_planes, squeeze_planes, 3, 3, 1, 1, 1, 1, 1, 1]
        self.expand3x3_conv_on_tt = TtConv(expand3x3_weight.reshape(-1).tolist(), self.expand3x3_params, self.device, expand3x3_bias)

        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if can_run_conv_on_device(list(x.size()), self.squeeze_params):
            print("Conv on tt device.")
            x = run_conv_on_tt_device(x, self.squeeze_conv_on_tt, self.squeeze_params, self.device, self.host)
        else:
            print("Conv on CPU.")
            x = self.squeeze(x)
        x = self.squeeze_activation(x)
        if can_run_conv_on_device(list(x.size()), self.expand1x1_params):
            print("Conv on tt device.")
            expand1x1_output = run_conv_on_tt_device(x, self.expand1x1_conv_on_tt, self.expand1x1_params, self.device, self.host)
        else:
            print("Conv on CPU.")
            expand1x1_output = self.expand1x1(x)
        expand1x1_output = self.expand1x1_activation(expand1x1_output)
        if can_run_conv_on_device(list(x.size()), self.expand3x3_params):
            print("Conv on tt device.")
            expand3x3_output = run_conv_on_tt_device(x, self.expand3x3_conv_on_tt, self.expand3x3_params, self.device, self.host)
        else:
            print("Conv on CPU.")
            expand3x3_output = self.expand3x3(x)
        expand3x3_output = self.expand3x3_activation(expand3x3_output)

        return torch.cat(
            [expand1x1_output, expand3x3_output], 1
        )


class SqueezeNet(nn.Module):
    def __init__(self, version: str = "1_0", num_classes: int = 1000, dropout: float = 0.5, state_dict=None, base_address="", device=None, host=None) -> None:
        super().__init__()

        self.num_classes = num_classes
        if version == "1_0":
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64, state_dict=state_dict, base_address=f"features.3", device=device, host=host),
                Fire(128, 16, 64, 64, state_dict=state_dict, base_address=f"features.4", device=device, host=host),
                Fire(128, 32, 128, 128, state_dict=state_dict, base_address=f"features.5", device=device, host=host),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128, state_dict=state_dict, base_address=f"features.7", device=device, host=host),
                Fire(256, 48, 192, 192, state_dict=state_dict, base_address=f"features.8", device=device, host=host),
                Fire(384, 48, 192, 192, state_dict=state_dict, base_address=f"features.9", device=device, host=host),
                Fire(384, 64, 256, 256, state_dict=state_dict, base_address=f"features.10", device=device, host=host),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256, state_dict=state_dict, base_address=f"features.12", device=device, host=host),
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64, state_dict=state_dict, base_address=f"features.3", device=device, host=host),
                Fire(128, 16, 64, 64, state_dict=state_dict, base_address=f"features.4", device=device, host=host),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128, state_dict=state_dict, base_address=f"features.6", device=device, host=host),
                Fire(256, 32, 128, 128, state_dict=state_dict, base_address=f"features.7", device=device, host=host),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192, state_dict=state_dict, base_address=f"features.9", device=device, host=host),
                Fire(384, 48, 192, 192, state_dict=state_dict, base_address=f"features.10", device=device, host=host),
                Fire(384, 64, 256, 256, state_dict=state_dict, base_address=f"features.11", device=device, host=host),
                Fire(512, 64, 256, 256, state_dict=state_dict, base_address=f"features.12", device=device, host=host),
            )
        else:
            raise ValueError(f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected")

        self.features[0].weight = nn.Parameter(state_dict["features.0.weight"])
        self.features[0].bias = nn.Parameter(state_dict["features.0.bias"])

        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier[1].weight = nn.Parameter(state_dict["classifier.1.weight"])
        self.classifier[1].bias = nn.Parameter(state_dict["classifier.1.bias"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _squeezenet(version: str, state_dict, device=None, host=None) -> SqueezeNet:

    model = SqueezeNet(version, state_dict=state_dict, device=device, host=host)

    return model


# weights: SqueezeNet1_0_Weights.IMAGENET1K_V1
def squeezenet1_0(state_dict, device=None, host=None) -> SqueezeNet:
    return _squeezenet("1_0", state_dict=state_dict, device=device, host=host)


# weights:  SqueezeNet1_1_Weights.IMAGENET1K_V1
def squeezenet1_1(state_dict, device=None, host=None) -> SqueezeNet:
    return _squeezenet("1_1", state_dict=state_dict, device=device, host=host)
