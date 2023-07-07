import torch
import torch.nn as nn
import torch.nn.init as init
import tt_lib
from python_api_testing.models.squeezenet_1.tt.squeezenet_conv2d import (
    TtSqueezenetConv2D,
)
from python_api_testing.models.squeezenet_1.tt.squeezenet_fire import TtFire
from utility_functions_new import tt2torch_tensor, torch2tt_tensor
from tt_lib.fallback_ops import fallback_ops


from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.init as init
from loguru import logger


class TtSqueezeNet(nn.Module):
    def __init__(
        self,
        device,
        hugging_face_reference_model,
        state_dict: dict,
        version: str = "1_0",
        num_classes: int = 1000,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.num_classes = num_classes
        self.device = device
        self.state_dict = state_dict
        self.hugging_face_reference_model = hugging_face_reference_model

        if version == "1_0":
            # started convolution
            # take parameters
            pt_conv2d_start = self.hugging_face_reference_model.features[0]

            in_channels = pt_conv2d_start.in_channels
            out_channels = pt_conv2d_start.out_channels
            kernel_size = pt_conv2d_start.kernel_size[0]
            stride = pt_conv2d_start.stride[0]
            padding = pt_conv2d_start.padding[0]
            groups = pt_conv2d_start.groups
            dilation = pt_conv2d_start.dilation

            # use parameters
            self.conv_start = TtSqueezenetConv2D(
                state_dict=hugging_face_reference_model.state_dict(),
                base_address=f"features.0",
                device=self.device,
                c1=in_channels,
                c2=out_channels,
                k=kernel_size,
                s=stride,
                p=padding,
                g=groups,
                d=dilation[0],
            )

            self.tt_maxpool2d_1 = fallback_ops.MaxPool2d(
                kernel_size=3, stride=2, ceil_mode=True
            )
            self.tt_fire_3 = TtFire(
                device,
                hugging_face_reference_model,
                fire_position=3,
                inplanes=96,
                squeeze_planes=16,
                expand1x1_planes=64,
                expand3x3_planes=64,
            )
            self.tt_fire_4 = TtFire(
                device,
                hugging_face_reference_model,
                fire_position=4,
                inplanes=128,
                squeeze_planes=16,
                expand1x1_planes=64,
                expand3x3_planes=64,
            )
            self.tt_fire_5 = TtFire(
                device,
                hugging_face_reference_model,
                fire_position=5,
                inplanes=128,
                squeeze_planes=32,
                expand1x1_planes=128,
                expand3x3_planes=128,
            )
            self.tt_maxpool2d_2 = fallback_ops.MaxPool2d(
                kernel_size=3, stride=2, ceil_mode=True
            )
            self.tt_fire_7 = TtFire(
                device,
                hugging_face_reference_model,
                fire_position=7,
                inplanes=256,
                squeeze_planes=32,
                expand1x1_planes=128,
                expand3x3_planes=128,
            )
            self.tt_fire_8 = TtFire(
                device,
                hugging_face_reference_model,
                fire_position=8,
                inplanes=256,
                squeeze_planes=48,
                expand1x1_planes=192,
                expand3x3_planes=192,
            )
            self.tt_fire_9 = TtFire(
                device,
                hugging_face_reference_model,
                fire_position=9,
                inplanes=384,
                squeeze_planes=48,
                expand1x1_planes=192,
                expand3x3_planes=192,
            )
            self.tt_fire_10 = TtFire(
                device,
                hugging_face_reference_model,
                fire_position=10,
                inplanes=384,
                squeeze_planes=64,
                expand1x1_planes=256,
                expand3x3_planes=256,
            )
            self.tt_maxpool2d_3 = fallback_ops.MaxPool2d(
                kernel_size=3, stride=2, ceil_mode=True
            )
            self.tt_fire_12 = TtFire(
                device,
                hugging_face_reference_model,
                fire_position=12,
                inplanes=512,
                squeeze_planes=64,
                expand1x1_planes=256,
                expand3x3_planes=256,
            )

        elif version == "1_1":
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError(
                f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected"
            )

        # Final convolution is initialized differently from the rest
        # take parameters
        pt_conv2d_final = self.hugging_face_reference_model.classifier[1]

        in_channels = pt_conv2d_final.in_channels
        out_channels = pt_conv2d_final.out_channels
        kernel_size = pt_conv2d_final.kernel_size[0]
        stride = pt_conv2d_final.stride[0]
        padding = pt_conv2d_final.padding[0]
        groups = pt_conv2d_final.groups
        dilation = pt_conv2d_final.dilation

        # use parameters
        self.final_conv = TtSqueezenetConv2D(
            state_dict=hugging_face_reference_model.state_dict(),
            base_address=f"classifier.1",
            device=self.device,
            c1=in_channels,
            c2=out_channels,
            k=kernel_size,
            s=stride,
            p=padding,
            g=groups,
            d=dilation[0],
        )

        self.adaptive_avgpool2d = fallback_ops.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # features (sequential) part
        x = self.conv_start(x)
        x = tt_lib.tensor.relu(x)
        x = self.tt_maxpool2d_1(x)

        x = self.tt_fire_3(x)
        x = self.tt_fire_4(x)
        x = self.tt_fire_5(x)

        x = self.tt_maxpool2d_2(x)

        x = self.tt_fire_7(x)
        x = self.tt_fire_8(x)
        x = self.tt_fire_9(x)
        x = self.tt_fire_10(x)

        x = self.tt_maxpool2d_3(x)

        x = self.tt_fire_12(x)

        x = self.final_conv(x)
        x = tt_lib.tensor.relu(x)
        x = self.adaptive_avgpool2d(x)

        x = tt2torch_tensor(x)
        return torch.flatten(x, 1)


def _squeenet_1(device, hugging_face_reference_model, state_dict):
    model = TtSqueezeNet(device, hugging_face_reference_model, state_dict)
    return model


def squeezenet_1_0(device, hugging_face_reference_model, state_dict):
    return _squeenet_1(device, hugging_face_reference_model, state_dict)
