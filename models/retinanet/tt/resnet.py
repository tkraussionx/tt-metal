from typing import Type, Union, Optional, List, Callable

import tt_lib
import torch
import torch.nn as nn

from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.helper_funcs import TtBottleneck, TtBasicBlock, GetBatchNorm
from models.retinanet.retinanet_mini_graphs import TtLinear


class TtResnet50(nn.Module):
    def __init__(
        self, block, layers: List[int], device=None, host=None, state_dict=None
    ) -> None:
        super(TtResnet50, self).__init__()
        self.device = device
        self.host = host
        self.state_dict = state_dict
        self.inplanes = 64

        self.conv1_weights = torch_to_tt_tensor_rm(
            state_dict["conv1.weight"], self.device, put_on_device=False
        )
        self.conv1 = tt_lib.fallback_ops.Conv2d(
            self.conv1_weights,
            biases=None,
            in_channels=3,
            out_channels=self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.bn1 = GetBatchNorm(self.inplanes, state_dict, "bn1", self.device)
        self.relu = tt_lib.tensor.relu
        self.maxpool = tt_lib.fallback_ops.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, 64, layers[0], name="layer1", state_dict=state_dict
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, name="layer2", state_dict=state_dict
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, name="layer3", state_dict=state_dict
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, name="layer4", state_dict=state_dict
        )

        self.avgpool = tt_lib.fallback_ops.AdaptiveAvgPool2d((1, 1))

        self.fc_weight = torch_to_tt_tensor_rm(
            state_dict["fc.weight"], self.device, put_on_device=False
        )
        self.fc_bias = torch_to_tt_tensor_rm(
            state_dict["fc.bias"], self.device, put_on_device=False
        )
        self.fc = TtLinear(self.fc_weight, self.fc_bias)

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        name: str = None,
        state_dict=None,
    ) -> nn.Sequential:
        layers = []
        layers.append(
            block(
                in_ch=self.inplanes,
                out_ch=planes,
                device=self.device,
                host=self.host,
                state_dict=self.state_dict,
                base_address=f"{name}.0",
                is_downsample=True,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    device=self.device,
                    host=self.host,
                    state_dict=self.state_dict,
                    base_address=f"{name}.{i}",
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = tt_lib.tensor.permute(x, 0, 3, 2, 1)
        x = self.fc(x)
        return x
