from typing import Type, Union, Optional, List, Callable
from collections import OrderedDict
import tt_lib
import torch
import torch.nn as nn

from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from tests.python_api_testing.models.resnet.resnetBlock import Bottleneck
from models.helper_funcs import GetBatchNorm
from models.retinanet.retinanet_mini_graphs import TtLinear


class TtIntermediateLayerGetter(nn.Module):
    def __init__(
        self,
        block,
        layers: List[int],
        device=None,
        host=None,
        state_dict=None,
        base_address=None,
    ) -> None:
        super(TtIntermediateLayerGetter, self).__init__()
        self.device = device
        self.host = host
        self.state_dict = state_dict
        self.inplanes = 64
        self.base_address = base_address

        self.conv1_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}conv1.weight"], self.device, put_on_device=False
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

        self.bn1 = GetBatchNorm(
            self.inplanes, state_dict, f"{base_address}bn1", self.device
        )
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
        conv_ds_weight = torch_to_tt_tensor_rm(
            state_dict[f"{self.base_address}{name}.0.downsample.0.weight"], self.device, put_on_device=False
        )
        self.conv_ds = tt_lib.fallback_ops.Conv2d(
                                        conv_ds_weight,
                                        biases=None,
                                        in_channels=self.inplanes,
                                        out_channels=planes * block.expansion,
                                        kernel_size=1,
                                        stride=stride,
                                        bias=False,
                                    )
        self.bn_ds = GetBatchNorm(planes * block.expansion, self.state_dict, f"{self.base_address}{name}.0.downsample.1", )

        self.downsample = nn.Sequential(
            self.conv_ds,
            self.bn_ds
        )
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=self.downsample,
                device=self.device,
                host=self.host,
                state_dict=self.state_dict,
                base_address=f"{self.base_address}{name}.0",
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    device=self.device,
                    host=self.host,
                    state_dict=self.state_dict,
                    base_address=f"{self.base_address}{name}.{i}",
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: tt_lib.tensor.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        out_2 = self.layer2(x)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)

        return [out_2, out_3, out_4]
