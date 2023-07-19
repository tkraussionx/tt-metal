import torch
import torch.nn as nn
import torch.nn.init as init
import tt_lib
from models.squeezenet.squeezenet_mini_graphs import (
    TtSqueezenetConv2D,
)
from models.utility_functions import (
    tt2torch_tensor,
    torch2tt_tensor,
)


class TtFire(nn.Module):
    def __init__(
        self,
        device,
        hugging_face_reference_model,
        fire_position: int,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
    ) -> None:
        super().__init__()
        self.device = device
        self.inplanes = inplanes

        # self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        fire_conv2d_1 = hugging_face_reference_model.features[fire_position].squeeze

        in_channels = fire_conv2d_1.in_channels
        out_channels = fire_conv2d_1.out_channels
        kernel_size = fire_conv2d_1.kernel_size[0]
        stride = fire_conv2d_1.stride[0]
        padding = fire_conv2d_1.padding[0]
        groups = fire_conv2d_1.groups
        dilation = fire_conv2d_1.dilation

        self.squeeze = TtSqueezenetConv2D(
            state_dict=hugging_face_reference_model.state_dict(),
            base_address=f"features.{fire_position}.squeeze",
            device=device,
            c1=in_channels,
            c2=out_channels,
            k=kernel_size,
            s=stride,
            p=padding,
            g=groups,
            d=dilation[0],
        )

        # self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        fire_conv2d_2 = hugging_face_reference_model.features[fire_position].expand1x1

        in_channels = fire_conv2d_2.in_channels
        out_channels = fire_conv2d_2.out_channels
        kernel_size = fire_conv2d_2.kernel_size[0]
        stride = fire_conv2d_2.stride[0]
        padding = fire_conv2d_2.padding[0]
        groups = fire_conv2d_2.groups
        dilation = fire_conv2d_2.dilation

        self.expand1x1 = TtSqueezenetConv2D(
            state_dict=hugging_face_reference_model.state_dict(),
            base_address=f"features.{fire_position}.expand1x1",
            device=device,
            c1=in_channels,
            c2=out_channels,
            k=kernel_size,
            s=stride,
            p=padding,
            g=groups,
            d=dilation[0],
        )

        # self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        fire_conv2d_3 = hugging_face_reference_model.features[fire_position].expand3x3

        in_channels = fire_conv2d_3.in_channels
        out_channels = fire_conv2d_3.out_channels
        kernel_size = fire_conv2d_3.kernel_size[0]
        stride = fire_conv2d_3.stride[0]
        padding = fire_conv2d_3.padding[0]
        groups = fire_conv2d_3.groups
        dilation = fire_conv2d_3.dilation

        self.expand3x3 = TtSqueezenetConv2D(
            state_dict=hugging_face_reference_model.state_dict(),
            base_address=f"features.{fire_position}.expand3x3",
            device=device,
            c1=in_channels,
            c2=out_channels,
            k=kernel_size,
            s=stride,
            p=padding,
            g=groups,
            d=dilation[0],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pytorch implementation ----------------------------
        # x = self.squeeze_activation(self.squeeze(x))
        # return torch.cat(
        #     [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        # )

        # TT implementation ---------------------------------
        # x = self.squeeze_activation(self.squeeze(x))
        x = self.squeeze(x)
        x = tt_lib.tensor.relu(x)

        # term_1 = self.expand1x1_activation(self.expand1x1(x))
        term_1 = self.expand1x1(x)
        term_1 = tt_lib.tensor.relu(term_1)
        term_1 = tt2torch_tensor(term_1)

        # term_2 = self.expand3x3_activation(self.expand3x3(x))
        term_2 = self.expand3x3(x)
        term_2 = tt_lib.tensor.relu(term_2)
        term_2 = tt2torch_tensor(term_2)

        return torch.cat([term_1, term_2], 1)
