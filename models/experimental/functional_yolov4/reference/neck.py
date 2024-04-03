# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        self.c2 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(1024)

        self.c3 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b3 = nn.BatchNorm2d(512)

        # 3 maxpools
        self.p1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.p2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
        self.p3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
        ####

        self.c4 = nn.Conv2d(2048, 512, 1, 1, 0, bias=False)
        self.b4 = nn.BatchNorm2d(512)

        self.c5 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.b5 = nn.BatchNorm2d(1024)

        self.c6 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b6 = nn.BatchNorm2d(512)

        self.c7 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b7 = nn.BatchNorm2d(256)

        # 2 upsample2d
        self.u = nn.Upsample(scale_factor=(2, 2), mode="nearest")

        self.c7_2 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b7_2 = nn.BatchNorm2d(256)

        self.c7_3 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b7_3 = nn.BatchNorm2d(256)

        self.c8 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.b8 = nn.BatchNorm2d(512)

        self.c7_4 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b7_4 = nn.BatchNorm2d(256)

        self.c8_2 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.b8_2 = nn.BatchNorm2d(512)

        self.c7_5 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b7_5 = nn.BatchNorm2d(256)

        self.c9 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.b9 = nn.BatchNorm2d(128)

        self.c9_2 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.b9_2 = nn.BatchNorm2d(128)
        self.c9_3 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.b9_3 = nn.BatchNorm2d(128)

        self.c10 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.b10 = nn.BatchNorm2d(256)

        self.c9_4 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.b9_4 = nn.BatchNorm2d(128)
        self.c10_2 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.b10_2 = nn.BatchNorm2d(256)
        self.c9_5 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.b9_5 = nn.BatchNorm2d(128)

    def forward(self, input: torch.Tensor):
        # 3 CBN blocks
        x1 = self.c1(input)
        x1_b = self.b1(x1)
        x1_m = self.relu(x1_b)

        x2 = self.c2(x1_m)
        x2_b = self.b2(x2)
        x2_m = self.relu(x2_b)

        x3 = self.c3(x2_m)
        x3_b = self.b3(x3)
        x3_m = self.relu(x3_b)

        # maxpools
        x4 = self.p1(x3_m)
        x5 = self.p2(x3_m)
        x6 = self.p3(x3_m)

        # concat the outputs of maxpool and x3_m
        conc1 = torch.cat([x4, x5, x6, x3_m], dim=1)

        # 4 back2back CBRs
        # CBR4-1
        x7 = self.c4(conc1)
        x7_b = self.b4(x7)
        x7_m = self.relu(x7_b)

        # CBR4-2
        x8 = self.c5(x7_m)
        x8_b = self.b5(x8)
        x8_m = self.relu(x8_b)

        # CBR4-3
        x9 = self.c6(x8_m)
        x9_b = self.b6(x9)
        x9_m = self.relu(x9_b)

        # CBR4-4
        x10 = self.c7(x9_m)
        x10_b = self.b7(x10)
        x10_m = self.relu(x10_b)

        # upsample
        u1 = self.u(x10_m)

        # Next CBR block to be concatinated with output of u1
        # gets the output of downsample4 module which is dimensions: [1, 512, 20, 20] - make a random tensor with that shape for the purpose of running the neck unit test stand-alone
        outDownSample4 = torch.rand([1, 512, 20, 20])
        # CBR block for conc2
        x11 = self.c7(outDownSample4)
        x11_b = self.b7(x11)
        x11_m = self.relu(x11_b)

        # concat CBR output with output from u1
        conc2 = torch.cat([u1, x11_m], dim=1)

        # 6 back2back CBRs
        # CBR6_1
        x12 = self.c7(conc2)
        x12_b = self.b7(x12)
        x12_m = self.relu(x12_b)

        # CBR6_2
        x13 = self.c8(x12_m)
        x13_b = self.b8(x13)
        x13_m = self.relu(x13_b)

        # CBR6_3
        x14 = self.c7(x13_m)
        x14_b = self.b7(x14)
        x14_m = self.relu(x14_b)

        # CBR6_4
        x15 = self.c8(x14_m)
        x15_b = self.b8(x15)
        x15_m = self.relu(x15_b)

        # CBR6_5
        x16 = self.c7(x15_m)
        x16_b = self.b7(x16)
        x16_m = self.relu(x16_b)

        # CBR6_6
        x17 = self.c9(x16_m)
        x17_b = self.b9(x17)
        x17_m = self.relu(x17_b)

        # upsample
        u2 = self.u(x17_m)

        # CBR block for conc3
        outDownSample3 = torch.rand([1, 256, 40, 40])
        x18 = self.c9(outDownSample3)
        x18_b = self.b9(x18)
        x18_m = self.relu(x18_b)

        # concat CBR output with output from u2
        conc3 = torch.cat([u2, x18_m], dim=1)

        # 5 CBR blocks
        # CBR5_1
        x19 = self.c9(conc3)
        x19_b = self.b9(x19)
        x19_m = self.relu(x19_b)

        # CBR5_2
        x20 = self.c10(x19_m)
        x20_b = self.b10(x20)
        x20_m = self.relu(x20_b)

        # CBR5_3
        x21 = self.c9(x20_m)
        x21_b = self.b9(x21)
        x21_m = self.relu(x21_b)

        # CBR5_4
        x22 = self.c10(x21_m)
        x22_b = self.b10(x22)
        x22_m = self.relu(x22_b)

        # CBR5_5
        x23 = self.c9(x22_m)
        x23_b = self.b9(x23)
        x23_m = self.relu(x23_b)

        return x23_m, x9_m, x16_m


### The following is just to text implementation
# TODO delete the following lines
state_dict = torch.load("tests/ttnn/integration_tests/yolov4/yolov4.pth")
neck_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("neek."))}
torch_input_tensor = torch.randn(1, 1024, 10, 10)  # Batch size of 1, 1024 input channels, 10x10 height and width
torch_model = Neck()


#########
params = list(torch_model.parameters())
for i, param in enumerate(params):
    print(f"Parameter {i}: {param.shape}")


#########


for layer in torch_model.children():
    print(layer)

new_state_dict = {}
keys = [name for name, parameter in torch_model.state_dict().items()]
values = [parameter for name, parameter in neck_state_dict.items()]
for i in range(len(keys)):
    new_state_dict[keys[i]] = values[i]

torch_model.load_state_dict(new_state_dict)
torch_model.eval()

torch_input_tensor = torch.randn(1, 1024, 10, 10)  # Batch size of 1, 1024 input channels, 10x10 height and width
torch_output_tensor = torch_model(torch_input_tensor)
print("\n\n\n\nthe ouput shape is: ")
print(torch_output_tensor[0].shape)
print(torch_output_tensor[1].shape)
print(torch_output_tensor[2].shape)
