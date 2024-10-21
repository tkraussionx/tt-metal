import ttnn
import torch
from models.experimental.functional_yolov7.ttnn.common import Conv


class ttnn_yolov7:
    def __init__(self, device, parameters) -> None:
        self.device = device
        self.parameters = parameters
        self.conv1 = Conv([1, 640, 640, 3], (3, 3, 1, 1, 1, 1, 1, 1), parameters["0"], act_block_h=32)
        self.conv2 = Conv([1, 640, 640, 32], (3, 3, 2, 2, 1, 1, 1, 1), parameters["1"])
        self.conv3 = Conv([1, 320, 320, 64], (3, 3, 1, 1, 1, 1, 1, 1), parameters["2"], act_block_h=32)
        self.conv4 = Conv([1, 320, 320, 64], (3, 3, 2, 2, 1, 1, 1, 1), parameters["3"])
        self.conv5 = Conv([1, 160, 160, 128], (1, 1, 1, 1, 0, 0, 1, 1), parameters["4"])
        self.conv6 = Conv([1, 160, 160, 128], (1, 1, 1, 1, 0, 0, 1, 1), parameters["5"])
        self.conv7 = Conv([1, 160, 160, 64], (3, 3, 1, 1, 1, 1, 1, 1), parameters["6"])
        self.conv8 = Conv([1, 160, 160, 64], (3, 3, 1, 1, 1, 1, 1, 1), parameters["7"])
        self.conv9 = Conv([1, 160, 160, 64], (3, 3, 1, 1, 1, 1, 1, 1), parameters["8"])
        self.conv10 = Conv([1, 160, 160, 64], (3, 3, 1, 1, 1, 1, 1, 1), parameters["9"])

        self.conv11 = Conv([1, 160, 160, 256], (1, 1, 1, 1, 0, 0, 1, 1), parameters["11"])
        self.conv12 = Conv([1, 80, 80, 256], (1, 1, 1, 1, 0, 0, 1, 1), parameters["13"])
        self.conv13 = Conv([1, 160, 160, 256], (1, 1, 1, 1, 0, 0, 1, 1), parameters["14"])
        self.conv14 = Conv([1, 160, 160, 128], (3, 3, 2, 2, 1, 1, 1, 1), parameters["15"])

    def __call__(self, input_tensor):
        conv1 = self.conv1(self.device, input_tensor)
        conv1 = ttnn.silu(conv1)

        conv2 = self.conv2(self.device, conv1)
        conv2 = ttnn.silu(conv2)
        ttnn.deallocate(conv1)

        conv3 = self.conv3(self.device, conv2)
        conv3 = ttnn.silu(conv3)
        ttnn.deallocate(conv2)

        conv4 = self.conv4(self.device, conv3)
        conv4 = ttnn.silu(conv4)
        ttnn.deallocate(conv3)

        conv5 = self.conv5(self.device, conv4)
        conv5 = ttnn.silu(conv5)

        conv6 = self.conv6(self.device, conv4)
        conv6 = ttnn.silu(conv6)

        conv7 = self.conv7(self.device, conv6)
        conv7 = ttnn.silu(conv7)

        conv8 = self.conv8(self.device, conv7)
        conv8 = ttnn.silu(conv8)

        conv9 = self.conv9(self.device, conv8)  # decrease in pcc - 0.988
        conv9 = ttnn.silu(conv9)

        conv10 = self.conv10(self.device, conv9)
        conv10 = ttnn.silu(conv10)  # decrease in pcc - 0.9856

        conv10 = ttnn.reshape(conv10, (1, 160, 160, 64))
        conv10 = ttnn.sharded_to_interleaved(conv10, ttnn.L1_MEMORY_CONFIG)

        conv8 = ttnn.reshape(conv8, (1, 160, 160, 64))
        conv8 = ttnn.sharded_to_interleaved(conv8, ttnn.L1_MEMORY_CONFIG)

        conv6 = ttnn.reshape(conv6, (1, 160, 160, 64))
        conv6 = ttnn.sharded_to_interleaved(conv6, ttnn.L1_MEMORY_CONFIG)

        conv5 = ttnn.reshape(conv5, (1, 160, 160, 64))
        conv5 = ttnn.sharded_to_interleaved(conv5, ttnn.L1_MEMORY_CONFIG)

        conv10 = ttnn.concat(
            [conv10, conv8, conv6, conv5], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG
        )  # pcc = 0.99 (pcc 0.00909 - when inputs are in row major)
        ttnn.deallocate(conv4)
        ttnn.deallocate(conv7)
        ttnn.deallocate(conv9)

        conv11 = self.conv11(self.device, conv10)
        conv11 = ttnn.silu(conv11)
        ttnn.deallocate(conv5)
        ttnn.deallocate(conv6)
        ttnn.deallocate(conv8)

        mp1 = ttnn.max_pool2d(
            input_tensor=conv11,
            batch_size=1,
            input_h=160,
            input_w=160,
            channels=256,
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
        )
        ttnn.deallocate(conv10)
        ttnn.deallocate(conv11)
        print("mp1 shape: ", mp1.shape)
        mp1 = ttnn.to_layout(mp1, layout=ttnn.ROW_MAJOR_LAYOUT)
        mp1 = ttnn.reshape(mp1, (1, 80, 80, 256))
        print("mp1 shape: ", mp1.shape)
        # mp1 = ttnn.to_device(mp1, device=self.device)
        # mp1 = ttnn.sharded_to_interleaved(mp1, ttnn.L1_MEMORY_CONFIG)

        conv12 = self.conv11(self.device, mp1)
        conv12 = ttnn.silu(conv12)

        conv13 = self.conv11(self.device, conv11)
        conv13 = ttnn.silu(conv13)

        conv14 = self.conv11(self.device, conv13)
        conv14 = ttnn.silu(conv14)

        return mp1
