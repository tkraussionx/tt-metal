import ttnn
import torch
import torch.nn as nn


class BottleNeck(nn.Module):
    def __init__(
        self,
        parameters,
        layer_num,
        block_num,
        inplanes,
        planes,
        base_width,
        stride,
        downsample,
        input_size,
        config_overrides,
        dtype,
        device,
    ):
        super().__init__()
        self.input_height = input_size[0]
        self.input_width = input_size[1]
        self.inplanes = inplanes
        self.planes = planes
        self.base_width = base_width
        self.stride = stride
        self.width = int(self.planes * (self.base_width / 64))
        self.expansion = 4
        self.config_overrides = config_overrides
        self.dtype = dtype
        self.device = device
        self.device_name = "wormhole_b0"
        self.parameters = parameters

        self.conv1_weights = torch.randn(self.width, self.inplanes, 1, 1, dtype=torch.bfloat16)
        self.bn1_weights = None
        self.bn1_bias = None
        if parameters is not None:
            self.conv1_weights = self.parameters["layers.{}.{}.conv1.weight".format(layer_num, block_num)]
            self.bn1_weights, self.bn1_bias = fuse_bn_into_conv_weights(
                self.conv1_weights,
                self.parameters["layers.{}.{}.bn1.weight".format(layer_num, block_num)],
                self.parameters["layers.{}.{}.bn1.bias".format(layer_num, block_num)],
                self.parameters["layers.{}.{}.bn1.running_mean".format(layer_num, block_num)],
                self.parameters["layers.{}.{}.bn1.running_var".format(layer_num, block_num)],
            )
            self.conv1_weights = ttnn.from_torch(self.bn1_weights, self.dtype)
            if self.bn1_bias.dim() != 4:
                while self.bn1_bias.dim() != 4:
                    self.bn1_bias = self.bn1_bias.unsqueeze(0)
            self.bn1_bias = ttnn.from_torch(self.bn1_bias, self.dtype)
        else:
            self.conv1_weights = ttnn.from_torch(self.conv1_weights, self.dtype)
        self.conv1_use_shallow_conv_variant = False
        self.conv1_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            # height_sharding=True,
            input_channels_alignment=16 if self.conv1_use_shallow_conv_variant else 32,
            fp32_dest_acc_enabled=False,
            activation="relu",
            deallocate_activation=True,
            reshard_if_not_optimal=False,
            transpose_shards=False,
            packer_l1_accum_enabled=True if self.device_name == "wormhole_b0" else False,
            act_block_h_override=32,
        )

        self.conv2_weights = torch.randn(self.width, self.width, 3, 3, dtype=torch.bfloat16)
        self.bn2_weights = None
        self.bn2_bias = None
        if parameters is not None:
            self.conv2_weights = parameters["layers.{}.{}.conv2.weight".format(layer_num, block_num)]
            if self.parameters["layers.{}.{}.bn2.weight".format(layer_num, block_num)] is not None:
                self.bn2_weights, self.bn2_bias = fuse_bn_into_conv_weights(
                    self.conv2_weights,
                    self.parameters["layers.{}.{}.bn2.weight".format(layer_num, block_num)],
                    self.parameters["layers.{}.{}.bn2.bias".format(layer_num, block_num)],
                    self.parameters["layers.{}.{}.bn2.running_mean".format(layer_num, block_num)],
                    self.parameters["layers.{}.{}.bn2.running_var".format(layer_num, block_num)],
                )
                self.conv2_weights = ttnn.from_torch(self.bn2_weights, self.dtype)
                if self.bn2_bias.dim() != 4:
                    while self.bn2_bias.dim() != 4:
                        self.bn2_bias = self.bn2_bias.unsqueeze(0)
                self.bn2_bias = ttnn.from_torch(self.bn2_bias, self.dtype)
        else:
            self.conv2_weights = ttnn.from_torch(self.conv2_weights, self.dtype)
        self.conv2_use_shallow_conv_variant = False
        self.conv2_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            # height_sharding=True,
            input_channels_alignment=16 if self.conv2_use_shallow_conv_variant else 32,
            fp32_dest_acc_enabled=False,
            activation="relu",
            deallocate_activation=True,
            reshard_if_not_optimal=False,
            transpose_shards=False,
            packer_l1_accum_enabled=True if self.device_name == "wormhole_b0" else False,
            act_block_h_override=32,
        )

        if self.stride != 1:
            input_size = [input_size[0] // 2, input_size[1] // 2]

        self.conv3_weights = torch.randn(self.planes * self.expansion, self.width, 1, 1, dtype=torch.bfloat16)
        self.bn3_weights = None
        self.bn3_bias = None
        if parameters is not None:
            self.conv3_weights = parameters["layers.{}.{}.conv3.weight".format(layer_num, block_num)]
            if self.parameters["layers.{}.{}.bn3.weight".format(layer_num, block_num)] is not None:
                self.bn3_weights, self.bn3_bias = fuse_bn_into_conv_weights(
                    self.conv3_weights,
                    self.parameters["layers.{}.{}.bn3.weight".format(layer_num, block_num)],
                    self.parameters["layers.{}.{}.bn3.bias".format(layer_num, block_num)],
                    self.parameters["layers.{}.{}.bn3.running_mean".format(layer_num, block_num)],
                    self.parameters["layers.{}.{}.bn3.running_var".format(layer_num, block_num)],
                )
                self.conv3_weights = ttnn.from_torch(self.bn3_weights, self.dtype)
                if self.bn3_bias.dim() != 4:
                    while self.bn3_bias.dim() != 4:
                        self.bn3_bias = self.bn3_bias.unsqueeze(0)
                self.bn3_bias = ttnn.from_torch(self.bn3_bias, self.dtype)
        else:
            self.conv3_weights = ttnn.from_torch(self.conv3_weights, self.dtype)
        self.conv3_use_shallow_conv_variant = False
        self.conv3_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            # height_sharding=True,
            input_channels_alignment=16 if self.conv3_use_shallow_conv_variant else 32,
            fp32_dest_acc_enabled=False,
            activation="relu",
            deallocate_activation=True,
            reshard_if_not_optimal=False,
            transpose_shards=False,
            packer_l1_accum_enabled=True if self.device_name == "wormhole_b0" else False,
            act_block_h_override=32,
        )

    def forward(self, inputs, conformance_mode=False):
        # x = self.conv1.copy_input_to_device(inputs)
        # inputs = torch.permute(inputs, (0, 3, 1, 2))    # Because of Maxpooling
        identity = inputs
        x = inputs

        print("Identity - ", identity.shape)

        x, feature_height, feature_width, self.conv1_weights, self.bn1_weights = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weights,
            device=self.device,
            in_channels=self.inplanes,
            out_channels=self.width,
            batch_size=1,
            input_height=self.input_height,
            input_width=self.input_width,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.bn1_weights,
            conv_config=self.conv1_config,
            conv_op_cache={},
            # debug=False,
        )
        print("\nConv1'ed", x.shape)
        print(x.memory_config())

        x, feature_height, feature_width, self.conv2_weights, self.bn2_weights = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv2_weights,
            device=self.device,
            in_channels=self.width,
            out_channels=self.width,
            batch_size=1,
            input_height=feature_height,
            input_width=feature_width,
            kernel_size=(3, 3),
            stride=(self.stride, self.stride),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            bias_tensor=self.bn2_weights,
            conv_config=self.conv2_config,
            conv_op_cache={},
            # debug=False,
        )
        print("\nConv2'ed", x.shape)
        print(x.memory_config())

        x, feature_height, feature_width, self.conv3_weights, self.bn3_weights = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv3_weights,
            device=self.device,
            in_channels=self.width,
            out_channels=self.planes * self.expansion,
            batch_size=1,
            input_height=feature_height,
            input_width=feature_width,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            bias_tensor=self.bn3_weights,
            conv_config=self.conv3_config,
            conv_op_cache={},
            # debug=False,
        )
        print("\nConv3'ed", x.shape)
        print(x.memory_config())

        return x


device = ttnn.open_device(device_id=0, l1_small_size=32768)
in_channels = 3
image_height, image_width = 224, 224
test_image = torch.randn(1, in_channels, image_height, image_width, dtype=torch.float)
# test_image = test_image.permute(0,2,3,1).reshape(1,1,-1,in_channels)
# test_image = ttnn.from_torch(test_image, ttnn.bfloat16)

# parameters = None
# layer_num = 0
# block_num = 0
# inplanes = 32
# planes = 64
# base_width = 64
# stride = 2
# downsample = False
# input_size = [image_height,image_width]
# config_overrides = None
# config_overrides = [None,{"act_block_h": 64},{"act_block_h": 64, "out_subblock_h": 64, "out_subblock_w": 64}]
dtype = ttnn.bfloat16


class RetinaNet(nn.Module):
    def __init__(self, image_size, in_channels, parameters, layers, dtype, device):
        super().__init__()
        self.in_channels = in_channels
        self.input_height = image_size[0]
        self.input_width = image_size[1]
        self.in_planes = 64
        self.expansion = 4
        self.layers = layers
        self.dtype = dtype
        self.device = device
        self.device_name = "wormhole_b0"

        self.first_conv_weights = None
        self.first_conv_bn_weights = None
        if parameters is not None:
            raise Exception("parameters not implemented for now!")
        else:
            self.first_conv_weights = torch.rand(self.in_planes, self.in_channels, 7, 7, dtype=torch.bfloat16)
            self.first_conv_weights = ttnn.from_torch(self.first_conv_weights, dtype=self.dtype)
        self.first_conv_use_shallow_conv_variant = False
        self.first_conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            # height_sharding=True,
            input_channels_alignment=16 if self.first_conv_use_shallow_conv_variant else 32,
            fp32_dest_acc_enabled=False,
            activation="relu",
            deallocate_activation=True,
            reshard_if_not_optimal=False,
            transpose_shards=False,
            packer_l1_accum_enabled=True if self.device_name == "wormhole_b0" else False,
            act_block_h_override=32,
        )

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).reshape(1, 1, -1, in_channels)
        x = ttnn.from_torch(inputs, ttnn.bfloat16)

        print("Input = ", x.shape)

        # if x.shape[2] == 50176 and x.shape[3] == 3:
        #     reshard_shard_spec = ttnn.ShardSpec(
        #         ttnn.CoreRangeSet({ttnn.CoreRange((0,0), (7,6))}),
        #         [896,32],
        #         ttnn.ShardOrientation.COL_MAJOR,
        #         False
        #     )
        #     reshard_mem_config = ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, reshard_shard_spec
        #     )
        #     # x = ttnn.reshard(x, reshard_mem_config)
        #     x = ttnn.to_device(x, self.device, reshard_mem_config)
        #     print(">>> Sharded")

        x, feature_height, feature_width, self.conv1_weights, self.bn1_weights = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.first_conv_weights,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.in_planes,
            batch_size=1,
            input_height=self.input_height,
            input_width=self.input_width,
            kernel_size=[7, 7],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.first_conv_bn_weights,
            conv_config=self.first_conv_config,
            conv_op_cache={},
            # debug=False,
        )
        print("\nFirst Conv = ", x.shape)
        print(x.memory_config())


# model = BottleNeck(parameters, layer_num, block_num, inplanes, planes, base_width, stride, downsample, input_size, config_overrides, dtype, device)
model = RetinaNet([image_height, image_width], in_channels, None, [3, 4, 6, 3], dtype, device)

print(model)

x = model(test_image)
