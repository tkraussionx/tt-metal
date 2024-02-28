# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def resnet_basic_block(x, *, parameters):
    identity = x

    # Relu and bn1 are fused with conv1
    conv1 = parameters.conv1(x)

    # Relu and bn2 are fused with conv1
    conv2 = parameters.conv2(conv1)
    ttnn.deallocate(conv1)

    if "downsample" in parameters and parameters.downsample is not None:
        identity = parameters.downsample(x)
        ttnn.deallocate(x)

    identity = ttnn.reshape(identity, conv2.shape)
    out = ttnn.add_and_apply_activation(conv2, identity, activation="relu", memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(conv2)
    if x is not identity:
        ttnn.deallocate(identity)

    return out


def resnet_bottleneck_block(x, parameters, device=None):
    identity = x

    conv1 = parameters.conv1(x)
    conv2 = parameters.conv2(conv1)
    if conv1.value.is_allocated():
        ttnn.deallocate(conv1)
    conv3 = parameters.conv3(conv2)
    ttnn.deallocate(conv2)

    if device is not None:
        ttnn.dump_device_memory_state(device)

    if "downsample" in parameters and parameters.downsample is not None:
        identity = parameters.downsample(x)
        ttnn.deallocate(x)

    conv3 = ttnn.reshape(conv3, identity.shape)
    out = ttnn.add_and_apply_activation(conv3, identity, activation="relu")  ##, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(conv3)

    if x is not identity:
        ttnn.deallocate(identity)

    return out
