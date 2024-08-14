# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def tt_all_reduce(input_tensor, device_mesh, cluster_axis, dim=0, num_links=2, memory_config=None):
    # Ensure the input tensor is in the correct memory configuration
    input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    gathered_tensor = ttnn.line_all_gather(
        input_tensor, dim, num_links=num_links, cluster_axis=cluster_axis, device_mesh=device_mesh
    )
    reduced_tensors = ttnn.experimental.tensor.fast_reduce_nc(
        gathered_tensor, dims=[dim], output=None, compute_kernel_config=None
    )

    return reduced_tensors


def tt_all_gather(input_tensor, device_mesh, cluster_axis, dim, num_links=2, memory_config=None):
    # Ensure the input tensor is in the correct memory configuration
    input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    return ttnn.line_all_gather(
        input_tensor, dim, num_links=num_links, cluster_axis=cluster_axis, device_mesh=device_mesh
    )


def get_expand_dim(configuration):
    multiple_of = configuration.multiple_of
    ffn_dim_multiplier = configuration.ffn_dim_multiplier

    hidden_dim = 4 * configuration.dim
    hidden_dim = int(2 * hidden_dim / 3)

    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim
