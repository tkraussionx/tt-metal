# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn
import os
import math
import hashlib
from elasticsearch import Elasticsearch
import time

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from tests.sweep_framework.statuses import VectorValidity, VectorStatus
import pathlib

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

TILE_WIDTH = 32
TILE_HEIGHT = 32

ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
ELASTIC_DEFAULT_URL = "http://yyz-elk:9200"
# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.

from collections import namedtuple

parameters = {
    "dict_softmax_trace_hash3": {
        "batch_sizes": [(1,)],
        "num_inputs": [1],
        "input_a_height": [1024],
        "input_a_width": [1024],
        "input_a_dtype": [ttnn.bfloat16, ttnn.float32],
        "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_a_sharding_strategy": [
            None,
            ttnn.ShardStrategy.BLOCK,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardStrategy.WIDTH,
        ],
        "multi_core_program_config": [ttnn.SoftmaxDefaultProgramConfig],
        "is_scale_causal_mask_hw_dims_softmax": [False],
        "is_inplace": [False],
        "is_causal_mask": [False],
        "input_a_shard_orientation": [None, ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
        "input_b_height": [None],
        "input_b_width": [None],
        "input_b_dtype": [None],
        "input_b_layout": [None],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_sharding_strategy": [None],
        "softmax_type": ["softmax"],
    },
}


def dict_softmax_in_place(input_tensor, program_config, compute_kernel_config):
    return {
        "batch_sizes": (input_tensor.shape[0],),
        "num_inputs": 1,
        "input_a_height": input_tensor.shape[1],
        "input_a_width": input_tensor.shape[2],
        "input_a_dtype": input_tensor.dtype,
        "input_a_layout": input_tensor.layout,
        "input_a_memory_config": ttnn.get_memory_config(input_tensor),
        "input_a_sharding_strategy": ttnn.get_memory_config(input_tensor).shard_spec,
        "multi_core_program_config": program_config,
        "is_scale_causal_mask_hw_dims_softmax": False,
        "is_inplace": True,
        "is_causal_mask": False,
        "input_a_shard_orientation": ttnn.get_memory_config(input_tensor).shard_spec.orientation
        if ttnn.get_memory_config(input_tensor).shard_spec
        else None,
        "softmax_type": ["softmax_in_place"],
    }


def dict_scale_mask_softmax_in_place(input_tensor, scale, mask, program_config, is_causal_mask, compute_kernel_config):
    return {
        "batch_sizes": (input_tensor.shape[0],),
        "num_inputs": 2,
        "input_a_height": input_tensor.shape[1],
        "input_a_width": input_tensor.shape[2],
        "input_b_height": mask.shape[1],
        "input_b_width": mask.shape[2],
        "input_a_dtype": input_tensor.dtype,
        "input_b_dtype": mask.dtype,
        "input_a_layout": input_tensor.layout,
        "input_b_layout": mask.layout,
        "input_a_memory_config": ttnn.get_memory_config(input_tensor),
        "input_b_memory_config": ttnn.get_memory_config(mask),
        "input_a_sharding_strategy": ttnn.get_memory_config(input_tensor).shard_spec,
        "input_b_sharding_strategy": ttnn.get_memory_config(mask).shard_spec,
        "multi_core_program_config": program_config,
        "is_scale_causal_mask_hw_dims_softmax": False,
        "is_inplace": True,
        "is_causal_mask": is_causal_mask,
        "input_a_shard_orientation": ttnn.get_memory_config(input_tensor).shard_spec.orientation
        if ttnn.get_memory_config(input_tensor).shard_spec
        else None,
        "softmax_type": ["scale_mask_softmax_in_place"],
    }


def dict_scale_causal_mask_hw_dims_softmax_in_place(input_tensor, scale, mask, program_config, compute_kernel_config):
    return {
        "batch_sizes": (input_tensor.shape[0],),
        "num_inputs": 2,
        "input_a_height": input_tensor.shape[1],
        "input_a_width": input_tensor.shape[2],
        "input_b_height": mask.shape[1],
        "input_b_width": mask.shape[2],
        "input_a_dtype": input_tensor.dtype,
        "input_b_dtype": mask.dtype,
        "input_a_layout": input_tensor.layout,
        "input_b_layout": mask.layout,
        "input_a_memory_config": ttnn.get_memory_config(input_tensor),
        "input_b_memory_config": ttnn.get_memory_config(mask),
        "input_a_sharding_strategy": ttnn.get_memory_config(input_tensor).shard_spec,
        "input_b_sharding_strategy": ttnn.get_memory_config(mask).shard_spec,
        "multi_core_program_config": program_config,
        "is_scale_causal_mask_hw_dims_softmax": True,
        "is_inplace": True,
        "is_causal_mask": True,
        "input_a_shard_orientation": ttnn.get_memory_config(input_tensor).shard_spec.orientation
        if ttnn.get_memory_config(input_tensor).shard_spec
        else None,
        "softmax_type": ["scale_causal_mask_hw_dims_softmax_in_place"],
    }


def dict_softmax(input_tensor, dim, output_mem_config=None, compute_kernel_config=None):
    return {
        "batch_sizes": (input_tensor.shape[0],),
        "num_inputs": 1,
        "input_a_height": input_tensor.shape[1],
        "input_a_width": input_tensor.shape[2],
        "input_a_dtype": input_tensor.dtype,
        "input_a_layout": input_tensor.layout,
        "input_a_memory_config": ttnn.get_memory_config(input_tensor),
        "input_a_sharding_strategy": ttnn.get_memory_config(input_tensor).shard_spec,
        "multi_core_program_config": ttnn.SoftmaxDefaultProgramConfig,
        "is_scale_causal_mask_hw_dims_softmax": False,
        "is_inplace": False,
        "is_causal_mask": False,
        "input_a_shard_orientation": ttnn.get_memory_config(input_tensor).shard_spec.orientation
        if ttnn.get_memory_config(input_tensor).shard_spec
        else None,
        "input_b_height": None,
        "input_b_width": None,
        "input_b_dtype": None,
        "input_b_layout": None,
        "input_b_memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "input_b_sharding_strategy": None,
        "softmax_type": ["softmax"],
    }


def dict_scale_mask_softmax(input_tensor, scale, mask, output_mem_config, is_causal_mask, compute_kernel_config):
    return {
        "batch_sizes": (input_tensor.shape[0],),
        "num_inputs": 2,
        "input_a_height": input_tensor.shape[1],
        "input_a_width": input_tensor.shape[2],
        "input_b_height": mask.shape[1],
        "input_b_width": mask.shape[2],
        "input_a_dtype": input_tensor.dtype,
        "input_b_dtype": mask.dtype,
        "input_a_layout": input_tensor.layout,
        "input_b_layout": mask.layout,
        "input_a_memory_config": ttnn.get_memory_config(input_tensor),
        "input_b_memory_config": ttnn.get_memory_config(mask),
        "input_a_sharding_strategy": ttnn.get_memory_config(input_tensor).shard_spec,
        "input_b_sharding_strategy": ttnn.get_memory_config(mask).shard_spec,
        "multi_core_program_config": ttnn.SoftmaxDefaultProgramConfig,
        "is_scale_causal_mask_hw_dims_softmax": False,
        "is_inplace": False,
        "is_causal_mask": is_causal_mask,
        "input_a_shard_orientation": ttnn.get_memory_config(input_tensor).shard_spec.orientation
        if ttnn.get_memory_config(input_tensor).shard_spec
        else None,
        "softmax_type": ["scale_mask_softmax"],
    }


def scale_mask_softmax_check(
    input_tensor, scale=None, mask=None, output_mem_config=None, is_causal_mask=False, compute_kernel_config=None
):
    print(
        invalidate_vector(
            dict_scale_mask_softmax(input_tensor, scale, mask, output_mem_config, is_causal_mask, compute_kernel_config)
        )[1]
    )
    return not invalidate_vector(
        dict_scale_mask_softmax(input_tensor, scale, mask, output_mem_config, is_causal_mask, compute_kernel_config)
    )[0]


def softmax_check(input_tensor, dim=-1, output_mem_config=None, compute_kernel_config=None):
    print(invalidate_vector(dict_softmax(input_tensor, dim, output_mem_config, compute_kernel_config))[1])
    return not invalidate_vector(dict_softmax(input_tensor, dim, output_mem_config, compute_kernel_config))[0]


def scale_causal_mask_hw_dims_softmax_in_place_check(
    input_tensor, scale=None, mask=None, program_config=ttnn.SoftmaxDefaultProgramConfig, compute_kernel_config=None
):
    print(
        invalidate_vector(
            dict_scale_causal_mask_hw_dims_softmax_in_place(
                input_tensor, scale, mask, program_config, compute_kernel_config
            )
        )[1]
    )
    return not invalidate_vector(
        dict_scale_causal_mask_hw_dims_softmax_in_place(
            input_tensor, scale, mask, program_config, compute_kernel_config
        )
    )[0]


def scale_mask_softmax_in_place_check(
    input_tensor,
    scale=None,
    mask=None,
    program_config=ttnn.SoftmaxDefaultProgramConfig,
    is_causal_mask=False,
    compute_kernel_config=None,
):
    print(
        invalidate_vector(
            dict_scale_mask_softmax_in_place(
                input_tensor, scale, mask, program_config, is_causal_mask, compute_kernel_config
            )
        )[1]
    )
    return not invalidate_vector(
        dict_scale_mask_softmax_in_place(
            input_tensor, scale, mask, program_config, is_causal_mask, compute_kernel_config
        )
    )[0]


def get_hash(test_vector):
    values = []
    for key, value in test_vector.items():
        if value is not None:
            if key not in ["validity", "invalid_reason", "suite_name", "status"]:
                values.append(str(key) + str(value))

    # Concatenate all string values
    concatenated_string = "".join(values)
    print("conc_string=", concatenated_string)
    # Convert concatenated string to hex using hashlib for a consistent length
    hex_result = hashlib.md5(concatenated_string.encode()).hexdigest()
    print("hex_result=", hex_result)
    return hex_result


def softmax_in_place_check(input_tensor, program_config=ttnn.SoftmaxDefaultProgramConfig, compute_kernel_config=None):
    print(invalidate_vector(dict_softmax_in_place(input_tensor, program_config, compute_kernel_config))[1])
    return not invalidate_vector(dict_softmax_in_place(input_tensor, program_config, compute_kernel_config))[0]


def are_shapes_same(input_shape_a: list, input_shape_b: list):
    return all(first_index == second_index for first_index, second_index in zip(input_shape_a, input_shape_b))


def calculate_volume_of_a_shape(input_shape_a: list):
    return math.prod(input_shape_a)


def softmax_elastic_search_check(input_tensor, dim=-1, output_mem_config=None, compute_kernel_config=None):
    dictionary = dict_softmax(input_tensor, dim, output_mem_config, compute_kernel_config)
    dictionary["suite_name"] = "dict_softmax_t"
    dictionary["validity"] = VectorValidity.VALID
    dictionary["invalid_reason"] = ""
    dictionary["status"] = VectorStatus.CURRENT
    dictionary["sweep_name"] = "softmax"
    vector_id = hashlib.sha224(str(dictionary).encode("utf-8")).hexdigest()
    matches = []
    matches.append({"match": {"vector_id": vector_id}})
    RESULT_INDEX_PREFIX = "ttnn_sweeps_test_results_"
    module_name = "softmax"
    results_index = RESULT_INDEX_PREFIX + module_name
    client = Elasticsearch(ELASTIC_DEFAULT_URL, basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    results = client.search(
        index=results_index,
        size=10000,
        sort=[{"timestamp.keyword": {"order": "asc"}}],
        query={"bool": {"must": matches}},
    )["hits"]["hits"]
    return results[0]["_source"]["status"]


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None
    if test_vector["num_inputs"] < 1 and test_vector["num_inputs"] > 2:
        return True, "Softmax must have from 1 to 2 outputs."
    if test_vector["input_a_layout"] != ttnn.TILE_LAYOUT:
        return True, "First input must be in tile layout."
    if (
        test_vector["input_a_dtype"] != ttnn.float32
        and test_vector["input_a_dtype"] != ttnn.bfloat16
        and test_vector["input_a_dtype"] != ttnn.bfloat8_b
    ):
        return True, "Data type of first input must be Float32, BFloat16 or BFloat8_b"
    if test_vector["num_inputs"] == 2:
        input_shape_a = [*test_vector["batch_sizes"], test_vector["input_a_height"], test_vector["input_a_width"]]
        input_shape_b = [*test_vector["batch_sizes"], test_vector["input_b_height"], test_vector["input_b_width"]]
        if test_vector["input_b_sharding_strategy"]:
            if test_vector["input_b_layout"] != ttnn.TILE_LAYOUT:
                return True, "Second input must be in tile layout."
            if not are_shapes_same(input_shape_a, input_shape_b):
                return True, "If mask is sharded, it has to have the shape as the first input."
        else:
            if test_vector["input_b_layout"] == ttnn.ROW_MAJOR_LAYOUT:
                expected_shape = [input_shape_b[0], 1, input_shape_b[-1] / TILE_WIDTH, TILE_WIDTH]
                if not are_shapes_same(expected_shape, input_shape_b):
                    return True, "The mask should have [batch_size, 1, input_1 / TILE_WIDTH, TILE_WIDTH] shape."
            elif test_vector["input_b_layout"] == ttnn.TILE_LAYOUT:
                shortened_input_shape = input_shape_b[1:-2]
                if not all(layout == 1 for layout in shortened_input_shape):
                    return True, "Mask input shape should be 1."
            else:
                return True, "The mask should be tilized on in row major layout."
        if test_vector["multi_core_program_config"] is ttnn.SoftmaxShardedMultiCoreProgramConfig:
            M = calculate_volume_of_a_shape(input_shape_a) / input_shape_a[-1]
            K = input_shape_a[-1]
            if M % TILE_HEIGHT != 0:
                return True, "M must be divisible by tile height."
            if K % TILE_WIDTH != 0:
                return True, "K must be divisible by tile width."
            if (
                test_vector["multi_core_program_config"].block_w % test_vector["multi_core_program_config"].subblock_w
                != 0
            ):
                return True, "block_w must be divisible by subblock_w."
            if not test_vector["is_inplace"]:
                return True, "Multi core softmax must be inplace."
            if not test_vector["is_scale_causal_mask_hw_dims_softmax"]:
                num_cores_r = test_vector["multi_core_program_config"].core_x
                num_cores_c = test_vector["multi_core_program_config"].core_y
                number_of_shards = (
                    M
                    * K
                    / (
                        (
                            test_vector["multi_core_program_config"].block_w
                            * test_vector["multi_core_program_config"].block_h
                        )
                        * (TILE_WIDTH * TILE_HEIGHT)
                    )
                )
                if number_of_shards != num_cores_r * num_cores_c:
                    return True, "Number of shards must equal to number of cores."
            else:
                if not test_vector["is_causal_mask"]:
                    return True, "Has to be a causal mask."
                if test_vector["input_b_layout"] != ttnn.TILE_LAYOUT:
                    return True, "Second input must be in tile layout."
                if test_vector["input_b_sharding_strategy"]:
                    return True, "Input mask cannot be sharded."
                if test_vector["input_a_layout"] != ttnn.TILE_LAYOUT:
                    return True, "First input must be in tile layout."
                if not test_vector["input_a_sharding_strategy"]:
                    return True, "First input must be sharded."
                if not test_vector["input_a_shard_orientation"] != ttnn.ShardOrientation.ROW_MAJOR:
                    return True, "First input must be row major sharded."
    else:
        if test_vector["is_scale_causal_mask_hw_dims_softmax"]:
            return True, "Must be scale causal softmax."
    return False, None


def run(
    batch_sizes,
    num_inputs,
    input_a_height,
    input_a_width,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_a_sharding_strategy,
    multi_core_program_config,
    is_scale_causal_mask_hw_dims_softmax,
    is_inplace,
    is_causal_mask,
    input_a_shard_orientation,
    input_b_height,
    input_b_width,
    input_b_dtype,
    input_b_layout,
    input_b_memory_config,
    input_b_sharding_strategy,
    softmax_type,
    *,
    device,
) -> list:
    input_shape = (*batch_sizes, input_a_height, input_a_width)
    input_tensor = torch.randn(input_shape).bfloat16()
    torch_output_tensor = torch.softmax(input_tensor, dim=-1)
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    input_tensor_a = ttnn.from_torch(
        input_tensor,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )
    start_time = start_measuring_time()
    throws_error = False
    try:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        tt_output_tensor_on_device = ttnn.softmax(input_tensor_a, compute_kernel_config=compute_kernel_config)
        captured_graph = ttnn.graph.end_graph_capture()
    except Exception as e:
        print(e)
        throws_error = True
    e2e_perf = stop_measuring_time(start_time)
    return [(not throws_error, ""), e2e_perf]
