// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
namespace ttnn {

struct Barrier {
    const bool is_starting_core;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const std::optional<chip_id_t> receiver_device_id;
    const std::optional<chip_id_t> sender_device_id;
    const MemoryConfig output_mem_config;
    const ttnn::ccl::Topology topology;

    //Required functions to all tensor op functions
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<SimpleShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
};

namespace ccl::barrier::detail {
//Template for the barrier_with_workers function
//Found in device/host/barrier_full_worker_grid.cpp
operation::ProgramWithCallbacks barrier_with_workers(
    const Tensor& input_tensors,
    const Tensor& output_tensors,
    const bool is_starting_core,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    ttnn::ccl::Topology topology);
}; // namespace ccl::barrier::detail

namespace operations::ccl{
    //Template for the barrier tensor found in device/barrier_op.cpp
    Tensor barrier(
    const Tensor &input_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring);
} // namespace operations::ccl

}// namespace ttnn
