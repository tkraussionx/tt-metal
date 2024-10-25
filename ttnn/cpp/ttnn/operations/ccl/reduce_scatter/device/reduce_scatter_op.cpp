// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"
#include "tt_metal/host_api.hpp"

#include <cstdint>

namespace ttnn {

void ReduceScatter::validate(const std::vector<Tensor>& input_tensors) const {
    for (auto const& t : input_tensors) {
        TT_FATAL(
            t.get_legacy_shape()[this->scatter_dim] / this->ring_size > 0,
            "Reduce scatter input tensor shape on dim {} must be divisible by ring size", this->scatter_dim);
        TT_FATAL(
            t.get_legacy_shape()[this->scatter_dim] % this->ring_size == 0,
            "Reduce scatter input tensor shape on dim {} must be divisible by ring size", this->scatter_dim);
    }
}

std::vector<ttnn::SimpleShape> ReduceScatter::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    auto shape = input_tensors[0].get_logical_shape();
    TT_FATAL(
        shape[this->scatter_dim] % this->ring_size == 0,
        "The size of the scatter dimension must be a multiple of the ring size. Dimension size: {}, ring Size: {}", shape[this->scatter_dim], this->ring_size);
    shape[this->scatter_dim] /= this->ring_size;
    return std::vector<ttnn::SimpleShape>(input_tensors.size(), shape);
}

std::vector<Tensor> ReduceScatter::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks ReduceScatter::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return ccl::reduce_scatter_detail::reduce_scatter_with_workers(
        input_tensors.at(0),
        output_tensors.at(0),
        this->binary_op_type,
        this->scatter_dim,
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->receiver_device_id,
        this->sender_device_id,
        this->topology,
        this->user_defined_num_workers,
        this->user_defined_num_buffers_per_channel);
}

static ttnn::operations::binary::BinaryOpType convert_reduce_type_to_eltwise_type(ttnn::operations::reduction::ReduceType reduce_op) {
    // Leaving switch statement for future support of additional types.
    switch (reduce_op) {
        case ttnn::operations::reduction::ReduceType::Sum:
            return ttnn::operations::binary::BinaryOpType::ADD;
        default:
            TT_THROW("Reduce scatter only supports reduce_type Sum. Op type {} not supported.", reduce_op);
            return ttnn::operations::binary::BinaryOpType::ADD;
    }
}

namespace operations{
namespace ccl{
Tensor reduce_scatter(
    const Tensor& input_tensor,
    const uint32_t scatter_dim,
    ttnn::operations::reduction::ReduceType math_op,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(math_op);
    TT_FATAL(std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "reduce_scatter op is only supported for Fast Dispatch");

    auto devices = input_tensor.get_workers();
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [binary_op_type, scatter_dim, num_links, output_mem_config, topology, devices, user_defined_num_workers, user_defined_num_buffers_per_channel](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {

            uint32_t num_devices = devices.size();
            if (num_devices == 2){
                topology = ttnn::ccl::Topology::Linear;
            }
            bool is_linear = topology == ttnn::ccl::Topology::Linear;

            const auto& input_tensor = input_tensors.at(0);
            uint32_t device_index = 0; // Initialize device index
            std::optional<chip_id_t> receiver_device_id = std::nullopt; // Initialize receiver device ID
            std::optional<chip_id_t> sender_device_id = std::nullopt; // Initialize sender device ID
            for (uint32_t i = 0; i < num_devices; ++i) {
                if (devices.at(i) == input_tensor.device()) {

                    bool is_last_chip_in_clockwise_direction = is_linear && i == (num_devices - 1);
                    bool is_last_chip_in_counter_clockwise_direction = is_linear && i == 0;
                    device_index = i;
                    receiver_device_id = is_last_chip_in_clockwise_direction ?
                        std::nullopt :
                        std::optional<chip_id_t>(devices.at((i + 1) % num_devices)->id());
                    sender_device_id = is_last_chip_in_counter_clockwise_direction ?
                        std::nullopt :
                        std::optional<chip_id_t>(devices.at((i + num_devices - 1) % num_devices)->id());
                    break;
                }
            }
            TT_FATAL(receiver_device_id != std::nullopt || sender_device_id != std::nullopt, "Error, Reduce-scatter was unable to identify either a sender or receiver device ID and atleast one must be identified for a valid Reduce-scatter configuration. The input mesh tensor or Reduce-scatter arguments may be incorrect");

            return operation::run(
                ttnn::ReduceScatter{
                    binary_op_type,
                    scatter_dim,
                    num_links,
                    num_devices,
                    device_index,
                    receiver_device_id,
                    sender_device_id,
                    output_mem_config,
                    topology,
                    user_defined_num_workers,
                    user_defined_num_buffers_per_channel},
                {input_tensor});
        },
     {input_tensor},
     output_tensors);
    return output_tensors.at(0);
}




Tensor reduce_scatter(
    const Tensor &input_tensor,
    const uint32_t scatter_dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::operations::reduction::ReduceType reduce_op,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(reduce_op);

    TT_FATAL(topology == ttnn::ccl::Topology::Linear, "This all_gather API with cluster_axis is currently supported only for the Linear topology");
    const auto mesh_view = mesh_device.get_view();
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view->num_rows() : mesh_view->num_cols();

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};

    operation::launch_op(
        [scatter_dim, binary_op_type, num_links, output_mem_config, mesh_view, cluster_axis, user_defined_num_workers, user_defined_num_buffers_per_channel, num_devices, topology](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {

            const auto& input_device_tensor = input_tensors.at(0);

            const auto coordinate = mesh_view->find_device(input_device_tensor.device()->id());
            const auto view_index = (cluster_axis == 0) ? coordinate.col : coordinate.row;
            const auto device_index = (cluster_axis == 0) ? coordinate.row : coordinate.col;

            auto get_chip_id = [&](std::size_t line_index) -> std::optional<chip_id_t> {
                auto new_coord = coordinate;
                if (cluster_axis == 0) {
                    new_coord.row = line_index % num_devices;
                } else {
                    new_coord.col = line_index % num_devices;
                }
                return mesh_view->find_device_id(new_coord);
            };

            bool is_last_chip_in_clockwise_direction = device_index == (num_devices - 1);
            bool is_last_chip_in_counter_clockwise_direction = device_index == 0;
            auto receiver_device_id = is_last_chip_in_clockwise_direction ? std::nullopt : get_chip_id(device_index + 1);
            auto sender_device_id = is_last_chip_in_counter_clockwise_direction ? std::nullopt : get_chip_id(device_index + num_devices - 1);

            return operation::run(
                ttnn::ReduceScatter{
                    binary_op_type,
                    scatter_dim,
                    num_links,
                    num_devices,
                    device_index,
                    receiver_device_id,
                    sender_device_id,
                    output_mem_config.value_or(input_device_tensor.memory_config()),
                    topology,
                    user_defined_num_workers,
                    user_defined_num_buffers_per_channel},
                {input_device_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);

}



} // namespace ccl
} // namespace operations

};  // namespace ttnn
