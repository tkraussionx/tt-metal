// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/all_reduce/device/all_reduce_op.hpp"
#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include <cstdint>

namespace ttnn {

void AllReduce::validate(const std::vector<Tensor>& input_tensors) const {
    for (auto const& t : input_tensors) {
        TT_FATAL(!t.is_sharded(), "Sharded tensors are not supported for all reduce currently");
    }
}

std::vector<ttnn::SimpleShape> AllReduce::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    auto shape = input_tensors[0].get_logical_shape();
    return std::vector<ttnn::SimpleShape>(input_tensors.size(), shape);
}

std::vector<Tensor> AllReduce::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks AllReduce::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return ccl::reduce_scatter_detail::reduce_scatter_with_workers(
        input_tensors.at(0),
        output_tensors.at(0),
        this->binary_op_type,
        0,
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
            TT_THROW("All reduce only supports reduce_type Sum. Op type {} not supported.", reduce_op);
            return ttnn::operations::binary::BinaryOpType::ADD;
    }
}

namespace operations{
namespace experimental{
namespace ccl{
Tensor all_reduce(
    const Tensor& input_tensor,
    ttnn::operations::reduction::ReduceType math_op,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(math_op);
    TT_FATAL(std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "This op is only supported for Fast Dispatch");
    TT_FATAL(topology == ttnn::ccl::Topology::Ring, "All Reduce op is currently supported only on Ring topology");

    auto devices = input_tensor.get_workers();
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [binary_op_type, num_links, output_mem_config, topology, devices, user_defined_num_workers, user_defined_num_buffers_per_channel](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            TT_FATAL(input_tensors.size() >= 1, "All reduce op expects an input tensor but it received none");
            bool is_linear = topology == ttnn::ccl::Topology::Linear;

            const auto& input_tensor = input_tensors.at(0);

            auto shape = input_tensor.get_logical_shape();
            auto rank = shape.rank();
            uint32_t num_devices = devices.size();

            uint32_t merged_dim_size = 1;
            for (uint32_t i = 0; i <= rank - 3; ++i) {
                merged_dim_size *= shape[i];
            }

            uint32_t all_reduce_dim = -1;
            for (uint32_t i = 0; i < rank; ++i) {
                if(shape[i] % num_devices == 0){
                    all_reduce_dim = i;
                }
            }
            TT_FATAL(all_reduce_dim != -1, "Atleast one dim should be divisible by num_devices {}", num_devices);

            std::vector<int32_t> new_shape{1, merged_dim_size, shape[rank - 2], shape[rank - 1]};

            auto reshaped_tensor = ttnn::reshape(input_tensor, new_shape);

            const auto& reduced_tensor = operation::run(
                create_reduce_scatter_struct(
                    reshaped_tensor,
                    binary_op_type,
                    all_reduce_dim,
                    num_links,
                    output_mem_config,
                    user_defined_num_workers,
                    user_defined_num_buffers_per_channel,
                    devices,
                    topology),
                {reshaped_tensor});

            const auto& gathered_tensor = operation::run(
                create_all_gather_struct(reduced_tensor.at(0), all_reduce_dim, num_links, output_mem_config, user_defined_num_workers, user_defined_num_buffers_per_channel, devices, topology),
                {reduced_tensor.at(0)});

            auto final_output = ttnn::reshape(gathered_tensor.at(0), shape);

            return {final_output};
            },
     {input_tensor},
     output_tensors);

    return output_tensors.at(0);
}

} // namespace ccl
} // namespace experimental
} // namespace operations

};  // namespace ttnn
