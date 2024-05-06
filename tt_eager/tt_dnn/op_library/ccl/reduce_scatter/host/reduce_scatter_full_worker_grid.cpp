// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#include "impl/buffers/buffer.hpp"
#include "impl/kernels/data_types.hpp"
#include "tt_dnn/op_library/ccl/ccl_common.hpp"
#include "tt_dnn/op_library/ccl/ccl_host_datastructures.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/common/constants.hpp"
#include "eth_l1_address_map.h"
#include "tensor/tensor_impl.hpp"

// Includes that need to be moved to CCL datastructures header
#include <vector>


using namespace tt::constants;

namespace tt {

namespace tt_metal {





/////////////////////////////////////////////////////////////

// TODO(snijjar): move enable_bidirectional to a topology specific config
std::size_t decide_number_of_edm_channels(ccl::CCLOpConfig const& ccl_op_config, std::size_t max_num_workers, bool enable_bidirectional) {
    return ccl_op_config.is_input_sharded() ?
        std::min<uint32_t>(ccl_op_config.get_shard_grid_size(), std::min<std::size_t>(max_num_workers, enable_bidirectional ? 8 : 4)) :
        std::min<std::size_t>(max_num_workers, enable_bidirectional ? 8 : 4);
}

struct ReduceScatterWorkerArgBuilder {
    ReduceScatterWorkerArgBuilder(InterleavedRingReduceScatterTensorSlicer const& tensor_slicer, uint32_t worker_idx) :
        tensor_slicer(tensor_slicer),
        worker_idx(worker_idx) {
    }

    std::vector<uint32_t> generate_reduce_op_kernel_ct_args() const;
    std::vector<uint32_t> generate_reduce_op_kernel_rt_args() const;
    std::vector<uint32_t> generate_receiver_kernel_ct_args() const;
    std::vector<uint32_t> generate_receiver_kernel_rt_args() const;
    std::vector<uint32_t> generate_sender_kernel_ct_args() const;
    std::vector<uint32_t> generate_sender_kernel_rt_args() const;

};

void add_worker_kernels(tt_metal::Program &program, tt_metal::Device const* device, ReduceScatterWorkerArgBuilder const& worker_arg_builder) {


}

struct EdmInterfaceAddresses {
    std::unordered_map<int, uint32_t> worker_sender_edm_semaphore_addresses;
    std::unordered_map<int, uint32_t> worker_sender_edm_buffer_addresses;
    std::unordered_map<int, uint32_t> worker_receiver_edm_semaphore_addresses;
    std::unordered_map<int, uint32_t> worker_receiver_edm_buffer_addresses;
};

// Future work: split this up further:
// 1) assign workers to EDM channel (with buffer sharing mode specified too)
// 2) Compute the semaphore and buffer addresses (for each EDM channel and worker)
// For now - the mapping between workers and EDM channels is 1:1
EdmInterfaceAddresses add_worker_config_to_edm_builders(
    Device *device,
    ccl::CCLOpConfig const& op_config,
    std::vector<CoreCoord> const& worker_cores,
    uint32_t num_channels_per_edm,

    std::vector<ccl::EriscDatamoverBuilder> &clockwise_edm_builders,
    std::vector<ccl::EriscDatamoverBuilder> &counter_clockwise_edm_builders,

    std::vector<uint32_t> &clockwise_link_buffer_num_messages_to_send,
    std::vector<uint32_t> &counter_clockwise_link_buffer_num_messages_to_send,

    uint32_t worker_sender_semaphore_address,
    uint32_t worker_receiver_semaphore_address,
    uint32_t link,
    std::function<bool(uint32_t)> is_buffer_in_clockwise_direction_fn
    ) {
    EdmInterfaceAddresses edm_interface_addresses;
    // uint32_t workers_per_link = all_gather_config.get_num_workers_per_link() / num_channels_per_edm;
    // Currently setup for non-full-worker-grid setup
    for (uint32_t c = 0; c < num_channels_per_edm; ++c) {
        uint32_t global_worker_idx = c + num_channels_per_edm * link;
        uint32_t num_workers_per_eth_buffer = 1;//std::min(workers_per_link, num_channels_per_edm );

        std::vector<ccl::WorkerXY> sender_worker_coords;
        std::vector<ccl::WorkerXY> receiver_worker_coords;
        for (uint32_t w = c * num_workers_per_eth_buffer; w < (c + 1) * num_workers_per_eth_buffer; ++w) {
            sender_worker_coords.push_back(
                ccl::WorkerXY(
                    device->worker_core_from_logical_core(worker_cores.at(w)).x,
                    device->worker_core_from_logical_core(worker_cores.at(w)).y));
            receiver_worker_coords.push_back(
                ccl::WorkerXY(
                    device->worker_core_from_logical_core(worker_cores.at(w)).x,
                    device->worker_core_from_logical_core(worker_cores.at(w)).y));

        }

        bool sender_enabled = true; // (!is_linear || !is_last_chip_in_chain); // update for linear
        if (sender_enabled) {
            auto &sender_edm_builder = is_buffer_in_clockwise_direction_fn(c) ? clockwise_edm_builders.at(link) : counter_clockwise_edm_builders.at(link);
            log_trace(tt::LogOp, "Adding sender EDM channel");
            ccl::EriscDatamoverBuilder::ChannelBufferInterface const& sender_channel_buffer_info =
                sender_edm_builder.add_sender_channel(
                    worker_sender_semaphore_address,
                    clockwise_link_buffer_num_messages_to_send.at(c),
                    sender_worker_coords);
            edm_interface_addresses.worker_sender_edm_semaphore_addresses[global_worker_idx] = sender_channel_buffer_info.eth_semaphore_l1_address;
            edm_interface_addresses.worker_sender_edm_buffer_addresses[global_worker_idx] = sender_channel_buffer_info.eth_buffer_l1_address;
        }

        bool receiver_enabled = true; //(!is_linear || !is_first_chip_in_chain);
        if (receiver_enabled) {
            auto &receiver_edm_builder = is_buffer_in_clockwise_direction_fn(c) ? counter_clockwise_edm_builders.at(link) : clockwise_edm_builders.at(link);
            log_trace(tt::LogOp, "Adding receiver EDM channel");
            ccl::EriscDatamoverBuilder::ChannelBufferInterface const& receiver_channel_buffer_info =
                receiver_edm_builder.add_receiver_channel(
                    worker_receiver_semaphore_address,
                    counter_clockwise_link_buffer_num_messages_to_send.at(c),
                    receiver_worker_coords);
            edm_interface_addresses.worker_receiver_edm_semaphore_addresses[global_worker_idx] = receiver_channel_buffer_info.eth_semaphore_l1_address;
            edm_interface_addresses.worker_receiver_edm_buffer_addresses[global_worker_idx] = receiver_channel_buffer_info.eth_buffer_l1_address;
        }
    }
}

void build_reduce_scatter_worker(
    tt_metal::Program &program,
    ccl::CCLOpConfig const& op_config,
    ReduceScatterWorkerArgBuilder const& worker_arg_builder,
    CoreCoord const& worker_core,
    std::map<string, string> const& worker_defines,
    tt_metal::ComputeConfig compute_config
    ) {
    std::string const& receiver_kernel_path = op_config.is_input_sharded() ?  "receiver_kernel_path0" : "reciever_kernel_path1";
    std::string const& reduce_kernel_path = "reduce_unpack_kernel_path";
    std::string const& sender_kernel_path = op_config.is_input_sharded() ?  "sender_kernel_path0" : "sender_kernel_path1";
    // This will be configurable by sharded/non-sharded but present the same arg builder

    { // Move to kernel builder component
        std::string const& receiver_kernel_path = op_config.is_input_sharded() ? "receiver_kernel_path0" : "reciever_kernel_path1";
        KernelHandle worker_receiver_kernel_id = tt_metal::CreateKernel(
            program,
            receiver_kernel_path,
            worker_core,
            tt_metal::ReaderDataMovementConfig(worker_arg_builder.generate_receiver_kernel_ct_args(), worker_defines));

        // worker_receiver_kernels.push_back(worker_receiver_reader_kernel_id);

        tt_metal::SetRuntimeArgs(
            program,
            worker_receiver_kernel_id,
            worker_core,
            worker_arg_builder.generate_receiver_kernel_rt_args());
    }

    {
        compute_config.compile_args = worker_arg_builder.generate_reduce_op_kernel_ct_args();
        KernelHandle worker_reduce_kernel_id = tt_metal::CreateKernel(
            program,
            reduce_kernel_path,
            worker_core,
            compute_config);

        // worker_reduce__kernels.push_back(worker_reduce_kernel_id);

        tt_metal::SetRuntimeArgs(
            program,
            worker_reduce_kernel_id,
            worker_core,
            worker_arg_builder.generate_reduce_op_kernel_rt_args());

    }

    { // Move to kernel builder component
        KernelHandle worker_sender_kernel_id = tt_metal::CreateKernel(
            program,
            sender_kernel_path,
            worker_core,
            tt_metal::ReaderDataMovementConfig(worker_arg_builder.generate_sender_kernel_ct_args(), worker_defines));

        // worker_sender_kernels.push_back(worker_sender_kernel_id);

        tt_metal::SetRuntimeArgs(
            program,
            worker_sender_kernel_id,
            worker_core,
            worker_arg_builder.generate_sender_kernel_rt_args());
    }

}


// Notes on abbreviations:
// CW = clockwise
// CCW = counter-clockwise
// edm = erisc data mover

// How this reduce_scatter op works:
// For each chip, we have a element range of the input tensor shape that will eventually scatter
// out to it. For all other chunks outside that range, the chip will forward the chunk to the next chip.
// While forwarding the data, the chip will also reduce it with the local input tensor chunk corresponding
// with that received chunk. It will forward the partially reduced chunk.
// Reduces along rank
operation::ProgramWithCallbacks reduce_scatter_with_workers(
    const Tensor& input_tensor,
    Tensor& output_tensor, ReduceOpMath reduce_op,
    const uint32_t scatter_split_dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    ccl::Topology topology) {

    TT_ASSERT(input_tensor.get_legacy_shape() == output_tensor.get_legacy_shape(), "Input and output tensor shapes must match");

    /// Constants/Configuration
    ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode = ccl::EriscDataMoverBufferSharingMode::ROUND_ROBIN;
    auto op_config = ccl::CCLOpConfig(input_tensor, output_tensor);
    bool is_linear = topology == tt::tt_metal::ccl::Topology::Linear;
    auto num_edm_channels = decide_number_of_edm_channels(op_config, 8, false);
    auto const& edm_builder = create_erisc_datamover_builder(num_edm_channels, op_config.get_page_size(), ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode);

    std::vector<ccl::EriscDatamoverBuilder> cw_per_link_edm_builders(num_links, edm_builder);
    std::vector<ccl::EriscDatamoverBuilder> ccw_per_link_edm_builders(num_links, edm_builder);
    //////////////////

    tt_metal::Program program{};
    const auto& device = input_tensor.device();

    auto worker_receiver_semaphore_address = tt_metal::CreateSemaphore(program, worker_core_range, 0);
    auto worker_sender_semaphore_address = tt_metal::CreateSemaphore(program, worker_core_range, 0);

    auto tensor_slicer = InterleavedRingReduceScatterTensorSlicer(...);

    // Configure the EDM builders
    for (std::size_t link = 0; link < num_links; link++) {
        add_worker_config_to_edm_builders(
            device,
            op_config,
            worker_cores,
            num_channels_per_edm,
            clockwise_edm_builders,
            counter_clockwise_edm_builders,
            clockwise_link_buffer_num_messages_to_send,
            counter_clockwise_link_buffer_num_messages_to_send,
            worker_sender_semaphore_address,
            worker_receiver_semaphore_address,
            link,
            std::function<bool(uint32_t)> is_buffer_in_clockwise_direction_fn
            );
    }
    tt_metal::ComputeConfig compute_config ...;
    for (std::size_t link = 0; link < num_links; link++) {
        uint32_t global_worker_index = link * num_edm_channels;
        // Add the worker kerneles
        for (std::size_t worker = 0; worker < num_edm_channels; worker++) {
            // This will be configurable by sharded/non-sharded but present the same arg builder
            auto worker_arg_builder = ReduceScatterWorkerArgBuilder(tensor_slicer, worker);

            build_reduce_scatter_worker(
                program,
                op_config,
                worker_arg_builder,
                worker_cores.at(global_worker_index),
                worker_defines,
                compute_config
                );
        }
    }

    // Generate the EDM kernels
    ccl::generate_edm_kernels_for_ring_or_linear_topology(
        program,
        device,
        cw_per_link_edm_builders,
        ccw_per_link_edm_builders,
        receiver_device_id,
        sender_device_id,
        num_links,
        ring_size,
        ring_index,
        is_linear);

    auto override_runtime_arguments_callback = [...] (//num_links, total_worker_core_pairs_used, worker_reader_sender_kernels, worker_writer_sender_kernels, worker_reader_receiver_kernels, worker_writer_receiver_kernels, all_worker_sender_cores, all_worker_receiver_cores] (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        // const auto& input = input_tensors.at(0);
        // const auto& output = output_tensors.at(0);
        // for (uint32_t i = 0; i < total_worker_core_pairs_used; ++i) {
        //     auto &worker_reader_sender_runtime_args = GetRuntimeArgs(program, worker_reader_sender_kernels.at(i), all_worker_sender_cores.at(i));
        //     worker_reader_sender_runtime_args.at(0) = input.buffer()->address();
        //     worker_reader_sender_runtime_args.at(1) = output.buffer()->address();
        //     auto &worker_writer_sender_runtime_args = GetRuntimeArgs(program, worker_writer_sender_kernels.at(i), all_worker_sender_cores.at(i));
        //     worker_writer_sender_runtime_args.at(0) = output.buffer()->address();

        //     auto &worker_writer_receiver_runtime_args = GetRuntimeArgs(program, worker_writer_receiver_kernels.at(i), all_worker_receiver_cores.at(i));
        //     worker_writer_receiver_runtime_args.at(0) = output.buffer()->address();
        // }
    };

    TT_ASSERT(false, "Not implemented yet");

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}
} // namespace tt_metal

} // namespace tt
