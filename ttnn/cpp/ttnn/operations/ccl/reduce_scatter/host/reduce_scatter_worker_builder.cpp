// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iterator>
#include <ranges>

#include "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/host/reduce_scatter_worker_builder.hpp"
#include "hostdevcommon/kernel_structs.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"

namespace ttnn {
namespace ccl {
namespace reduce_scatter_detail {

ReduceScatterWorkerArgBuilder::ReduceScatterWorkerArgBuilder (
    Device const* device,
    ttnn::ccl::CCLOpConfig const& op_config,
    ttnn::ccl::RingTopology const& topology_config,
    ttnn::ccl::InterleavedTensorWorkerSlice const& worker_input_slice,
    WorkerTransferInfo const& worker_transfer_info,
    ttnn::ccl::EriscDataMoverTerminationMode edm_termination_mode,
    std::size_t scatter_dim,
    std::size_t cb_num_pages_per_packet,
    uint32_t worker_sender_semaphore_id,
    uint32_t worker_receiver_semaphore_id,
    std::optional<uint32_t> receiver_worker_partial_ready_semaphore_id,
    std::size_t num_buffers_per_channel
    ) :
    device(device),
    op_config(op_config),
    topology_config(topology_config),
    worker_input_slice(worker_input_slice),
    worker_transfer_info(worker_transfer_info),
    edm_termination_mode(edm_termination_mode),
    cb_num_pages_per_packet(cb_num_pages_per_packet),
    worker_sender_semaphore_id(worker_sender_semaphore_id),
    worker_receiver_semaphore_id(worker_receiver_semaphore_id),
    num_buffers_per_channel(num_buffers_per_channel),
    receiver_worker_partial_ready_semaphore_id(receiver_worker_partial_ready_semaphore_id),
    scatter_dim(scatter_dim) {
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_reduce_op_kernel_ct_args() const {
    log_trace(tt::LogOp, "Reduce Scatter Worker CT Args: None");
    return {};
}

std::size_t ReduceScatterWorkerArgBuilder::get_total_num_math_pages(WorkerAttributes const& worker_attrs) const {
    // This algorithm assumes that the worker slices are sized such that they start at the same x offsets for each
    // new row they slice into (as they stride through the tensor)
    std::size_t num_slice_iterations =
        worker_input_slice.compute_num_worker_slice_iterations(worker_transfer_info.num_workers);
    std::size_t worker_slice_num_pages =
        worker_input_slice.worker_slice_shape.x * worker_input_slice.worker_slice_shape.y;
    std::size_t pages_per_full_chunk = worker_transfer_info.get_num_pages_per_full_chunk(worker_attrs);
    std::size_t num_filler_pages_per_slice = pages_per_full_chunk - (worker_slice_num_pages % pages_per_full_chunk);
    uint32_t total_num_math_pages = (worker_input_slice.get_worker_slice_num_pages() + num_filler_pages_per_slice) *
                                    num_slice_iterations * (topology_config.ring_size - 1);
    return total_num_math_pages;
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_reduce_op_kernel_rt_args(
    WorkerAttributes const& worker_attrs, std::size_t ring_size
) const {
    log_trace(tt::LogOp, "generate_reduce_op_kernel_rt_args");


    uint32_t total_num_math_pages = get_total_num_math_pages(worker_attrs);
    auto const& args = std::vector<uint32_t>{total_num_math_pages, 1, 0};

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Worker RT Args:");
    log_trace(tt::LogOp, "\tblock_size: {}", args.at(i++));
    log_trace(tt::LogOp, "\ttotal_num_math_pages: {}", args.at(i++));
    log_trace(tt::LogOp, "\tacc_to_dst: {}", args.at(i++));
    TT_ASSERT(args.size() == i, "Missed some args");

    return args;
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_receiver_kernel_ct_args() const {
    auto const& local_input_tensor = this->op_config.get_input_tensor(0);
    auto const& local_output_tensor = this->op_config.get_output_tensor(0);
    auto args = std::vector<uint32_t>{

        static_cast<uint32_t>(this->op_config.is_input_sharded() ? 1 : 0),

        static_cast<uint32_t>(local_input_tensor.memory_config().buffer_type == BufferType::DRAM ? 1 : 0),

        static_cast<uint32_t>(local_input_tensor.memory_config().memory_layout),

        static_cast<uint32_t>(local_output_tensor.memory_config().buffer_type == BufferType::DRAM ? 1 : 0),

        static_cast<uint32_t>(local_output_tensor.memory_config().memory_layout),

        static_cast<uint32_t>(this->num_buffers_per_channel),
        static_cast<uint32_t>(this->topology_config.is_linear ? 1 : 0),
            };

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Receiver Worker CT Args:");
    log_trace(tt::LogOp, "\tis_sharded: {}", args.at(i++));
    log_trace(tt::LogOp, "\tsrc_is_dram: {}", args.at(i++));
    log_trace(tt::LogOp, "\tinput_tensor_memory_layout: {}", args.at(i++));
    log_trace(tt::LogOp, "\tdest_is_dram: {}", args.at(i++));
    log_trace(tt::LogOp, "\toutput_tensor_memory_layout: {}", args.at(i++));
    log_trace(tt::LogOp, "\tnum_buffers_per_channel: {}", args.at(i++));
    log_trace(tt::LogOp, "\tis_linear: {}", args.at(i++));

    TT_ASSERT(args.size() == i, "Missed some args");
    if (local_input_tensor.is_sharded()) {
        std::ranges::copy(ShardedAddrGenArgBuilder::emit_ct_args(local_input_tensor), std::back_inserter(args));
        std::ranges::copy(ShardedAddrGenArgBuilder::emit_ct_args(local_output_tensor), std::back_inserter(args));

        ShardedAddrGenArgBuilder::log_sharded_tensor_kernel_args(local_input_tensor, "input");
        ShardedAddrGenArgBuilder::log_sharded_tensor_kernel_args(local_output_tensor, "output");
    }

    return args;
}

// TODO: need to get worker input slice
std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_receiver_kernel_rt_args(
    ttnn::ccl::WorkerXY const& edm_core,
    uint32_t edm_core_semaphore_address,
    uint32_t edm_core_buffer_address,
    WorkerAttributes const& worker_attrs) const {
    TT_ASSERT(edm_core_semaphore_address > 0);
    TT_ASSERT(edm_core_buffer_address > 0);
    auto const& local_input_tensor = this->op_config.get_input_tensor(0);
    auto const& local_output_tensor = this->op_config.get_output_tensor(0);
    bool is_in_clockwise_direction = worker_attrs.direction == Direction::CLOCKWISE;
    uint32_t starting_ring_index =
        is_in_clockwise_direction ? (this->topology_config.ring_index == 0 ? this->topology_config.ring_size - 1
                                                                            : this->topology_config.ring_index - 1)
                                    : (this->topology_config.ring_index == this->topology_config.ring_size - 1
                                            ? 0
                                            : this->topology_config.ring_index + 1);
    uint32_t num_transfers = this->topology_config.is_linear
                                 ? (this->topology_config.ring_size - (is_in_clockwise_direction ? this->topology_config.ring_index : (this->topology_config.ring_size - 1 - this->topology_config.ring_index)))
                                 : this->topology_config.ring_size;
    uint32_t total_num_math_pages = get_total_num_math_pages(worker_attrs);
    auto args = std::vector<uint32_t>{
        static_cast<uint32_t>(local_input_tensor.buffer()->address()),
        static_cast<uint32_t>(local_output_tensor.buffer()->address()),
        static_cast<uint32_t>(num_transfers),
        static_cast<uint32_t>(this->worker_transfer_info.get_num_pages_per_full_chunk(worker_attrs)),
        static_cast<uint32_t>(this->op_config.get_page_size()),
        static_cast<uint32_t>(starting_ring_index),
        static_cast<uint32_t>(this->topology_config.ring_size),
        static_cast<uint32_t>(this->worker_receiver_semaphore_id),
        static_cast<uint32_t>(is_in_clockwise_direction ? 1 : 0),
        static_cast<uint32_t>(this->cb_num_pages_per_packet),
        static_cast<uint32_t>(edm_core.x),
        static_cast<uint32_t>(edm_core.y),
        static_cast<uint32_t>(edm_core_semaphore_address),
        static_cast<uint32_t>(edm_core_buffer_address),

        static_cast<uint32_t>(worker_transfer_info.num_workers),

        static_cast<uint32_t>(this->worker_input_slice.tensor_shape.x),
        static_cast<uint32_t>(this->worker_input_slice.tensor_shape.y),

        static_cast<uint32_t>(this->worker_input_slice.tensor_slice_shape.x),
        static_cast<uint32_t>(this->worker_input_slice.tensor_slice_shape.y),

        static_cast<uint32_t>(this->worker_input_slice.worker_slice_shape.x),
        static_cast<uint32_t>(this->worker_input_slice.worker_slice_shape.y),

        static_cast<uint32_t>(this->worker_input_slice.worker_slice_offset.x),
        static_cast<uint32_t>(this->worker_input_slice.worker_slice_offset.y),

        total_num_math_pages};

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Receiver Worker RT Args:");
    log_trace(tt::LogOp, "\tinput_tensor_address: {}", args.at(i++));
    log_trace(tt::LogOp, "\toutput_tensor_address: {}", args.at(i++));
    log_trace(tt::LogOp, "\tnum_transfers: {}", args.at(i++));
    log_trace(tt::LogOp, "\tfull_chunk_num_pages: {}", args.at(i++));
    log_trace(tt::LogOp, "\tpage_size: {}", args.at(i++));
    log_trace(tt::LogOp, "\tmy_ring_idx: {}", args.at(i++));
    log_trace(tt::LogOp, "\tring_size: {}", args.at(i++));
    log_trace(tt::LogOp, "\tsem_addr: {}", args.at(i++));
    log_trace(tt::LogOp, "\tis_clockwise_direction: {}", args.at(i++));
    log_trace(tt::LogOp, "\thalf_cb_n_pages: {}", args.at(i++));

    log_trace(tt::LogOp, "\tedm_core_noc0_core_x: {}", args.at(i++));
    log_trace(tt::LogOp, "\tedm_core_noc0_core_y: {}", args.at(i++));
    log_trace(tt::LogOp, "\tedm_core_semaphore_address: {}", args.at(i++));
    log_trace(tt::LogOp, "\tedm_core_buffer_address: {}", args.at(i++));
    log_trace(tt::LogOp, "\tnum_concurrent_workers: {}", args.at(i++));

    log_trace(tt::LogOp, "\tinput_tensor_shape.x={}", args.at(i++));
    log_trace(tt::LogOp, "\tinput_tensor_shape.y={}", args.at(i++));
    log_trace(tt::LogOp, "\ttensor_slice_shape.x={}", args.at(i++));
    log_trace(tt::LogOp, "\ttensor_slice_shape.y={}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_shape.x={}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_shape.y={}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_offset.x={}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_offset.y={}", args.at(i++));
    log_trace(tt::LogOp, "\ttotal_num_math_pages={}", args.at(i++));

    TT_ASSERT(args.size() == i, "Missed some args");

    if (local_input_tensor.is_sharded()) {
        std::ranges::copy(ShardedAddrGenArgBuilder::emit_rt_args(device, local_input_tensor), std::back_inserter(args));
        ShardedAddrGenArgBuilder::log_sharded_tensor_kernel_args(local_input_tensor, "input");

        if (this->topology_config.is_linear) {
            std::ranges::copy(ShardedAddrGenArgBuilder::emit_rt_args(device, local_output_tensor), std::back_inserter(args));
            ShardedAddrGenArgBuilder::log_sharded_tensor_kernel_args(local_output_tensor, "output");
        }
    }

    return args;
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_sender_kernel_ct_args() const {
    auto const& local_output_tensor = this->op_config.get_output_tensor(0);

    auto args = std::vector<uint32_t>{
        static_cast<uint32_t>(this->op_config.is_input_sharded() ? 1 : 0),
        static_cast<uint32_t>(
            this->op_config.get_output_tensor(0).memory_config().buffer_type == BufferType::DRAM ? 1 : 0),
        static_cast<uint32_t>(this->num_buffers_per_channel),
        static_cast<uint32_t>(local_output_tensor.memory_config().memory_layout),
        static_cast<uint32_t>(this->topology_config.is_linear)
    };

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Sender Worker CT Args:");
    log_trace(tt::LogOp, "\tis_sharded: {}", args.at(i++));
    log_trace(tt::LogOp, "\tdst_is_dram: {}", args.at(i++));
    log_trace(tt::LogOp, "\tnum_buffers_per_channel: {}", args.at(i++));
    log_trace(tt::LogOp, "\ttensor_memory_layout: {}", args.at(i++));
    log_trace(tt::LogOp, "\tis_linear: {}", args.at(i++));
    TT_ASSERT(args.size() == i, "Missed some args");

    if (local_output_tensor.is_sharded()) {
        auto const& shard_ct_args = ShardedAddrGenArgBuilder::emit_ct_args(local_output_tensor);
        std::copy(shard_ct_args.begin(), shard_ct_args.end(), std::back_inserter(args));
    }
    return args;
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_sender_kernel_rt_args(
    WorkerEdmInterfaceArgs const& edm_interface,
    WorkerAttributes const& worker_attrs) const {

    bool is_clockwise = worker_attrs.direction == Direction::CLOCKWISE;
    TT_ASSERT(edm_interface.edm_semaphore_address > 0);
    TT_ASSERT(edm_interface.edm_buffer_base_address > 0);
    auto const& local_output_tensor = this->op_config.get_output_tensor(0);
    uint32_t num_transfers = this->topology_config.is_linear
        ? (this->topology_config.ring_size - (is_clockwise ? this->topology_config.ring_index : (this->topology_config.ring_size - 1 - this->topology_config.ring_index)))
        : this->topology_config.ring_size - 1;


    bool distance_from_start_of_line = is_clockwise ? this->topology_config.ring_index : this->topology_config.ring_size - 1 - this->topology_config.ring_index;
    bool distance_from_end_of_line = is_clockwise ? this->topology_config.ring_size - 1 - this->topology_config.ring_index : this->topology_config.ring_index;
    bool is_closer_to_start_of_line =
        (distance_from_start_of_line < distance_from_end_of_line) ||
        (distance_from_start_of_line == distance_from_end_of_line && is_clockwise);
    bool signal_reader_on_output_tensor_partial_writes =
            this->topology_config.is_linear && is_closer_to_start_of_line;

    uint32_t total_num_math_pages = get_total_num_math_pages(worker_attrs);
    auto args = std::vector<uint32_t>{
        static_cast<uint32_t>(local_output_tensor.buffer()->address()),
        static_cast<uint32_t>(edm_interface.edm_buffer_base_address),
        static_cast<uint32_t>(edm_interface.edm_semaphore_address),
        static_cast<uint32_t>(edm_interface.edm_noc_x),
        static_cast<uint32_t>(edm_interface.edm_noc_y),
        static_cast<uint32_t>(this->topology_config.ring_size - 1),  // num_transfers),

        static_cast<uint32_t>(this->op_config.get_page_size()),
        static_cast<uint32_t>(this->worker_transfer_info.get_num_pages_per_full_chunk(worker_attrs)),

        static_cast<uint32_t>(this->worker_sender_semaphore_id),
        static_cast<uint32_t>(this->cb_num_pages_per_packet),

        static_cast<uint32_t>(worker_transfer_info.num_workers),

        // For sender side, all worker slice info is the same except for the tensor shape
        // and for sender side specifically, there is only one tensor_slice_shape for the output
        // tensor (as opposed to `ring_size` tensor_slice_shapes for the input tensor), so we can
        // directly use it as the output tensor shape
        static_cast<uint32_t>(this->worker_input_slice.tensor_slice_shape.x),
        static_cast<uint32_t>(this->worker_input_slice.tensor_slice_shape.y),
        static_cast<uint32_t>(this->worker_input_slice.worker_slice_shape.x),
        static_cast<uint32_t>(this->worker_input_slice.worker_slice_shape.y),
        static_cast<uint32_t>(this->worker_input_slice.worker_slice_offset.x),
        static_cast<uint32_t>(this->worker_input_slice.worker_slice_offset.y),

        total_num_math_pages,
        static_cast<uint32_t>(signal_reader_on_output_tensor_partial_writes ? 1 : 0)};

    if (signal_reader_on_output_tensor_partial_writes) {
        TT_ASSERT(receiver_worker_partial_ready_semaphore_id.has_value());
        auto associated_worker_core = worker_attrs.associated_worker_core_logical;
        // Asserts here
        TT_ASSERT(associated_worker_core.has_value());
        auto const& worker_core_xy = this->device->worker_core_from_logical_core(associated_worker_core.value());

        args.push_back(static_cast<uint32_t>(worker_core_xy.x));
        args.push_back(static_cast<uint32_t>(worker_core_xy.y));
        args.push_back(static_cast<uint32_t>(receiver_worker_partial_ready_semaphore_id.value()));
    }

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Sender Worker RT Args (signal_reader_on_output_tensor_partial_writes={}):", signal_reader_on_output_tensor_partial_writes);
    log_trace(tt::LogOp, "\tdst_addr: {}", args.at(i++));
    log_trace(tt::LogOp, "\teth_sender_l1_base_addr: {}", args.at(i++));
    log_trace(tt::LogOp, "\teth_sender_l1_sem_addr: {}", args.at(i++));
    log_trace(tt::LogOp, "\teth_sender_noc_x: {}", args.at(i++));
    log_trace(tt::LogOp, "\teth_sender_noc_y: {}", args.at(i++));
    log_trace(tt::LogOp, "\tnum_transfers: {}", args.at(i++));
    log_trace(tt::LogOp, "\tpage_size: {}", args.at(i++));
    log_trace(tt::LogOp, "\tfull_chunk_num_pages: {}", args.at(i++));
    log_trace(tt::LogOp, "\twriter_send_sem_addr: {}", args.at(i++));
    log_trace(tt::LogOp, "\thalf_cb_n_pages: {}", args.at(i++));
    log_trace(tt::LogOp, "\tnum_concurrent_workers: {}", args.at(i++));

    log_trace(tt::LogOp, "\toutput_tensor_shape.x: {}", args.at(i++));
    log_trace(tt::LogOp, "\toutput_tensor_shape.y: {}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_shape.x: {}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_shape.y: {}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_offset.x: {}", args.at(i++));
    log_trace(tt::LogOp, "\tworker_slice_offset.y: {}", args.at(i++));

    log_trace(tt::LogOp, "\ttotal_num_math_pages={}", args.at(i++));
    log_trace(tt::LogOp, "\tsignal_reader_on_output_tensor_partial_writes={}", args.at(i++));

    if (signal_reader_on_output_tensor_partial_writes) {
        log_trace(tt::LogOp, "\teth_receiver_noc_x: {}", args.at(i++));
        log_trace(tt::LogOp, "\teth_receiver_noc_y: {}", args.at(i++));
        log_trace(tt::LogOp, "\teth_receiver_sem_addr: {}", args.at(i++));
    }

    TT_ASSERT(args.size() == i, "Missed some args");

    if (local_output_tensor.is_sharded()) {
        auto const& shard_rt_args = ShardedAddrGenArgBuilder::emit_rt_args(device, local_output_tensor);
        std::copy(shard_rt_args.begin(), shard_rt_args.end(), std::back_inserter(args));

        ShardedAddrGenArgBuilder::log_sharded_tensor_kernel_args(local_output_tensor, "output");
    }
    return args;
}


static void convert_slices_to_ccl_commands() {

}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_line_start_sender_kernel_rt_args(
    WorkerEdmInterfaceArgs const& edm_interface,
    std::size_t scatter_dim,
    WorkerAttributes const& worker_attrs) const
{
    static constexpr std::size_t max_args = 256; // need to get from tt_metal

    std::size_t num_commands = this->topology_config.ring_size - 1;
    auto &input_tensor = this->op_config.get_input_tensor(0);
    TT_ASSERT(input_tensor.get_legacy_shape().size() == 4, "Only 4D tensors are supported for reduce scatter");
    ttnn::ccl::Shape4D<uint32_t> input_tensor_shape = {input_tensor.get_legacy_shape()[0], input_tensor.get_legacy_shape()[1],input_tensor.get_legacy_shape()[2],input_tensor.get_legacy_shape()[3]};

    std::vector<uint32_t> args = {
        static_cast<uint32_t>(input_tensor.buffer()->address()),
        static_cast<uint32_t>(num_commands)
    };
    std::size_t logged_arg_idx = 0;
    log_trace(tt::LogOp, "ccl_send arg[{}]: buffer_address = {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;
    log_trace(tt::LogOp, "ccl_send arg[{}]: num_commands = {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;

    auto const& edm_interface_args = ttnn::ccl::emit_runtime_args(edm_interface);
    std::ranges::copy(edm_interface_args, std::back_inserter(args));
    for (auto const& arg : edm_interface_args) {
        log_trace(tt::LogOp, "ccl_send arg[{}]: edm_interface_args[] {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;
    }
    // ttnn::ccl::log_runtime_args(edm_interface, "edm_interface");

    std::ranges::copy(std::vector<uint32_t>{this->worker_transfer_info.get_num_pages_per_full_chunk(worker_attrs)}, std::back_inserter(args));
    log_trace(tt::LogOp, "ccl_send arg[{}]: pages_per_packet {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;

    std::ranges::copy(std::vector<uint32_t>{this->op_config.get_page_size()}, std::back_inserter(args));
    log_trace(tt::LogOp, "ccl_send arg[{}]: page_size {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;

    auto const& addr_gen_rt_args = ttnn::ccl::emit_address_generator_runtime_args(this->device, input_tensor);
    std::ranges::copy(addr_gen_rt_args, std::back_inserter(args));
    for (auto const& arg : addr_gen_rt_args) {
        log_trace(tt::LogOp, "ccl_send arg[{}]: addr_gen_rt_args[] {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;
    }

    std::ranges::copy(std::vector<uint32_t>{this->worker_sender_semaphore_id}, std::back_inserter(args));
    log_trace(tt::LogOp, "ccl_send arg[{}]: semaphore_id {}", logged_arg_idx, args[logged_arg_idx]);logged_arg_idx++;

    // If we are on device zero, we send n-1 chunks in ascending order

    auto const& tensor_shape = this->worker_input_slice.tensor_shape;
    auto const& tensor_slice_shape = this->worker_input_slice.tensor_slice_shape;

    auto num_slices = topology_config.ring_size;
    auto start_slice_index = topology_config.ring_index == 0 ? 0 : topology_config.ring_size - 1;
    TT_ASSERT(start_slice_index == 0 || start_slice_index == topology_config.ring_size - 1);
    auto end_slice_index_exclusive = topology_config.ring_size - start_slice_index; //-  start_slice_index == 0 ? num_slices : 0;

    // Add the command args
    auto const& slices = generate_slice_sequence_on_dim(
        tensor_shape,
        worker_input_slice.worker_slice_shape,
        scatter_dim,
        num_slices,
        start_slice_index,
        end_slice_index_exclusive,
        worker_attrs.channel
    );
    for (std::size_t i = 0; i < slices.size(); i++) {
        auto const& slice = slices[i];
        // Copy the header
        // TODO: encapsulate into a proper ccl command (generator) object
        // but for now lets just get something going that we can flush out the
        // flow with
        if (i == 0) {
            std::size_t args_index_old = args.size();
            args.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandHeader::to_uint32(
                ttnn::ccl::cmd::CclCommandHeader{ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_EDM, 1})));
            args.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES));
            auto const& ccl_command_tensor = ttnn::ccl::cmd::CclCommandTensor{
                {1, 1, slice.tensor_shape.y, slice.tensor_shape.x},
                {1, 1, slice.tensor_slice_shape.y, slice.tensor_slice_shape.x},
                {1, 1, slice.tensor_slice_offset.y, slice.tensor_slice_offset.x},
                {1, 1, slice.worker_slice_offset.y, slice.worker_slice_offset.x},
                slice.worker_slice_shape.x * slice.worker_slice_shape.y};
            auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::size_in_words();
            log_trace(tt::LogOp, "Emitting {} args for full tensor slice command", num_words_for_args);
            args.resize(args.size() + num_words_for_args);
            ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::
                pack_to(
                    &args[args.size() - num_words_for_args],
                    ccl_command_tensor
                    );
            std::size_t args_index_new = args.size();

            log_trace(tt::LogOp, "ccl_send command {} has {} args: ", i, args_index_new - args_index_old);
            for (std::size_t j = args_index_old; j < args_index_new; j++) {
                log_trace(tt::LogOp, "\targ {}: {}", j - args_index_old, args[j]);
            }
            // We can reused cached values for the first slice
        } else {
            auto const& last_slice = slices[i - 1];
            std::size_t args_index_old = args.size();
            auto header_index = args.size();
            args.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandHeader::to_uint32(
                ttnn::ccl::cmd::CclCommandHeader{ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_EDM, 1})));
            std::size_t num_args = 0;

            // tensor shape
            if (last_slice.tensor_shape != slice.tensor_shape) {
                args.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for tensor_shape field", num_words_for_args);
                args.resize(args.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::pack_to(
                    &args[args.size() - num_words_for_args],
                    {1, 1, slice.tensor_shape.y, slice.tensor_shape.x}
                );

                num_args++;
            }

            // tensor slice shape
            if (last_slice.tensor_slice_shape != slice.tensor_slice_shape) {
                args.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for tensor_slice_shape field", num_words_for_args);
                args.resize(args.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::pack_to(
                    &args[args.size() - num_words_for_args],
                    {1, 1, slice.tensor_slice_shape.y, slice.tensor_slice_shape.x}
                );

                num_args++;
            }

            // tensor slice offset
            if (last_slice.tensor_slice_offset != slice.tensor_slice_offset) {
                args.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for tensor_slice_offset field", num_words_for_args + 1);
                args.resize(args.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::pack_to(
                    &args[args.size() - num_words_for_args],
                    {1, 1, slice.tensor_slice_offset.y, slice.tensor_slice_offset.x}
                );

                num_args++;
            }

            // worker slice offset
            if (last_slice.worker_slice_offset != slice.worker_slice_offset) {
                args.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for worker_slice_offset field", num_words_for_args);
                args.resize(args.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::pack_to(
                    &args[args.size() - num_words_for_args],
                    {1, 1, slice.worker_slice_offset.y, slice.worker_slice_offset.x}
                );

                num_args++;
            }

            // worker_pages_per_slice
            if (last_slice.worker_slice_shape != slice.worker_slice_shape) {
                args.push_back(static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE));
                auto num_words_for_args = ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::size_in_words();
                log_trace(tt::LogOp, "Emitting {} args for worker_pages_per_slice field", num_words_for_args);
                args.resize(args.size() + num_words_for_args);
                ttnn::ccl::cmd::CclCommandArg<ttnn::ccl::cmd::CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::pack_to(
                    &args[args.size() - num_words_for_args],
                    slice.worker_slice_shape.y * slice.worker_slice_shape.x
                );

                num_args++;
            }

            args[header_index] = static_cast<uint32_t>(ttnn::ccl::cmd::CclCommandHeader::to_uint32(
                ttnn::ccl::cmd::CclCommandHeader{ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_EDM, 1}));

            std::size_t args_index_new = args.size();
            log_trace(tt::LogOp, "ccl_send command {} has {} args: ", i, args_index_new - args_index_old);
            for (std::size_t j = args_index_old; j < args_index_new; j++) {
                log_trace(tt::LogOp, "\targ {}: {}", j, args[j]);
            }
        }
    }

    log_trace(tt::LogOp, "Reduce Scatter Sender Worker has {} RT Args: {}", args.size(), args);


    TT_FATAL(args.size() < max_args, "Too many command args. The cluster size is too large for reduce scatter");

    return args;
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_line_start_sender_kernel_ct_args() const
{
    std::vector<uint32_t> args = {
        static_cast<uint32_t>(this->op_config.get_input_tensor(0).memory_config().memory_layout), // tensor memory layout
        static_cast<uint32_t>(this->op_config.get_input_tensor(0).buffer()->buffer_type()), // buffer type
        static_cast<uint32_t>(this->op_config.get_input_tensor(0).layout()), // page layout
        static_cast<uint32_t>(this->edm_termination_mode), // (EDM) termination mode
        static_cast<uint32_t>(tt::CB::c_in0) // cb_id
    };

    return args;
}

} // namespace reduce_scatter_detail
} // namespace ccl
} // namespace ttnn
