// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/host/reduce_scatter_worker_builder.hpp"
#include <cstdint>
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_host.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"


namespace ttnn {
namespace ccl {
namespace reduce_scatter_detail {

ReduceScatterWorkerArgBuilder::ReduceScatterWorkerArgBuilder (
    Device const* device,
    ttnn::ccl::CCLOpConfig const& op_config,
    ttnn::ccl::RingTopology const& topology_config,
    ttnn::ccl::InterleavedTensorWorkerSlice const& worker_input_slice,
    WorkerTransferInfo const& worker_transfer_info,
    uint32_t worker_idx,
    uint32_t link,
    uint32_t cb_num_pages_per_packet,
    uint32_t worker_sender_semaphore_id,
    uint32_t worker_receiver_semaphore_id) :
    device(device),
    op_config(op_config),
    topology_config(topology_config),
    worker_input_slice(worker_input_slice),
    worker_transfer_info(worker_transfer_info),
    cb_num_pages_per_packet(cb_num_pages_per_packet),
    worker_sender_semaphore_id(worker_sender_semaphore_id),
    worker_receiver_semaphore_id(worker_receiver_semaphore_id) {
    // This algorithm assumes that the worker slices are sized such that they start at the same x offsets for each
    // new row they slice into (as they stride through the tensor)
    std::size_t num_slice_iterations =
        worker_input_slice.compute_num_worker_slice_iterations(worker_transfer_info.num_workers);
    std::size_t worker_slice_num_pages =
        worker_input_slice.worker_slice_shape.x * worker_input_slice.worker_slice_shape.y;
    std::size_t pages_per_full_chunk = worker_transfer_info.get_num_pages_per_full_chunk(link, worker_idx);
    std::size_t num_filler_pages_per_slice = pages_per_full_chunk - (worker_slice_num_pages % pages_per_full_chunk);
    this->total_num_math_pages = (worker_input_slice.get_worker_slice_num_pages() + num_filler_pages_per_slice) *
                                    num_slice_iterations * (topology_config.ring_size - 1);

    log_trace(tt::LogOp, "ReduceScatterWorkerArgBuilder: total_num_math_pages: {}", this->total_num_math_pages);
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_reduce_op_kernel_ct_args() const {
    log_trace(tt::LogOp, "Reduce Scatter Worker CT Args: None");
    return {};
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_reduce_op_kernel_rt_args(
    uint32_t link, uint32_t worker_index, uint32_t ring_size) const {
    log_trace(tt::LogOp, "generate_reduce_op_kernel_rt_args");

    auto const& args = std::vector<uint32_t>{total_num_math_pages, 1, 0};

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Worker RT Args:");
    log_trace(tt::LogOp, "\tblock_size: {}", args.at(i++));
    log_trace(tt::LogOp, "\ttotal_num_math_pages: {}", args.at(i++));
    log_trace(tt::LogOp, "\tacc_to_dst: {}", args.at(i++));

    return args;
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_receiver_kernel_ct_args() const {
    auto const& local_input_tensor = this->op_config.get_input_tensor(0);
    auto args = std::vector<uint32_t>{
        static_cast<uint32_t>(this->op_config.is_input_sharded() ? 1 : 0),
        static_cast<uint32_t>(
            this->op_config.get_input_tensor(0).memory_config().buffer_type == BufferType::DRAM ? 1 : 0),
            static_cast<uint32_t>(perform_readback_accumulation ? 1 : 0)
            };

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Receiver Worker CT Args:");
    log_trace(tt::LogOp, "\tis_sharded: {}", args.at(i++));
    log_trace(tt::LogOp, "\tsrc_is_dram: {}", args.at(i++));
    TT_ASSERT(args.size() == i, "Missed some args");

    if (local_input_tensor.is_sharded()) {
        auto const& shard_ct_args = ShardedAddrGenArgBuilder::emit_ct_args(local_input_tensor);
        std::copy(shard_ct_args.begin(), shard_ct_args.end(), std::back_inserter(args));
    } else {
        args.push_back(static_cast<uint32_t>(local_input_tensor.memory_config().memory_layout));
    }
    return args;
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_receiver_kernel_rt_args(
    ttnn::ccl::WorkerXY const& edm_core,
    uint32_t edm_core_semaphore_address,
    uint32_t edm_core_buffer_address,
    uint32_t link,
    uint32_t worker_index,
    bool is_in_clockwise_direction) const {
    TT_ASSERT(edm_core_semaphore_address > 0);
    TT_ASSERT(edm_core_buffer_address > 0);
    auto const& local_input_tensor = this->op_config.get_input_tensor(0);
    uint32_t starting_ring_index =
        is_in_clockwise_direction ? (this->topology_config.ring_index == 0 ? this->topology_config.ring_size - 1
                                                                            : this->topology_config.ring_index - 1)
                                    : (this->topology_config.ring_index == this->topology_config.ring_size - 1
                                            ? 0
                                            : this->topology_config.ring_index + 1);
    auto args = std::vector<uint32_t>{
        static_cast<uint32_t>(local_input_tensor.buffer()->address()),
        static_cast<uint32_t>(this->topology_config.ring_size),  // num_transfers
        static_cast<uint32_t>(this->worker_transfer_info.get_num_pages_per_full_chunk(link, worker_index)),
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

        this->total_num_math_pages};

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Receiver Worker RT Args:");
    log_trace(tt::LogOp, "\tsrc_addr: {}", args.at(i++));
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
        auto const& shard_rt_args = ShardedAddrGenArgBuilder::emit_rt_args(device, local_input_tensor);
        std::copy(shard_rt_args.begin(), shard_rt_args.end(), std::back_inserter(args));
    }
    return args;
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_sender_kernel_ct_args() const {
    auto const& local_output_tensor = this->op_config.get_output_tensor(0);
    auto args = std::vector<uint32_t>{
        static_cast<uint32_t>(this->op_config.is_input_sharded() ? 1 : 0),
        static_cast<uint32_t>(
            this->op_config.get_output_tensor(0).memory_config().buffer_type == BufferType::DRAM ? 1 : 0),
            static_cast<uint32_t>(signal_reader_on_output_tensor_partial_writes ? 1 : 0)
            };

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Sender Worker CT Args:");
    log_trace(tt::LogOp, "\tis_sharded: {}", args.at(i++));
    log_trace(tt::LogOp, "\tdst_is_dram: {}", args.at(i++));
    TT_ASSERT(args.size() == i, "Missed some args");

    if (local_output_tensor.is_sharded()) {
        auto const& shard_ct_args = ShardedAddrGenArgBuilder::emit_ct_args(local_output_tensor);
        std::copy(shard_ct_args.begin(), shard_ct_args.end(), std::back_inserter(args));
    } else {
        args.push_back(static_cast<uint32_t>(local_output_tensor.memory_config().memory_layout));
    }
    return args;
}

std::vector<uint32_t> ReduceScatterWorkerArgBuilder::generate_sender_kernel_rt_args(
    ttnn::ccl::WorkerXY edm_core,
    uint32_t edm_core_semaphore_address,
    uint32_t edm_core_buffer_address,
    uint32_t link,
    uint32_t worker_index,
    bool is_clockwise) const {
    TT_ASSERT(edm_core_semaphore_address > 0);
    TT_ASSERT(edm_core_buffer_address > 0);
    auto const& local_output_tensor = this->op_config.get_output_tensor(0);
    auto args = std::vector<uint32_t>{
        static_cast<uint32_t>(local_output_tensor.buffer()->address()),
        static_cast<uint32_t>(edm_core_buffer_address),
        static_cast<uint32_t>(edm_core_semaphore_address),
        static_cast<uint32_t>(edm_core.x),
        static_cast<uint32_t>(edm_core.y),
        static_cast<uint32_t>(this->topology_config.ring_size - 1),  // num_transfers),

        static_cast<uint32_t>(this->op_config.get_page_size()),
        static_cast<uint32_t>(this->worker_transfer_info.get_num_pages_per_full_chunk(link, worker_index)),

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

        total_num_math_pages};

    if (signal_reader_on_output_tensor_partial_writes) {
        args.push_back(static_cast<uint32_t>(reader_noc_x));
        args.push_back(static_cast<uint32_t>(reader_noc_y));
        args.push_back(static_cast<uint32_t>(reader_semaphore_id));
    }

    std::size_t i = 0;
    log_trace(tt::LogOp, "Reduce Scatter Sender Worker RT Args:");
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

    TT_ASSERT(args.size() == i, "Missed some args");

    if (local_output_tensor.is_sharded()) {
        auto const& shard_rt_args = ShardedAddrGenArgBuilder::emit_rt_args(device, local_output_tensor);
        std::copy(shard_rt_args.begin(), shard_rt_args.end(), std::back_inserter(args));
    }
    return args;
}



} // namespace reduce_scatter_detail
} // namespace ccl
} // namespace ttnn
