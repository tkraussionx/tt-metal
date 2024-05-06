// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#include "common/core_coord.h"
#include "eth_l1_address_map.h"
#include "impl/buffers/buffer.hpp"
#include "impl/kernels/data_types.hpp"
#include "tensor/tensor_impl.hpp"
#include "tt_dnn/op_library/ccl/ccl_common.hpp"
#include "tt_dnn/op_library/ccl/ccl_host_datastructures.hpp"
#include "tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/circular_buffer_types.hpp"

// Includes that need to be moved to CCL datastructures header
#include <vector>

using namespace tt::constants;

// Notes on abbreviations:
// cw = clockwise
// ccw = counter-clockwise
// edm = erisc data mover

// How this reduce_scatter op works:
// For each chip, we have a element range of the input tensor shape that will eventually scatter
// out to it. For all other chunks outside that range, the chip will forward the chunk to the next chip.
// While forwarding the data, the chip will also reduce it with the local input tensor chunk corresponding
// with that received chunk. It will forward the partially reduced chunk.
// Reduces along rank

namespace tt {

namespace tt_metal {

struct WorkerTransferInfo {
    WorkerTransferInfo(
        std::vector<uint32_t> pages_per_full_chunk_per_worker,
        std::vector<uint32_t> num_messages_per_worker,
        std::vector<uint32_t> remaining_num_pages_per_worker,
        uint32_t num_links,
        uint32_t num_workers) :
        pages_per_full_chunk_per_worker(pages_per_full_chunk_per_worker),
        num_messages_per_worker(num_messages_per_worker),
        remaining_num_pages_per_worker(remaining_num_pages_per_worker),
        num_links(num_links),
        num_workers(num_workers) {}

    uint32_t get_num_pages_per_full_chunk(uint32_t link, uint32_t worker_idx) const {
        return pages_per_full_chunk_per_worker.at(link * num_workers + worker_idx);
    }
    uint32_t get_num_remaining_pages(uint32_t link, uint32_t worker_idx) const {
        return remaining_num_pages_per_worker.at(link * num_workers + worker_idx);
    }
    uint32_t get_num_transfers(uint32_t link, uint32_t worker_idx) const {
        return get_num_full_chunks(link, worker_idx);
    }
    uint32_t get_num_full_chunks(uint32_t link, uint32_t worker_idx) const {
        return num_messages_per_worker.at(link * num_workers + worker_idx) -
               (get_num_remaining_pages(link, worker_idx) > 0 ? 1 : 0);
    }
    uint32_t get_num_pages_per_ring_index(uint32_t link, uint32_t worker_idx) const {
        return get_num_full_chunks(link, worker_idx) * get_num_pages_per_full_chunk(link, worker_idx) +
               get_num_remaining_pages(link, worker_idx);
    }

    std::vector<uint32_t> pages_per_full_chunk_per_worker;
    std::vector<uint32_t> num_messages_per_worker;
    std::vector<uint32_t> remaining_num_pages_per_worker;
    uint32_t num_links;
    uint32_t num_workers;
};

/////////////////////////////////////////////////////////////

// TODO(snijjar): move enable_bidirectional to a topology specific config
static std::size_t decide_number_of_edm_channels(
    ccl::CCLOpConfig const& ccl_op_config, std::size_t max_num_workers, bool enable_bidirectional) {
    return ccl_op_config.is_input_sharded() ? std::min<uint32_t>(
                                                  ccl_op_config.get_shard_grid_size(),
                                                  std::min<std::size_t>(max_num_workers, enable_bidirectional ? 8 : 4))
                                            : std::min<std::size_t>(max_num_workers, enable_bidirectional ? 8 : 4);
}

struct ReduceScatterWorkerArgBuilder {
    ReduceScatterWorkerArgBuilder(
        ccl::CCLOpConfig const& op_config,
        ccl::RingTopology const& topology_config,
        ccl::LegacyCclTensorSlicer& tensor_slicer,
        WorkerTransferInfo const& worker_transfer_info,
        uint32_t worker_idx,
        uint32_t cb_num_pages,
        uint32_t worker_receiver_semaphore_address,
        uint32_t worker_sender_semaphore_address) :
        op_config(op_config),
        topology_config(topology_config),
        tensor_slicer(tensor_slicer),
        worker_transfer_info(worker_transfer_info),
        cb_num_pages(cb_num_pages),
        worker_receiver_semaphore_address(worker_receiver_semaphore_address),
        worker_sender_semaphore_address(worker_sender_semaphore_address) {}

    std::vector<uint32_t> generate_reduce_op_kernel_ct_args() const {
        log_trace(tt::LogOp, "Reduce Scatter Worker CT Args: None");
        return {};
    }

    std::vector<uint32_t> generate_reduce_op_kernel_rt_args(uint32_t link, uint32_t worker_index) const {
        auto const& args = std::vector<uint32_t>{
            this->worker_transfer_info.get_num_pages_per_ring_index(link, worker_index),
            this->tensor_slicer.input_page_size};

        std::size_t i = 0;
        log_trace(tt::LogOp, "Reduce Scatter Worker RT Args:");
        log_trace(tt::LogOp, "\tnum_pages: {}", args.at(i++));
        log_trace(tt::LogOp, "\tpage_size: {}", args.at(i++));

        return args;
    }

    std::vector<uint32_t> generate_receiver_kernel_ct_args() const {
        auto const& args = std::vector<uint32_t>{
            static_cast<uint32_t>(
                this->op_config.get_input_tensor(0).memory_config().buffer_type == BufferType::DRAM ? 1 : 0),
            static_cast<uint32_t>(
                this->op_config.get_output_tensor(0).memory_config().buffer_type == BufferType::DRAM ? 1 : 0)};

        std::size_t i = 0;
        log_trace(tt::LogOp, "Reduce Scatter Receiver Worker CT Args:");
        log_trace(tt::LogOp, "\tsrc_is_dram: {}", args.at(i++));
        log_trace(tt::LogOp, "\tdst_is_dram: {}", args.at(i++));
        TT_ASSERT(args.size() == i, "Missed some args");

        return args;
    }

    std::vector<uint32_t> generate_receiver_kernel_rt_args(
        ccl::WorkerXY edm_core,
        uint32_t edm_core_semaphore_address,
        uint32_t edm_core_buffer_address,
        uint32_t link,
        uint32_t worker_index,
        bool is_in_clockwise_direction) const {
        uint32_t last_output_page_offset = (this->topology_config.ring_size - 1) * tensor_slicer.output_page_offset;
        uint32_t last_output_addr_offset = (this->topology_config.ring_size - 1) * tensor_slicer.output_addr_offset;
        auto args = std::vector<uint32_t>{
            static_cast<uint32_t>(
                this->op_config.get_input_tensor(this->topology_config.ring_index).buffer()->address()),
            static_cast<uint32_t>(
                this->op_config.get_output_tensor(this->topology_config.ring_index).buffer()->address()),
            static_cast<uint32_t>(this->topology_config.ring_size),  // num_transfers
            static_cast<uint32_t>(this->worker_transfer_info.get_num_full_chunks(link, worker_index)),
            static_cast<uint32_t>(this->op_config.get_page_size()),
            static_cast<uint32_t>(this->op_config.get_page_size()),  // output_page_size
            static_cast<uint32_t>(this->worker_transfer_info.get_num_pages_per_full_chunk(link, worker_index)),
            static_cast<uint32_t>(this->worker_transfer_info.get_num_remaining_pages(link, worker_index)),
            static_cast<uint32_t>(this->tensor_slicer.input_start_page_idx),
            static_cast<uint32_t>(this->tensor_slicer.output_start_page_idx),
            static_cast<uint32_t>(this->tensor_slicer.output_start_addr_offset),
            static_cast<uint32_t>(this->tensor_slicer.row_idx),
            static_cast<uint32_t>(this->tensor_slicer.col_idx),
            static_cast<uint32_t>(this->tensor_slicer.row_offset),
            static_cast<uint32_t>(this->tensor_slicer.col_offset),
            static_cast<uint32_t>(this->tensor_slicer.num_rows),
            static_cast<uint32_t>(this->tensor_slicer.num_cols),
            static_cast<uint32_t>(last_output_page_offset),  //
            static_cast<uint32_t>(this->tensor_slicer.output_page_offset),
            static_cast<uint32_t>(last_output_addr_offset),  //
            static_cast<uint32_t>(this->tensor_slicer.output_addr_offset),
            static_cast<uint32_t>(this->tensor_slicer.output_addr_offset),   // input_start_ring_idx), //
            static_cast<uint32_t>(this->worker_receiver_semaphore_address),  //
            static_cast<uint32_t>(is_in_clockwise_direction ? 1 : 0),        //
            static_cast<uint32_t>(this->cb_num_pages / 2),
            static_cast<uint32_t>(this->topology_config.ring_size),
            static_cast<uint32_t>(edm_core.x),
            static_cast<uint32_t>(edm_core.y),
            static_cast<uint32_t>(edm_core_semaphore_address),
            static_cast<uint32_t>(edm_core_buffer_address)};

        std::size_t i = 0;
        log_trace(tt::LogOp, "Reduce Scatter Receiver Worker RT Args:");
        log_trace(tt::LogOp, "\tsrc_addr: {}", args.at(i++));
        log_trace(tt::LogOp, "\tdst_addr: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_transfers: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_full_chunks: {}", args.at(i++));
        log_trace(tt::LogOp, "\tpage_size: {}", args.at(i++));
        log_trace(tt::LogOp, "\toutput_page_size: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_pages: {}", args.at(i++));
        log_trace(tt::LogOp, "\trem_num_pages: {}", args.at(i++));
        log_trace(tt::LogOp, "\tinput_start_idx: {}", args.at(i++));
        log_trace(tt::LogOp, "\toutput_start_idx: {}", args.at(i++));
        log_trace(tt::LogOp, "\toutput_start_addr_offset: {}", args.at(i++));
        log_trace(tt::LogOp, "\trow_start_idx: {}", args.at(i++));
        log_trace(tt::LogOp, "\tcol_start_idx: {}", args.at(i++));
        log_trace(tt::LogOp, "\trow_offset: {}", args.at(i++));
        log_trace(tt::LogOp, "\tcol_offset: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_rows: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_cols: {}", args.at(i++));
        log_trace(tt::LogOp, "\tlast_output_page_offset: {}", args.at(i++));
        log_trace(tt::LogOp, "\toutput_page_offset: {}", args.at(i++));
        log_trace(tt::LogOp, "\tlast_output_addr_offset: {}", args.at(i++));
        log_trace(tt::LogOp, "\toutput_addr_offset: {}", args.at(i++));
        log_trace(tt::LogOp, "\tinput_start_ring_idx: {}", args.at(i++));
        log_trace(tt::LogOp, "\tsem_addr: {}", args.at(i++));
        log_trace(tt::LogOp, "\tis_clockwise_direction: {}", args.at(i++));
        log_trace(tt::LogOp, "\thalf_cb_n_pages: {}", args.at(i++));
        log_trace(tt::LogOp, "\tring_size: {}", args.at(i++));
        log_trace(tt::LogOp, "\tedm_core_noc0_core_x: {}", args.at(i++));
        log_trace(tt::LogOp, "\tedm_core_noc0_core_y: {}", args.at(i++));
        log_trace(tt::LogOp, "\tedm_core_semaphore_address: {}", args.at(i++));
        log_trace(tt::LogOp, "\tedm_core_buffer_address: {}", args.at(i++));
        TT_ASSERT(args.size() == i, "Missed some args");

        return args;
    }

    std::vector<uint32_t> generate_sender_kernel_ct_args() const {
        auto const& args = std::vector<uint32_t>{static_cast<uint32_t>(
            this->op_config.get_output_tensor(0).memory_config().buffer_type == BufferType::DRAM ? 1 : 0)};

        std::size_t i = 0;
        log_trace(tt::LogOp, "Reduce Scatter Sender Worker CT Args:");
        log_trace(tt::LogOp, "\tdst_is_dram: {}", args.at(i++));
        TT_ASSERT(args.size() == i, "Missed some args");

        return args;
    }

    std::vector<uint32_t> generate_sender_kernel_rt_args(
        ccl::WorkerXY edm_core,
        uint32_t edm_core_semaphore_address,
        uint32_t edm_core_buffer_address,
        uint32_t link,
        uint32_t worker_index,
        bool is_clockwise) const {
        uint32_t last_output_page_offset = (this->topology_config.ring_size - 1) * tensor_slicer.output_page_offset;
        uint32_t last_output_addr_offset = (this->topology_config.ring_size - 1) * tensor_slicer.output_addr_offset;
        auto const& args = std::vector<uint32_t>{
            static_cast<uint32_t>(
                this->op_config.get_output_tensor(this->topology_config.ring_index).buffer()->address()),
            static_cast<uint32_t>(edm_core_buffer_address),
            static_cast<uint32_t>(edm_core_semaphore_address),
            static_cast<uint32_t>(this->topology_config.ring_size - 1),  // num_transfers),
            static_cast<uint32_t>(this->worker_transfer_info.get_num_full_chunks(link, worker_index)),
            static_cast<uint32_t>(this->op_config.get_page_size()),
            static_cast<uint32_t>(this->op_config.get_page_size()),
            static_cast<uint32_t>(this->worker_transfer_info.get_num_pages_per_full_chunk(link, worker_index)),
            static_cast<uint32_t>(this->worker_transfer_info.get_num_remaining_pages(link, worker_index)),
            static_cast<uint32_t>(this->tensor_slicer.input_start_page_idx),   // input_start_idx),
            static_cast<uint32_t>(this->tensor_slicer.output_start_page_idx),  // output_start_idx),
            static_cast<uint32_t>(this->tensor_slicer.output_start_addr_offset),
            static_cast<uint32_t>(this->tensor_slicer.row_idx),  // row_start_idx),
            static_cast<uint32_t>(this->tensor_slicer.col_idx),  // col_start_idx),
            static_cast<uint32_t>(this->tensor_slicer.row_offset),
            static_cast<uint32_t>(this->tensor_slicer.col_offset),
            static_cast<uint32_t>(this->tensor_slicer.num_rows),
            static_cast<uint32_t>(this->tensor_slicer.num_cols),

            static_cast<uint32_t>(this->topology_config.ring_index),  // input_start_ring_idx),
            static_cast<uint32_t>(this->worker_sender_semaphore_address),
            static_cast<uint32_t>(edm_core.x),
            static_cast<uint32_t>(edm_core.y),
            static_cast<uint32_t>(this->cb_num_pages / 2),
            static_cast<uint32_t>(this->topology_config.ring_size),
            static_cast<uint32_t>(this->tensor_slicer.output_start_page_idx),
            static_cast<uint32_t>(last_output_addr_offset),           //
            static_cast<uint32_t>(last_output_page_offset),           //
            static_cast<uint32_t>(tensor_slicer.output_addr_offset),  //
            static_cast<uint32_t>(tensor_slicer.output_page_offset),  //
            static_cast<uint32_t>(is_clockwise ? 1 : 0)};

        std::size_t i = 0;
        log_trace(tt::LogOp, "Reduce Scatter Sender Worker RT Args:");
        log_trace(tt::LogOp, "\tdst_addr: {}", args.at(i++));
        log_trace(tt::LogOp, "\teth_sender_l1_base_addr: {}", args.at(i++));
        log_trace(tt::LogOp, "\teth_sender_l1_sem_addr: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_transfers: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_full_chunks: {}", args.at(i++));
        log_trace(tt::LogOp, "\tpage_size: {}", args.at(i++));
        log_trace(tt::LogOp, "\toutput_page_size: {}", args.at(i++));
        log_trace(tt::LogOp, "\tfull_chunk_num_pages: {}", args.at(i++));
        log_trace(tt::LogOp, "\trem_num_pages: {}", args.at(i++));
        log_trace(tt::LogOp, "\tinput_start_idx: {}", args.at(i++));
        log_trace(tt::LogOp, "\toutput_start_idx: {}", args.at(i++));
        log_trace(tt::LogOp, "\toutput_start_addr_offset: {}", args.at(i++));
        log_trace(tt::LogOp, "\trow_start_idx: {}", args.at(i++));
        log_trace(tt::LogOp, "\tcol_start_idx: {}", args.at(i++));
        log_trace(tt::LogOp, "\trow_offset: {}", args.at(i++));
        log_trace(tt::LogOp, "\tcol_offset: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_rows: {}", args.at(i++));
        log_trace(tt::LogOp, "\tnum_cols: {}", args.at(i++));
        log_trace(tt::LogOp, "\tinput_start_ring_idx: {}", args.at(i++));
        log_trace(tt::LogOp, "\twriter_send_sem_addr: {}", args.at(i++));
        log_trace(tt::LogOp, "\teth_sender_noc_x: {}", args.at(i++));
        log_trace(tt::LogOp, "\teth_sender_noc_y: {}", args.at(i++));
        log_trace(tt::LogOp, "\thalf_cb_n_pages: {}", args.at(i++));
        log_trace(tt::LogOp, "\tring_size: {}", args.at(i++));
        log_trace(tt::LogOp, "\toutput_start_page_idx: {}", args.at(i++));
        log_trace(tt::LogOp, "\tlast_output_addr_offset: {}", args.at(i++));
        log_trace(tt::LogOp, "\tlast_output_page_offset: {}", args.at(i++));
        log_trace(tt::LogOp, "\toutput_addr_offset: {}", args.at(i++));
        log_trace(tt::LogOp, "\toutput_page_offset: {}", args.at(i++));
        log_trace(tt::LogOp, "\tis_clockwise_direction: {}", args.at(i++));
        TT_ASSERT(args.size() == i, "Missed some args");

        return args;
    }

    ccl::RingTopology const& topology_config;
    ccl::CCLOpConfig const& op_config;
    ccl::LegacyCclTensorSlicer& tensor_slicer;
    WorkerTransferInfo const& worker_transfer_info;
    uint32_t cb_num_pages;
    uint32_t worker_receiver_semaphore_address;
    uint32_t worker_sender_semaphore_address;
    bool src_is_dram;
    bool dst_is_dram;
};

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
static void add_worker_config_to_edm_builders(
    Device* device,
    ccl::CCLOpConfig const& op_config,
    std::vector<CoreCoord> const& worker_cores,
    uint32_t num_channels_per_edm,

    std::vector<ccl::EriscDatamoverBuilder>& clockwise_edm_builders,
    std::vector<ccl::EriscDatamoverBuilder>& counter_clockwise_edm_builders,

    std::vector<uint32_t> const& cw_edm_channel_num_messages_to_send,
    std::vector<uint32_t> const& ccw_edm_channel_num_messages_to_send,

    uint32_t worker_sender_semaphore_address,
    uint32_t worker_receiver_semaphore_address,
    uint32_t link,
    std::function<bool(uint32_t)> is_buffer_in_clockwise_direction_fn,

    EdmInterfaceAddresses& edm_interface_addresses) {
    // uint32_t workers_per_link = all_gather_config.get_num_workers_per_link() / num_channels_per_edm;
    // Currently setup for non-full-worker-grid setup
    for (uint32_t c = 0; c < num_channels_per_edm; ++c) {
        uint32_t global_worker_idx = c + num_channels_per_edm * link;
        uint32_t num_workers_per_eth_buffer = 1;  // std::min(workers_per_link, num_channels_per_edm );

        std::vector<ccl::WorkerXY> sender_worker_coords;
        std::vector<ccl::WorkerXY> receiver_worker_coords;
        for (uint32_t w = c * num_workers_per_eth_buffer; w < (c + 1) * num_workers_per_eth_buffer; ++w) {
            sender_worker_coords.push_back(ccl::WorkerXY(
                device->worker_core_from_logical_core(worker_cores.at(w)).x,
                device->worker_core_from_logical_core(worker_cores.at(w)).y));
            receiver_worker_coords.push_back(ccl::WorkerXY(
                device->worker_core_from_logical_core(worker_cores.at(w)).x,
                device->worker_core_from_logical_core(worker_cores.at(w)).y));
        }

        bool sender_enabled = true;  // (!is_linear || !is_last_chip_in_chain); // update for linear
        if (sender_enabled) {
            auto& sender_edm_builder = is_buffer_in_clockwise_direction_fn(c) ? clockwise_edm_builders.at(link)
                                                                              : counter_clockwise_edm_builders.at(link);
            log_trace(tt::LogOp, "Adding sender EDM channel");
            ccl::EriscDatamoverBuilder::ChannelBufferInterface const& sender_channel_buffer_info =
                sender_edm_builder.add_sender_channel(
                    worker_sender_semaphore_address, cw_edm_channel_num_messages_to_send.at(c), sender_worker_coords);
            edm_interface_addresses.worker_sender_edm_semaphore_addresses[global_worker_idx] =
                sender_channel_buffer_info.eth_semaphore_l1_address;
            edm_interface_addresses.worker_sender_edm_buffer_addresses[global_worker_idx] =
                sender_channel_buffer_info.eth_buffer_l1_address;
        }

        bool receiver_enabled = true;  //(!is_linear || !is_first_chip_in_chain);
        if (receiver_enabled) {
            auto& receiver_edm_builder = is_buffer_in_clockwise_direction_fn(c)
                                             ? counter_clockwise_edm_builders.at(link)
                                             : clockwise_edm_builders.at(link);
            log_trace(tt::LogOp, "Adding receiver EDM channel");
            ccl::EriscDatamoverBuilder::ChannelBufferInterface const& receiver_channel_buffer_info =
                receiver_edm_builder.add_receiver_channel(
                    worker_receiver_semaphore_address,
                    ccw_edm_channel_num_messages_to_send.at(c),
                    receiver_worker_coords);
            edm_interface_addresses.worker_receiver_edm_semaphore_addresses[global_worker_idx] =
                receiver_channel_buffer_info.eth_semaphore_l1_address;
            edm_interface_addresses.worker_receiver_edm_buffer_addresses[global_worker_idx] =
                receiver_channel_buffer_info.eth_buffer_l1_address;
        }
    }
}

static void build_reduce_scatter_worker(
    tt_metal::Program& program,
    Device const* device,
    ccl::RingTopology const& topology_config,
    ccl::CCLOpConfig const& op_config,
    ReduceScatterWorkerArgBuilder const& worker_arg_builder,
    std::vector<ccl::EriscDatamoverBuilder>& cw_edm_builders,
    std::vector<ccl::EriscDatamoverBuilder>& ccw_edm_builders,
    EdmInterfaceAddresses const& edm_interface_addresses,
    CoreCoord const& worker_core,
    uint32_t num_edm_channels,
    uint32_t link,
    uint32_t worker_index,
    std::map<string, string> const& worker_defines,
    BinaryOpType binary_math_op) {
    static std::string const& receiver_kernel_path =
        "tt_eager/tt_dnn/op_library/ccl/reduce_scatter/kernels/worker_interleaved_ring_reduce_scatter_reader.cpp";
    static std::string const& sender_kernel_path =
        "tt_eager/tt_dnn/op_library/ccl/reduce_scatter/kernels/worker_interleaved_ring_reduce_scatter_sender.cpp";

    // This will be configurable by sharded/non-sharded but present the same arg builder

    bool is_in_clockwise_direction = true;
    uint32_t global_worker_index = link * num_edm_channels + worker_index;
    {
        CoreCoord const& receiver_edm = is_in_clockwise_direction ? topology_config.eth_receiver_cores.at(link)
                                                                  : topology_config.eth_sender_cores.at(link);
        ccl::WorkerXY receiver_edm_noc_coord = ccl::WorkerXY(
            device->ethernet_core_from_logical_core(receiver_edm).x,
            device->ethernet_core_from_logical_core(receiver_edm).y);
        const uint32_t edm_core_semaphore_address =
            is_in_clockwise_direction
                ? edm_interface_addresses.worker_receiver_edm_semaphore_addresses.at(global_worker_index)
                : edm_interface_addresses.worker_sender_edm_semaphore_addresses.at(global_worker_index);
        const uint32_t edm_core_buffer_address =
            is_in_clockwise_direction
                ? edm_interface_addresses.worker_receiver_edm_buffer_addresses.at(global_worker_index)
                : edm_interface_addresses.worker_sender_edm_buffer_addresses.at(global_worker_index);
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
            worker_arg_builder.generate_receiver_kernel_rt_args(
                receiver_edm_noc_coord,  // ccl::WorkerXY edm_core,
                edm_core_semaphore_address,
                edm_core_buffer_address,
                link,
                worker_index,
                is_in_clockwise_direction));
    }

    {
        std::map<string, string> eltwise_defines =
            eltwise_binary_op_utils::get_defines(binary_math_op, std::nullopt /*fused_activations*/);
        KernelHandle worker_reduce_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/compute/eltwise_binary.cpp",
            worker_core,
            tt_metal::ComputeConfig{.defines = eltwise_defines});

        tt_metal::SetRuntimeArgs(
            program,
            worker_reduce_kernel_id,
            worker_core,
            worker_arg_builder.generate_reduce_op_kernel_rt_args(link, worker_index));
    }

    {
        CoreCoord sender_edm = is_in_clockwise_direction ? topology_config.eth_sender_cores.at(link)
                                                         : topology_config.eth_receiver_cores.at(link);
        ccl::WorkerXY sender_edm_noc_coord = ccl::WorkerXY(
            device->ethernet_core_from_logical_core(sender_edm).x,
            device->ethernet_core_from_logical_core(sender_edm).y);
        const uint32_t edm_core_semaphore_address =
            is_in_clockwise_direction
                ? edm_interface_addresses.worker_sender_edm_semaphore_addresses.at(global_worker_index)
                : edm_interface_addresses.worker_receiver_edm_semaphore_addresses.at(global_worker_index);
        const uint32_t edm_core_buffer_address =
            is_in_clockwise_direction
                ? edm_interface_addresses.worker_sender_edm_buffer_addresses.at(global_worker_index)
                : edm_interface_addresses.worker_receiver_edm_buffer_addresses.at(global_worker_index);
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
            worker_arg_builder.generate_sender_kernel_rt_args(
                sender_edm_noc_coord,  // ccl::WorkerXY
                edm_core_semaphore_address,
                edm_core_buffer_address,
                link,
                worker_index,
                is_in_clockwise_direction));
    }
}

CoreRangeSet select_worker_cores(ccl::CCLOpConfig const& op_config, uint32_t num_links) {
    switch (op_config.get_topology()) {
        case tt::tt_metal::ccl::Topology::Linear:
            // 4 per direction per link
            return CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(num_links - 1, 7))});
        case tt::tt_metal::ccl::Topology::Ring:
            // 4 per direction per link
            return CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(num_links - 1, 7))});
        default: TT_ASSERT(false, "Unsupported topology"); return CoreRangeSet({});
    };
}

// map: (CW) link -> (CW) edm num messages to send per channel
// map: (CCW) link -> (CCW) edm num messages to send per channel
// There's a bit of a mutual dependence here between the number of workers and the number of channels,
// and the number of channels and the channel buffer size and the buffer size and the number of transfers
WorkerTransferInfo compute_num_edm_messages_per_channel(
    ccl::CCLOpConfig const& op_config,
    uint32_t const page_size_in_bytes,
    uint32_t const pages_per_slice,

    std::vector<ccl::EriscDatamoverBuilder> const& cw_per_link_edm_builders,
    std::vector<ccl::EriscDatamoverBuilder> const& ccw_per_link_edm_builders,
    std::size_t const num_edm_channels,
    std::size_t const num_links) {
    auto get_iter_begin = [num_edm_channels](
                              std::vector<uint32_t>& vec, std::size_t link) -> std::vector<uint32_t>::iterator {
        return vec.begin() + (link * num_edm_channels);
    };

    auto get_iter_end = [num_edm_channels, num_links](
                            std::vector<uint32_t>& vec, std::size_t link) -> std::vector<uint32_t>::iterator {
        bool last_link = link == num_links - 1;
        TT_ASSERT(
            (!last_link && ((link + 1) * num_edm_channels < vec.size())) ||
            (last_link && ((link + 1) * num_edm_channels == vec.size())));
        return last_link ? vec.end() : vec.begin() + ((link + 1) * num_edm_channels);
    };

    std::unordered_map<int, std::vector<uint32_t>> cw_edm_channel_num_messages_to_send;
    std::unordered_map<int, std::vector<uint32_t>> ccw_edm_channel_num_messages_to_send;

    std::size_t const total_num_pages = pages_per_slice;
    std::vector<uint32_t> pages_per_link(num_links, total_num_pages / num_links);
    for (std::size_t i = 0; i < total_num_pages % num_links; i++) {
        pages_per_link.at(i)++;
    }

    // Pages per EDM channel
    std::size_t total_num_edm_channels = num_links * num_edm_channels;
    std::vector<uint32_t> num_pages_per_edm_channel(total_num_edm_channels, 0);

    for (std::size_t link = 0; link < num_links; link++) {
        std::fill(
            get_iter_begin(num_pages_per_edm_channel, link),
            get_iter_end(num_pages_per_edm_channel, link),
            pages_per_link.at(link) / num_edm_channels);
        for (std::size_t i = 0; i < pages_per_link.at(link) % num_edm_channels; i++) {
            num_pages_per_edm_channel.at(link * num_edm_channels + i)++;
        }
    }

    std::vector<uint32_t> num_messages_per_edm_channel;
    std::vector<uint32_t> num_pages_per_full_chunk(num_pages_per_edm_channel.size(), 0);
    std::vector<uint32_t> remaining_num_pages_per_edm_channel;
    num_messages_per_edm_channel.reserve(num_pages_per_edm_channel.size());
    remaining_num_pages_per_edm_channel.reserve(num_pages_per_edm_channel.size());
    for (std::size_t link = 0; link < num_links; link++) {
        std::size_t edm_channel_size_in_bytes = cw_per_link_edm_builders.at(link).get_eth_buffer_size_bytes();
        std::size_t num_pages_per_edm_buffer = edm_channel_size_in_bytes / page_size_in_bytes;

        std::transform(
            get_iter_begin(num_pages_per_edm_channel, link),
            get_iter_end(num_pages_per_edm_channel, link),
            std::back_inserter(num_messages_per_edm_channel),
            [num_pages_per_edm_buffer](uint32_t num_pages) {
                return ((num_pages - 1) / num_pages_per_edm_buffer) + 1;
            });
        std::transform(
            get_iter_begin(num_pages_per_edm_channel, link),
            get_iter_end(num_pages_per_edm_channel, link),
            std::back_inserter(remaining_num_pages_per_edm_channel),
            [num_pages_per_edm_buffer](uint32_t num_pages) { return num_pages % num_pages_per_edm_buffer; });
        std::fill(
            get_iter_begin(num_pages_per_full_chunk, link),
            get_iter_end(num_pages_per_full_chunk, link),
            num_pages_per_edm_buffer);
    }

    log_trace(tt::LogOp, "WorkerTransferInfo");
    log_trace(tt::LogOp, "-- num_pages_per_full_chunk:");
    for (std::size_t l = 0; l < num_links; l++) {
        for (std::size_t w = 0; w < num_edm_channels; w++) {
            log_trace(
                tt::LogOp, "\t\t(link={},worker={}): {}", l, w, num_pages_per_full_chunk.at(l * num_edm_channels + w));
        }
    }
    log_trace(tt::LogOp, "-- num_messages_per_edm_channel:");
    for (std::size_t l = 0; l < num_links; l++) {
        for (std::size_t w = 0; w < num_edm_channels; w++) {
            log_trace(
                tt::LogOp,
                "\t\t(link={},worker={}): {}",
                l,
                w,
                num_messages_per_edm_channel.at(l * num_edm_channels + w));
        }
    }
    log_trace(tt::LogOp, "-- remaining_num_pages_per_edm_channel:");
    for (std::size_t l = 0; l < num_links; l++) {
        for (std::size_t w = 0; w < num_edm_channels; w++) {
            log_trace(
                tt::LogOp,
                "\t\t(link={},worker={}): {}",
                l,
                w,
                remaining_num_pages_per_edm_channel.at(l * num_edm_channels + w));
        }
    }

    return WorkerTransferInfo(
        num_pages_per_full_chunk,
        num_messages_per_edm_channel,
        remaining_num_pages_per_edm_channel,
        num_links,
        num_edm_channels);
}

static std::tuple<CBHandle, CBHandle, CBHandle> create_worker_circular_buffers(
    Tensor const& input_tensor,
    ccl::CCLOpConfig const& op_config,
    CoreRangeSet const& worker_core_range,
    uint32_t cb_num_pages,
    tt_metal::Program& program) {
    tt::DataFormat df = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t page_size_bytes = op_config.get_page_size();
    // Input 0 CB
    uint32_t src0_cb_index = CB::c_in0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(cb_num_pages * page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, page_size_bytes);
    CBHandle cb_src0_workers = CreateCircularBuffer(program, worker_core_range, cb_src0_config);
    // Input 1 CB
    uint32_t src1_cb_index = CB::c_in1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(cb_num_pages * page_size_bytes, {{src1_cb_index, df}})
            .set_page_size(src1_cb_index, page_size_bytes);
    CBHandle cb_src1_workers = CreateCircularBuffer(program, worker_core_range, cb_src1_config);
    // Dataflow Writer Kernel input CB
    uint32_t cb_dst0_index = CB::c_out0;
    tt_metal::CircularBufferConfig cb_dst0_config =
        tt_metal::CircularBufferConfig(cb_num_pages * page_size_bytes, {{cb_dst0_index, df}})
            .set_page_size(cb_dst0_index, page_size_bytes);
    CBHandle cb_dst0_sender_workers = CreateCircularBuffer(program, worker_core_range, cb_dst0_config);

    return {cb_src0_workers, cb_src1_workers, cb_dst0_sender_workers};
}

operation::ProgramWithCallbacks reduce_scatter_with_workers(
    const std::vector<Tensor>& input_tensors,
    const std::vector<Tensor>& output_tensors,
    BinaryOpType reduce_op,
    const uint32_t scatter_split_dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    ccl::Topology topology) {
    log_trace(tt::LogOp, "reduce_scatter_with_workers entry");
    TT_ASSERT(
        input_tensors.at(0).get_legacy_shape()[scatter_split_dim] ==
            output_tensors.at(0).get_legacy_shape()[scatter_split_dim] * ring_size,
        "Input and output tensor shapes must match");
    TT_ASSERT(
        input_tensors.at(0).buffer()->num_pages() % ring_size == 0,
        "Reduce scatter current only supports even divisibility of input tensor(s) across ranks");

    /////////////// Constants/Configuration
    /// Constants/Configuration
    ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode = ccl::EriscDataMoverBufferSharingMode::ROUND_ROBIN;
    auto const& op_config = ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto num_edm_channels = decide_number_of_edm_channels(op_config, 8, false);
    auto const& edm_builder =
        create_erisc_datamover_builder(num_edm_channels, op_config.get_page_size(), buffer_sharing_mode);
    TT_ASSERT(num_edm_channels > 0);

    std::map<string, string> worker_defines;
    std::vector<KernelHandle> worker_receiver_kernels;
    std::vector<KernelHandle> worker_sender_kernels;
    std::vector<ccl::EriscDatamoverBuilder> cw_per_link_edm_builders(num_links, edm_builder);
    std::vector<ccl::EriscDatamoverBuilder> ccw_per_link_edm_builders(num_links, edm_builder);

    bool rm = input_tensors.at(ring_index).get_layout() == Layout::ROW_MAJOR;
    if (rm) {
        worker_defines["RM_INTERLEAVED"] = "1";
    } else {
        worker_defines["TILE_INTERLEAVED"] = "1";
    }

    //////////////////
    tt_metal::Program program{};
    const auto& device = input_tensors.at(ring_index).device();

    auto const& topology_config =
        ccl::RingTopology(device, topology, sender_device_id, receiver_device_id, num_links, ring_size, ring_index);

    auto dim_slice_factors = Shape(std::vector<uint32_t>(input_tensors.at(ring_index).get_legacy_shape().rank(), 1));
    dim_slice_factors[-1] = 8;

    auto tensor_slicer = ccl::InterleavedRingAllGatherTensorSlicer(
        input_tensors.at(ring_index), output_tensors.at(ring_index), scatter_split_dim, ring_index);

    // Not per buffer because the buffer sharing mode may cause some buffers to share EDM transfers
    WorkerTransferInfo const& worker_transfer_info = compute_num_edm_messages_per_channel(
        op_config,
        input_tensors.at(ring_index).buffer()->page_size(),
        input_tensors.at(ring_index).buffer()->num_pages() / ring_size,  // pages_per_slice,
        cw_per_link_edm_builders,
        ccw_per_link_edm_builders,
        num_edm_channels,
        num_links);

    CoreRangeSet const& worker_core_range = select_worker_cores(op_config, num_links);
    auto const& worker_cores = corerange_to_cores(worker_core_range, std::nullopt, true);

    // Semaphores && CBs
    uint32_t cb_num_pages =
        cw_per_link_edm_builders.at(0).get_eth_buffer_size_bytes() / input_tensors.at(ring_index).buffer()->page_size();
    auto worker_receiver_semaphore_address = tt_metal::CreateSemaphore(program, worker_core_range, 0);
    auto worker_sender_semaphore_address = tt_metal::CreateSemaphore(program, worker_core_range, 0);
    auto const& [cb_src0_workers, cb_src1_workers, cb_dst0_sender_workers] = create_worker_circular_buffers(
        input_tensors.at(ring_index), op_config, worker_core_range, cb_num_pages, program);

    // Configure the EDM builders
    EdmInterfaceAddresses edm_interface_addresses;
    for (std::size_t link = 0; link < num_links; link++) {
        add_worker_config_to_edm_builders(
            device,
            op_config,
            worker_cores,
            num_edm_channels,

            cw_per_link_edm_builders,
            ccw_per_link_edm_builders,

            std::vector<uint32_t>(
                worker_transfer_info.num_messages_per_worker.begin() + link * num_edm_channels,
                worker_transfer_info.num_messages_per_worker.begin() + (link + 1) * num_edm_channels),
            std::vector<uint32_t>(
                worker_transfer_info.num_messages_per_worker.begin() + link * num_edm_channels,
                worker_transfer_info.num_messages_per_worker.begin() + (link + 1) * num_edm_channels),

            worker_sender_semaphore_address,
            worker_receiver_semaphore_address,
            link,
            [](uint32_t x) { return true; },  // std::function<bool(uint32_t)> is_buffer_in_clockwise_direction_fn

            edm_interface_addresses);
    }

    // build the worker kernels
    tt_metal::ComputeConfig compute_config;
    for (std::size_t link = 0; link < num_links; link++) {
        uint32_t global_worker_index = link * num_edm_channels;
        log_trace(tt::LogOp, "==============================================");
        log_trace(tt::LogOp, "------------------ Link: {} ------------------", link);
        for (std::size_t worker = 0; worker < num_edm_channels; worker++) {
            std::size_t global_worker_index = worker + link * num_edm_channels;
            log_trace(tt::LogOp, "------ Worker: {} (global ID={})", worker, global_worker_index);
            // This will be configurable by sharded/non-sharded but present the same arg builder
            auto worker_arg_builder = ReduceScatterWorkerArgBuilder(
                op_config,
                topology_config,
                tensor_slicer,
                worker_transfer_info,
                worker,
                cb_num_pages,
                worker_receiver_semaphore_address,
                worker_sender_semaphore_address);

            build_reduce_scatter_worker(
                program,
                device,
                topology_config,
                op_config,
                worker_arg_builder,
                cw_per_link_edm_builders,
                ccw_per_link_edm_builders,
                edm_interface_addresses,
                worker_cores.at(global_worker_index),
                num_edm_channels,
                link,
                worker,
                worker_defines,
                reduce_op);

            uint32_t pages_per_worker = worker_transfer_info.get_num_pages_per_ring_index(link, worker);
            log_trace(tt::LogOp, "- tensor_slicer increment {} pages", pages_per_worker);
            tensor_slicer.increment(pages_per_worker);
        }
    }

    // Generate the EDM kernels
    ccl::generate_edm_kernels_for_ring_or_linear_topology(
        program,
        device,
        topology_config,
        cw_per_link_edm_builders,
        ccw_per_link_edm_builders,
        receiver_device_id,
        sender_device_id);

    uint32_t total_num_workers = worker_cores.size();
    auto override_runtime_arguments_callback =
        [topology_config, worker_receiver_kernels, worker_sender_kernels, worker_cores, total_num_workers, ring_index](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors.at(ring_index);
            const auto& output = output_tensors.at(ring_index);
            for (uint32_t i = 0; i < total_num_workers; ++i) {
                auto& worker_receiver_runtime_args =
                    GetRuntimeArgs(program, worker_receiver_kernels.at(i), worker_cores.at(i));
                worker_receiver_runtime_args.at(0) = input.buffer()->address();
                worker_receiver_runtime_args.at(1) = input.buffer()->address();

                auto& worker_sender_runtime_args =
                    GetRuntimeArgs(program, worker_sender_kernels.at(i), worker_cores.at(i));
                worker_sender_runtime_args.at(0) = output.buffer()->address();
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}
}  // namespace tt_metal

}  // namespace tt
