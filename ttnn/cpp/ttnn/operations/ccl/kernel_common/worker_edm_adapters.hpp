// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include "dataflow_api.h"

#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_utils.hpp"
#include "tt_metal/hw/inc/ethernet/dataflow_api.h"

namespace ccl {
namespace edm {

template <ttnn::ccl::EriscDataMoverTerminationMode termination_mode>
struct WorkerToEdmReader{
    constexpr WorkerToEdmReader (
        ttnn::ccl::WorkerXY edm_worker_xy,
        std::size_t edm_buffer_base_addr,
        std::size_t num_buffers_per_channel,
        std::size_t edm_l1_sem_addr,
        std::size_t buffer_size_bytes,
        volatile uint32_t * const worker_sem_addr
    ) :
        edm_buffer_addr(get_noc_addr(edm_worker_xy.x, edm_worker_xy.y, edm_buffer_base_addr)),
        edm_semaphore_addr(get_noc_addr(edm_worker_xy.x, edm_worker_xy.y, edm_l1_sem_addr)),
        worker_sem_addr(worker_sem_addr),
        edm_buffer_base_addr(edm_buffer_base_addr),
        num_buffers_per_channel(num_buffers_per_channel),
        last_buffer_index(num_buffers_per_channel - 1),
        edm_l1_sem_addr(edm_l1_sem_addr),
        buffer_size_bytes(buffer_size_bytes),
        buffer_index(0),
        packet_sizing_mode(ttnn::ccl::EriscDataMoverPacketSizingMode::FIXED_SIZE)
    {}
    constexpr WorkerToEdmReader (
        ttnn::ccl::WorkerXY edm_worker_xy,
        std::size_t edm_buffer_base_addr,
        std::size_t num_buffers_per_channel,
        std::size_t edm_l1_sem_addr,
        std::size_t buffer_size_bytes,
        volatile uint32_t * const worker_sem_addr,
        ttnn::ccl::EriscDataMoverPacketSizingMode packet_sizing_mode
    ) :
        edm_buffer_addr(get_noc_addr(edm_worker_xy.x, edm_worker_xy.y, edm_buffer_base_addr)),
        edm_semaphore_addr(get_noc_addr(edm_worker_xy.x, edm_worker_xy.y, edm_l1_sem_addr)),
        worker_sem_addr(worker_sem_addr),
        edm_buffer_base_addr(edm_buffer_base_addr),
        num_buffers_per_channel(num_buffers_per_channel),
        last_buffer_index(num_buffers_per_channel - 1),
        edm_l1_sem_addr(edm_l1_sem_addr),
        buffer_size_bytes(buffer_size_bytes),
        buffer_index(0),
        packet_sizing_mode(packet_sizing_mode)
    {}

    FORCE_INLINE void wait_for_payload_available() const {
        // In the future, if packet_sizing_mode is VARIABLE_SIZE,
        // the semaphore value will contain the number of pages
        // which will tell the adapter how many pages to read.
        // For today, we will take the perf hit and read the max
        // number of pages
        noc_semaphore_wait(this->worker_sem_addr, 1);
        noc_semaphore_set(this->worker_sem_addr, 0);
    }

    /*
     * Returns the number of pages obtained from the EDM
     */
    FORCE_INLINE std::size_t fetch_payload_blocking(uint32_t cb_id, uint32_t num_pages, uint32_t page_size, bool last_message) {
        std::size_t overhead_per_buffer = ((packet_sizing_mode == ttnn::ccl::EriscDataMoverPacketSizingMode::VARIABLE_SIZE) * 16) + sizeof(eth_channel_sync_t);
        uint64_t buffer_address = this->edm_buffer_addr + (this->buffer_index * (this->buffer_size_bytes + overhead_per_buffer));

        if (packet_sizing_mode == ttnn::ccl::EriscDataMoverPacketSizingMode::VARIABLE_SIZE) {
            // Step past the header
            buffer_address += 16;
        }

        DPRINT << "Reader fetching payload from buffer " << (uint32_t)this->buffer_index << " at address " << (uint32_t)buffer_address << "\n";
        fetch_chunk(cb_id, num_pages, page_size, buffer_address);
        if constexpr (termination_mode == ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED) {
            if (!last_message) {
                noc_semaphore_inc(edm_semaphore_addr, ttnn::ccl::EriscDataMoverWorkerSignal::NEXT_MESSAGE_AVAILABLE);
            }
        } else {
            noc_semaphore_inc(edm_semaphore_addr, ttnn::ccl::EriscDataMoverWorkerSignal::NEXT_MESSAGE_AVAILABLE);
        }
        this->buffer_index = (this->buffer_index == this->last_buffer_index) ? 0 : this->buffer_index + 1;

        return num_pages * page_size;
    }

    FORCE_INLINE void close() {
        if constexpr (termination_mode == ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED) {
            noc_semaphore_inc(edm_semaphore_addr, ttnn::ccl::EriscDataMoverWorkerSignal::TERMINATE_IMMEDIATELY);
        }
    }

    uint64_t edm_buffer_addr;
    uint64_t edm_semaphore_addr;
    volatile uint32_t * const worker_sem_addr;
    std::size_t edm_buffer_base_addr;
    std::size_t num_buffers_per_channel;
    std::size_t last_buffer_index;
    std::size_t edm_l1_sem_addr;
    std::size_t buffer_size_bytes;
    std::size_t buffer_index;
    ttnn::ccl::EriscDataMoverPacketSizingMode packet_sizing_mode;
};


template <ttnn::ccl::EriscDataMoverTerminationMode termination_mode>
struct WorkerToEdmSender{
    constexpr WorkerToEdmSender (
        ttnn::ccl::WorkerXY edm_worker_xy,
        std::size_t edm_buffer_base_addr,
        std::size_t num_buffers_per_channel,
        std::size_t edm_l1_sem_addr,
        std::size_t buffer_size_bytes,
        volatile uint32_t * const worker_sem_addr
    ) :
        edm_buffer_addr(get_noc_addr(edm_worker_xy.x, edm_worker_xy.y, edm_buffer_base_addr)),
        edm_semaphore_addr(get_noc_addr(edm_worker_xy.x, edm_worker_xy.y, edm_l1_sem_addr)),
        worker_sem_addr(worker_sem_addr),
        edm_buffer_base_addr(edm_buffer_base_addr),
        num_buffers_per_channel(num_buffers_per_channel),
        last_buffer_index(num_buffers_per_channel - 1),
        edm_l1_sem_addr(edm_l1_sem_addr),
        buffer_size_bytes(buffer_size_bytes),
        buffer_index(0),
        packet_sizing_mode(ttnn::ccl::EriscDataMoverPacketSizingMode::FIXED_SIZE)
    {
        ASSERT(buffer_size_bytes > 0);
    }
    constexpr WorkerToEdmSender (
        ttnn::ccl::WorkerXY edm_worker_xy,
        std::size_t edm_buffer_base_addr,
        std::size_t num_buffers_per_channel,
        std::size_t edm_l1_sem_addr,
        std::size_t buffer_size_bytes,
        volatile uint32_t * const worker_sem_addr,
        ttnn::ccl::EriscDataMoverPacketSizingMode packet_sizing_mode
    ) :
        edm_buffer_addr(get_noc_addr(edm_worker_xy.x, edm_worker_xy.y, edm_buffer_base_addr)),
        edm_semaphore_addr(get_noc_addr(edm_worker_xy.x, edm_worker_xy.y, edm_l1_sem_addr)),
        worker_sem_addr(worker_sem_addr),
        edm_buffer_base_addr(edm_buffer_base_addr),
        num_buffers_per_channel(num_buffers_per_channel),
        last_buffer_index(num_buffers_per_channel - 1),
        edm_l1_sem_addr(edm_l1_sem_addr),
        buffer_size_bytes(buffer_size_bytes),
        buffer_index(0),
        packet_sizing_mode(packet_sizing_mode)
    {
        ASSERT(buffer_size_bytes > 0);
    }

    FORCE_INLINE void wait_for_empty_write_slot() const {
        noc_semaphore_wait(this->worker_sem_addr, 1);
        noc_semaphore_set(this->worker_sem_addr, 0);
    }

    FORCE_INLINE void send_payload_blocking(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        send_payload_impl<ttnn::ccl::EDM_IO_BLOCKING_MODE::BLOCKING>(cb_id, num_pages, page_size);
    }

    // Does not wait for CB. Assumes caller handles CB data availability
    FORCE_INLINE void send_payload_non_blocking(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        send_payload_impl<ttnn::ccl::EDM_IO_BLOCKING_MODE::NON_BLOCKING>(cb_id, num_pages, page_size);
    }

    FORCE_INLINE void close() {
        if constexpr (termination_mode == ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED) {
            this->wait_for_empty_write_slot();
            noc_semaphore_inc(edm_semaphore_addr, ttnn::ccl::EriscDataMoverWorkerSignal::TERMINATE_IMMEDIATELY);
        }
    }

    uint64_t edm_buffer_addr;
    uint64_t edm_semaphore_addr;
    volatile uint32_t * const worker_sem_addr;
    std::size_t edm_buffer_base_addr;
    std::size_t num_buffers_per_channel;
    std::size_t last_buffer_index;
    std::size_t edm_l1_sem_addr;
    std::size_t buffer_size_bytes;
    std::size_t buffer_index;
    ttnn::ccl::EriscDataMoverPacketSizingMode packet_sizing_mode;

   private:
    template<ttnn::ccl::EDM_IO_BLOCKING_MODE blocking_mode>
    FORCE_INLINE void send_payload_impl(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        std::size_t overhead_per_buffer = ((packet_sizing_mode == ttnn::ccl::EriscDataMoverPacketSizingMode::VARIABLE_SIZE) * 16) + sizeof(eth_channel_sync_t);
        uint64_t buffer_address = this->edm_buffer_addr + (this->buffer_index * (this->buffer_size_bytes + overhead_per_buffer));

        if (packet_sizing_mode == ttnn::ccl::EriscDataMoverPacketSizingMode::VARIABLE_SIZE) {
            DPRINT << "Sender setting size " << (uint32_t)(num_pages * page_size + 16) << " at address " << (uint32_t)buffer_address << "\n";
            noc_semaphore_inc(buffer_address, (num_pages * page_size) + 16);
            buffer_address += 16;
        }
        DPRINT << "Sender writing payload to buffer " << (uint32_t)this->buffer_index << " at address " << (uint32_t)buffer_address << " of size " << num_pages * page_size << "\n";

        send_chunk<blocking_mode>(cb_id, num_pages, page_size, buffer_address);
        noc_semaphore_inc(edm_semaphore_addr, 1);
        this->buffer_index = (this->buffer_index == this->last_buffer_index) ? 0 : this->buffer_index + 1;
    }
};


} // namespace edm
} // namespace ccl
