// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_eager/tt_dnn/op_library/ccl/kernel_common/worker_edm_utils.hpp"


void kernel_main() {
    DPRINT << "RR START\n";
    const uint32_t eth_receiver_l1_base_addr = get_compile_time_arg_val(0);
    const uint32_t eth_receiver_l1_sem_addr = get_compile_time_arg_val(1);
    const uint32_t num_buffers_per_channel = get_compile_time_arg_val(2);
    const uint32_t num_pages_per_read_chunk = get_arg_val<uint32_t>(0);
    const uint32_t total_pages_to_read = get_arg_val<uint32_t>(1);
    const uint32_t page_size = get_arg_val<uint32_t>(2);
    const uint32_t receiver_erisc_datamover_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t receiver_erisc_datamover_noc_y = get_arg_val<uint32_t>(4);
    // Worker local L1 semaphore that erisc datamover signals to
    const uint32_t receiver_read_sem_addr = get_arg_val<uint32_t>(5);

    DPRINT << "RR: num_buffers_per_channel: " << num_buffers_per_channel <<
                "\n\tnum_pages_per_read_chunk: " << num_pages_per_read_chunk <<
                "\n\ttotal_pages_to_read: " << total_pages_to_read <<
                "\n\tpage_size: " << page_size << "\n";

    std::array<uint64_t, num_buffers_per_channel> eth_buffer_addresses;
    const uint32_t buffer_size_bytes = num_pages_per_read_chunk * page_size;
    DPRINT << "RR: eth_receiver_l1_base_addr: " << eth_receiver_l1_base_addr << "\n";
    DPRINT << "RR: buffer_size_bytes: " << buffer_size_bytes << "\n";
    for (uint32_t i = 0; i < num_buffers_per_channel; i++) {
        eth_buffer_addresses[i] = get_noc_addr(
            receiver_erisc_datamover_noc_x,
            receiver_erisc_datamover_noc_y,
            eth_receiver_l1_base_addr + (i * (buffer_size_bytes + 16)));//sizeof(eth_channel_sync_t))));
        DPRINT << "RR: eth buf addr[" << i << "]: " << eth_buffer_addresses[i] << "\n";
        ASSERT((eth_buffer_addresses[i] & 0xF) == 0);
    }


    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    // Eth receiver will set this semaphore when data is available
    volatile tt_l1_ptr uint32_t* receiver_read_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_read_sem_addr);

    // Address of the buffer on the eth receiver, this is different per receiver worker core
    // const uint64_t eth_receiver_l1_base_noc_addr =
    //     get_noc_addr(receiver_erisc_datamover_noc_x, receiver_erisc_datamover_noc_y, eth_receiver_l1_base_addr);
    // Address of the semaphore on the eth receiver, this is the same per receiver worker core
    const uint64_t eth_receiver_l1_semaphore_noc_addr =
        get_noc_addr(receiver_erisc_datamover_noc_x, receiver_erisc_datamover_noc_y, eth_receiver_l1_sem_addr);

    uint32_t  buffer_index = 0;
    // So we can zone scope
    noc_semaphore_wait(receiver_read_semaphore_addr_ptr, 1);
    {
        DeviceZoneScopedN("RX-ACTIVE");
    for (uint32_t i = 0; i < total_pages_to_read; i += num_pages_per_read_chunk) {
        uint32_t num_pages_to_read = std::min(total_pages_to_read - i, num_pages_per_read_chunk);
        noc_semaphore_wait(receiver_read_semaphore_addr_ptr, 1);
        noc_semaphore_set(receiver_read_semaphore_addr_ptr, 0);
        // DPRINT << "RR FETCH_CHUNK from buffer index: " << buffer_index << " @ " << eth_buffer_addresses[buffer_index] << "\n";
        if ((eth_buffer_addresses[buffer_index] & 0xF) != 0) {
            DPRINT << "RR: eth_buffer_addresses" << buffer_index << "] is not aligned to 16 bytes. Value is " << eth_buffer_addresses[buffer_index] << "\n";
        }
        fetch_chunk(cb_id_in0, num_pages_to_read, page_size, eth_buffer_addresses[buffer_index]);
        noc_semaphore_inc(eth_receiver_l1_semaphore_noc_addr, 1);
        buffer_index = (buffer_index + 1) % num_buffers_per_channel;
    }
    }

    DPRINT << "RR DONE\n";
}
