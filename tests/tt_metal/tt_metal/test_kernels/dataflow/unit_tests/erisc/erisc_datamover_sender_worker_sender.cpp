// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "noc_nonblocking_api.h"
#include "noc_parameters.h"
#include "tt_eager/tt_dnn/op_library/ccl/kernel_common/worker_edm_utils.hpp"



// Worker core - Data Movement Writer -> Sends to Erisc Data Mover (sender side).
// -> takes input from local cb and pushes to erisc L1
void kernel_main() {
    const uint32_t eth_l1_base_addr = get_arg_val<uint32_t>(0);
    // erisc l1 semaphore address
    const uint32_t eth_sender_l1_sem_addr = get_arg_val<uint32_t>(1);
    const uint32_t writer_send_sem_addr = get_arg_val<uint32_t>(2);
    const uint32_t eth_sender_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t eth_sender_noc_y = get_arg_val<uint32_t>(4);

    constexpr uint32_t num_pages_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t total_pages_to_send = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_buffers_per_channel = get_compile_time_arg_val(3);
    DPRINT << "SW num_buffers_per_channel: " << num_buffers_per_channel << "\n";

    std::array<uint64_t, num_buffers_per_channel> eth_buffer_addresses;
    for (uint32_t i = 0; i < num_buffers_per_channel; i++) {
        eth_buffer_addresses[i] = get_noc_addr(
            eth_sender_noc_x,
            eth_sender_noc_y,
            eth_l1_base_addr + (i * ((num_pages_per_send * page_size) + 16)));//sizeof(eth_channel_sync_t))));
    }


    DPRINT << " sws: args:" <<
        "\n\teth_sender_l1_base_addr="<<eth_l1_base_addr<<
        "\n\teth_sender_l1_sem_addr="<<eth_sender_l1_sem_addr<<
        "\n\twriter_send_sem_addr="<<writer_send_sem_addr<<
        "\n\teth_sender_noc_x="<<eth_sender_noc_x<<
        "\n\teth_sender_noc_y="<<eth_sender_noc_y<<
        "\n\tnum_pages_per_send="<<num_pages_per_send<<
        "\n\ttotal_pages_to_send="<<total_pages_to_send<<
        "\n\tpage_size="<<page_size<<"\n";

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    // Used to wait until eth sender has space available
    volatile tt_l1_ptr uint32_t* writer_send_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_send_sem_addr);

    // This is different per writer core
    // const uint64_t eth_l1_sender_base_noc_addr =
    //     get_noc_addr(eth_sender_noc_x, eth_sender_noc_y, eth_l1_base_addr);
    // Used to signal eth sender that data is available. This is different per writer core
    const uint64_t eth_l1_sender_semaphore_addr =
        get_noc_addr(eth_sender_noc_x, eth_sender_noc_y, eth_sender_l1_sem_addr);

    // num_transfers = num_devices - 1
    uint32_t num_pages_sent = 0;
    DPRINT << " sws: noc_index " << (uint32_t)noc_index << "\n";
    DPRINT << " sws: my_x[0],my_y[0] " << (uint32_t)my_x[0] << "," << (uint32_t)my_y[0] << "\n";
    DPRINT << " sws: my_x[1],my_y[1] " << (uint32_t)my_x[1] << "," << (uint32_t)my_y[1] << "\n";

    uint32_t buffer_index = 0;
    for (uint32_t p = 0; p < total_pages_to_send; p += num_pages_per_send) {
        uint32_t num_pages_to_send = std::min(num_pages_per_send, total_pages_to_send - p);
        noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
        noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
        // DPRINT << "SW SEND_CHUNK from buffer index: " << buffer_index << " @ " << eth_buffer_addresses[buffer_index] << " x " << (num_pages_to_send * page_size) << "\n";
        send_chunk(cb_id_in0, num_pages_to_send, page_size, eth_buffer_addresses[buffer_index]);
        noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);

        buffer_index = (buffer_index + 1) % num_buffers_per_channel;
    }
    DPRINT << "SW DONE\n";
}
