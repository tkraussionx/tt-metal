// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t num_writes_per_cmd_buff = get_compile_time_arg_val(0);
    constexpr uint32_t addr_inc_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t transaction_size_bytes = get_compile_time_arg_val(2);

    constexpr uint32_t addr_inc = addr_inc_bytes/sizeof(uint32_t);

    constexpr uint32_t words_per_transaction = transaction_size_bytes / sizeof(uint32_t);

    uint32_t rcvr_noc_x = get_arg_val<uint32_t>(0);
    uint32_t rcvr_noc_y = get_arg_val<uint32_t>(1);
    uint32_t first_dst_addr  = get_arg_val<uint32_t>(2);
    uint32_t second_dst_addr  = get_arg_val<uint32_t>(3);
    uint32_t first_send_local_address = get_arg_val<uint32_t>(4);
    uint32_t second_send_local_address = get_arg_val<uint32_t>(5);
    uint32_t second_write_cmd_buf = get_arg_val<uint32_t>(6);

    uint64_t first_dst_noc_addr = get_noc_addr(rcvr_noc_x, rcvr_noc_y, first_dst_addr);
    uint64_t second_dst_noc_addr = get_noc_addr(rcvr_noc_x, rcvr_noc_y, second_dst_addr);

    volatile uint32_t* sem0 = reinterpret_cast<volatile uint32_t*>(get_semaphore(0));
    noc_semaphore_wait(sem0, 1);

    for (uint32_t i = 0; i < num_writes_per_cmd_buff; i++) {

        ncrisc_noc_fast_write_any_len(
            noc_index,
            NCRISC_WR_CMD_BUF,
            first_send_local_address,
            first_dst_noc_addr,
            transaction_size_bytes,
            NOC_UNICAST_WRITE_VC,
            false,  // mcast
            false,  // linked
            1,      // num_dests 1 for non-mcast
            true    // multicast_path_reserve
        );

        // noc_async_write_barrier();

        ncrisc_noc_fast_write_any_len(
            noc_index,
            second_write_cmd_buf, // alternate between NCRISC_WR_CMD_BUF and NCRISC_WR_REG_CMD_BUF
            second_send_local_address,
            second_dst_noc_addr,
            transaction_size_bytes,
            NOC_UNICAST_WRITE_VC,
            false,  // mcast
            false,  // linked
            1,      // num_dests 1 for non-mcast
            true    // multicast_path_reserve
        );

        // for (uint32_t data_idx = 0; data_idx < words_per_transaction; data_idx++) {
        //     first_local_address[data_idx] += 2;
        //     second_local_address[data_idx] += 2;
        // }

        first_send_local_address += addr_inc_bytes;
        second_send_local_address += addr_inc_bytes;
        first_dst_addr += addr_inc_bytes;
        second_dst_addr += addr_inc_bytes;
        first_dst_noc_addr = get_noc_addr(rcvr_noc_x, rcvr_noc_y, first_dst_addr);
        second_dst_noc_addr = get_noc_addr(rcvr_noc_x, rcvr_noc_y, second_dst_addr);
    }

    noc_async_write_barrier();
}
