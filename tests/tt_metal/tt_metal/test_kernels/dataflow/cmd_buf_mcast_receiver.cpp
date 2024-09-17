// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/ring_buffer.h"

void kernel_main() {
    constexpr uint32_t num_writes_per_cmd_buff = get_compile_time_arg_val(0);
    constexpr uint32_t addr_inc_bytes = get_compile_time_arg_val(1);

    constexpr uint32_t addr_inc = addr_inc_bytes/sizeof(uint32_t);
    constexpr uint32_t timeout = 1000000;

    uint32_t src_noc_x = get_arg_val<uint32_t>(0);
    uint32_t src_noc_y = get_arg_val<uint32_t>(1);
    uint32_t first_expected_val  = get_arg_val<uint32_t>(2);
    uint32_t second_expected_val  = get_arg_val<uint32_t>(3);
    uint32_t first_initial_receive_local_address = get_arg_val<uint32_t>(4);
    uint32_t second_initial_receive_local_address = get_arg_val<uint32_t>(5);

    volatile tt_l1_ptr uint32_t* first_local_address = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(first_initial_receive_local_address);
    volatile tt_l1_ptr uint32_t* second_local_address = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(second_initial_receive_local_address);

    uint32_t src_noc_xy = uint32_t(NOC_XY_ENCODING(src_noc_x, src_noc_y));

    bool correct_order = true;

    WAYPOINT("NAVY");

    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(noc_index, 0, NOC_NODE_ID);
    uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;

    DPRINT << "DM0 receiver (" << my_x << ", " << my_y << ") sem inc on (" << src_noc_x << ", " << src_noc_y << ")" << ENDL();

    noc_semaphore_inc(get_noc_addr_helper(src_noc_xy, get_semaphore(0)), 1);

    for (uint32_t i = 0; i < num_writes_per_cmd_buff; i++) {

        // DPRINT << "first_local_address " << (uint32_t)first_local_address
        //        << " second_local_address " << (uint32_t)second_local_address
        //        << " expecting " << first_expected_val << " and " << second_expected_val << ENDL();

        WAYPOINT("SVWS");
        WATCHER_RING_BUFFER_PUSH(second_expected_val);
        uint32_t timeout_counter = 0;
        while (*second_local_address != second_expected_val and timeout_counter < timeout) {
            // DPRINT << "second local address " << *second_local_address << ENDL();
            // WATCHER_RING_BUFFER_PUSH(second_expected_val);
            // DPRINT << "timeout_counter " << timeout_counter << ENDL();
            invalidate_l1_cache();
            timeout_counter++;
        }
        WAYPOINT("SVWD");

        // DPRINT << "HI" << ENDL();
        invalidate_l1_cache();

        if (*first_local_address != first_expected_val) {
            correct_order = false;
            break;
        }

        first_expected_val += 2;
        second_expected_val += 2;
        first_local_address += addr_inc;
        second_local_address += addr_inc;
    }


    // if (not correct_order) {
    //     volatile tt_l1_ptr uint32_t* result_address_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(L1_UNRESERVED_BASE);
    //     *result_address_ptr = 0xBADC00DE;
    // }

    DPRINT << "rcv-out " << my_x << ", " << my_y << ENDL();
}
