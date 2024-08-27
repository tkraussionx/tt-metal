// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t l1_buffer_src_addr = get_arg_val<uint32_t>(0);
    uint32_t input_dram_buffer_addr = get_arg_val<uint32_t>(1);
    uint32_t input_dram_noc_x = get_arg_val<uint32_t>(2);
    uint32_t input_dram_noc_y = get_arg_val<uint32_t>(3);
    uint32_t dram_buffer_size      = get_arg_val<uint32_t>(4);
    uint32_t l1_buffer_dest_addr = get_arg_val<uint32_t>(5);
    uint32_t dest_noc_x_start = get_arg_val<uint32_t>(6);
    uint32_t dest_noc_y_start = get_arg_val<uint32_t>(7);
    uint32_t dest_noc_x_end = get_arg_val<uint32_t>(8);
    uint32_t dest_noc_y_end = get_arg_val<uint32_t>(9);
    uint32_t num_dests = get_arg_val<uint32_t>(10);

    // step1. read dram input buffer into local L1 buffer
    uint64_t input_dram_buffer_noc_addr = get_noc_addr(input_dram_noc_x, input_dram_noc_y, input_dram_buffer_addr);
    noc_async_read(input_dram_buffer_noc_addr, l1_buffer_src_addr, dram_buffer_size);
    noc_async_read_barrier();

    // step2. multicast local L1 buffer to all destination cores
    /* TODO: fill this seciont */
    uint64_t dst_noc_addr_multicast = get_noc_multicast_addr(dest_noc_x_start, dest_noc_y_start, dest_noc_x_end, dest_noc_y_end, l1_buffer_dest_addr);
    noc_async_write_multicast(l1_buffer_src_addr, dst_noc_addr_multicast, dram_buffer_size, num_dests);
    noc_async_write_barrier();
}
