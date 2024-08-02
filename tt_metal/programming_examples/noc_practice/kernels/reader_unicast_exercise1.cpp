// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    std::uint32_t l1_buffer_addr        = get_arg_val<uint32_t>(0);

    std::uint32_t dram_buffer_input_addr  = get_arg_val<uint32_t>(1);
    std::uint32_t dram_input_noc_x        = get_arg_val<uint32_t>(2);
    std::uint32_t dram_input_noc_y        = get_arg_val<uint32_t>(3);

    std::uint32_t dram_buffer_output_addr  = get_arg_val<uint32_t>(4);
    std::uint32_t dram_output_noc_x        = get_arg_val<uint32_t>(5);
    std::uint32_t dram_output_noc_y        = get_arg_val<uint32_t>(6);

    std::uint32_t dram_buffer_size      = get_arg_val<uint32_t>(7);

    // DRAM NOC input address
    std::uint64_t dram_buffer_input_noc_addr = get_noc_addr(dram_input_noc_x, dram_input_noc_y, dram_buffer_input_addr);
    noc_async_read(dram_buffer_input_noc_addr, l1_buffer_addr, dram_buffer_size);
    noc_async_read_barrier();

    // DRAM NOC output address
    std::uint64_t dram_buffer_output_noc_addr = get_noc_addr(dram_output_noc_x, dram_output_noc_y, dram_buffer_output_addr);
    noc_async_write(l1_buffer_addr, dram_buffer_output_noc_addr, dram_buffer_size);
    noc_async_write_barrier();
}
