// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    std::uint32_t l1_buffer_addr        = get_arg_val<uint32_t>(0);
    std::uint32_t input_dram_buffer_addr  = get_arg_val<uint32_t>(1);
    // 1. Fill this section
    std::uint32_t input_dram_noc_xy_x = get_arg_val<uint32_t>(2);
    std::uint32_t input_dram_noc_xy_y = get_arg_val<uint32_t>(3);
    std::uint64_t input_dram_noc = get_noc_addr(input_dram_noc_xy_x, input_dram_noc_xy_y, input_dram_buffer_addr);

    std::uint32_t output_dram_buffer_addr  = get_arg_val<uint32_t>(4);
    // 1. Fill this section
    std::uint32_t output_dram_noc_xy_x = get_arg_val<uint32_t>(5);
    std::uint32_t output_dram_noc_xy_y = get_arg_val<uint32_t>(6);
    std::uint64_t output_dram_noc = get_noc_addr(output_dram_noc_xy_x, output_dram_noc_xy_y, output_dram_buffer_addr);

    std::uint32_t dram_buffer_size      = get_arg_val<uint32_t>(7);

    // Read data from DRAM to SRAM
    // 2. Fill this section
    noc_async_read(input_dram_noc, l1_buffer_addr, dram_buffer_size);
    noc_async_read_barrier();

    // Write data from SRAM to DRAM
    // 3. Fill this section
    noc_async_write(l1_buffer_addr, output_dram_noc, dram_buffer_size);
    noc_async_write_barrier();
}
