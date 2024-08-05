// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    std::uint32_t l1_buffer_addr        = get_arg_val<uint32_t>(0);

    std::uint32_t input_dram_buffer_addr  = get_arg_val<uint32_t>(1);
    std::uint32_t input_dram_noc_x        = get_arg_val<uint32_t>(2);
    std::uint32_t input_dram_noc_y        = get_arg_val<uint32_t>(3);

    std::uint32_t output_dram_buffer_addr  = get_arg_val<uint32_t>(4);
    std::uint32_t output_dram_noc_x        = get_arg_val<uint32_t>(5);
    std::uint32_t output_dram_noc_y        = get_arg_val<uint32_t>(6);

    std::uint32_t dram_buffer_size      = get_arg_val<uint32_t>(7);

    // Read data from DRAM to L1
    std::uint64_t input_dram_buffer_noc_addr = get_noc_addr(input_dram_noc_x, input_dram_noc_y, input_dram_buffer_addr);
    noc_async_read(input_dram_buffer_noc_addr, l1_buffer_addr, dram_buffer_size);
    noc_async_read_barrier();

    // Write data from L1 to DRAM
    std::uint64_t output_dram_buffer_noc_addr = get_noc_addr(output_dram_noc_x, output_dram_noc_y, output_dram_buffer_addr);
    noc_async_write(l1_buffer_addr, output_dram_buffer_noc_addr, dram_buffer_size);
    noc_async_write_barrier();
}
