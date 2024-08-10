// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    std::uint32_t l1_buffer_addr        = get_arg_val<uint32_t>(0);
    std::uint32_t input_dram_buffer_addr  = get_arg_val<uint32_t>(1);
    // 1. Fill this section

    std::uint32_t output_dram_buffer_addr  = get_arg_val<uint32_t>(4);
    // 1. Fill this section

    std::uint32_t dram_buffer_size      = get_arg_val<uint32_t>(7);

    // Read data from DRAM to SRAM
    // 2. Fill this section

    // Write data from SRAM to DRAM
    // 3. Fill this section
}
