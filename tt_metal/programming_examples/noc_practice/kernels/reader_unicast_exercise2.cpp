// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    std::uint32_t input_dram_buffer_addr = get_arg_val<uint32_t>(0);
    std::uint32_t input_dram_noc_x = get_arg_val<uint32_t>(1);
    std::uint32_t input_dram_noc_y = get_arg_val<uint32_t>(2);
    std::uint32_t num_tiles = get_arg_val<uint32_t>(3);
    std::uint32_t tile_size = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb0_id = 0;

    std::uint32_t input_addr = input_dram_buffer_addr;
    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb0_id, 1);
        const auto cb0_l1_addr = get_write_ptr(cb0_id);

        // TODO: get input_dram_buffer_noc_addr and read tile
        std::uint64_t input_dram_buffer_noc_addr = get_noc_addr(input_dram_noc_x, input_dram_noc_y, input_addr);
        noc_async_read(input_dram_buffer_noc_addr, cb0_l1_addr, tile_size);
        noc_async_read_barrier();

        // This section is reserved for kernel debug print practice session.
        #if 0
        if (i == 0) {
            auto l1_write_addr = get_write_ptr(cb0_id);
            auto l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(l1_write_addr);
            for (int idx = 0; idx < 10; ++idx) {
                DPRINT << "reader kernel cb0_id tile index [" << idx << "] = " << BF16(l1_ptr[idx]) << "\n";
            }
        }
        #endif

        cb_push_back(cb0_id, 1);

        // TODO: need to update input_addr
        input_addr += tile_size;
    }
}
