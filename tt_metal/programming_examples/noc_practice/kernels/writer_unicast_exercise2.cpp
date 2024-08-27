// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    std::uint32_t output_dram_buffer_addr = get_arg_val<uint32_t>(0);
    std::uint32_t output_dram_noc_x = get_arg_val<uint32_t>(1);
    std::uint32_t output_dram_noc_y = get_arg_val<uint32_t>(2);
    std::uint32_t num_tiles = get_arg_val<uint32_t>(3);
    std::uint32_t tile_size = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb1_id = 16; // cb_out0 idx

    std::uint32_t output_addr = output_dram_buffer_addr;
    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb1_id, 1);
        const auto cb1_l1_addr = get_read_ptr(cb1_id);


        // This section is reserved for kernel debug print practice session.
        #if 0
        if (i == 0) {
            auto l1_read_addr = get_read_ptr(cb1_id);
            auto l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(l1_read_addr);
            for (int idx = 0; idx < 10; ++idx) {
                DPRINT << "writer kernel cb1_id tile index [" << idx << "] = " << BF16(l1_ptr[idx]) << "\n";
            }
        }
        #endif

        // TODO: get output_dram_buffer_noc_addr and write tile
        std::uint64_t output_dram_buffer_noc_addr = get_noc_addr(output_dram_noc_x, output_dram_noc_y, output_addr);
        noc_async_write(cb1_l1_addr, output_dram_buffer_noc_addr, tile_size);
        noc_async_write_barrier();

        cb_pop_front(cb1_id, 1);

        // TODO: need to update output_addr
        output_addr += tile_size;
    }
}
