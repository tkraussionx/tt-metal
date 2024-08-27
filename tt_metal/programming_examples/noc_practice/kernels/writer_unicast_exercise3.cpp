// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    std::uint32_t output_dram_buffer_addr = get_arg_val<uint32_t>(0);
    std::uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb1_id = 16; // cb_out0 idx
    const uint32_t cb1_page_size = get_tile_size(cb1_id);
    const auto cb1_data_format = get_dataformat(cb1_id);
    /* TODO: fill this section
    const InterleavedAddrGenFast<true> dram_buffer1_addrg = {
        .bank_base_address = , .page_size = , .data_format = };
        */
    const InterleavedAddrGenFast<true> dram_buffer1_addrg = {
        .bank_base_address = output_dram_buffer_addr, .page_size = cb1_page_size, .data_format = cb1_data_format};

    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb1_id, 1);
        const auto cb1_l1_addr = get_read_ptr(cb1_id);

        // TODO: write tile using dram_buffer1_addrg
        std::uint64_t output_dram_noc_addr =  dram_buffer1_addrg.get_noc_addr(i);
        noc_async_write(cb1_l1_addr, output_dram_noc_addr, cb1_page_size);
        noc_async_write_barrier();

        cb_pop_front(cb1_id, 1);
    }
}
