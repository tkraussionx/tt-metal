// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    std::uint32_t dram_input_buffer_addr = get_arg_val<uint32_t>(0);
    std::uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb0_id = 0;

    const uint32_t cb0_page_size = get_tile_size(cb0_id);
    const auto cb0_data_format = get_dataformat(cb0_id);
    const InterleavedAddrGenFast<true> input_addrg = {
        .bank_base_address = dram_input_buffer_addr, .page_size = cb0_page_size, .data_format = cb0_data_format};

    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb0_id, 1);
        const auto cb0_l1_addr = get_write_ptr(cb0_id);

        noc_async_read_tile(i, input_addrg, cb0_l1_addr, 0 /*offset*/);
        noc_async_read_barrier();

        cb_push_back(cb0_id, 1);
    }
}
