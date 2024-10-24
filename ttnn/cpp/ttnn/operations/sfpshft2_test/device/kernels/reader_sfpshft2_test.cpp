// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr bool input_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t input_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);

    constexpr uint32_t cb_input = tt::CB::c_in0;

    const uint32_t input_tile_size = get_tile_size(cb_input);
    const DataFormat input_data_format = get_dataformat(cb_input);

    const InterleavedAddrGenFast<input_is_dram> s = {
        .bank_base_address = input_addr, .page_size = input_tile_size, .data_format = input_data_format};

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_input, 1);
        const uint32_t cb_input_addr = get_write_ptr(cb_input);
        noc_async_read_tile(i, s, cb_input_addr);
        noc_async_read_barrier();
        cb_push_back(cb_input, 1);
    }
}
