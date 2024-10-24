// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t output_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);

    constexpr uint32_t cb_output = tt::CB::c_out0;

    const uint32_t tile_size = get_tile_size(cb_output);
    const DataFormat data_format = get_dataformat(cb_output);

    const InterleavedAddrGenFast<output_is_dram> s = {
        .bank_base_address = output_addr, .page_size = tile_size, .data_format = data_format};

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_output, 1);
        const uint32_t cb_output_addr = get_read_ptr(cb_output);
        noc_async_write_tile(i, s, cb_output_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_output, 1);
    }
}
