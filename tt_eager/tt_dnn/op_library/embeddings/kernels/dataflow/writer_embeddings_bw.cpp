// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

// #include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr bool out_is_dram = get_compile_time_arg_val(1) == 1;

    uint32_t out_addr = get_arg_val<uint32_t>(0);

    const InterleavedAddrGenFast<out_is_dram> s = {
        .bank_base_address = out_addr,
        .page_size = get_tile_size(cb_out),
        .data_format = get_dataformat(cb_out),
    };

    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_out, onetile);
    uint32_t l1_read_addr = get_read_ptr(cb_out);
    noc_async_write_tile(0, s, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_out, onetile);
}
