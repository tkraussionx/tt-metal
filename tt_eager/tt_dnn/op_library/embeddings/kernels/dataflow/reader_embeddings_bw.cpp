// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

// #include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t cb_weights = get_compile_time_arg_val(0);
    constexpr uint32_t cb_index = get_compile_time_arg_val(1);
    constexpr bool weights_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool index_is_dram = get_compile_time_arg_val(3) == 1;

    uint32_t weight_addr = get_arg_val<uint32_t>(0);
    uint32_t index_addr = get_arg_val<uint32_t>(1);

    const InterleavedAddrGenFast<weights_is_dram> s_w = {
        .bank_base_address = weight_addr,
        .page_size = get_tile_size(cb_weights),
        .data_format = get_dataformat(cb_weights),
    };

    const InterleavedAddrGenFast<index_is_dram> s_i = {
        .bank_base_address = index_addr,
        .page_size = get_tile_size(cb_index),
        .data_format = get_dataformat(cb_index),
    };

    auto read_one_tile = [&](uint32_t cb_id, const auto& addr_gen) {
        constexpr uint32_t onetile = 1;

        cb_reserve_back(cb_id, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        noc_async_read_tile(0, addr_gen, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id, onetile);
    };

    read_one_tile(cb_weights, s_w);
    read_one_tile(cb_index, s_i);
}
