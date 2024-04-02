// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/kernels/dataflow/moreh_common.hpp"
#include "debug/dprint.h"


void kernel_main() {
    ArgFetcher arg_fetcher;
    const auto output_addr{arg_fetcher.get_next_arg_val<uint32_t>()};
    const auto num_tiles{arg_fetcher.get_next_arg_val<uint32_t>()};
    const auto start_id{arg_fetcher.get_next_arg_val<uint32_t>()};

    constexpr uint32_t onetile{1};
    constexpr uint32_t cb_id_out0{16};

    // set output tensor to out0 cb
    uint32_t l1_read_addr;
    uint32_t output_tile_bytes{static_cast<uint32_t>(get_tile_size(cb_id_out0))};
    const auto output_data_format{get_dataformat(cb_id_out0)};
    const InterleavedAddrGenFast<true> dram_output_addrg = {
        .bank_base_address = output_addr,
        .page_size = output_tile_bytes,
        .data_format = output_data_format};

    DPRINT << "writer cb out0 data_format : " << (uint32_t)output_data_format << ENDL();
    DPRINT << "writer cb out0 tile bytes : " << output_tile_bytes << ENDL();
    DPRINT << "writer output_addr " << output_addr << ENDL();
    DPRINT << "writer num_tiles " << num_tiles << ENDL();
    DPRINT << "writer start_id " << start_id << ENDL();

    for (uint32_t i = 0; i < num_tiles; ++i) {
        auto write_tile_id{start_id + i};
        cb_wait_front(cb_id_out0, onetile);
        l1_read_addr = get_read_ptr(cb_id_out0);
        noc_async_write_tile(write_tile_id, dram_output_addrg, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, onetile);
    }

    DPRINT << "writer done " << ENDL();
}
