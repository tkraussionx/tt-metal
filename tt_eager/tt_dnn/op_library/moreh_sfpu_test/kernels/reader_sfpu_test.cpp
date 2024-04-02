// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/kernels/dataflow/moreh_common.hpp"
#include "debug/dprint.h"

void kernel_main() {
    ArgFetcher arg_fetcher;
    const auto input_addr{arg_fetcher.get_next_arg_val<uint32_t>()};
    const auto num_tiles{arg_fetcher.get_next_arg_val<uint32_t>()};
    const auto start_id{arg_fetcher.get_next_arg_val<uint32_t>()};

    constexpr uint32_t onetile{1};
    constexpr uint32_t cb_id_in0{0};

    uint32_t l1_write_addr;
    uint32_t in0_tile_bytes{static_cast<uint32_t>(get_tile_size(cb_id_in0))};
    const auto in0_data_format{get_dataformat(cb_id_in0)};
    const InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_addr,
        .page_size = in0_tile_bytes,
        .data_format = in0_data_format};

    DPRINT << "reader cb in0 data_format : " << (uint32_t)in0_data_format << ENDL();
    DPRINT << "reader cb in0 tile bytes : " << in0_tile_bytes << ENDL();
    DPRINT << "reader input_addr " << input_addr << ENDL();
    DPRINT << "reader num_tiles " << num_tiles << ENDL();
    DPRINT << "reader start_id " << start_id << ENDL();

    for (uint32_t i = 0; i < num_tiles; ++i) {
        auto read_tile_id{start_id};
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile(read_tile_id, dram_input_addrg, l1_write_addr);

        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
    }

    DPRINT << "readeer done " << ENDL();
}
