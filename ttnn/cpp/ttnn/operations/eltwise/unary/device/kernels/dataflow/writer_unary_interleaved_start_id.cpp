// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    DPRINT << "wr1" << ENDL();
    #ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, num_tiles);
    DPRINT << "wr2" << ENDL();
    #else

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    DPRINT << "wr3" << ENDL();
    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    #ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; -- i) {
    DPRINT << "wr4" << ENDL();
    #else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++ i) {
    DPRINT << "wr5" << ENDL();
    #endif
        cb_wait_front(cb_id_out, onetile);
	DPRINT << "wr6" << ENDL();
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
	DPRINT << "wr7" << ENDL();
        noc_async_write_tile(i, s, l1_read_addr);
	DPRINT << "wr8" << ENDL();
        noc_async_write_barrier();
	DPRINT << "wr9" << ENDL();
        cb_pop_front(cb_id_out, onetile);
	DPRINT << "wr10" << ENDL();
    }
    DPRINT << "wr11" << ENDL();
    #endif
    DPRINT << "wr12" << ENDL();
}
