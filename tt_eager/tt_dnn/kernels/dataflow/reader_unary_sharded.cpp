// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    //uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    uint32_t tile_size = get_arg_val<uint32_t>(1);

    //constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    //DPRINT << "printing values here any problem sir" << ENDL();
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(1);
    DPRINT << "reader Unary_1 asbvksdk  " <<  num_tiles_per_core << "    " << tile_size << ENDL();
    DPRINT << "Checking CB value " << cb_id_in0 << " " << out_cb_id  << ENDL();

    uint32_t l1_read_addr = get_read_ptr(cb_id_in0);
    uint32_t l1_write_addr = get_write_ptr(out_cb_id);
    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    for (uint32_t i = 0; i < num_tiles_per_core; ++ i) {
    //#endif
        cb_reserve_back(out_cb_id, onetile);
        DPRINT << "28 " << ENDL();
        uint64_t dst_noc_addr = get_noc_addr(l1_write_addr);
        DPRINT << "Addrs " << l1_read_addr << " " << l1_write_addr << " "<< dst_noc_addr << ENDL();
        noc_async_write(l1_read_addr, dst_noc_addr, tile_size);
        DPRINT << "32 " << ENDL();
        l1_read_addr += tile_size;
        l1_write_addr += tile_size;
        DPRINT << "34 " << ENDL();
        noc_async_write_barrier();
        DPRINT << "37 " << ENDL();
        cb_push_back(out_cb_id, onetile);
        DPRINT << "39 " << ENDL();
    }

    DPRINT << "Completed " << ENDL();
}
