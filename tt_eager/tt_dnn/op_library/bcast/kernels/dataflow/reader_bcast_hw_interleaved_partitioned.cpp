// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t src0_addr          = get_arg_val<uint32_t>(0);
    uint32_t src1_addr          = get_arg_val<uint32_t>(1);
    uint32_t num_tiles          = get_arg_val<uint32_t>(2);
    uint32_t HtWt               = get_arg_val<uint32_t>(3);
    uint32_t base_start_id_HtWt = get_arg_val<uint32_t>(4);
    uint32_t curr_id_from_base  = get_arg_val<uint32_t>(5);
    uint32_t bcast_id           = get_arg_val<uint32_t>(6);

    #ifndef  IN0_SHARDED
    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    #endif

    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    // single-tile ublocks
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat in1_data_format = get_dataformat(cb_id_in1);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    #ifndef IN0_SHARDED
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr,
        .page_size = in0_tile_bytes,
        .data_format = in0_data_format
    };
    #else
        // DPRINT << "PUSHING " << num_tiles << " Tiles to CB0 as part of sharding" << ENDL();
        cb_reserve_back(cb_id_in0, num_tiles);
        cb_push_back(cb_id_in0, num_tiles);
        if (num_tiles == 1) {
            // SliceRange sr = SliceRange{.h0 = 0, .h1 = (uint16_t)(1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
            // DPRINT << "Reader CB0 Tile: " << 0 << TileSlice(cb_id_in0, 0, sr, true, true) << ENDL();
        }
    #endif

    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr,
        .page_size = in1_tile_bytes,
        .data_format = in1_data_format
    };

    #ifdef BCAST_SCALAR
    cb_reserve_back(cb_id_in1, onetile);
    l1_write_addr_in1 = get_write_ptr(cb_id_in1);
    noc_async_read_tile(bcast_id, s1, l1_write_addr_in1);
    noc_async_read_barrier();
    cb_push_back(cb_id_in1, onetile);
    if (num_tiles == 1) {
        SliceRange sr = SliceRange{.h0 = 0, .h1 = (uint16_t)(1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT << "Reader CB1 Bcast Tile: " << TileSlice(cb_id_in1, 0, sr, true, true) << ENDL();
    }
    #endif

    for (uint32_t i = 0; i < num_tiles; i++) {
        uint32_t curr_id = base_start_id_HtWt + curr_id_from_base;

        DPRINT << "In loop" << ENDL();

        #ifndef IN0_SHARDED
        DPRINT << "Should not be here!" << ENDL();
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(curr_id, s0, l1_write_addr_in0);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
        #endif

        curr_id_from_base++;

        #ifndef BCAST_SCALAR
        DPRINT << "SHOULD NOT BE HERE!" << ENDL();
        cb_reserve_back(cb_id_in1, onetile);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read_tile(bcast_id, s1, l1_write_addr_in1);
        noc_async_read_barrier();
        cb_push_back(cb_id_in1, onetile);

        if (curr_id_from_base == HtWt) {
            bcast_id++;
        #else
        if (curr_id_from_base == HtWt) {
        #endif
            base_start_id_HtWt += HtWt;
            curr_id_from_base = 0;
        }
    }
}
