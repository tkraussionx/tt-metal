// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    if (is_ncrisc){
        DPRINT << "ncrisc reader clone " << (uint)noc_index_to_dram_bank_map[0] << ENDL();

        uint32_t offset = (noc_index << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(NIU_MST_REQS_OUTSTANDING_ID(0));
        volatile uint32_t* ptr = (volatile uint32_t*)offset;
        DPRINT << *ptr << ENDL();

        offset = (noc_index << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(NIU_MST_REQS_OUTSTANDING_ID(1));
        ptr = (volatile uint32_t*)offset;
        DPRINT << *ptr << ENDL();

        offset = (noc_index << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(NIU_MST_REQS_OUTSTANDING_ID(2));
        ptr = (volatile uint32_t*)offset;
        DPRINT << *ptr << ENDL();
    } else {
        DPRINT << "brisc reader clone " << (uint)noc_index_to_dram_bank_map[0] << ENDL();
    }
    volatile uint32_t cnt;


    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    #ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; -- i) {
    #else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++ i) {
    #endif
        cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile(i, s, l1_write_addr);
        // noc_async_read_tile(i, s, l1_write_addr);
        noc_async_read_tile_barrier();
        // noc_async_read_barrier();

        // DPRINT  << TSLICE(cb_id_in0, 0, SliceRange::h0_w0_32()) << ENDL();

        cb_push_back(cb_id_in0, onetile);
    }
}
