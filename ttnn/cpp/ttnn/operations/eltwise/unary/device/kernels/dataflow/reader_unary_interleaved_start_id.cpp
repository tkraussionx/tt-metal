// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

//#include "debug/dprint.h"

inline void RISC_POST_STATUS_1(uint32_t status, uint32_t addr = 0xFFB2010C) {
    volatile uint32_t *ptr = (volatile uint32_t *)(addr);
    ptr[0] = status;
}

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    // RISC_POST_STATUS((0xa << 16) | (num_tiles) << 8 | (start_id));
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
    RISC_POST_STATUS_1(tile_bytes, PRINT_BUFFER_START + 4);
    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    #ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; -- i) {
    #else
    uint32_t end_id = start_id + num_tiles;
    // RISC_POST_STATUS((end_id << 16) | (num_tiles) << 8 | (start_id));
    int count = 0;
    for (uint32_t i = start_id; i < end_id; ++ i) {
    #endif
        count++;
        // RISC_POST_STATUS_1((0xa << 16) | ((end_id - start_id) << 8) | count, PRINT_BUFFER_START + 4);
        cb_reserve_back(cb_id_in0, onetile);
        // RISC_POST_STATUS_1((0xb << 16) | ((end_id - start_id) << 8) | count, PRINT_BUFFER_START + 4);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        //  RISC_POST_STATUS_1(l1_write_addr, PRINT_BUFFER_START + 4);
        noc_async_read_tile(i, s, l1_write_addr);
        noc_async_read_barrier();
        // RISC_POST_STATUS_1((0xc << 16) | ((end_id - start_id) << 8) |count, PRINT_BUFFER_START + 4);
        cb_push_back(cb_id_in0, onetile);
        // RISC_POST_STATUS_1((0xe << 16) | ((end_id - start_id) << 8) |count, PRINT_BUFFER_START + 4);
    }
    // RISC_POST_STATUS_1(0xdddd0000 | count, PRINT_BUFFER_START + 4);
}
