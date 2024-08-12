// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

inline void RISC_POST_STATUS_2(uint32_t status, uint32_t addr = 0xFFB2010C) {
    volatile uint32_t *ptr = (volatile uint32_t *)(addr);
    ptr[0] = status;
}


void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    #ifdef OUT_SHARDED
    RISC_POST_STATUS_2((0xa << 16) | (num_tiles) << 8 | (start_id), PRINT_BUFFER_START + 8);
    cb_wait_front(cb_id_out, num_tiles);
    RISC_POST_STATUS_2((0xb << 16) | (num_tiles) << 8 | (start_id), PRINT_BUFFER_START + 8);
    #else

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };
    int ct = 0;
    #ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; -- i) {
    #else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++ i) {
    #endif
        ct++;
        RISC_POST_STATUS_2((0xc << 16) | (end_id - start_id) << 8 | (onetile), PRINT_BUFFER_START + 8);
        cb_wait_front(cb_id_out, onetile);
        RISC_POST_STATUS_2((0xd << 16) | (end_id - start_id) << 8 | (onetile), PRINT_BUFFER_START + 8);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        RISC_POST_STATUS_2((0xe << 16) | (end_id - start_id) << 8 | (l1_read_addr), PRINT_BUFFER_START + 8);
        noc_async_write_tile(i, s, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, onetile);
    }
    #endif
    RISC_POST_STATUS_2(0xdddd0000 | ct , PRINT_BUFFER_START + 8);
}
