// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"



void kernel_main() {

    // 1. calculate which group of 4s family this core belongs to and at what offset
    // 2. perform remote data movement -- get a quarter chunk from each of the 4s family

    constexpr uint32_t chunk_size_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t read_chunk_offset_bytes = get_compile_time_arg_val(1);

    tt_l1_ptr uint32_t* noc_x_0 = (tt_l1_ptr uint32_t*) get_arg_addr(0);
    tt_l1_ptr uint32_t* noc_x_1 = (tt_l1_ptr uint32_t*) get_arg_addr(1);
    tt_l1_ptr uint32_t* noc_x_2 = (tt_l1_ptr uint32_t*) get_arg_addr(2);
    tt_l1_ptr uint32_t* noc_x_3 = (tt_l1_ptr uint32_t*) get_arg_addr(3);
    tt_l1_ptr uint32_t* noc_y_0 = (tt_l1_ptr uint32_t*) get_arg_addr(4);
    tt_l1_ptr uint32_t* noc_y_1 = (tt_l1_ptr uint32_t*) get_arg_addr(5);
    tt_l1_ptr uint32_t* noc_y_2 = (tt_l1_ptr uint32_t*) get_arg_addr(6);
    tt_l1_ptr uint32_t* noc_y_3 = (tt_l1_ptr uint32_t*) get_arg_addr(7);

    constexpr auto cb_in = tt::CB::c_in0;
    constexpr auto cb_out = tt::CB::c_intermed0;

    uint32_t l1_read_addr = get_read_ptr(cb_in0) + read_chunk_offset_bytes;
    uint64_t read_noc_addr0 = get_noc_addr(noc_x_0, noc_y_0, l1_read_addr);
    uint64_t read_noc_addr1 = get_noc_addr(noc_x_1, noc_y_1, l1_read_addr);
    uint64_t read_noc_addr2 = get_noc_addr(noc_x_2, noc_y_2, l1_read_addr);
    uint64_t read_noc_addr3 = get_noc_addr(noc_x_3, noc_y_3, l1_read_addr);

    cb_reserve_back(cb_out, 4 * chunk_size_bytes);

    uint32_t l1_write_addr = get_write_ptr(cb_out);
    noc_async_read(read_noc_addr0, l1_write_addr, chunk_size_bytes);
    l1_write_addr += chunk_size_bytes;
    noc_async_read(read_noc_addr1, l1_write_addr, chunk_size_bytes);
    l1_write_addr += chunk_size_bytes;
    noc_async_read(read_noc_addr2, l1_write_addr, chunk_size_bytes);
    l1_write_addr += chunk_size_bytes;
    noc_async_read(read_noc_addr3, l1_write_addr, chunk_size_bytes);

    noc_async_read_barrier();
    cb_push_back(cb_out, 4 * chunk_size_bytes);

    // now do a CH transpose locally
    cb_wait_front(cb_out, 4 * chunk_size_bytes);
    cb_reserve_back(cb_in, 4 * chunk_size_bytes);

    uint32_t stick_offset = 0;
    l1_read_addr = get_read_ptr(cb_out);
    l1_write_addr = get_write_ptr(cb_in);
    uint32_t stick_size_bytes = 512;  // 1 stick of 256 length = 512 bytes
    for (uint32_t i = 0; i < 56; ++i) {     // each chunk is 224 / 4 = 56 sticks of 256 length = 512 bytes
        // copy 4 sticks, 1 from each chunk to the output as contiguous sticks
        auto curr_l1_read_addr = l1_read_addr + stick_offset;
        auto curr_l1_write_addr = l1_write_addr + i * 4 * stick_size_bytes;
        noc_async_write(curr_l1_read_addr, get_noc_addr(curr_l1_write_addr), stick_size_bytes);
        curr_l1_read_addr += chunk_size_bytes;
        curr_l1_write_addr += stick_size_bytes;
        noc_async_write(curr_l1_read_addr, get_noc_addr(curr_l1_write_addr), stick_size_bytes);
        curr_l1_read_addr += chunk_size_bytes;
        curr_l1_write_addr += stick_size_bytes;
        noc_async_write(curr_l1_read_addr, get_noc_addr(curr_l1_write_addr), stick_size_bytes);
        curr_l1_read_addr += chunk_size_bytes;
        curr_l1_write_addr += stick_size_bytes;
        noc_async_write(curr_l1_read_addr, get_noc_addr(curr_l1_write_addr), stick_size_bytes);
        stick_offset += stick_size_bytes;
    }

    cb_push_back(cb_in, 4 * chunk_size_bytes);
    cb_pop_front(cb_out, 4 * chunk_size_bytes);
}
