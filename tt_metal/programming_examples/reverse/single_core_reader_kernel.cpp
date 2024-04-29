// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <dataflow_api.h>
void kernel_main() {
    std::uint32_t l1_buffer_addr        = get_arg_val<uint32_t>(0);

    std::uint32_t dram_buffer_src_addr  = get_arg_val<uint32_t>(1);
    std::uint32_t dram_src_noc_x        = get_arg_val<uint32_t>(2);
    std::uint32_t dram_src_noc_y        = get_arg_val<uint32_t>(3);

    std::uint32_t dram_buffer_size      = get_arg_val<uint32_t>(4);
    auto page_size                      = get_arg_val<uint32_t>(5);
    volatile tt_l1_ptr uint8_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_buffer_addr);



    int chunk_size = page_size;
    int half_cs = chunk_size/2;
    auto remaining_size = dram_buffer_size;

    const InterleavedAddrGen<true> s0 = {
        .bank_base_address = dram_buffer_src_addr,
        .page_size = page_size
    };



    uint32_t src_page_index = 0;
    constexpr uint32_t cb_in_id = 0;

    while(remaining_size>0)
    {
        std::uint64_t dram_buffer_src_noc_addr = get_noc_addr(src_page_index,s0);

        cb_reserve_back(cb_in_id,1);
        noc_async_read(dram_buffer_src_noc_addr, get_write_ptr(cb_in_id), chunk_size);
        noc_async_read_barrier();
        cb_push_back(cb_in_id,1);
        remaining_size-=chunk_size;
        src_page_index++;
    }
}
