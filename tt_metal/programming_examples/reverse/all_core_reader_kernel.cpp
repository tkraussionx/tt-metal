// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <dataflow_api.h>
void kernel_main() {

    std::uint32_t dram_buffer_src_base_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t dram_src_noc_x             = get_arg_val<uint32_t>(1);
    std::uint32_t dram_src_noc_y             = get_arg_val<uint32_t>(2);

    std::uint32_t page_size                  = get_arg_val<uint32_t>(3);
    std::uint32_t start_page                 = get_arg_val<uint32_t>(4);
    std::uint32_t num_pages                  = get_arg_val<uint32_t>(5);


    auto remaining_pages = num_pages;

    const InterleavedAddrGen<true> s0 = {
        .bank_base_address = dram_buffer_src_base_addr,
        .page_size = page_size
    };



    uint32_t src_page_index = start_page;
    constexpr uint32_t cb_in_id = 0;

    while(remaining_pages>0)
    {
        std::uint64_t dram_buffer_src_noc_addr = get_noc_addr(src_page_index,s0);

        cb_reserve_back(cb_in_id,1);
        noc_async_read(dram_buffer_src_noc_addr, get_write_ptr(cb_in_id), page_size);
        noc_async_read_barrier();
        cb_push_back(cb_in_id,1);
        remaining_pages--;
        src_page_index++;
    }
}
