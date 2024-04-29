// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <dataflow_api.h>
void kernel_main() {

    std::uint32_t dram_buffer_dst_base_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t dram_dst_noc_x             = get_arg_val<uint32_t>(1);
    std::uint32_t dram_dst_noc_y             = get_arg_val<uint32_t>(2);

    std::uint32_t page_size                  = get_arg_val<uint32_t>(3);
    std::uint32_t start_page                 = get_arg_val<uint32_t>(4);
    std::uint32_t num_pages                  = get_arg_val<uint32_t>(5);


    int chunk_size = page_size;
    int half_cs = chunk_size/2;


    auto remaining_pages = num_pages;

    const InterleavedAddrGen<true> d0 = {
        .bank_base_address = dram_buffer_dst_base_addr,
        .page_size = page_size
    };

    uint32_t dst_page_index = start_page;
    constexpr uint32_t cb_in_id = 0;

    while(remaining_pages>0)
    {
        cb_wait_front(cb_in_id,1);
        auto read_ptr = get_read_ptr(cb_in_id);
        volatile tt_l1_ptr uint8_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(read_ptr);
        for(int index = 0; index < half_cs; index++)
        {
            auto temp = ptr[index];
            ptr[index]=ptr[chunk_size-index-1];
            ptr[chunk_size-index-1]=temp;
        }
        std::uint64_t dram_buffer_dst_noc_addr = get_noc_addr(dst_page_index,d0);
        noc_async_write(read_ptr, dram_buffer_dst_noc_addr, chunk_size);
        noc_async_write_barrier();
        cb_pop_front(cb_in_id,1);
        remaining_pages--;
        dst_page_index--;
    }
}
