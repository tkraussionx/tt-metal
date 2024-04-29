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

    std::uint32_t dram_buffer_dst_addr  = get_arg_val<uint32_t>(4);
    std::uint32_t dram_dst_noc_x        = get_arg_val<uint32_t>(5);
    std::uint32_t dram_dst_noc_y        = get_arg_val<uint32_t>(6);

    std::uint32_t dram_buffer_size      = get_arg_val<uint32_t>(7);
    auto page_size                      = get_arg_val<uint32_t>(8);
    volatile tt_l1_ptr uint8_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_buffer_addr);



    int chunk_size = page_size;
    int half_cs = chunk_size/2;
    auto remaining_size = dram_buffer_size;

    const InterleavedAddrGen<true> s0 = {
        .bank_base_address = dram_buffer_src_addr,
        .page_size = page_size
    };

    const InterleavedAddrGen<true> d0 = {
        .bank_base_address = dram_buffer_dst_addr,
        .page_size = page_size
    };

    uint32_t src_page_index = 0;
    uint32_t dst_page_index = (dram_buffer_size/page_size)-1;
    while(remaining_size>0)
    {
        std::uint64_t dram_buffer_src_noc_addr = get_noc_addr(src_page_index,s0);
        std::uint64_t dram_buffer_dst_noc_addr = get_noc_addr(dst_page_index,d0);

        noc_async_read(dram_buffer_src_noc_addr, l1_buffer_addr, chunk_size);
        noc_async_read_barrier();
        for(int index = 0; index < half_cs; index++)
        {
            auto temp = ptr[index];
            ptr[index]=ptr[chunk_size-index-1];
            ptr[chunk_size-index-1]=temp;
        }

        noc_async_write(l1_buffer_addr, dram_buffer_dst_noc_addr, chunk_size);
        noc_async_write_barrier();
        remaining_size-=chunk_size;
        src_page_index++;
        dst_page_index--;
    }
}
