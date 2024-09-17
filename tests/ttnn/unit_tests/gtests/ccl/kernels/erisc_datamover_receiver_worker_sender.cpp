// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const ttnn::ccl::EriscDataMoverPacketSizingMode packet_sizing_mode =
        static_cast<ttnn::ccl::EriscDataMoverPacketSizingMode>(get_arg_val<uint32_t>(1));

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t num_pages_total = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_edm_buffer = get_compile_time_arg_val(3);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    InterleavedAddrGen<dst_is_dram> dest_addr_generator = {
        .bank_base_address = dst_addr, .page_size = page_size};

    for (uint32_t p = 0; p < num_pages_total; p += pages_per_edm_buffer) {
        uint32_t num_pages_to_send = std::min<uint32_t>(pages_per_edm_buffer, num_pages_total - p);
        cb_wait_front(cb_id_in0, num_pages_to_send);
        uint32_t l1_read_addr = get_read_ptr(cb_id_in0);

        for (uint32_t i = 0; i < num_pages_to_send; ++i) {
            uint64_t dst_noc_addr = get_noc_addr(p + i, dest_addr_generator);
            if (packet_sizing_mode == ttnn::ccl::EriscDataMoverPacketSizingMode::FIXED_SIZE) {
                noc_async_write(l1_read_addr, dst_noc_addr, page_size);
            } else {
                noc_async_write(dst_noc_addr, l1_read_addr, page_size + 16);
            }
            l1_read_addr += page_size;
        }
        noc_async_write_barrier();

        cb_pop_front(cb_id_in0, num_pages_to_send);
    }

}
