// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t output_is_dram = get_compile_time_arg_val(1) == 1;

    const auto output_addr = get_arg_val<uint32_t>(0);

    uint32_t cb_id{16};
    const auto cb_id_output = cb_id++;

    // output
    const uint32_t output_tile_bytes = get_tile_size(cb_id_output);
    const auto output_data_format = get_dataformat(cb_id_output);

    const InterleavedAddrGenFast<output_is_dram> output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    constexpr uint32_t onetile = 1;

    const auto output_l1_read_ptr = get_read_ptr(cb_id_output);

    for (uint32_t idx = 0; idx < num_tiles; ++idx) {
        cb_wait_front(cb_id_output, onetile);
        noc_async_write_tile(idx, output_addrg, output_l1_read_ptr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_output, onetile);
    }

}  // void kernel_main()
