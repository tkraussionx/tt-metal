// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"
#include "tt_eager/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t input_is_dram = get_compile_time_arg_val(1) == 1;

    const auto input_addr = get_arg_val<uint32_t>(0);

    uint32_t cb_id{0};
    const auto cb_id_input = cb_id++;

    // input
    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);
    const auto input_data_format = get_dataformat(cb_id_input);

    const InterleavedAddrGenFast<input_is_dram> input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    constexpr uint32_t onetile = 1;

    const auto input_l1_write_ptr = get_write_ptr(cb_id_input);

    // for (uint32_t idx = 0; idx < num_tiles; ++idx) {
    //     cb_reserve_back(cb_id_input, onetile);
    //     auto noc_addr = get_noc_addr(idx, input_addrg, 96 * 2);
    //     DPRINT << "input_addr: " << input_addr << ENDL();
    //     DPRINT << "input_l1_write_ptr: " << input_l1_write_ptr << ENDL();
    //     DPRINT << "noc_addr: " << (uint32_t)noc_addr << ENDL();
    //     noc_async_read(noc_addr, input_l1_write_ptr, 4 * 2);
    //     noc_async_read_barrier();
    //     cb_push_back(cb_id_input, onetile);
    // }

    // for (uint32_t idx = 0; idx < num_tiles; ++idx) {
    //     cb_reserve_back(cb_id_input, onetile);
    //     auto noc_addr = get_noc_addr(idx, input_addrg, 98 * 2);
    //     DPRINT << "input_addr: " << input_addr << ENDL();
    //     DPRINT << "input_l1_write_ptr: " << input_l1_write_ptr << ENDL();
    //     DPRINT << "noc_addr: " << (uint32_t)noc_addr << ENDL();
    //     noc_async_read(noc_addr, input_l1_write_ptr, 3 * 2);
    //     noc_async_read_barrier();
    //     cb_push_back(cb_id_input, onetile);
    // }

    // for (uint32_t idx = 0; idx < num_tiles; ++idx) {
    //     cb_reserve_back(cb_id_input, onetile);
    //     auto noc_addr = get_noc_addr(idx, input_addrg, 98 * 2);
    //     noc_async_read(noc_addr, input_l1_write_ptr + 2 * 2, 3 * 2);

    //     noc_addr = get_noc_addr(idx, input_addrg, 114 * 2);
    //     noc_async_read(noc_addr, input_l1_write_ptr + 18 * 2, 3 * 2);
    //     noc_async_read_barrier();
    //     // auto ptr = reinterpret_cast<uint16_t *>(input_l1_write_ptr);
    //     // ptr[0] = ptr[2];
    //     // ptr[1] = ptr[3];
    //     // ptr[2] = ptr[4];
    //     // ptr[3] = 0;
    //     // ptr[4] = 0;
    //     cb_push_back(cb_id_input, onetile);
    // }

    for (uint32_t idx = 0; idx < num_tiles; ++idx) {
        cb_reserve_back(cb_id_input, onetile);
        auto noc_addr = get_noc_addr(idx, input_addrg, 98 * 2);
        DPRINT << "input_addr: " << input_addr << ENDL();
        DPRINT << "input_l1_write_ptr: " << input_l1_write_ptr << ENDL();
        DPRINT << "noc_addr: " << (uint32_t)noc_addr << ENDL();
        noc_async_read(noc_addr, input_l1_write_ptr + 2 * 2, 3 * 2);
        noc_async_read_barrier();
        cb_push_back(cb_id_input, onetile);
    }

    // for (uint32_t idx = 0; idx < num_tiles; ++idx) {
    //     cb_reserve_back(cb_id_input, onetile);
    //     auto noc_addr = get_noc_addr(idx, input_addrg, 98 * 2);
    //     DPRINT << "input_addr: " << input_addr << ENDL();
    //     DPRINT << "input_l1_write_ptr: " << input_l1_write_ptr << ENDL();
    //     DPRINT << "noc_addr: " << (uint32_t)noc_addr << ENDL();
    //     noc_async_read(noc_addr, input_l1_write_ptr + 18 * 2, 3 * 2);
    //     noc_async_read_barrier();
    //     cb_push_back(cb_id_input, onetile);
    // }

    // for (uint32_t idx = 0; idx < num_tiles; ++idx) {
    //     cb_reserve_back(cb_id_input, onetile);
    //     auto noc_addr = get_noc_addr(0, input_addrg, 114 * 2);
    //     noc_async_read(noc_addr, input_l1_write_ptr + 114 * 2, 2);
    //     noc_async_read_barrier();
    //     cb_push_back(cb_id_input, onetile);
    // }

    // for (uint32_t idx = 0; idx < num_tiles; ++idx) {
    //     cb_reserve_back(cb_id_input, onetile);
    //     auto noc_addr = get_noc_addr(0, input_addrg, 3 * 2);
    //     noc_async_read(noc_addr, input_l1_write_ptr + 3 * 2, 2 * 2);
    //     noc_async_read_barrier();
    //     cb_push_back(cb_id_input, onetile);
    // }

}  // void kernel_main()
