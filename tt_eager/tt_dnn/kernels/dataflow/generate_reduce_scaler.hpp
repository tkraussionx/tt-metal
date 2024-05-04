// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"

FORCE_INLINE void generate_reduce_scaler(const uint32_t cb_id, const uint32_t scaler) {
    cb_reserve_back(cb_id, 1);

    // Currently hardcoded to only support tile with 2 byte size elements
    constexpr uint32_t tile_size = 2048;
    static_assert(tile_size % MEM_ZEROS_SIZE == 0 || MEM_ZEROS_SIZE > 2048);
    static_assert(MEM_ZEROS_SIZE <= NOC_MAX_BURST_SIZE);
    constexpr uint32_t read_size = MEM_ZEROS_SIZE > 2048 ? 2048 : MEM_ZEROS_SIZE;
    constexpr uint32_t num_zeros_reads = tile_size / read_size;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

    // Fill tile with zeros
    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();

    if (scaler != 0) {
        for (uint32_t k = 0; k < 4; ++k) {
            uint32_t idx = k << 7;
            for (uint32_t j = 0; j < 8; ++j) {
                ptr[idx + j] = scaler;
            }
        }
    }
    cb_push_back(cb_id, 1);
}
