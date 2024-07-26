// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"

FORCE_INLINE void generate_reduce_mm_scaler(const uint32_t cb_id, const uint32_t scaler) {
    cb_reserve_back(cb_id, 1);

    constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

    // Fill tile with zeros
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read(zeros_noc_addr, write_addr, MEM_ZEROS_SIZE);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();

    // for (int i = 0; i < 512; i++) {
    //     ptr[i] = 0x3f803f80;
    // }

    // First row ones, zeros elsewhere
    if (scaler != 0) {
        for (int k = 0; k < 2; ++k) {
            uint32_t idx = k << 7;
            for (int j = 0; j < 8; ++j) {
                ptr[idx + j] = scaler;
            }
        }
    }

    // // First column ones, zeros elsewhere
    // uint32_t scaler_single = scaler >> 16;
    // if (scaler != 0) {
    //     for (int k = 0; k < 4; k+=2) {
    //         uint32_t idx = k << 7;
    //         for (int j = 0; j < 128; ++j) {
    //             if (j % 8 == 0) {
    //                 ptr[idx + j] = scaler_single;
    //             }
    //         }
    //     }
    // }

    cb_push_back(cb_id, 1);
}
