/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

inline __attribute__((always_inline)) uint32_t get_flat_id(uint32_t coreX, uint32_t coreY)
{
    constexpr uint32_t DRAM_ROW = 6;
    constexpr uint32_t MULTIPLIER = 12;
    uint32_t coreFlatID = 0;

    if (coreY > DRAM_ROW){
        coreFlatID = ((coreY - 2) * MULTIPLIER + (coreX - 1));
    }
    else{
        coreFlatID = ((coreY - 1) * MULTIPLIER + (coreX - 1));
    }
    return coreFlatID;
}

namespace kernel_profiler{
/**
 * L1 buffer structure for profiler markers
 * _____________________________________________________________________________________________________
 *|                  |                        |              |             |             |              |

 *|__________________|________________________|______________|_____________|_____________|______________|
 *
 * */

enum BufferIndex {SYNC_VAL_H, SYNC_VAL_L, FW_START, KERNEL_START, KERNEL_END, FW_END};

enum ControlBuffer {DRAM_BUFFER_NUM, FW_RESET_H, FW_RESET_L};



}
