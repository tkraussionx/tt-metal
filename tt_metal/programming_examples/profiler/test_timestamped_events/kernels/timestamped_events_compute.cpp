// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint16_t THE_ID=12345;
namespace NAMESPACE {
    void MAIN {
        for (int i = 0; i < LOOP_COUNT; i ++)
        {
            DeviceZoneScopedN("TEST-FULL");
            DeviceTimestampedData( THE_ID, i + ((uint64_t)1 << 32));
            DeviceRecordEvent( i );
//Max unroll size
#pragma GCC unroll 65534
            for (int j = 0 ; j < LOOP_SIZE; j++)
            {
                asm("nop");
            }
        }
    }
} // NAMESPACE
