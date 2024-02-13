// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    for (int i = 0; i < LOOP_COUNT; i ++)
    {
//Max unroll size
//#pragma GCC unroll 65534
        for (int j = 0 ; j < LOOP_SIZE; j++)
        {
            DeviceZoneScopedN("TEST-FULL-BUFFER");
            asm("nop");
        }
    }
}
