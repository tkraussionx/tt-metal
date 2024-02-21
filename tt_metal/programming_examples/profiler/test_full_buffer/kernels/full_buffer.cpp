// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
//Max unroll size
#if defined(COMPILE_FOR_NCRISC)
#pragma GCC unroll 65534
    for (int j = 0 ; j < 100; j++)
    {
        DeviceZoneScopedN("TEST-FULL-BUFFER");
    }
#endif
}
