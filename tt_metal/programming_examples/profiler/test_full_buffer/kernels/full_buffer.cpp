// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
//Max unroll size
#pragma GCC unroll 65534
    for (int j = 0 ; j < 3; j++)
    {
        DeviceZoneScopedN("TEST-FULL");
    }
}
