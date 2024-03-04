// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "packet_queue.hpp"

void kernel_main() {



    volatile uint32_t* reg_ptr = reinterpret_cast<volatile uint32_t*>(0xFFB40000);
    reg_ptr[0] = 0xc00 + arg1;
}
