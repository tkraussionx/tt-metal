// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"

void kernel_main() {
    ///////////////////////////////////////////////////////////////////////
    // This section is reserved for kernel debug print practice session. //
    ///////////////////////////////////////////////////////////////////////
    #if 1
    // Direct printing is supported for const char*/char/uint32_t/float
    DPRINT << "Test string" << 'a' << 5 << 0.123456f << ENDL();

    // BF16 type printing is supported via a macro
    uint16_t my_bf16_val = 0x3dfb;  // Equivalent to 0.122559
    DPRINT << BF16(my_bf16_val) << ENDL();

    // Printing a bool type directly causes a kernel crash.
    // To avoid this, cast the bool variable to uint32_t.
    bool my_bool_flag = true;
    DPRINT << static_cast<uint32_t>(my_bool_flag) << ENDL();

    // HEX/DEC/OCT macros corresponding to std::hex/std::dec/std::oct
    DPRINT << HEX() << 15 << " " << DEC() << 15 << " " << OCT() << 15 << ENDL();

    // The following prints only occur on a particular RISCV core:
    DPRINT_MATH(DPRINT << "this is the math kernel" << ENDL());
    DPRINT_PACK(DPRINT << "this is the pack kernel" << ENDL());
    DPRINT_UNPACK(DPRINT << "this is the unpack kernel" << ENDL());
    DPRINT_DATA0(DPRINT << "this is the data movement kernel on noc 0" << ENDL());
    DPRINT_DATA1(DPRINT << "this is the data movement kernel on noc 1" << ENDL());
    #endif
}
