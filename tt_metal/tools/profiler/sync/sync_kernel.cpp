// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    volatile tt_reg_ptr uint32_t *p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t *> (RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile tt_l1_ptr uint32_t *profiler_control_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
    volatile tt_l1_ptr uint32_t *briscBuffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_BR + kernel_profiler::CUSTOM_MARKERS * sizeof(uint32_t));

    uint32_t syncTimeBufferIndex = 0;

    constexpr int FIRST_READ_COUNT = 2;

    while ( syncTimeBufferIndex < FIRST_READ_COUNT) {
        uint32_t deviceTime = p_reg[0];

        uint32_t hostTime = profiler_control_buffer[kernel_profiler::FW_RESET_L];
        if (hostTime > 0)
        {
            briscBuffer[syncTimeBufferIndex++] = p_reg[1];
            briscBuffer[syncTimeBufferIndex++] = deviceTime;
            briscBuffer[syncTimeBufferIndex++] = deviceTime;
            briscBuffer[syncTimeBufferIndex++] = hostTime;
            profiler_control_buffer[kernel_profiler::FW_RESET_L] = 0;
        }
    }

    while ( syncTimeBufferIndex < ((SAMPLE_COUNT + 1) * 2) ) {
        uint32_t deviceTime = p_reg[0];

        uint32_t hostTime = profiler_control_buffer[kernel_profiler::FW_RESET_L];
        if (hostTime > 0)
        {
            briscBuffer[syncTimeBufferIndex++] = deviceTime;
            briscBuffer[syncTimeBufferIndex++] = hostTime;
            profiler_control_buffer[kernel_profiler::FW_RESET_L] = 0;
        }
    }
}
