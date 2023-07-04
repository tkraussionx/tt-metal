/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include <climits>

#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC)
#include "risc_common.h"
#else
#include "ckernel.h"
#endif

#ifdef PROFILE_KERNEL
#include "debug_print_buffer.h" // only needed because the address is shared
#endif

#include "hostdevcommon/profiler_common.h"
#include "src/firmware/riscv/common/risc_attribs.h"

namespace kernel_profiler{

    extern uint32_t wIndex;

#if defined(COMPILE_FOR_BRISC)
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_BR;
#elif defined(COMPILE_FOR_NCRISC)
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_NC;
#elif COMPILE_FOR_TRISC == 0
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_T0;
#elif COMPILE_FOR_TRISC == 1
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_T1;
#elif COMPILE_FOR_TRISC == 2
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_T2;
#endif

    inline __attribute__((always_inline)) void init_profiler()
    {
#if defined(PROFILE_KERNEL)
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC)
        uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#else
        uint32_t time_L = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#endif
        volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(profilerBuffer);
        buffer[SYNC_VAL_L] = time_L;
        buffer[SYNC_VAL_H] = time_H;
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_time(uint32_t timer_id)
    {
#if defined(PROFILE_KERNEL)
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC)
        uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
#else
        uint32_t time_L = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
#endif
        volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(profilerBuffer);
        buffer[timer_id] = time_L;
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_time_once(uint32_t timer_id, bool * one_time)
    {
#if defined(PROFILE_KERNEL)
        if (*one_time)
        {
            mark_time(timer_id);
        }
        *one_time = false;
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_BR_fw_first_start()
    {
#if defined(PROFILE_KERNEL) & defined(COMPILE_FOR_BRISC)
        uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);

        volatile uint32_t *profiler_control_buffer = reinterpret_cast<uint32_t*>(PROFILER_L1_BUFFER_CONTROL);

        profiler_control_buffer[FW_RESET_L] = time_L;
        profiler_control_buffer[FW_RESET_H] = time_H;
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_fw_start()
    {
#if defined(PROFILE_KERNEL)
        mark_time(FW_START);
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_fw_end()
    {
#if defined(PROFILE_KERNEL)
        mark_time(FW_END);
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_kernel_start()
    {
#if defined(PROFILE_KERNEL)
        mark_time(KERNEL_START);
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_kernel_end()
    {
#if defined(PROFILE_KERNEL)
        mark_time(KERNEL_END);
#endif //PROFILE_KERNEL
    }
}
