#pragma once

#include <climits>

#include "risc_common.h"
#include "hostdevcommon/profiler_common.h"

//#define PROFILER_OPTIONS KERNEL_FUNCT_MARKER
#define PROFILER_OPTIONS KERNEL_FUNCT_MARKER | MAIN_FUNCT_MARKER

//TODO: Add mechanism for selecting PROFILER markers

namespace kernel_profiler{

    volatile uint32_t *buffer;
    uint32_t wIndex;
    uint32_t time_H_global;

    inline __attribute__((always_inline)) void init_profiler()
    {
#ifdef PROFILE_KERNEL
        buffer = reinterpret_cast<uint32_t*>(get_debug_print_buffer());
        wIndex = MARKER_DATA_START;
        buffer [BUFFER_END_INDEX] = wIndex;
        buffer [DROPPED_MARKER_COUNTER] = 0;
        //buffer [TIMER_H_GLOBAL] = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_time(uint32_t timer_id)
    {
#ifdef PROFILE_KERNEL
	if (wIndex < PRINT_BUFFER_SIZE) {
	    buffer[wIndex] = (timer_id<<26) | (((1<<26)-1) & reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L));
	} else {
            buffer [DROPPED_MARKER_COUNTER]++;
	    return;
	}
        buffer [BUFFER_END_INDEX] = ++wIndex;
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_time_once(uint32_t timer_id, bool * one_time)
    {
#ifdef PROFILE_KERNEL
        if (*one_time)
        {
            mark_time(timer_id);
        }
        *one_time = false;
#endif //PROFILE_KERNEL
    }
}
