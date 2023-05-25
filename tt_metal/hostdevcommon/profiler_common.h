/*
*
* Ennums and defines shared between host and device profiler.
*
*/
#pragma once

#define MAIN_FUNCT_MARKER   (1U << 0)
#define KERNEL_FUNCT_MARKER (1U << 1)

#define CC_MAIN_START          1U
#define CC_KERNEL_MAIN_START   2U
#define CC_KERNEL_MAIN_END     3U
#define CC_MAIN_END            4U

#define PROFILER_VERSION_MAJOR 1U
#define PROFILER_VERSION_MINOR 0U
#define PROFILER_VERSION_PATCH 0U

//Pofiler version struct
struct ProfilerVersion {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};

namespace kernel_profiler{
/**
 * L1 buffer structure for profiler markers
 * _____________________________________________________________________________________________________
 *|                  |                        |              |             |             |              |
 *| Buffer end index | Dropped marker counter | 1st timer_id | 1st timer_L | 1st timer_H | 2nd timer_id | ...
 *|__________________|________________________|______________|_____________|_____________|______________|
 *
 * */

enum BufferIndex {BUFFER_END_INDEX, DROPPED_MARKER_COUNTER, MARKER_DATA_START};

enum TimerDataIndex {TIMER_ID, TIMER_VAL_L, TIMER_VAL_H, TIMER_DATA_UINT32_SIZE};

constexpr ProfilerVersion profiler_version = {1,0,0};

}
