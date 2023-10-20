// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "llrt/tt_debug_print_server.hpp"

#include "tools/profiler/profiler.hpp"
#include "hostdevcommon/profiler_common.h"

#include "tt_metal/detail/tt_metal.hpp"

#include "tt_metal/third_party/tracy/public/tracy/TracyOpenCL.hpp"

namespace tt {

namespace tt_metal {

namespace detail {

Profiler tt_metal_profiler;

void InitDeviceProfiler(Device *device){
#if defined(PROFILER)
    ZoneScoped;

    CoreCoord compute_with_storage_size = device->logical_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};

    //std::vector<uint32_t> zero_buffer(PROFILER_RISC_COUNT * PROFILER_L1_VECTOR_SIZE + PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
    //{
        //ZoneScopedN("Clearing_profiler_L1");
        //for (size_t x=start_core.x; x <= end_core.x; x++)
        //{
            //for (size_t y=start_core.y; y <= end_core.y; y++)
            //{
                //CoreCoord curr_core = {x, y};
                //tt_metal::detail::WriteToDeviceL1(device, curr_core, PROFILER_L1_BUFFER_BR, zero_buffer);
            //}
        //}
    //}

    vector<uint32_t> huge_zero_buffer(PROFILER_FULL_BUFFER_SIZE / sizeof(uint32_t), 2);
    tt::Cluster::instance().write_sysmem_vec(huge_zero_buffer, PROFILER_HUGE_PAGE_ADDRESS, 0);

#endif
}

void DumpDeviceProfileResults(Device *device) {
#if defined(PROFILER)
    CoreCoord compute_with_storage_size = device->logical_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};

    std::vector<CoreCoord> workerCores;
    for (size_t x=start_core.x; x <= end_core.x; x++)
    {
        for (size_t y=start_core.y; y <= end_core.y; y++)
        {
            CoreCoord logical_core = {x, y};
            workerCores.push_back(device->worker_core_from_logical_core(logical_core));
        }
    }
    DumpDeviceProfileResults(device, workerCores);
#endif
}

void DumpDeviceProfileResults(Device *device, const Program &program)
{
#if defined(PROFILER)
    auto worker_cores_used_in_program =\
                                       device->worker_cores_from_logical_cores(program.logical_cores());
    DumpDeviceProfileResults(device, worker_cores_used_in_program);
#endif
}


void DumpDeviceProfileResults(Device *device, vector<CoreCoord>& worker_cores) {
#if defined(PROFILER)
    ZoneScoped;
    if (getDeviceProfilerState())
    {
        auto device_id = device->id();
        tt_metal_profiler.setDeviceArchitecture(device->arch());
        tt_metal_profiler.dumpDeviceResults(device_id, worker_cores);

        tt_metal_profiler.tracyTTCtx->PopulateCLContext();

        for (auto& data: tt_metal_profiler.device_data)
        {
            ZoneScopedNC("Marker",tracy::Color::Red);
            uint64_t threadID = 100*(data.first/100);
            uint64_t row = int(threadID / 1000000);
            uint64_t col = int((threadID-row*1000000)/10000);
            uint64_t risc = int ((threadID-row*1000000-col*10000)/100);
            uint64_t markerID = data.first - threadID;

            if (row == 0 && col == 0 && markerID == 1)
            {
                for (auto event : data.second)
                {
                    switch (risc)
                    {
                        case 0:
                            {
                                TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "FW", tracy::Color::Red3,threadID);
                                {
                                    TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "KERNEL", tracy::Color::Red2,threadID);
                                    TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,0));
                                }
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,1));
                            }
                            break;
                        case 1:
                            {
                                TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "FW", tracy::Color::Green4,threadID);
                                {
                                    TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "KERNEL", tracy::Color::Green3,threadID);
                                    TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,0));
                                }
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,1));
                            }
                            break;
                        case 2:
                            {
                                TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "FW", tracy::Color::Blue4,threadID);
                                {
                                    TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "KERNEL", tracy::Color::Blue3,threadID);
                                    TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,0));
                                }
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,1));
                            }
                            break;
                        case 3:
                            {
                                TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "FW", tracy::Color::Purple3,threadID);
                                {
                                    TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "KERNEL", tracy::Color::Purple2,threadID);
                                    TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,0));
                                }
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,1));
                            }
                            break;
                        case 4:
                            {
                                TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "FW", tracy::Color::Yellow4,threadID);
                                {
                                    TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "KERNEL", tracy::Color::Yellow3,threadID);
                                    TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,0));
                                }
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(device_id,row,col,risc,1));
                            }
                            break;

                        default:
                            break;
                    }
                }
            }
        }

        TracyCLCollect(tt_metal_profiler.tracyTTCtx, tt_metal_profiler.device_data);
    }
#endif
}

void SetProfilerDir(std::string output_dir){
#if defined(PROFILER)
     tt_metal_profiler.setOutputDir(output_dir);
#endif
}

void FreshProfilerHostLog(){
#if defined(PROFILER)
     tt_metal_profiler.setHostNewLogFlag(true);
#endif
}

void FreshProfilerDeviceLog(){
#if defined(PROFILER)
     tt_metal_profiler.setDeviceNewLogFlag(true);
#endif
}

ProfileTTMetalScope::ProfileTTMetalScope (const string& scopeNameArg) : scopeName(scopeNameArg){
#if defined(PROFILER)
    tt_metal_profiler.markStart(scopeName);
#endif
}

ProfileTTMetalScope::~ProfileTTMetalScope ()
{
#if defined(PROFILER)
    tt_metal_profiler.markStop(scopeName);
#endif
}

}  // namespace detail

}  // namespace tt_metal

}  // namespace tt
