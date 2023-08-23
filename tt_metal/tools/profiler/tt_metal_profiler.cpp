#include "tt_metal/host_api.hpp"
#include "llrt/tt_debug_print_server.hpp"

#include "tools/profiler/profiler.hpp"

#include "tt_metal/detail/tt_metal.hpp"

#include "tt_metal/third_party/tracy/public/tracy/TracyOpenCL.hpp"

namespace tt {

namespace tt_metal {

namespace detail {

static Profiler tt_metal_profiler = Profiler();

cl_device_id deviceTEST;
cl_context context;
void DumpDeviceProfileResults(Device *device, const Program &program) {
#if defined(PROFILER)
    ZoneScoped;
    if (getDeviceProfilerState())
    {
        ProfileTTMetalScope profile_this = ProfileTTMetalScope("DumpDeviceProfileResults");
        //TODO: (MO) This global is temporary need to update once the new interface is in
        if (GLOBAL_CQ) {
            Finish(*GLOBAL_CQ);
        }
        TT_ASSERT(tt_is_print_server_running() == false, "Debug print server is running, cannot dump device profiler data");
        auto worker_cores_used_in_program =\
            device->worker_cores_from_logical_cores(program.logical_cores());
        auto cluster = device->cluster();
        auto pcie_slot = device->pcie_slot();
        tt_metal_profiler.dumpDeviceResults(cluster, pcie_slot, worker_cores_used_in_program);

        static TracyCLCtx tracyCLCtx = TracyCLContext(context, deviceTEST);

        for (auto& data: tt_metal_profiler.device_data)
        {
            uint64_t threadID = 100*(data.first/100);
            uint64_t row = int(threadID / 1000000);
            uint64_t col = int((threadID-row*1000000)/10000);
            uint64_t risc = int ((threadID-row*1000000-col*10000)/100);
            uint64_t markerID = data.first - threadID;

            if (row < 1 && col < 1 && markerID == 1)
            {
                for (auto event : data.second)
                {
                    switch (risc)
                    {
                        case 1:
                            {
                                TracyCLZoneC(tracyCLCtx, "FW", tracy::Color::Red3,threadID);
                                {
                                    TracyCLZoneC(tracyCLCtx, "KERNEL", tracy::Color::Red2,threadID);
                                    TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,0));
                                }
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,1));
                            }
                            break;
                        case 2:
                            {
                                TracyCLZoneC(tracyCLCtx, "FW", tracy::Color::Green4,threadID);
                                {
                                    TracyCLZoneC(tracyCLCtx, "KERNEL", tracy::Color::Green3,threadID);
                                    TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,0));
                                }
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,1));
                            }
                            break;
                        case 3:
                            {
                                TracyCLZoneC(tracyCLCtx, "FW", tracy::Color::Blue4,threadID);
                                {
                                    TracyCLZoneC(tracyCLCtx, "KERNEL", tracy::Color::Blue3,threadID);
                                    TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,0));
                                }
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,1));
                            }
                            break;
                        case 4:
                            {
                                TracyCLZoneC(tracyCLCtx, "FW", tracy::Color::Purple3,threadID);
                                {
                                    TracyCLZoneC(tracyCLCtx, "KERNEL", tracy::Color::Purple2,threadID);
                                    TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,0));
                                }
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,1));
                            }
                            break;
                        case 5:
                            {
                                TracyCLZoneC(tracyCLCtx, "FW", tracy::Color::Yellow4,threadID);
                                {
                                    TracyCLZoneC(tracyCLCtx, "KERNEL", tracy::Color::Yellow3,threadID);
                                    TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,0));
                                }
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,1));
                            }
                            break;

                        default:
                            break;
                    }
                }
            }
        }

        TracyCLCollect(tracyCLCtx, tt_metal_profiler.device_data);
        //TracyCLDestroy(tracyCLCtx);
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
