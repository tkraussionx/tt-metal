#include "tt_metal/host_api.hpp"
#include "llrt/tt_debug_print_server.hpp"

#include "tools/profiler/profiler.hpp"

#include "tt_metal/detail/tt_metal.hpp"

#include "tt_metal/third_party/tracy/public/tracy/TracyOpenCL.hpp"

namespace tt {

namespace tt_metal {

namespace detail {

static Profiler tt_metal_profiler = Profiler();

void InitDeviceProfiler(Device *device){

    const size_t byte_size = 1024 * 120;

    tt_metal_profiler.output_dram_buffer = tt_metal::Buffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    dram_buffer_start_addr = tt_metal_profiler.output_dram_buffer.address();

    //std::cout << dram_buffer_start_addr << std::endl;

    std::vector<uint32_t> inputs_DRAM(byte_size, 3);
    tt_metal::WriteToBuffer(tt_metal_profiler.output_dram_buffer, inputs_DRAM);
}


void DumpDeviceProfileResults(Device *device, vector<CoreCoord>& worker_cores) {
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
        auto cluster = device->cluster();
        auto pcie_slot = device->pcie_slot();
        tt_metal_profiler.dumpDeviceResults(device, pcie_slot, worker_cores);

        tt_metal_profiler.tracyTTCtx->PopulateCLContext();

        for (auto& data: tt_metal_profiler.device_data)
        {
            ZoneScopedNC("Marker",tracy::Color::Red);
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
                                TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "FW", tracy::Color::Red3,threadID);
                                {
                                    TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "KERNEL", tracy::Color::Red2,threadID);
                                    TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,0));
                                }
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,1));
                            }
                            break;
                        case 2:
                            {
                                TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "FW", tracy::Color::Green4,threadID);
                                {
                                    TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "KERNEL", tracy::Color::Green3,threadID);
                                    TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,0));
                                }
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,1));
                            }
                            break;
                        case 3:
                            {
                                TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "FW", tracy::Color::Blue4,threadID);
                                {
                                    TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "KERNEL", tracy::Color::Blue3,threadID);
                                    TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,0));
                                }
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,1));
                            }
                            break;
                        case 4:
                            {
                                TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "FW", tracy::Color::Purple3,threadID);
                                {
                                    TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "KERNEL", tracy::Color::Purple2,threadID);
                                    TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,0));
                                }
                                TracyCLZoneSetEvent(tracy::TTDeviceEvent(pcie_slot,row,col,risc,1));
                            }
                            break;
                        case 5:
                            {
                                TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "FW", tracy::Color::Yellow4,threadID);
                                {
                                    TracyCLZoneC(tt_metal_profiler.tracyTTCtx, "KERNEL", tracy::Color::Yellow3,threadID);
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

        TracyCLCollect(tt_metal_profiler.tracyTTCtx, tt_metal_profiler.device_data);
    }
#endif
}

void DumpDeviceProfileResults(Device *device, const Program &program)
{
    auto worker_cores_used_in_program =\
                                       device->worker_cores_from_logical_cores(program.logical_cores());
    DumpDeviceProfileResults(device, worker_cores_used_in_program);
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
