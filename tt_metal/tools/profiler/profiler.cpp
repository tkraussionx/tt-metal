#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>


#include "tt_metal/host_api.hpp"
#include "tools/profiler/profiler.hpp"
#include "tools/profiler/profiler_state.hpp"
#include "hostdevcommon/profiler_common.h"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

#define HOST_SIDE_LOG "profile_log_host.csv"
#define DEVICE_SIDE_LOG "profile_log_device.csv"

namespace tt {

namespace tt_metal {

TimerPeriodInt Profiler::timerToTimerInt(TimerPeriod period)
{
    TimerPeriodInt ret;

    ret.start = duration_cast<nanoseconds>(period.start.time_since_epoch()).count();
    ret.stop = duration_cast<nanoseconds>(period.stop.time_since_epoch()).count();
    ret.delta = duration_cast<nanoseconds>(period.stop - period.start).count();

    return ret;
}

void Profiler::dumpHostResults(const std::string& timer_name, const std::vector<std::pair<std::string,std::string>>& additional_fields)
{
    auto timer = name_to_timer_map[timer_name];

    auto timer_period_ns = timerToTimerInt(timer);
    TT_ASSERT (timer_period_ns.start != 0 , "Timer start cannot be zero on : " + timer_name);
    TT_ASSERT (timer_period_ns.stop != 0 , "Timer stop cannot be zero on : " + timer_name);

    std::filesystem::path log_path = output_dir / HOST_SIDE_LOG;
    std::ofstream log_file;

    if (host_new_log|| !std::filesystem::exists(log_path))
    {
        log_file.open(log_path);

        log_file << "Name" << ", ";
        log_file << "Start timer count [ns]"  << ", ";
        log_file << "Stop timer count [ns]"  << ", ";
        log_file << "Delta timer count [ns]";

        for (auto &field: additional_fields)
        {
            log_file  << ", "<< field.first;
        }

        log_file << std::endl;
        host_new_log = false;
    }
    else
    {
        log_file.open(log_path, std::ios_base::app);
    }

    log_file << timer_name << ", ";
    log_file << timer_period_ns.start  << ", ";
    log_file << timer_period_ns.stop  << ", ";
    log_file << timer_period_ns.delta;

    for (auto &field: additional_fields)
    {
        log_file  << ", "<< field.second;
    }

    log_file << std::endl;

    log_file.close();
}

void Profiler::readRiscProfilerResults(
        vector<std::uint32_t> profile_buffer,
        int pcie_slot,
        const CoreCoord &worker_core,
        int risc_num,
        std::string risc_name,
        int risc_print_buffer_addr){

    ZoneScoped;
    uint32_t end_index;
    uint32_t dropped_marker_counter;

    uint32_t startIndex = risc_print_buffer_addr * PRINT_BUFFER_SIZE/sizeof(uint32_t);
    end_index = profile_buffer[startIndex + kernel_profiler::BUFFER_END_INDEX];
    TT_ASSERT (end_index < (PRINT_BUFFER_SIZE/sizeof(uint32_t)));
    dropped_marker_counter = profile_buffer[startIndex + kernel_profiler::DROPPED_MARKER_COUNTER];

    if(dropped_marker_counter > 0){
        log_debug(
                tt::LogDevice,
                "{} device markers on device {} worker core {},{} risc {} were dropped. End index {}",
                dropped_marker_counter,
                pcie_slot,
                worker_core.x,
                worker_core.y,
                risc_name,
                end_index);
    }

    for (int i = kernel_profiler::MARKER_DATA_START; i < end_index; i+=kernel_profiler::TIMER_DATA_UINT32_SIZE) {
        dumpDeviceResultToFile(
                pcie_slot,
                worker_core.x,
                worker_core.y,
                risc_num,
                risc_name,
                (uint64_t(profile_buffer[startIndex + i+kernel_profiler::TIMER_VAL_H]) << 32) | profile_buffer[startIndex + i+kernel_profiler::TIMER_VAL_L],
                profile_buffer[startIndex + i+kernel_profiler::TIMER_ID]);
    }
}



void Profiler::dumpDeviceResultToFile(
        int chip_id,
        int core_x,
        int core_y,
        int risc,
        std::string hart_name,
        uint64_t timestamp,
        uint32_t timer_id){

    ZoneScoped;
    //std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
    //std::ofstream log_file;


    //if (device_new_log || !std::filesystem::exists(log_path))
    //{
        //log_file.open(log_path);
        //log_file << "Chip clock is at 1.2 GHz" << std::endl;
        //log_file << "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset]" << std::endl;
        //device_new_log = false;
    //}
    //else
    //{
        //log_file.open(log_path, std::ios_base::app);
    //}

    constexpr int DRAM_ROW = 6;
    if (core_y > DRAM_ROW){
       core_y = core_y - 2;
    }
    else{
       core_y--;
    }
    core_x--;

    uint64_t threadID = core_x*1000000+core_y*10000+risc*100;
    uint64_t eventID = timer_id + threadID;

    if (device_data.find (eventID) != device_data.end())
    {
        ZoneScopedNC("eventFound",tracy::Color::Green);
        device_data.at(eventID).push_back(timestamp);
    }
    else
    {
        ZoneScopedNC("eventNotFound",tracy::Color::Red);
        device_data.emplace(eventID,std::list<uint64_t>{timestamp});
    }

    //log_file << chip_id << ", " << core_x << ", " << core_y << ", " << hart_name << ", ";
    //log_file << timer_id << ", ";
    //log_file << timestamp;
    //log_file << std::endl;
    //log_file.close();
}

Profiler::Profiler()
{
#if defined(PROFILER)
    host_new_log = true;
    device_new_log = true;
    output_dir = std::filesystem::path("tt_metal/tools/profiler/logs");
    std::filesystem::create_directories(output_dir);
#endif
}

void Profiler::markStart(const std::string& timer_name)
{
#if defined(PROFILER)
    name_to_timer_map[timer_name].start = steady_clock::now();
#endif
}

void Profiler::markStop(const std::string& timer_name, const std::vector<std::pair<std::string,std::string>>& additional_fields)
{
#if defined(PROFILER)
    name_to_timer_map[timer_name].stop = steady_clock::now();
    dumpHostResults(timer_name, additional_fields);
#endif
}

void Profiler::setDeviceNewLogFlag(bool new_log_flag)
{
#if defined(PROFILER)
    device_new_log = new_log_flag;
#endif
}

void Profiler::setHostNewLogFlag(bool new_log_flag)
{
#if defined(PROFILER)
    host_new_log = new_log_flag;
#endif
}

void Profiler::setOutputDir(const std::string& new_output_dir)
{
#if defined(PROFILER)
    std::filesystem::create_directories(new_output_dir);
    output_dir = new_output_dir;
#endif
}

void Profiler::dumpDeviceResults (
        Device *device,
        int pcie_slot,
        const vector<CoreCoord> &worker_cores){
#if defined(PROFILER)
    ZoneScoped;
    for (const auto &worker_core : worker_cores) {

        auto dram_noc_xy = output_dram_buffer.noc_coordinates();

        std::cout << dram_noc_xy.x << "," << dram_noc_xy.y << std::endl;


        vector<std::uint32_t> profile_buffer;
	tt_metal::ReadFromBuffer(output_dram_buffer, profile_buffer);
        vector<std::uint32_t> profile_buffer_2 = tt::llrt::read_hex_vec_from_core(
		device->cluster(),
		pcie_slot,
		worker_core,
		(uint32_t)(106*1024),
		(uint32_t)1024);


        for (int i=0; i < profile_buffer.size(); i++)
        {
            std::cout << profile_buffer[i] << "," << profile_buffer_2 [i] << std::endl;
        }
    return;

        readRiscProfilerResults(
            profile_buffer,
            pcie_slot,
            worker_core,
            2,
            "NCRISC",
            0);
        readRiscProfilerResults(
            profile_buffer,
            pcie_slot,
            worker_core,
            1,
            "BRISC",
            4);
        readRiscProfilerResults(
            profile_buffer,
            pcie_slot,
            worker_core,
            3,
            "TRISC_0",
            1);
	readRiscProfilerResults(
	    profile_buffer,
	    pcie_slot,
	    worker_core,
            4,
	    "TRISC_1",
	    2);
	readRiscProfilerResults(
	    profile_buffer,
	    pcie_slot,
	    worker_core,
            5,
	    "TRISC_2",
	    3);
    }
#endif
}

bool getHostProfilerState ()
{
    bool profile_host = false;
#if defined(PROFILER)
    profile_host = true;
#endif
    return profile_host;
}

bool getDeviceProfilerState ()
{
    bool profile_device = false;
#if defined(PROFILER)
    const char *TT_METAL_DEVICE_PROFILER = std::getenv("TT_METAL_DEVICE_PROFILER");
    if (TT_METAL_DEVICE_PROFILER != nullptr && TT_METAL_DEVICE_PROFILER[0] == '1')
    {
        profile_device = true;
    }
#endif
    return profile_device;
}

}  // namespace tt_metal

}  // namespace tt
