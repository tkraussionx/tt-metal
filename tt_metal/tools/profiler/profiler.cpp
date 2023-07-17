#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>

#include "tools/profiler/profiler.hpp"
#include "hostdevcommon/profiler_common.h"

#define HOST_SIDE_LOG "profile_log_host.csv"
#define DEVICE_SIDE_LOG "profile_log_device.csv"

Profiler::Profiler()
{
    do_profile_host = false;
    host_new_log = true;
    device_new_log = true;
    output_dir = std::filesystem::path("tt_metal/tools/profiler/logs");
    std::filesystem::create_directories(output_dir);
}

void Profiler::setHostDoProfile(bool profile_flag)
{
    do_profile_host = profile_flag;
}

void Profiler::setDeviceNewLogFlag(bool new_log_flag)
{
    device_new_log = new_log_flag;
}

void Profiler::setHostNewLogFlag(bool new_log_flag)
{
    host_new_log = new_log_flag;
}

void Profiler::setOutputDir(std::string new_output_dir)
{
    std::filesystem::create_directories(new_output_dir);
    output_dir = new_output_dir;
}

TimerPeriodInt Profiler::timerToTimerInt(TimerPeriod period)
{
    TimerPeriodInt ret;

    ret.start = duration_cast<nanoseconds>(period.start.time_since_epoch()).count();
    ret.stop = duration_cast<nanoseconds>(period.stop.time_since_epoch()).count();
    ret.delta = duration_cast<nanoseconds>(period.stop - period.start).count();

    return ret;
}

void Profiler::markStart(std::string timer_name)
{
    if (do_profile_host)
    {
        name_to_timer_map[timer_name].start = steady_clock::now();
    }
}

void Profiler::markStop(std::string timer_name, bool dumpResults)
{
    if (do_profile_host)
    {
        name_to_timer_map[timer_name].stop = steady_clock::now();
        if (dumpResults)
        {
            dumpHostResults();
        }
    }
}


void Profiler::dumpHostResults(const std::vector<std::pair<std::string,std::string>>& additional_fields, std::string name_prepend)
{
    if (do_profile_host)
    {
        const int large_width = 30;
        const int medium_width = 25;

        for (auto timer : name_to_timer_map)
        {
            auto timer_period_ns = timerToTimerInt(timer.second);
            TT_ASSERT (timer_period_ns.start != 0 , "Timer start cannot be zero on : " + timer.first);
            TT_ASSERT (timer_period_ns.stop != 0 , "Timer stop cannot be zero on : " + timer.first);
        }
        std::filesystem::path log_path = output_dir / HOST_SIDE_LOG;
        std::ofstream log_file;

        if (host_new_log || !std::filesystem::exists(log_path))
        {
            log_file.open(log_path);

            log_file << "Section Name" << ", ";
            log_file << "Function Name" << ", ";
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

        for (auto timer : name_to_timer_map)
        {
            auto timer_period_ns = timerToTimerInt(timer.second);
            log_file << name_prepend << ", ";
            log_file << timer.first << ", ";
            log_file << timer_period_ns.start  << ", ";
            log_file << timer_period_ns.stop  << ", ";
            log_file << timer_period_ns.delta;

            for (auto &field: additional_fields)
            {
                log_file  << ", "<< field.second;
            }

            log_file << std::endl;
        }

        log_file.close();

        name_to_timer_map.clear();
    }
}

void Profiler::dumpHostResults(std::string name_prepend)
{
    dumpHostResults({}, name_prepend);
}

void Profiler::dumpDeviceResults (
        tt_cluster *cluster,
        int pcie_slot,
        const vector<CoreCoord> &worker_cores){

    for (const auto &worker_core : worker_cores) {
        readRiscProfilerResults(
            cluster,
            pcie_slot,
            worker_core,
            "NCRISC",
            PRINT_BUFFER_NC);
        readRiscProfilerResults(
            cluster,
            pcie_slot,
            worker_core,
            "BRISC",
            PRINT_BUFFER_BR);
        readRiscProfilerResults(
            cluster,
            pcie_slot,
            worker_core,
            "TRISC_0",
            PRINT_BUFFER_T0);
	readRiscProfilerResults(
	    cluster,
	    pcie_slot,
	    worker_core,
	    "TRISC_1",
	    PRINT_BUFFER_T1);
	readRiscProfilerResults(
	    cluster,
	    pcie_slot,
	    worker_core,
	    "TRISC_2",
	    PRINT_BUFFER_T2);
    }
}

void Profiler::readRiscProfilerResults(
        tt_cluster *cluster,
        int pcie_slot,
        const CoreCoord &worker_core,
        std::string risc_name,
        int risc_print_buffer_addr){

    vector<std::uint32_t> profile_buffer;
    uint32_t end_index;
    uint32_t dropped_marker_counter;

    profile_buffer = tt::llrt::read_hex_vec_from_core(
            cluster,
            pcie_slot,
            worker_core,
            risc_print_buffer_addr,
            PRINT_BUFFER_SIZE);

    end_index = profile_buffer[kernel_profiler::BUFFER_END_INDEX];
    TT_ASSERT (end_index < (PRINT_BUFFER_SIZE/sizeof(uint32_t)));
    dropped_marker_counter = profile_buffer[kernel_profiler::DROPPED_MARKER_COUNTER];

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
                risc_name,
                (uint64_t(profile_buffer[i+kernel_profiler::TIMER_VAL_H]) << 32) | profile_buffer[i+kernel_profiler::TIMER_VAL_L],
                profile_buffer[i+kernel_profiler::TIMER_ID]);
    }
}

void Profiler::dumpDeviceResultToFile(
        int chip_id,
        int core_x,
        int core_y,
        std::string hart_name,
        uint64_t timestamp,
        uint32_t timer_id){

    std::filesystem::path log_path = output_dir / DEVICE_SIDE_LOG;
    std::ofstream log_file;

    if (device_new_log || !std::filesystem::exists(log_path))
    {
        log_file.open(log_path);
        log_file << "Chip clock is at 1.2 GHz" << std::endl;
        log_file << "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset]" << std::endl;
        device_new_log = false;
    }
    else
    {
        log_file.open(log_path, std::ios_base::app);
    }

    constexpr int DRAM_ROW = 6;
    if (core_y > DRAM_ROW){
       core_y = core_y - 2;
    }
    else{
       core_y--;
    }
    core_x--;

    log_file << chip_id << ", " << core_x << ", " << core_y << ", " << hart_name << ", ";
    log_file << timer_id << ", ";
    log_file << timestamp;
    log_file << std::endl;
    log_file.close();
}
