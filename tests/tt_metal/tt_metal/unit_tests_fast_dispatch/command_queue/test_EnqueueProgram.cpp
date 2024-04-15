// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "impl/buffers/buffer.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/scoped_timer.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt::tt_metal;

struct CBConfig {
    uint32_t cb_id;
    uint32_t num_pages;
    uint32_t page_size;
    tt::DataFormat data_format;
};

struct DummyProgramConfig {
    CoreRangeSet cr_set;
    CBConfig cb_config;
    uint32_t num_cbs;
    uint32_t num_sems;
};

struct DummyProgramMultiCBConfig {
    CoreRangeSet cr_set;
    std::vector<CBConfig> cb_config_vector;
    uint32_t num_sems;
};


namespace local_test_functions {

void initialize_dummy_kernels(Program& program, const CoreRangeSet& cr_set) {
    auto dummy_reader_kernel = CreateKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp", cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto dummy_writer_kernel = CreateKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp", cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto dummy_compute_kernel = CreateKernel(program, "tt_metal/kernels/compute/blank.cpp", cr_set, ComputeConfig{});
}

bool cb_config_successful(Device* device, const DummyProgramMultiCBConfig & program_config){

    bool pass = true;

    // Need to use old APIs to read since we cannot allocate a buffer in the reserved space we're trying
    // to read from
    vector<uint32_t> cb_config_vector;
    uint32_t cb_config_buffer_size = NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);

    for (const CoreRange& core_range : program_config.cr_set.ranges()) {
        CoresInCoreRangeGenerator core_range_generator(core_range, device->compute_with_storage_grid_size());

        bool terminate;
        do {
            auto [core_coord, terminate_] = core_range_generator();

            terminate = terminate_;
            tt::tt_metal::detail::ReadFromDeviceL1(
                device, core_coord, CIRCULAR_BUFFER_CONFIG_BASE, cb_config_buffer_size, cb_config_vector);

            uint32_t cb_addr = L1_UNRESERVED_BASE;
            for (uint32_t i = 0; i < program_config.cb_config_vector.size(); i ++) {
                uint32_t index = program_config.cb_config_vector[i].cb_id * sizeof(uint32_t);
                uint32_t cb_num_pages = program_config.cb_config_vector[i].num_pages;
                uint32_t cb_size = cb_num_pages * program_config.cb_config_vector[i].page_size;
                bool addr_match = cb_config_vector.at(index) == ((cb_addr) >> 4);
                bool size_match = cb_config_vector.at(index + 1) == (cb_size >> 4);
                bool num_pages_match = cb_config_vector.at(index + 2) == cb_num_pages;
                pass &= (addr_match and size_match and num_pages_match);

                cb_addr += cb_size;
            }

        } while (not terminate);
    }

    return pass;



}

bool test_dummy_EnqueueProgram_with_cbs(Device* device, CommandQueue& cq, DummyProgramMultiCBConfig& program_config) {

    Program program;


    for (uint32_t i = 0; i < program_config.cb_config_vector.size(); i++) {
        uint32_t cb_id = program_config.cb_config_vector[i].cb_id;
        uint32_t cb_num_pages = program_config.cb_config_vector[i].num_pages;
        uint32_t cb_size = cb_num_pages * program_config.cb_config_vector[i].page_size;
        auto df = program_config.cb_config_vector[i].data_format;
        uint32_t page_size = program_config.cb_config_vector[i].page_size;
        CircularBufferConfig cb_config = CircularBufferConfig(cb_size, {{cb_id, df}}).set_page_size(cb_id, page_size);
        auto cb = CreateCircularBuffer(program, program_config.cr_set, cb_config);
    }

    initialize_dummy_kernels(program, program_config.cr_set);
    EnqueueProgram(cq, program, false);
    Finish(cq);
    return cb_config_successful(device, program_config);

}

bool test_dummy_EnqueueProgram_with_cbs_update_size(Device* device, CommandQueue& cq, const DummyProgramMultiCBConfig& program_config) {

    Program program;


    std::vector<CBHandle> cb_ids;
    for (uint32_t i = 0; i < program_config.cb_config_vector.size(); i++) {
        uint32_t cb_id = program_config.cb_config_vector[i].cb_id;
        uint32_t cb_num_pages = program_config.cb_config_vector[i].num_pages;
        uint32_t cb_size = cb_num_pages * program_config.cb_config_vector[i].page_size;
        auto df = program_config.cb_config_vector[i].data_format;
        uint32_t page_size = program_config.cb_config_vector[i].page_size;
        CircularBufferConfig cb_config = CircularBufferConfig(cb_size, {{cb_id, df}}).set_page_size(cb_id, page_size);
        auto cb = CreateCircularBuffer(program, program_config.cr_set, cb_config);
        cb_ids.push_back(cb);
    }

    initialize_dummy_kernels(program, program_config.cr_set);
    EnqueueProgram(cq, program, false);
    Finish(cq);

    auto pass_1 = cb_config_successful(device, program_config);

    DummyProgramMultiCBConfig program_config_2 = program_config;
    for (auto & cb_config: program_config_2.cb_config_vector)
        cb_config.num_pages *=2;
    for (uint32_t buffer_id = 0; buffer_id < program_config.cb_config_vector.size(); buffer_id++) {
        auto cb_size = program_config_2.cb_config_vector[buffer_id].num_pages * program_config_2.cb_config_vector[buffer_id].page_size;
        UpdateCircularBufferTotalSize(program, cb_ids[buffer_id], cb_size);
    }


    EnqueueProgram(cq, program, false);
    Finish(cq);


    auto pass_2 = cb_config_successful(device, program_config_2);
    return pass_1 && pass_2;

}


bool test_dummy_EnqueueProgram_with_sems(Device* device, CommandQueue& cq, const DummyProgramConfig& program_config) {
    bool pass = true;

    Program program;

    for (uint32_t sem_id = 0; sem_id < program_config.num_sems; sem_id++) {
        CreateSemaphore(program, program_config.cr_set, sem_id);
    }

    EnqueueProgram(cq, program, false);
    Finish(cq);

    vector<uint32_t> sem_vector;
    uint32_t sem_buffer_size = program_config.num_sems * L1_ALIGNMENT;

    for (const CoreRange& core_range : program_config.cr_set.ranges()) {
        CoresInCoreRangeGenerator core_range_generator(core_range, device->compute_with_storage_grid_size());

        bool terminate;
        do {
            auto [core_coord, terminate_] = core_range_generator();

            terminate = terminate_;
            tt::tt_metal::detail::ReadFromDeviceL1(device, core_coord, SEMAPHORE_BASE, sem_buffer_size, sem_vector);

            uint32_t sem_id = 0;
            for (uint32_t i = 0; i < sem_vector.size(); i += (L1_ALIGNMENT / sizeof(uint32_t))) {

                bool sem_match = sem_vector.at(i) == sem_id;
                sem_id++;

                pass &= sem_match;
            }
        } while (not terminate);
    }

    return pass;
}

// KCM - Update this too.
bool test_dummy_EnqueueProgram_with_runtime_args(Device* device, CommandQueue& cq, const DummyProgramConfig& program_config, uint32_t num_runtime_args_for_kernel0, uint32_t num_runtime_args_for_kernel1, uint32_t num_iterations) {
    Program program;
    bool pass = true;

    CoreRangeSet cr_set = program_config.cr_set;

    auto dummy_kernel0 = CreateKernel(
        program, "tests/tt_metal/tt_metal/gtest_unit_tests/command_queue/test_kernels/runtime_args_kernel0.cpp", cr_set, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto dummy_kernel1 = CreateKernel(
        program, "tests/tt_metal/tt_metal/gtest_unit_tests/command_queue/test_kernels/runtime_args_kernel1.cpp", cr_set, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto dummy_compute_kernel = CreateKernel(program, "tt_metal/kernels/compute/blank.cpp", cr_set, ComputeConfig{});

    vector<uint32_t> dummy_kernel0_args;
    vector<uint32_t> dummy_kernel1_args;

    uint32_t idx;
    for (idx = 0; idx < num_runtime_args_for_kernel0; idx++) {
        dummy_kernel0_args.push_back(idx);
    }

    for (; idx < num_runtime_args_for_kernel0 + num_runtime_args_for_kernel1; idx++) {
        dummy_kernel1_args.push_back(idx);
    }

    for (const CoreRange& core_range : program_config.cr_set.ranges()) {
        CoresInCoreRangeGenerator core_range_generator(core_range, device->compute_with_storage_grid_size());

        bool terminate;
        do {
            auto [core_coord, terminate_] = core_range_generator();

            SetRuntimeArgs(program, dummy_kernel0, core_coord, dummy_kernel0_args);
            SetRuntimeArgs(program, dummy_kernel1, core_coord, dummy_kernel1_args);

            terminate = terminate_;
        } while (not terminate);
    }

    tt::tt_metal::detail::CompileProgram(device, program);
    for (uint32_t i = 0; i < num_iterations; i++) {
        EnqueueProgram(cq, program, false);
    }
    Finish(cq);

    for (const CoreRange& core_range : program_config.cr_set.ranges()) {
        CoresInCoreRangeGenerator core_range_generator(core_range, device->compute_with_storage_grid_size());

        bool terminate;
        do {
            auto [core_coord, terminate_] = core_range_generator();
            terminate = terminate_;

            vector<uint32_t> dummy_kernel0_args_readback;
            tt::tt_metal::detail::ReadFromDeviceL1(
                device, core_coord, BRISC_L1_ARG_BASE, dummy_kernel0_args.size() * sizeof(uint32_t), dummy_kernel0_args_readback);
            pass &= (dummy_kernel0_args == dummy_kernel0_args_readback);

            vector<uint32_t> dummy_kernel1_args_readback;
            tt::tt_metal::detail::ReadFromDeviceL1(
                device, core_coord, NCRISC_L1_ARG_BASE, dummy_kernel1_args.size() * sizeof(uint32_t), dummy_kernel1_args_readback);
            pass &= (dummy_kernel1_args == dummy_kernel1_args_readback);
        } while (not terminate);
    }

    return pass;
}

// KCM - FIXME Update this.
bool test_dummy_EnqueueProgram_with_runtime_args_multi_crs(
    Device* device,
    CommandQueue& cq,
    const DummyProgramConfig& program_config,
    uint32_t num_runtime_args_for_cr0,
    uint32_t num_runtime_args_for_cr1,
    uint32_t num_iterations) {
    Program program;
    bool pass = true;

    CoreRangeSet cr_set = program_config.cr_set;

    auto dummy_kernel0 = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/gtest_unit_tests/command_queue/test_kernels/runtime_args_kernel0.cpp",
        cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto dummy_kernel1 = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/gtest_unit_tests/command_queue/test_kernels/runtime_args_kernel1.cpp",
        cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto dummy_compute_kernel = CreateKernel(program, "tt_metal/kernels/compute/blank.cpp", cr_set, ComputeConfig{});

    vector<uint32_t> dummy_cr0_args;
    vector<uint32_t> dummy_cr1_args;
    bool terminate = false;

    auto it = program_config.cr_set.ranges().begin();
    CoreRange core_range_0 = *it;
    std::advance(it, 1);
    CoreRange core_range_1 = *it;

    for (uint32_t i = 0; i < num_iterations; i++) {
        dummy_cr0_args.clear();
        dummy_cr1_args.clear();
        uint32_t idx;
        for (idx = 0; idx < num_runtime_args_for_cr0; idx++) {
            dummy_cr0_args.push_back(idx * (i + 1));
        }

        for (; idx < num_runtime_args_for_cr0 + num_runtime_args_for_cr1; idx++) {
            dummy_cr1_args.push_back(idx * (i + 1));
        }

        CoresInCoreRangeGenerator core_range_generator_0(core_range_0, device->compute_with_storage_grid_size());

        do {
            auto [core_coord, terminate_] = core_range_generator_0();

            SetRuntimeArgs(program, dummy_kernel0, core_coord, dummy_cr0_args);
            log_info(tt::LogTest, "KCM Setting {} args for core: {}", dummy_cr0_args.size(), core_coord.str());
            // SetRuntimeArgs(program, dummy_kernel1, core_coord, dummy_cr0_args);

            terminate = terminate_;
        } while (not terminate);

        // Only do this if a 2nd core-range set was provided.
        if (program_config.cr_set.ranges().size() > 1) {

            CoresInCoreRangeGenerator core_range_generator_1(core_range_1, device->compute_with_storage_grid_size());

            terminate = false;
            do {
                auto [core_coord, terminate_] = core_range_generator_1();

                SetRuntimeArgs(program, dummy_kernel0, core_coord, dummy_cr1_args);
                SetRuntimeArgs(program, dummy_kernel1, core_coord, dummy_cr1_args);

                terminate = terminate_;
            } while (not terminate);
        }

        tt::tt_metal::detail::CompileProgram(device, program);
        EnqueueProgram(cq, program, false);
    }
    Finish(cq);

    terminate = false;
    CoresInCoreRangeGenerator core_range_generator_0(core_range_0, device->compute_with_storage_grid_size());
    CoresInCoreRangeGenerator core_range_generator_1(core_range_1, device->compute_with_storage_grid_size());
    do {
        auto [core_coord, terminate_] = core_range_generator_0();

        vector<uint32_t> dummy_kernel0_args_readback;
        tt::tt_metal::detail::ReadFromDeviceL1(
            device,
            core_coord,
            BRISC_L1_ARG_BASE,
            dummy_cr0_args.size() * sizeof(uint32_t),
            dummy_kernel0_args_readback);
        pass &= (dummy_cr0_args == dummy_kernel0_args_readback);

        vector<uint32_t> dummy_kernel1_args_readback;
        tt::tt_metal::detail::ReadFromDeviceL1(
            device,
            core_coord,
            NCRISC_L1_ARG_BASE,
            dummy_cr0_args.size() * sizeof(uint32_t),
            dummy_kernel1_args_readback);
        pass &= (dummy_cr0_args == dummy_kernel1_args_readback);

        terminate = terminate_;
    } while (not terminate);

    terminate = false;
    do {
        auto [core_coord, terminate_] = core_range_generator_1();

        vector<uint32_t> dummy_kernel0_args_readback;
        tt::tt_metal::detail::ReadFromDeviceL1(
            device,
            core_coord,
            BRISC_L1_ARG_BASE,
            dummy_cr1_args.size() * sizeof(uint32_t),
            dummy_kernel0_args_readback);
        pass &= (dummy_cr1_args == dummy_kernel0_args_readback);

        vector<uint32_t> dummy_kernel1_args_readback;
        tt::tt_metal::detail::ReadFromDeviceL1(
            device,
            core_coord,
            NCRISC_L1_ARG_BASE,
            dummy_cr1_args.size() * sizeof(uint32_t),
            dummy_kernel1_args_readback);
        pass &= (dummy_cr1_args == dummy_kernel1_args_readback);

        terminate = terminate_;
    } while (not terminate);

    return pass;
}

bool test_EnqueueWrap_on_EnqueueWriteBuffer(Device* device, CommandQueue& cq, const TestBufferConfig& config) {
    EnqueueWriteBuffer_prior_to_wrap(device, cq, config);

    /*
    This just ensures we don't hang on the subsequent EnqueueWriteBuffer
    */
    size_t buf_size = config.num_pages * config.page_size;
    Buffer buffer(device, buf_size, config.page_size, config.buftype);

    vector<uint32_t> src(buf_size / sizeof(uint32_t), 0);

    for (uint32_t i = 0; i < src.size(); i++) {
        src.at(i) = i;
    }
    EnqueueWriteBuffer(cq, buffer, src, false);
    Finish(cq);

    return true;
}

bool test_EnqueueWrap_on_Finish(Device* device, CommandQueue& cq, const TestBufferConfig& config) {
    bool pass = true;
    EnqueueWriteBuffer_prior_to_wrap(device, cq, config);

    return pass;
}

bool test_EnqueueWrap_on_EnqueueProgram(Device* device, CommandQueue& cq, const TestBufferConfig& config) {
    bool pass = true;
    EnqueueWriteBuffer_prior_to_wrap(device, cq, config);

    return pass;
}


// Verify RT args for a core at a given address by comparing to expected values.
bool verify_rt_args(bool unique, Device* device, CoreCoord logical_core, uint32_t addr, std::vector<uint32_t> expected_rt_args, std::vector<uint32_t> increments) {
    bool pass = true;
    std::string label = unique ? "Unique" : "Common";
    log_info(tt::LogTest, "Verifying {} {} RT args for {} at addr: 0x{:x} w/ increments: {}", expected_rt_args.size(), label, logical_core.str(), addr, increments);

    std::vector<uint32_t> written_args;
    tt::tt_metal::detail::ReadFromDeviceL1(device, logical_core, addr, expected_rt_args.size() * sizeof(uint32_t), written_args);

    for(int i=0; i<expected_rt_args.size(); i++){
        uint32_t incr_amount = i < increments.size() ? increments.at(i) : 0;
        uint32_t expected_val = expected_rt_args.at(i) + incr_amount;
        log_info(tt::LogTest, "Checking {} RT Arg. i: {} expected: {} observed: {} (incr_amount: {})", label, i, expected_val, written_args[i], incr_amount);
        EXPECT_EQ(written_args[i], expected_val);
        pass &= (written_args[i] == expected_val);
    }
    return pass;
}

// Write unique and common RT args, increment in kernel, and verify correctness via readback.
bool test_increment_runtime_args_sanity(Device* device, const DummyProgramConfig& program_config) {

    Program program;
    bool pass = true;
    CoreRangeSet cr_set = program_config.cr_set;

    auto compute_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/increment_runtime_arg.cpp",
        cr_set,
        tt::tt_metal::ComputeConfig{});
    const auto kernel = tt::tt_metal::detail::GetKernel(program, compute_kernel_id);

    // Unique Runtime Args.
    std::vector<uint32_t> unique_runtime_args = {101, 202};
    std::vector<uint32_t> unique_arg_increments = {87, 216};

    // Hack for local testing. This works.
    uint32_t num_rt_args = tt::parse_env("NUM_RT_ARGS", 2);
    if (num_rt_args != 2) {
        unique_runtime_args.clear();
        for (uint32_t i=0; i<num_rt_args; i++){
            unique_runtime_args.push_back(i);
        }
    }

    SetRuntimeArgs(program, 0, cr_set, unique_runtime_args);

    // Setup Common Runtime Args.
    std::vector<uint32_t> common_runtime_args = {11, 22, 33, 44};
    std::vector<uint32_t> common_arg_increments = {123, 234, 345, 456};

    // Hack to not overflow.
    if (num_rt_args + common_runtime_args.size() > 255) {
        common_runtime_args.clear();
        common_arg_increments.clear();
        SetCommonRuntimeArgs(program, 0, common_runtime_args);
    }


    // Compile and Launch the Program now.
    EnqueueProgram(device->command_queue(), program, false);
    Finish(device->command_queue());
    auto common_args_addr = TRISC_L1_ARG_BASE + kernel->get_common_runtime_args_offset();

    // Verify Unique/Common Args.
    for (auto &core_range : kernel->logical_coreranges()) {
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                CoreCoord core_coord(x, y);
                pass &= verify_rt_args(true, device, core_coord, TRISC_L1_ARG_BASE, unique_runtime_args, unique_arg_increments);
                pass &= verify_rt_args(false, device, core_coord, common_args_addr, common_runtime_args, common_arg_increments);
            }
        }
    }

    return pass;
}

}  // namespace local_test_functions

namespace basic_tests {

namespace compiler_workaround_hardware_bug_tests {

TEST_F(CommandQueueSingleCardFixture, TestArbiterDoesNotHang) {
    for (Device *device : devices_) {
        Program program;

        CoreRange cr({0, 0}, {0, 0});
        CoreRangeSet cr_set({cr});
        // Add an NCRISC blank manually, but in compile program, the BRISC blank will be
        // added separately
        auto dummy_reader_kernel = CreateKernel(
            program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/arbiter_hang.cpp", cr_set, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        EnqueueProgram(device->command_queue(), program, false);
        Finish(device->command_queue());
    }
}
}
namespace single_core_tests {

TEST_F(CommandQueueSingleCardFixture, TestSingleCbConfigCorrectlySentSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config = {.cb_id=0, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    DummyProgramMultiCBConfig config = {.cr_set = cr_set, .cb_config_vector = {cb_config} };

    for (Device *device : devices_) {
        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_cbs(device, device->command_queue(), config));
    }
}

TEST_F(CommandQueueSingleCardFixture, TestMultiCbSeqConfigCorrectlySentSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config_0 = {.cb_id = 0, .num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 1, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 2, .num_pages = 2, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 3, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};


    DummyProgramMultiCBConfig config = {.cr_set = cr_set, .cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3}};

    for (Device *device : devices_) {
        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_cbs(device, device->command_queue(), config));
    }
}

TEST_F(CommandQueueSingleCardFixture, TestMultiCbRandomConfigCorrectlySentSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config_0 = {.cb_id = 1, .num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 0, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 24, .num_pages = 2, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 16, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};


    DummyProgramMultiCBConfig config = {.cr_set = cr_set, .cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3}};

    for (Device *device : devices_) {
        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_cbs(device, device->command_queue(), config));
    }
}

TEST_F(CommandQueueSingleCardFixture, TestMultiCBSharedAddressSpaceSentSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    uint32_t intermediate_cb = 24;
    uint32_t out_cb = 16;
    std::map<uint8_t, tt::DataFormat> intermediate_and_out_data_format_spec = {
        {intermediate_cb, tt::DataFormat::Float16_b},
        {out_cb, tt::DataFormat::Float16_b}
    };
    uint32_t num_bytes_for_df = 2;
    uint32_t single_tile_size = num_bytes_for_df * 1024;
    uint32_t num_tiles = 2;
    uint32_t cb_size = num_tiles * single_tile_size;

    uint32_t cb_config_buffer_size = NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);
    CoreCoord core_coord(0,0);

    for (Device *device : devices_) {
        Program program;
        CircularBufferConfig cb_config = CircularBufferConfig(cb_size, intermediate_and_out_data_format_spec)
            .set_page_size(intermediate_cb, single_tile_size)
            .set_page_size(out_cb, single_tile_size);
        auto cb = CreateCircularBuffer(program, cr_set, cb_config);

        local_test_functions::initialize_dummy_kernels(program, cr_set);

        EnqueueProgram(device->command_queue(), program, false);

        Finish(device->command_queue());

        vector<uint32_t> cb_config_vector;

        tt::tt_metal::detail::ReadFromDeviceL1(
            device, core_coord, CIRCULAR_BUFFER_CONFIG_BASE, cb_config_buffer_size, cb_config_vector);
        uint32_t cb_addr = L1_UNRESERVED_BASE;
        uint32_t intermediate_index = intermediate_cb * sizeof(uint32_t);

        bool addr_match_intermediate = cb_config_vector.at(intermediate_index) == ((cb_addr) >> 4);
        bool size_match_intermediate = cb_config_vector.at(intermediate_index + 1) == (cb_size >> 4);
        bool num_pages_match_intermediate = cb_config_vector.at(intermediate_index + 2) == num_tiles;
        bool pass_intermediate = (addr_match_intermediate and size_match_intermediate and num_pages_match_intermediate);
        EXPECT_TRUE(pass_intermediate);

        uint32_t out_index = out_cb * sizeof(uint32_t);
        bool addr_match_out = cb_config_vector.at(out_index) == ((cb_addr) >> 4);
        bool size_match_out = cb_config_vector.at(out_index + 1) == (cb_size >> 4);
        bool num_pages_match_out = cb_config_vector.at(out_index + 2) == num_tiles;
        bool pass_out = (addr_match_out and size_match_out and num_pages_match_out);
        EXPECT_TRUE(pass_out);
    }
}

TEST_F(CommandQueueSingleCardFixture, TestSingleCbConfigCorrectlyUpdateSizeSentSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config = {.cb_id=0, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};

    DummyProgramMultiCBConfig config = {.cr_set = cr_set, .cb_config_vector = {cb_config}};

    for (Device *device : devices_) {
        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_cbs_update_size(device, device->command_queue(), config));
    }
}

TEST_F(CommandQueueSingleCardFixture, TestSingleSemaphoreConfigCorrectlySentSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    DummyProgramConfig config = {.cr_set = cr_set, .num_sems = NUM_SEMAPHORES};

    for (Device *device : devices_) {
        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_sems(device, device->command_queue(), config));
    }
}

TEST_F(CommandQueueSingleCardFixture, TestAutoInsertedBlankBriscKernelInDeviceDispatchMode) {
    for (Device *device : devices_) {
        Program program;

        CoreRange cr({0, 0}, {0, 0});
        CoreRangeSet cr_set({cr});
        // Add an NCRISC blank manually, but in compile program, the BRISC blank will be
        // added separately
        auto dummy_reader_kernel = CreateKernel(
            program, "tt_metal/kernels/dataflow/blank.cpp", cr_set,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        EnqueueProgram(device->command_queue(), program, false);
        Finish(device->command_queue());
    }
}

// Sanity test for setting and verifying common and unique runtime args to a single core, the simplest case.
TEST_F(CommandQueueSingleCardFixture, IncrementRuntimeArgsSanitySingleCore) {
    CoreRange cr0({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr0});
    DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
    for (Device *device : devices_) {
        EXPECT_TRUE(local_test_functions::test_increment_runtime_args_sanity(device, dummy_program_config));
    }
}

// Sanity test for setting and verifying common and unique runtime args to multiple cores.
TEST_F(CommandQueueSingleCardFixture, IncrementRuntimeArgsSanityMultiCore) {
    CoreRange cr0({1, 1}, {2, 2});
    CoreRange cr1({3, 3}, {4, 4});
    CoreRangeSet cr_set({cr0, cr1});
    DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
    for (Device *device : devices_) {
        EXPECT_TRUE(local_test_functions::test_increment_runtime_args_sanity(device, dummy_program_config));
    }
}

TEST_F(CommandQueueSingleCardFixture, TestRuntimeArgsCorrectlySentSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
    for (Device *device : devices_) {
        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_runtime_args(
            device, device->command_queue(), dummy_program_config, 9, 12, 1));
    }
}

}  // end namespace single_core_tests

namespace multicore_tests {
TEST_F(CommandQueueSingleCardFixture, TestAllCbConfigsCorrectlySentMultiCore) {
    CBConfig cb_config = {.num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    std::vector <CBConfig> cb_config_vector(NUM_CIRCULAR_BUFFERS, cb_config);
    for(int i=0; i<NUM_CIRCULAR_BUFFERS; i++)
        cb_config_vector[i].cb_id = i;

    for (Device *device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set({cr});

        DummyProgramMultiCBConfig config = {
            .cr_set = cr_set, .cb_config_vector = cb_config_vector};

        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_cbs(device, device->command_queue(), config));
    }
}

TEST_F(CommandQueueSingleCardFixture, TestAllCbConfigsCorrectlySentUpdateSizeMultiCore) {
    CBConfig cb_config = {.num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    std::vector <CBConfig> cb_config_vector(NUM_CIRCULAR_BUFFERS, cb_config);
    for(int i=0; i<NUM_CIRCULAR_BUFFERS; i++)
        cb_config_vector[i].cb_id = i;

    for (Device *device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set({cr});

        DummyProgramMultiCBConfig config = {
            .cr_set = cr_set, .cb_config_vector = cb_config_vector  };

        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_cbs_update_size(device, device->command_queue(), config));
    }
}

TEST_F(CommandQueueSingleCardFixture, TestMultiCbConfigsCorrectlySentUpdateSizeMultiCore) {
    CBConfig cb_config_0 = {.cb_id = 0, .num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 1, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 2, .num_pages = 2, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 3, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    std::vector <CBConfig> cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3};

    for (Device *device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set({cr});

        DummyProgramMultiCBConfig config = {
            .cr_set = cr_set, .cb_config_vector = cb_config_vector  };

        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_cbs_update_size(device, device->command_queue(), config));
    }
}

TEST_F(CommandQueueSingleCardFixture, TestAllSemConfigsCorrectlySentMultiCore) {
    for (Device *device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set({cr});

        DummyProgramConfig config = {.cr_set = cr_set, .num_sems = NUM_SEMAPHORES};

        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_sems(device, device->command_queue(), config));
    }
}

TEST_F(CommandQueueSingleCardFixture, TestAllRuntimeArgsCorrectlySentMultiCore) {
    for (Device *device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set({cr});

        DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_runtime_args(device, device->command_queue(), dummy_program_config, 9, 12, 1));
    }
}

TEST_F(CommandQueueSingleCardFixture, TestSendRuntimeArgsMultiCoreRange) {
    for (Device* device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr0({0, 0}, {worker_grid_size.x - 1, 3});
        CoreRange cr1({0, 4}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set({cr0, cr1});

        DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_runtime_args_multi_crs(
            device, device->command_queue(), dummy_program_config, 9, 12, 1));
    }
}

// Fails
TEST_F(CommandQueueSingleCardFixture, TestSendRuntimeArgsSingleCoreRange255_RISCV01) {
    for (Device* device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr0({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set({cr0});

        DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_runtime_args_multi_crs(
            device, device->command_queue(), dummy_program_config, 255, 255, 1));
    }
}

// Also Fails
TEST_F(CommandQueueSingleCardFixture, TestSendRuntimeArgsMultiCoreRange255_RISCV0) {
    for (Device* device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr0({0, 0}, {worker_grid_size.x - 1, 3});
        CoreRange cr1({0, 4}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set({cr0, cr1});

        DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_runtime_args_multi_crs(
            device, device->command_queue(), dummy_program_config, 255, 10, 1));
    }
}

// Still Fails
TEST_F(CommandQueueSingleCardFixture, TestSendRuntimeArgsMultiCoreRange255_RISCV1) {
    for (Device* device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr0({0, 0}, {worker_grid_size.x - 1, 3});
        CoreRange cr1({0, 4}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set({cr0, cr1});

        DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_runtime_args_multi_crs(
            device, device->command_queue(), dummy_program_config, 10, 255, 1));
    }
}

TEST_F(CommandQueueSingleCardFixture, TestUpdateRuntimeArgsMultiCoreRange) {
    for (Device* device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr0({0, 0}, {worker_grid_size.x - 1, 3});
        CoreRange cr1({0, 4}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set({cr0, cr1});

        DummyProgramConfig dummy_program_config = {.cr_set = cr_set};
        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_runtime_args_multi_crs(
            device, device->command_queue(), dummy_program_config, 9, 12, 2));
    }
}

}  // end namespace multicore_tests
}

namespace stress_tests {


TEST_F(CommandQueueSingleCardFixture, DISABLED_TestFillDispatchCoreBuffer) {
    uint32_t NUM_ITER = 100000;
    for (Device *device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();

        CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet cr_set({cr});

        DummyProgramConfig dummy_program_config = {.cr_set = cr_set};

        EXPECT_TRUE(local_test_functions::test_dummy_EnqueueProgram_with_runtime_args(device, device->command_queue(), dummy_program_config, 256, 256, NUM_ITER));
    }
}

TEST_F(CommandQueueFixture, TestRandomizedProgram) {
    uint32_t NUM_PROGRAMS = 5;
    uint32_t MAX_LOOP = 100;
    uint32_t MAX_RUNTIME_ARGS = 255;
    uint32_t page_size = 1024;

    // Make random
    // srand((unsigned int)time(NULL));

    CoreCoord worker_grid_size = this->device_->compute_with_storage_grid_size();
    CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set({cr});

    auto create_runtime_args = [&](uint32_t num_rt_args, uint32_t base = 0) -> vector<uint32_t> {
        vector<uint32_t> runtime_args;
        for (uint32_t i = 0; i < num_rt_args; i++) {
            runtime_args.push_back(base + i);
        }
        return runtime_args;
    };

    vector<Program> programs;
    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        programs.push_back(Program());
        Program& program = programs.back();

        std::map<string, string> data_movement_defines = {{"DATA_MOVEMENT", "1"}};
        std::map<string, string> compute_defines = {{"COMPUTE", "1"}};

        // brisc
        uint32_t BRISC_OUTER_LOOP, BRISC_MIDDLE_LOOP, BRISC_INNER_LOOP, NUM_CBS, NUM_SEMS, NUM_URT_ARGS, NUM_CRT_ARGS;
        if (i == 0) {
            // Ensures that we get at least one compilation with the max amount to
            // ensure it compiles and runs
            BRISC_OUTER_LOOP = MAX_LOOP;
            BRISC_MIDDLE_LOOP = MAX_LOOP;
            BRISC_INNER_LOOP = MAX_LOOP;
            NUM_CBS = NUM_CIRCULAR_BUFFERS;
            NUM_SEMS = NUM_SEMAPHORES;
            // NUM_URT_ARGS = MAX_RUNTIME_ARGS; // FIXME - Not working.
            NUM_URT_ARGS = 8;
            NUM_CRT_ARGS = 8;
        } else {
            BRISC_OUTER_LOOP = rand() % (MAX_LOOP) + 1;
            BRISC_MIDDLE_LOOP = rand() % (MAX_LOOP) + 1;
            BRISC_INNER_LOOP = rand() % (MAX_LOOP) + 1;
            NUM_CBS = rand() % (NUM_CIRCULAR_BUFFERS) + 1;
            NUM_SEMS = rand() % (NUM_SEMAPHORES) + 1;
            NUM_URT_ARGS = rand() % (MAX_RUNTIME_ARGS) + 1;
            uint32_t num_urt_args_aligned = align(NUM_URT_ARGS, L1_ALIGNMENT/sizeof(uint32_t));
            NUM_CRT_ARGS = rand() % (MAX_RUNTIME_ARGS - num_urt_args_aligned) + 1;
        }

        log_info(tt::LogTest, "Compiling program {}/{} w/ BRISC_OUTER_LOOP: {} BRISC_MIDDLE_LOOP: {} BRISC_INNER_LOOP: {} NUM_CBS: {} NUM_SEMS: {} NUM_URT_ARGS: {} NUM_CRT_ARGS: {}",
                 i+1, NUM_PROGRAMS, BRISC_OUTER_LOOP, BRISC_MIDDLE_LOOP, BRISC_INNER_LOOP, NUM_CBS, NUM_SEMS, NUM_URT_ARGS, NUM_CRT_ARGS);

        for (uint32_t j = 0; j < NUM_CBS; j++) {
            CircularBufferConfig cb_config = CircularBufferConfig(page_size * (j + 1), {{j, tt::DataFormat::Float16_b}}).set_page_size(j, page_size * (j + 1));
            auto cb = CreateCircularBuffer(program, cr_set, cb_config);
        }

        for (uint32_t j = 0; j < NUM_SEMS; j++) {
            CreateSemaphore(program, cr_set, j + 1);
        }

        vector<uint32_t> brisc_compile_args = {BRISC_OUTER_LOOP, BRISC_MIDDLE_LOOP, BRISC_INNER_LOOP, NUM_CBS, NUM_SEMS, NUM_URT_ARGS, NUM_CRT_ARGS, page_size};
        vector<uint32_t> brisc_unique_runtime_args = create_runtime_args(NUM_URT_ARGS);
        vector<uint32_t> brisc_common_runtime_args = create_runtime_args(NUM_CRT_ARGS, 100);
        // ncrisc
        uint32_t NCRISC_OUTER_LOOP, NCRISC_MIDDLE_LOOP, NCRISC_INNER_LOOP;
        if (i == 0) {
            NCRISC_OUTER_LOOP = MAX_LOOP;
            NCRISC_MIDDLE_LOOP = MAX_LOOP;
            NCRISC_INNER_LOOP = MAX_LOOP;
        } else {
            NCRISC_OUTER_LOOP = rand() % (MAX_LOOP) + 1;
            NCRISC_MIDDLE_LOOP = rand() % (MAX_LOOP) + 1;
            NCRISC_INNER_LOOP = rand() % (MAX_LOOP) + 1;
        }
        vector<uint32_t> ncrisc_compile_args = {NCRISC_OUTER_LOOP, NCRISC_MIDDLE_LOOP, NCRISC_INNER_LOOP, NUM_CBS, NUM_SEMS, NUM_URT_ARGS, NUM_CRT_ARGS, page_size};
        vector<uint32_t> ncrisc_unique_runtime_args = create_runtime_args(NUM_URT_ARGS);
        vector<uint32_t> ncrisc_common_runtime_args = create_runtime_args(NUM_CRT_ARGS, 100);

        // trisc
        uint32_t TRISC_OUTER_LOOP, TRISC_MIDDLE_LOOP, TRISC_INNER_LOOP;
        if (i == 0) {
            TRISC_OUTER_LOOP = MAX_LOOP;
            TRISC_MIDDLE_LOOP = MAX_LOOP;
            TRISC_INNER_LOOP = MAX_LOOP;
        } else {
            TRISC_OUTER_LOOP = rand() % (MAX_LOOP) + 1;
            TRISC_MIDDLE_LOOP = rand() % (MAX_LOOP) + 1;
            TRISC_INNER_LOOP = rand() % (MAX_LOOP) + 1;
        }

        vector<uint32_t> trisc_compile_args = {TRISC_OUTER_LOOP, TRISC_MIDDLE_LOOP, TRISC_INNER_LOOP, NUM_CBS, NUM_SEMS, NUM_URT_ARGS, NUM_CRT_ARGS, page_size};
        vector<uint32_t> trisc_unique_runtime_args = create_runtime_args(NUM_URT_ARGS);
        vector<uint32_t> trisc_common_runtime_args = create_runtime_args(NUM_CRT_ARGS, 100);

        bool at_least_one_kernel = false;
        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_brisc_kernel = CreateKernel(
                program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp", cr_set, DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = brisc_compile_args, .defines = data_movement_defines});
            SetRuntimeArgs(program, dummy_brisc_kernel, cr_set, brisc_unique_runtime_args);
            SetCommonRuntimeArgs(program, dummy_brisc_kernel, brisc_common_runtime_args);
            at_least_one_kernel = true;
        }

        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_ncrisc_kernel = CreateKernel(
                program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp", cr_set, DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = ncrisc_compile_args, .defines = data_movement_defines});
            SetRuntimeArgs(program, dummy_ncrisc_kernel, cr_set, ncrisc_unique_runtime_args);
            SetCommonRuntimeArgs(program, dummy_ncrisc_kernel, ncrisc_common_runtime_args);
            at_least_one_kernel = true;
        }

        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_trisc_kernel = CreateKernel(
                program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp", cr_set, ComputeConfig{
                    .math_approx_mode = false,
                    .compile_args = trisc_compile_args,
                    .defines = compute_defines
                });
            SetRuntimeArgs(program, dummy_trisc_kernel, cr_set, trisc_unique_runtime_args);
            SetCommonRuntimeArgs(program, dummy_trisc_kernel, trisc_common_runtime_args);
            at_least_one_kernel = true;
        }

        if (not at_least_one_kernel) {
            uint32_t random_risc = rand() % 3 + 1;
            if (random_risc == 1) {
                auto dummy_brisc_kernel = CreateKernel(
                    program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp", cr_set, DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = brisc_compile_args, .defines = data_movement_defines});
                SetRuntimeArgs(program, dummy_brisc_kernel, cr_set, brisc_unique_runtime_args);
                SetCommonRuntimeArgs(program, dummy_brisc_kernel, brisc_common_runtime_args);
            } else if (random_risc == 2) {
                auto dummy_ncrisc_kernel = CreateKernel(
                    program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp", cr_set, DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = ncrisc_compile_args, .defines = data_movement_defines});
                SetRuntimeArgs(program, dummy_ncrisc_kernel, cr_set, ncrisc_unique_runtime_args);
                // SetCommonRuntimeArgs(program, dummy_ncrisc_kernel, ncrisc_common_runtime_args);
            } else if (random_risc == 3) {
                auto dummy_trisc_kernel = CreateKernel(
                    program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp", cr_set, ComputeConfig{
                        .math_approx_mode = false,
                        .compile_args = trisc_compile_args,
                        .defines = compute_defines
                    });
                SetRuntimeArgs(program, dummy_trisc_kernel, cr_set, trisc_unique_runtime_args);
                SetCommonRuntimeArgs(program, dummy_trisc_kernel, trisc_common_runtime_args);
            } else {
                TT_ASSERT("Invalid");
            }
        }

        tt::tt_metal::detail::CompileProgram(this->device_, program);
    }

    log_info(tt::LogTest, "Caching {} programs now with initial EnqueueProgram()", programs.size());

    // This loop caches program and runs
    for (Program& program: programs) {
        EnqueueProgram(this->device_->command_queue(), program, false);
    }

    // This loops assumes already cached
    uint32_t NUM_ITERATIONS = 50; // TODO(agrebenisan): Bump this to 5000, saw hangs for very large number of iterations, need to come back to that
    log_info(tt::LogTest, "Running {} programs now for {} iterations...", programs.size(), NUM_ITERATIONS);
    for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(programs), std::end(programs), rng);
        for (Program& program: programs) {
            EnqueueProgram(this->device_->command_queue(), program, false);
        }
    }

    log_info(tt::LogTest, "Calling finish now...");
    Finish(this->device_->command_queue());
}

}  // namespace stress_tests
