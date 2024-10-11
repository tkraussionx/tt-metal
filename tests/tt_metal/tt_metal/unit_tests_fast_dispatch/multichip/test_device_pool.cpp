// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tests/tt_metal/tt_metal/unit_tests/common/basic_fixture.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/impl/device/device_pool.hpp"

using namespace tt;
using namespace tt::test_utils;

namespace dp_test_functions {

vector<uint32_t> generate_arange_vector(uint32_t size_bytes) {
    TT_FATAL(size_bytes % sizeof(uint32_t) == 0, "Error");
    vector<uint32_t> src(size_bytes / sizeof(uint32_t), 0);

    for (uint32_t i = 0; i < src.size(); i++) {
        src.at(i) = i;
    }
    return src;
}

struct TestBufferConfig {
    uint32_t num_pages;
    uint32_t page_size;
    tt::tt_metal::BufferType buftype;
};

bool test_EnqueueWriteBuffer_and_EnqueueReadBuffer(Device* device, vector<std::reference_wrapper<CommandQueue>>& cqs, const TestBufferConfig& config) {
    bool pass = true;
    for (const bool use_void_star_api: {true, false}) {

        size_t buf_size = config.num_pages * config.page_size;
        std::vector<std::unique_ptr<Buffer>> buffers;
        std::vector<std::vector<uint32_t>> srcs;
        for (uint i = 0; i < cqs.size(); i++) {
            buffers.push_back(std::make_unique<Buffer>(device, buf_size, config.page_size, config.buftype));
            srcs.push_back(generate_arange_vector(buffers[i]->size()));
            if (use_void_star_api) {
                EnqueueWriteBuffer(cqs[i], *buffers[i], srcs[i].data(), false);
            } else {
                EnqueueWriteBuffer(cqs[i], *buffers[i], srcs[i], false);
            }
        }

        for (uint i = 0; i < cqs.size(); i++) {
            std::vector<uint32_t> result;
            if (use_void_star_api) {
                result.resize(buf_size / sizeof(uint32_t));
                EnqueueReadBuffer(cqs[i], *buffers[i], result.data(), true);
            } else {
                EnqueueReadBuffer(cqs[i], *buffers[i], result, true);
            }
            bool local_pass = (srcs[i] == result);
            pass &= local_pass;
        }
    }

    return pass;
}
}

TEST_F(FDBasicFixture, DevicePoolOpenClose) {
    std::vector<chip_id_t> device_ids{0};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    std::vector<tt_metal::Device *> devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }

    // Close then get devices again
    for (const auto& dev: devices) {
        dev->close();
    }
    devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
    for (const auto& dev: devices) {
        dev->close();
    }
}

TEST_F(FDBasicFixture, DevicePoolReconfigDevices) {
    std::vector<chip_id_t> device_ids{0};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    std::vector<Device *> devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }

    // Close then get devices with different configs
    for (const auto& dev: devices) {
        dev->close();
    }
    l1_small_size = 2048;
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE(dev->is_initialized());
    }
    for (const auto& dev: devices) {
        dev->close();
    }
}

TEST_F(FDBasicFixture, DevicePoolAddDevices) {
    if (tt::tt_metal::GetNumAvailableDevices() != 8) {
        GTEST_SKIP();
    }
    std::vector<chip_id_t> device_ids{0};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    std::vector<Device *> devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }

    // Close then get more devices
    for (const auto& dev: devices) {
        dev->close();
    }
    device_ids = {0, 1, 2, 3};
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    devices = tt::DevicePool::instance().get_all_active_devices();
    ASSERT_TRUE(devices.size() >= 4);
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
    for (const auto& dev: devices) {
        dev->close();
    }
}

TEST_F(FDBasicFixture, DevicePoolReduceDevices) {
    if (tt::tt_metal::GetNumAvailableDevices() != 8) {
        GTEST_SKIP();
    }
    std::vector<chip_id_t> device_ids{0, 1, 2, 3};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    std::vector<Device *> devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }

    // Close then get less devices
    for (const auto& dev: devices) {
        dev->close();
    }
    device_ids = {0};
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    auto dev = tt::DevicePool::instance().get_active_device(0);
    ASSERT_TRUE(dev->id() == 0);
    ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
    ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
    ASSERT_TRUE(dev->is_initialized());
    tt::DevicePool::instance().close_device(0);
}

TEST_F(FDBasicFixture, DevicePoolShutdownSubmesh) {
    if (tt::tt_metal::GetNumAvailableDevices() != 32) {
        GTEST_SKIP();
    }
    chip_id_t mmio_device_id = 0;
    std::vector<chip_id_t> device_ids{mmio_device_id};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    std::vector<Device *> devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
    std::vector<Device*> tunnel_0;
    std::vector<Device*> tunnel_1;
    auto mmio_dev_handle = tt::DevicePool::instance().get_active_device(mmio_device_id);
    auto tunnels_from_mmio = mmio_dev_handle->tunnels_from_mmio_;
    //iterate over all tunnels origination from this mmio device
    for (uint32_t ts = tunnels_from_mmio[0].size() - 1; ts > 0; ts--) {
        tunnel_0.push_back(tt::DevicePool::instance().get_active_device(tunnels_from_mmio[0][ts]));
    }
    for (uint32_t ts = tunnels_from_mmio[1].size() - 1; ts > 0; ts--) {
        tunnel_1.push_back(tt::DevicePool::instance().get_active_device(tunnels_from_mmio[1][ts]));
    }

    tt::DevicePool::instance().close_devices(tunnel_0);
    for (const auto& dev: tunnel_1) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
    tt::DevicePool::instance().close_devices(tunnel_1);
}

TEST_F(FDBasicFixture, DevicePoolReopenOneDevice) {
    if (tt::tt_metal::GetNumAvailableDevices() != 8) {
        GTEST_SKIP();
    }
    chip_id_t mmio_device_id = 0;
    std::vector<chip_id_t> device_ids{mmio_device_id, 4};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    std::cout << "test 1" << std::endl;
    std::vector<Device *> devices = tt::DevicePool::instance().get_all_active_devices();
    std::cout << "test 1" << std::endl;
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
    // Test Body
    Device* dev = tt::DevicePool::instance().get_active_device(4);
    dp_test_functions::TestBufferConfig config = {.num_pages = 1, .page_size = 2048, .buftype = BufferType::DRAM};
    tt::log_info("Running On Device {}", dev->id());
    CommandQueue& a = dev->command_queue(0);
    vector<std::reference_wrapper<CommandQueue>> cqs = {a};
    // Simple read and write buffer test
    EXPECT_TRUE(dp_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(dev, cqs, config));

    std::cout << "test 1" << std::endl;
    dev->close();
    tt::DevicePool::instance().activate_device(dev->id());
    tt::DevicePool::instance().initialize_device(dev);
    std::cout << " done re init " << std::endl;
    // Simple read and write buffer test
    CommandQueue&b = dev->command_queue(0);
    cqs = {b};
   // EXPECT_TRUE(dp_test_functions::test_EnqueueWriteBuffer_and_EnqueueReadBuffer(dev, cqs, config));
    ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
    ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
    ASSERT_TRUE(dev->is_initialized());
    tt::DevicePool::instance().close_device(0);
}

TEST_F(FDBasicFixture, DevicePoolReopenSubmesh) {
    if (tt::tt_metal::GetNumAvailableDevices() != 32) {
        GTEST_SKIP();
    }

    chip_id_t mmio_device_id = 0;
    std::vector<chip_id_t> device_ids{mmio_device_id};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    std::vector<Device *> devices = tt::DevicePool::instance().get_all_active_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
    std::vector<Device*> tunnel_0;
    std::vector<Device*> tunnel_1;
    auto mmio_dev_handle = tt::DevicePool::instance().get_active_device(mmio_device_id);
    auto tunnels_from_mmio = mmio_dev_handle->tunnels_from_mmio_;
    //iterate over all tunnels origination from this mmio device
    for (uint32_t ts = tunnels_from_mmio[0].size() - 1; ts > 0; ts--) {
        tunnel_0.push_back(tt::DevicePool::instance().get_active_device(tunnels_from_mmio[0][ts]));
    }
    for (uint32_t ts = tunnels_from_mmio[1].size() - 1; ts > 0; ts--) {
        tunnel_1.push_back(tt::DevicePool::instance().get_active_device(tunnels_from_mmio[1][ts]));
    }

    tt::DevicePool::instance().close_devices(tunnel_0);
    for (const auto& dev: tunnel_1) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
    tt::DevicePool::instance().close_devices(tunnel_1);
    tt::DevicePool::instance().close_devices(tunnel_0);
}
