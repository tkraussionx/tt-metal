// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tests/tt_metal/tt_metal/unit_tests/common/basic_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/dispatch_address_map.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;

TEST_F(FDBasicFixture, DevicePoolUninitialized) {
    std::vector<chip_id_t> device_ids{0};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    ASSERT_ANY_THROW(std::vector<Device *> devices = tt::DevicePool::instance().get_all_devices());
}

TEST_F(FDBasicFixture, DevicePoolOpenClose) {
    std::vector<chip_id_t> device_ids{0};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size);
    std::vector<Device *> devices = tt::DevicePool::instance().get_all_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }

    // Close then get devices again
    for (const auto& dev: devices) {
        dev->close();
    }
    devices = tt::DevicePool::instance().get_all_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
}

TEST_F(FDBasicFixture, DevicePoolAddDevices) {
    std::vector<chip_id_t> device_ids{0};
    int num_hw_cqs = 1;
    int l1_small_size = 1024;
    tt::DevicePool::initialize(device_ids, num_hw_cqs, l1_small_size);
    std::vector<Device *> devices = tt::DevicePool::instance().get_all_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }

    // Close then get devices again
    for (const auto& dev: devices) {
        dev->close();
    }
    devices.clear();
    devices = tt::DevicePool::instance().get_all_devices();
    for (const auto& dev: devices) {
      ASSERT_TRUE((int)(dev->get_l1_small_size()) == l1_small_size);
      ASSERT_TRUE((int)(dev->num_hw_cqs()) == num_hw_cqs);
      ASSERT_TRUE(dev->is_initialized());
    }
}
