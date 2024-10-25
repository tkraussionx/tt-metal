// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking that the DPRINT server can detect an invalid core.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt::tt_metal;

TEST(DPrintErrorChecking, TensixTestPrintInvalidCore) {
    // Set DPRINT enabled on a mix of invalid and valid cores. Previously this would hang during
    // device setup, but not the print server should simply ignore the invalid cores.
    std::map<CoreType, std::vector<CoreCoord>> dprint_cores;
    dprint_cores[CoreType::WORKER] = {{0, 0}, {1, 1}, {100, 100}};
    tt::llrt::OptionsG.set_feature_cores(tt::llrt::RunTimeDebugFeatureDprint, dprint_cores);
    tt::llrt::OptionsG.set_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint, true);

    const int device_id = 0;
    Device* device = nullptr;
    const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
    device = tt::tt_metal::CreateDevice(device_id, tt::llrt::OptionsG.get_num_hw_cqs(), DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);

    // We expect that even though illegal worker cores were requested, device setup did not hang.
    // So just make sure that device setup worked and then close the device.
    EXPECT_TRUE(device != nullptr);
    tt::llrt::OptionsG.set_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint, false);
    tt::tt_metal::CloseDevice(device);
}
