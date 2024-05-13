// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/detail/tt_metal.hpp"
#include "ttnn/types.hpp"
#include "ttnn/device_pool.hpp"
namespace ttnn {

namespace device {

Device &open_device(int device_id, size_t l1_small_size = DEFAULT_L1_SMALL_SIZE);
void close_device(Device &device);
void enable_program_cache(Device &device);
void disable_and_clear_program_cache(Device &device);
void begin_trace_capture(Device* device, const uint32_t trace_buff_size, const uint8_t cq_id = 0);
void end_trace_capture(Device *device, const uint8_t cq_id = 0);
void execute_trace(Device *device, const uint8_t cq_id = 0, bool blocking = true);
void release_trace(Device* device, const uint8_t cq_id = 0);

}  // namespace device

using namespace device;

}  // namespace ttnn
