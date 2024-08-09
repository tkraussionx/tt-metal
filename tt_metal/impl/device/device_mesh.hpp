// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/device/device_mesh_view.hpp"

namespace tt::tt_metal {

using DeviceGrid = std::pair<int, int>;
using DeviceIds = std::vector<int>;

constexpr size_t DEFAULT_NUM_COMMAND_QUEUES = 1;

class DeviceMesh {
   public:
    DeviceMesh(
        const DeviceGrid &device_grid,
        const DeviceIds &device_ids,
        std::size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        std::size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        std::size_t num_command_queues = DEFAULT_NUM_COMMAND_QUEUES);
    ~DeviceMesh();

    // Deleted copy and move constructors/assignments
    DeviceMesh(const DeviceMesh &) = delete;
    DeviceMesh &operator=(const DeviceMesh &) = delete;
    DeviceMesh(DeviceMesh &&) = delete;
    DeviceMesh &operator=(DeviceMesh &&) = delete;

    // Get device grid information
    std::vector<Device *> get_devices() const;
    Device *get_device(int logical_device_id) const;
    Device *get_device(int row_idx, int col_idx) const;
    std::vector<Device *> get_devices_on_row(int row_idx) const;
    std::vector<Device *> get_devices_on_column(int col_idx) const;

    std::optional<Coordinate> find_device(int device_id) const;

    const DeviceIds get_device_ids() const;

    // Get device grid information
    int num_devices() const;
    int num_rows() const;
    int num_cols() const;
    DeviceGrid shape() const;

    // Convenience methods (assume homogeneous devices)
    CoreCoord compute_with_storage_grid_size() const;
    CoreCoord dram_grid_size() const;
    tt::ARCH arch() const;

    // Device-specific operations
    void close_devices();
    void enable_async(bool enable);

   private:
    DeviceGrid device_grid;
    DeviceMeshView view;

    std::map<chip_id_t, Device *> managed_devices;
    std::vector<std::pair<int, Device *>> mesh_devices;
};

bool validate_worker_modes(const std::vector<Device *> &workers);

}  // namespace tt::tt_metal
