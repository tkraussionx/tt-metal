// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "mesh_device_view.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/device/mesh_device_view.hpp"

namespace tt::tt_metal {

using DeviceIds = std::vector<int>;
using MeshDeviceID = std::size_t;
class MeshDeviceView;


class MeshDevice : public std::enable_shared_from_this<MeshDevice> {
    MeshDeviceID mesh_id;
    MeshShape virtual_mesh_shape;
    std::shared_ptr<MeshDeviceView> primary_view;
    std::vector<Device*> devices;

    void initialize(
        size_t l1_small_size,
        size_t trace_region_size,
        size_t num_command_queues,
        DispatchCoreType dispatch_core_type,
        const std::pair<size_t, size_t>& offset);

   public:
    MeshDevice(const MeshShape& virtual_mesh_shape);
    ~MeshDevice();

    MeshDevice(const MeshDevice &) = delete;
    MeshDevice &operator=(const MeshDevice &) = delete;

    MeshDevice(MeshDevice &&) = delete;
    MeshDevice &operator=(MeshDevice &&) = delete;

    std::vector<Device *> get_devices() const;
    Device *get_device(int logical_device_id) const;
    Device *get_device(int row_idx, int col_idx) const;
    std::vector<Device *> get_devices_on_row(int row_idx) const;
    std::vector<Device *> get_devices_on_column(int col_idx) const;

    const DeviceIds get_device_ids() const;

    int num_devices() const;
    int num_rows() const;
    int num_cols() const;
    MeshShape shape() const;

    CoreCoord compute_with_storage_grid_size() const;

    CoreCoord dram_grid_size() const;

    tt::ARCH arch() const;

    void close_devices();
    std::shared_ptr<const MeshDeviceView> get_view() const;
    std::shared_ptr<MeshDeviceView> get_view();

    std::string to_string() const;
    MeshDeviceID get_mesh_id() const;

    static std::shared_ptr<MeshDevice> create(
        const MeshShape& virtual_mesh_shape,
        size_t l1_small_size,
        size_t trace_region_size,
        size_t num_command_queues,
        DispatchCoreType dispatch_core_type,
        const std::pair<size_t, size_t>& offset = {0, 0});
};

std::ostream &operator<<(std::ostream &os, const MeshDevice &mesh_device);
bool validate_worker_modes(const std::vector<Device *> &workers);
std::vector<int> get_t3k_physical_device_ids_ring();

}  // namespace tt::tt_metal
