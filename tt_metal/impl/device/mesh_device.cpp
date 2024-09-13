// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/mesh_device.hpp"

#include <memory>
#include <unordered_map>

#include "device/tt_cluster_descriptor_types.h"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/mesh_device_view.hpp"

namespace tt::tt_metal {

using LogicalCoordinate = Coordinate;
using PhysicalCoordinate = eth_coord_t;

static std::string get_config_path(const std::string& filename) {
    std::string root_path = getenv("TT_METAL_HOME") ? getenv("TT_METAL_HOME") : "./";
    return root_path + "/tt_metal/impl/device/mesh_configurations/" + filename;
}

static std::map<LogicalCoordinate, PhysicalCoordinate> load_translation_map(const std::string& filename, const std::string& key) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    nlohmann::json j;
    try {
        file >> j;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("JSON parsing error in file " + filename + ": " + e.what());
    }

    if (!j.contains(key)) {
        throw std::runtime_error("Key '" + key + "' not found in JSON file: " + filename);
    }

    std::map<LogicalCoordinate, PhysicalCoordinate> result;
    for (const auto& mapping : j[key]) {
        if (mapping.size() != 2 || mapping[0].size() != 2 || mapping[1].size() != 4) {
            throw std::runtime_error("Invalid coordinate format in JSON file: " + filename);
        }
        result.emplace(LogicalCoordinate{mapping[0][0], mapping[0][1]}, PhysicalCoordinate{mapping[1][1], mapping[1][0], mapping[1][2], mapping[1][3]});
    }

    return result;
}

class LogicalMesh {
    std::unordered_map<MeshDeviceID, std::vector<chip_id_t>> assigned_devices;
    std::unordered_map<MeshDeviceID, std::shared_ptr<MeshDevice>> assigned_virtual_mesh_devices;
    std::unordered_map<chip_id_t, Device*> assigned_physical_id_to_device;

    // Logical mesh shape and coordinates
    MeshShape logical_mesh_shape;
    std::map<LogicalCoordinate, PhysicalCoordinate> logical_to_physical_coordinates;

    // Handling of physical coordinates
    std::unordered_map<PhysicalCoordinate, chip_id_t> physical_coordinate_to_device_id;
    std::unordered_map<chip_id_t, PhysicalCoordinate> physical_device_id_to_coordinate;

    LogicalMesh();
    LogicalMesh(const LogicalMesh&) = delete;
    LogicalMesh& operator=(const LogicalMesh&) = delete;
    LogicalMesh(LogicalMesh&&) = delete;
    LogicalMesh& operator=(LogicalMesh&&) = delete;

    // For now, simple way to figure out logical shape because we only have a few supported system configurations:
    const std::unordered_map<std::size_t, MeshShape> num_devices_to_logical_shape = {
        {1, MeshShape{1, 1}},   // single-device
        {2, MeshShape{1, 2}},   // N300
        {8, MeshShape{2, 4}},   // T3000; as ring to match existing tests
        {32, MeshShape{8, 4}},  // TG
        {64, MeshShape{8, 8}},  // TGG
    };
    const std::unordered_map<std::size_t, std::string> num_devices_to_translation_map = {
        {1, "single_device.json"},
        {2, "N300.json"},
        {8, "T3000.json"},
        {32, "TG.json"},
        {64, "TGG.json"},
    };

   public:
    static LogicalMesh& instance() {
        static LogicalMesh instance;
        return instance;
    }
    const MeshShape& get_shape() const;
    std::vector<Device*> map_virtual_mesh(
        std::shared_ptr<MeshDevice> virtual_mesh,
        size_t num_command_queues,
        size_t l1_small_size,
        size_t trace_region_size,
        DispatchCoreType dispatch_core_type,
        const std::pair<size_t, size_t>& offset = {0, 0});
    void unmap_virtual_mesh(const std::shared_ptr<MeshDevice>& virtual_mesh);
};

LogicalMesh::LogicalMesh() {
    std::size_t min_x, min_y, max_x, max_y;
    min_x = min_y = std::numeric_limits<std::size_t>::max();
    max_x = max_y = std::numeric_limits<std::size_t>::min();
    auto physical_coordinate_to_device_id = tt::Cluster::instance().get_user_chip_ethernet_coordinates();
    auto num_devices = physical_coordinate_to_device_id.size();
    TT_FATAL(
        this->num_devices_to_logical_shape.contains(num_devices), "Unsupported number of devices: {}", num_devices);
    this->logical_mesh_shape = this->num_devices_to_logical_shape.at(num_devices);

    std::set<std::size_t> shelf_ids;
    for (const auto& [chip_id, physical_coordinate] : physical_coordinate_to_device_id) {
        this->physical_coordinate_to_device_id[physical_coordinate] = chip_id;
        this->physical_device_id_to_coordinate[chip_id] = physical_coordinate;
        shelf_ids.insert(std::get<3>(physical_coordinate));
    }

    log_info("logical mesh shape: {}x{}", this->logical_mesh_shape.first, this->logical_mesh_shape.second);
}

const MeshShape& LogicalMesh::get_shape() const { return this->logical_mesh_shape; }

std::vector<Device*> LogicalMesh::map_virtual_mesh(
    std::shared_ptr<MeshDevice> virtual_mesh,
    size_t num_command_queues,
    size_t l1_small_size,
    size_t trace_region_size,
    DispatchCoreType dispatch_core_type,
    const std::pair<size_t, size_t>& offset) {

    auto [requested_num_rows, requested_num_cols] = virtual_mesh->shape();
    auto [max_num_rows, max_num_cols] = this->logical_mesh_shape;
    TT_FATAL(requested_num_rows <= max_num_rows, "Requested too many rows: {} > {}", requested_num_rows, max_num_rows);
    TT_FATAL(requested_num_rows*requested_num_cols <= max_num_rows*max_num_cols, "Requested submesh is too big: {}x{}", requested_num_rows, requested_num_cols);

    this->logical_to_physical_coordinates = load_translation_map(
        get_config_path(num_devices_to_translation_map.at(max_num_rows * max_num_cols)),
        "logical_to_physical_coordinates");
    // Print logical_to_physical_coordinate entries
    log_info("Logical to Physical Coordinate Mapping:");
    for (const auto& [logical, physical] : this->logical_to_physical_coordinates) {
        log_info("Logical ({}, {}) -> Physical ({}, {}, {}, {})",
                 logical.row, logical.col,
                 std::get<0>(physical), std::get<1>(physical), std::get<2>(physical), std::get<3>(physical));
    }
    // Print physical_coordinate to device_id mapping
    log_info("Physical Coordinate to Device ID Mapping:");
    for (const auto& [physical_coordinate, device_id] : this->physical_coordinate_to_device_id) {
        log_info("Physical ({}, {}, {}, {}) -> Device ID {}",
                 std::get<0>(physical_coordinate), std::get<1>(physical_coordinate),
                 std::get<2>(physical_coordinate), std::get<3>(physical_coordinate),
                 device_id);
    }

    this->assigned_virtual_mesh_devices.insert({virtual_mesh->get_mesh_id(), virtual_mesh});

    std::vector<chip_id_t> physical_device_ids;
    auto [row_offset, col_offset] = offset;
    log_info("Mapping virtual mesh with offset: {}, {}", row_offset, col_offset);
    log_info("Requested mesh shape: {}x{}", requested_num_rows, requested_num_cols);
    for (int row = 0; row < requested_num_rows; row++) {
        for (int col = 0; col < requested_num_cols; col++) {
            auto logical_device_id = (row + row_offset) * max_num_cols + (col + col_offset);
            auto logical_coordinate = Coordinate{logical_device_id / max_num_cols, logical_device_id % max_num_cols};
            auto physical_coordinate = this->logical_to_physical_coordinates.at(logical_coordinate);
            auto physical_device_id = this->physical_coordinate_to_device_id.at(physical_coordinate);
            physical_device_ids.push_back(physical_device_id);

            log_info("Logical device ID: {}, Logical coordinate: {}, Physical coordinate: {}, Physical device ID: {}",
                     logical_device_id, logical_coordinate, physical_coordinate, physical_device_id);

        }
    }

    // TODO(jchu): need to double-check whether we need to add gateway devices here..
    auto physical_id_to_device = tt::tt_metal::detail::CreateDevices(
        physical_device_ids, num_command_queues, l1_small_size, trace_region_size, dispatch_core_type);

    std::vector<Device*> mesh_devices;
    for (auto physical_device_id : physical_device_ids) {
        mesh_devices.push_back(physical_id_to_device.at(physical_device_id));
        this->assigned_devices[virtual_mesh->get_mesh_id()].push_back(physical_device_id);
        this->assigned_physical_id_to_device.insert({physical_device_id, mesh_devices.back()});
    }
    return mesh_devices;
}

void LogicalMesh::unmap_virtual_mesh(const std::shared_ptr<MeshDevice>& virtual_mesh) {
    auto mesh_id = virtual_mesh->get_mesh_id();

    // Construct the map of devices to close
    std::map<chip_id_t, Device*> devices_to_close;
    for (const auto& physical_id : this->assigned_devices[mesh_id]) {
        auto it = this->assigned_physical_id_to_device.find(physical_id);
        if (it != this->assigned_physical_id_to_device.end()) {
            devices_to_close[physical_id] = it->second;
        }
    }

    // Close the devices
    tt::tt_metal::detail::CloseDevices(devices_to_close);

    // Clean up all state related to this virtual mesh
    this->assigned_devices.erase(mesh_id);
    this->assigned_virtual_mesh_devices.erase(mesh_id);

    // Remove the devices from assigned_physical_id_to_device
    for (const auto& [physical_id, device] : devices_to_close) {
        this->assigned_physical_id_to_device.erase(physical_id);
    }
}

static MeshDeviceID generate_unique_mesh_id() {
    static std::atomic<MeshDeviceID> next_id{0};
    return next_id++;
}

MeshDevice::MeshDevice(const MeshShape& virtual_mesh_shape) : virtual_mesh_shape(virtual_mesh_shape), mesh_id(generate_unique_mesh_id()) {}

std::shared_ptr<MeshDevice> MeshDevice::create(
    const MeshShape& virtual_mesh_shape,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    DispatchCoreType dispatch_core_type,
    const std::pair<size_t, size_t>& offset)
{
    auto mesh_device = std::make_shared<MeshDevice>(virtual_mesh_shape);
    mesh_device->initialize(l1_small_size, trace_region_size, num_command_queues, dispatch_core_type, offset);

    return mesh_device;
}

void MeshDevice::initialize(
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    DispatchCoreType dispatch_core_type,
    const std::pair<size_t, size_t>& offset)
{
    auto [num_rows, num_cols] = this->shape();
    auto num_requested_devices = num_rows * num_cols;
    auto num_available_devices = tt::tt_metal::GetNumAvailableDevices();
    TT_FATAL(
        num_requested_devices <= num_available_devices,
        fmt::format(
            "User has requested more devices than available: {} requested, {} available",
            num_requested_devices,
            num_available_devices));

    std::cout << "Instance: " << std::endl;
    auto& instance = LogicalMesh::instance();
    std::cout << "Done instance: " << std::endl;

    std::cout << "Mapping virtual mesh: " << std::endl;
    this->devices = instance.map_virtual_mesh(
        shared_from_this(), num_command_queues, l1_small_size, trace_region_size, dispatch_core_type, offset);
    std::cout << "Done mapping virtual mesh: " << std::endl;
    this->primary_view = std::make_unique<tt::tt_metal::MeshDeviceView>(*this);
}

MeshDevice::~MeshDevice() {
    if (not this->devices.empty()) {
        this->close_devices();
    }
}

Device* MeshDevice::get_device(int logical_device_id) const {
    TT_FATAL(logical_device_id >= 0 and logical_device_id < num_devices(), "Invalid device index");
    return this->devices.at(logical_device_id);
}

std::vector<Device*> MeshDevice::get_devices() const { return this->devices; }

Device* MeshDevice::get_device(int row_idx, int col_idx) const {
    return this->get_device(row_idx * num_cols() + col_idx);
}

std::vector<Device*> MeshDevice::get_devices_on_row(int row_idx) const {
    return this->primary_view->get_devices_on_row(row_idx);
}

std::vector<Device*> MeshDevice::get_devices_on_column(int col_idx) const {
    return this->primary_view->get_devices_on_column(col_idx);
}

const DeviceIds MeshDevice::get_device_ids() const {
    DeviceIds device_ids;
    for (auto device : this->get_devices()) {
        device_ids.push_back(device->id());
    }
    return device_ids;
}

int MeshDevice::num_devices() const { return num_rows() * num_cols(); }

CoreCoord MeshDevice::compute_with_storage_grid_size() const { return get_device(0)->compute_with_storage_grid_size(); }

CoreCoord MeshDevice::dram_grid_size() const { return get_device(0)->dram_grid_size(); }

tt::ARCH MeshDevice::arch() const { return get_device(0)->arch(); }

int MeshDevice::num_rows() const { return this->virtual_mesh_shape.first; }

int MeshDevice::num_cols() const { return this->virtual_mesh_shape.second; }

MeshShape MeshDevice::shape() const { return this->virtual_mesh_shape; }

void MeshDevice::close_devices() {
    LogicalMesh::instance().unmap_virtual_mesh(shared_from_this());
    this->devices.clear();
}

std::string MeshDevice::to_string() const {
    return fmt::format("MeshDevice({}x{} grid, {} devices)", this->num_rows(), this->num_cols(), this->num_devices());
}

std::shared_ptr<const MeshDeviceView> MeshDevice::get_view() const { return this->primary_view; }

std::shared_ptr<MeshDeviceView> MeshDevice::get_view() { return this->primary_view; }

MeshDeviceID MeshDevice::get_mesh_id() const { return this->mesh_id; }

std::ostream& operator<<(std::ostream& os, const MeshDevice& mesh_device) { return os << mesh_device.to_string(); }

bool validate_worker_modes(const std::vector<Device*>& workers) {
    bool worker_modes_match = true;
    auto first_worker_mode = workers.at(0)->get_worker_mode();
    for (auto worker : workers) {
        worker_modes_match &= (worker->get_worker_mode() == first_worker_mode);
    }
    return worker_modes_match;
}

std::vector<int> get_t3k_physical_device_ids_ring() {
    auto virtual_mesh_shape = MeshShape{2, 4};
    // todo
    return {};
}

}  // namespace tt::tt_metal
