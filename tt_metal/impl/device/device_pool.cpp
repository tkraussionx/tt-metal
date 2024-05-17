// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/device_pool.hpp"
#include "tt_metal/detail/tt_metal.hpp"

namespace tt {
/*
ActiveDevices::~ActiveDevices() {
    for (size_t i = 0; i < devices_.size(); i++) {
        if (devices_[i] == ActiveState::ACTIVE) {
            TT_THROW("Process tear down with device {} still active", i);
        }
    }
}*/

void DevicePool::activate_device(chip_id_t id) {
    const std::lock_guard<std::mutex> lock(this->lock);
    if (this->devices.size() < id + 1) {
       this->devices.resize(id + 1);
    }
    if (this->devices[id] == nullptr) {
        auto dev = new Device(id, this->num_hw_cqs, this->l1_small_size);
        this->devices[id] = std::unique_ptr<Device>(dev);
        detail::InitDeviceProfiler(dev);
    } else if (this->devices[id]->state() == ActiveState::ACTIVE) {
        TT_THROW("Cannot re-initialize device {}, must first call close()", id);
    }
}

void DevicePool::deactivate_device(chip_id_t id) {
    const std::lock_guard<std::mutex> lock(this->lock);
   // this->devices[id] = ActiveState::INACTIVE;
}

bool DevicePool::is_device_active(chip_id_t id) const {
    if (this->devices.size() < id + 1) {
        return false;
    } else {
        return this->devices[id]->state() == ActiveState::ACTIVE;
    }
}

const DevicePool& DevicePool::instance(std::vector<chip_id_t> device_ids, const uint8_t num_hw_cqs, size_t l1_small_size) {
    static DevicePool device_pool(device_ids, num_hw_cqs, l1_small_size);
    return device_pool;
}

DevicePool::DevicePool(std::vector<chip_id_t> device_ids, const uint8_t num_hw_cqs, size_t l1_small_size) {
    ZoneScoped;
    this->l1_small_size = l1_small_size;
    this->num_hw_cqs = num_hw_cqs;
    for (const auto& device_id : device_ids) {
        const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        for (const auto& mmio_controlled_device_id :
             tt::Cluster::instance().get_devices_controlled_by_mmio_device(mmio_device_id)) {
            if (not this->is_device_active(mmio_controlled_device_id)) {
                this->activate_device(mmio_controlled_device_id);
            }
        }
    }
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
}

Device* DevicePool::get_device(chip_id_t device_id) const {
    TT_ASSERT(
        this->is_device_active(device_id), "DevicePool does not contain active device {}", device_id);
    Device* dev = this->devices[device_id].get();
    if (not dev->is_initialized()) {
        dev->initialize(this->l1_small_size);
    }
    return dev;
}

std::vector<Device*> DevicePool::get_all_devices() const {
    std::vector<Device*> user_devices;
    for (const auto& dev : this->devices) {
        if (dev != nullptr) {
            if (not dev->is_initialized()) {
                dev->initialize(this->l1_small_size);
            }
            user_devices.emplace_back(dev.get());
        }
    }
    return user_devices;
}

bool DevicePool::close_device(chip_id_t device_id) const {
    auto device = this->get_device(device_id);
    return device->close();
}

DevicePool::~DevicePool() {
  std::cout << " Device pool destructor " << std::endl;
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
    // TODO: should this be done explicitly here?

    for (const auto& dev : this->devices) {
          if (dev != nullptr and dev->is_initialized()) {
              std::cout << " calling device close " << std::endl;
              dev->close();
        }
    }
    std::cout << " clearing " << std::endl;
    this->devices.clear();
}

}  // namespace tt
