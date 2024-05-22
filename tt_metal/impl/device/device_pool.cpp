// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/device_pool.hpp"
#include "tt_metal/detail/tt_metal.hpp"

namespace tt {

DevicePool* DevicePool::_inst = nullptr;

void DevicePool::initialize_device(Device *dev) const {
    //TODO: temp, all these can be removed
    dev->initialize_and_launch_firmware();

    // Create system memory writer for this device to have an associated interface to hardware command queue (i.e. hugepage)
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        detail::DispatchStateCheck(true);
        dev->initialize_command_queue();
    } else {
        detail::DispatchStateCheck(false);
        dev->initialize_synchronous_sw_cmd_queue();
        TT_ASSERT(dev->num_hw_cqs() == 1, "num_hw_cqs must be 1 in slow dispatch");
    }
    detail::InitDeviceProfiler(dev);
}

void DevicePool::activate_device(chip_id_t id) {

    const std::lock_guard<std::mutex> lock(this->lock);
    if (this->devices.size() < id + 1) {
       this->devices.resize(id + 1);
    }
    if (this->devices[id] == nullptr) {
      std::cout << " DP activate device " << id << std::endl;
        auto dev = new Device(id, this->num_hw_cqs, this->l1_small_size, this->l1_bank_remap);
        dev->build_firmware();
        this->initialize_device(dev);
        this->devices[id] = std::unique_ptr<Device>(dev);

    } else {
      const auto& dev = this->devices[id];
      std::cout << " DP re-init device " << id << std::endl;
      if (not dev->is_initialized()) {
          dev->initialize(num_hw_cqs, this->l1_small_size, this->l1_bank_remap);
          this->initialize_device(dev.get());
      } else {
          TT_THROW("Cannot re-initialize device {}, must first call close()", id);
      }
    }

}

bool DevicePool::is_device_active(chip_id_t id) const {
    if (this->devices.size() < id + 1 || this->devices[id] == nullptr) {
        return false;
    } else {
        return this->devices[id]->state() == ActiveState::ACTIVE;
    }
}

void DevicePool::add_devices_to_pool(std::vector<chip_id_t> device_ids, const uint8_t num_hw_cqs, size_t l1_small_size, const std::vector<uint32_t> &l1_bank_remap) {
    this->l1_small_size = l1_small_size;
    this->num_hw_cqs = num_hw_cqs;
    this->l1_bank_remap = l1_bank_remap;
    for (const auto& device_id : device_ids) {
        const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        for (const auto& mmio_controlled_device_id :
             tt::Cluster::instance().get_devices_controlled_by_mmio_device(mmio_device_id)) {
          std::cout << " Checking device " << mmio_controlled_device_id << std::endl;
            if (not this->is_device_active(mmio_controlled_device_id)) {
                this->activate_device(mmio_controlled_device_id);
            }
        }
    }
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
}

DevicePool::DevicePool(std::vector<chip_id_t> device_ids, const uint8_t num_hw_cqs, size_t l1_small_size, const std::vector<uint32_t> &l1_bank_remap) {
  std::cout << " device pool ctor  " << std::endl;
    ZoneScoped;
    this->add_devices_to_pool(device_ids, num_hw_cqs, l1_small_size, l1_bank_remap);
}

Device* DevicePool::get_active_device(chip_id_t device_id) const {
    TT_ASSERT(
        this->is_device_active(device_id), "DevicePool does not contain active device {}", device_id);
    std::cout << " Get active device " << this->devices[device_id]->id() << std::endl;
    return this->devices[device_id].get();
}

std::vector<Device*> DevicePool::get_all_active_devices() const {
    std::vector<Device*> user_devices;
    for (int id=0; id < this->devices.size(); id++) {
      if(this->is_device_active(id)) {
        user_devices.emplace_back(this->devices[id].get());
      }
    }
    return user_devices;
}

bool DevicePool::close_device(chip_id_t device_id) const {
  bool pass = true;
  const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
  for (const auto& mmio_controlled_device_id :
       tt::Cluster::instance().get_devices_controlled_by_mmio_device(mmio_device_id)) {
    auto device = this->get_active_device(mmio_controlled_device_id);
    pass &= device->close();
  }
  return pass;
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
