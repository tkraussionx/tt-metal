// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/third_party/umd/device/tt_cluster_descriptor.h"
namespace tt {


using Device = tt_metal::Device;
class DevicePool {
   public:
    DevicePool &operator=(const DevicePool &) = delete;
    DevicePool &operator=(DevicePool &&other) noexcept = delete;
    DevicePool(const DevicePool &) = delete;
    DevicePool(DevicePool &&other) noexcept = delete;


		static DevicePool& instance() noexcept {
        TT_ASSERT(_inst != nullptr, "Trying to get DevicePool without initializing it");
        return *_inst;
    }

    static void initialize(
        std::vector<chip_id_t> device_ids,
        const uint8_t num_hw_cqs,
        size_t l1_small_size) noexcept {
      std::cout << " DP initialize " << std::endl;
        static DevicePool device_pool(device_ids, num_hw_cqs, l1_small_size);
        _inst = &device_pool;
    }

    Device* get_active_device(chip_id_t device_id) const;
    std::vector<Device*> get_all_devices() const;
    bool close_device(chip_id_t device_id) const;

   private:
    ~DevicePool();
    DevicePool(std::vector<chip_id_t> device_ids, const uint8_t num_hw_cqs, size_t l1_small_size);
    uint8_t num_hw_cqs;
    size_t l1_small_size;
    std::mutex lock;
    std::vector<std::unique_ptr<Device>> devices;

    void activate_device(chip_id_t id);
    void deactivate_device(chip_id_t id);
    bool is_device_active(chip_id_t id) const;
    void initialize_device_after_close(Device *dev) const;
    static DevicePool* _inst;

};


}  // namespace tt
