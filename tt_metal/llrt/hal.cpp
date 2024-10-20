// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hal.hpp"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"
#include "tt_metal/third_party/umd/device/tt_arch_types.h"

namespace tt {

namespace tt_metal {

Hal hal;

// This back poitner is a little clunky but necessary at least for now
Hal::Hal() : initialized_(false) {
}

void Hal::initialize(tt::ARCH arch) {

    const std::lock_guard<std::mutex> lock(this->lock);

    if (!this->initialized_) {
        switch (arch) {
        case tt::ARCH::GRAYSKULL:
            initialize_gs();
            break;

        case tt::ARCH::WORMHOLE_B0:
            initialize_wh();
            break;

        case tt::ARCH::BLACKHOLE:
            initialize_bh();
            break;

        default:
            TT_THROW("Unsupported arch for HAL");
        }

        this->initialized_ = true;
    }
}

uint32_t Hal::get_programmable_core_type_index(HalProgrammableCoreType programmable_core_type_index) const {
    uint32_t index = static_cast<uint32_t>(programmable_core_type_index);

    // TODO: this assumes unused entries occur at the end
    // Assumes unused indices go at the end
    if (index >= core_info_.size()) {
        return -1;
    } else {
        return index;
    }
}

std::string_view Hal::get_name(HalProgrammableCoreType core_type) const {
    switch (core_type) {
    case HalProgrammableCoreType::TENSIX:
        return "tensix";
    case HalProgrammableCoreType::ACTIVE_ETH:
        return "active ethernet";
    case HalProgrammableCoreType::IDLE_ETH:
        return "idle ethernet";
    case HalProgrammableCoreType::COUNT:
        TT_ASSERT(0);
        return "";
    }
    TT_ASSERT(0);
    return "";
}

HalCoreInfoType::HalCoreInfoType(HalProgrammableCoreType programmable_core_type,
                                 CoreType core_type,
                                 uint32_t core_proc_count,
                                 const std::vector<DeviceAddr>& mem_map_bases,
                                 const std::vector<uint32_t>& mem_map_sizes,
                                 bool supports_cbs) :
    programmable_core_type_(programmable_core_type),
    core_type_(core_type),
    proc_count_(core_proc_count),
    mem_map_bases_(mem_map_bases),
    mem_map_sizes_(mem_map_sizes),
    supports_cbs_(supports_cbs) {
}

}  // namespace tt_metal
}  // namespace tt
