/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tt_metal/common/base.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/src/firmware/riscv/grayskull/noc/noc_overlay_parameters.h"

using namespace tt::tt_metal;

struct SystemMemoryCQWriteInterface {
    // Equation for fifo size is
    // | fifo_wr_ptr + command size B - fifo_rd_ptr |
    // Space available would just be fifo_limit - fifo_size
    const uint32_t fifo_size = ((DeviceCommand::HUGE_PAGE_SIZE) - CQ_START) >> 4;
    const uint32_t fifo_limit = ((DeviceCommand::HUGE_PAGE_SIZE) >> 4) - 1;  // Last possible FIFO address

    uint32_t fifo_wr_ptr;
    bool fifo_wr_toggle;
};

inline uint32_t get_cq_rd_ptr(chip_id_t chip_id) {
    vector<uint32_t> recv;
    tt::Cluster::instance().read_sysmem_vec(recv, HOST_CQ_READ_PTR, 4, chip_id);
    return recv[0];
}

class SystemMemoryWriter {
   public:
    char* hugepage_start;
    SystemMemoryCQWriteInterface cq_write_interface;
    SystemMemoryWriter();

    void cq_reserve_back(Device* device, uint32_t cmd_size_B) const {
        uint32_t cmd_size_16B = (((cmd_size_B - 1) | 31) + 1) >> 4; // Terse way to find next multiple of 32 in 16B words
        uint32_t rd_ptr_and_toggle;
        uint32_t rd_ptr;
        uint32_t rd_toggle;
        do {
            rd_ptr_and_toggle = get_cq_rd_ptr(device->id());
            rd_ptr = rd_ptr_and_toggle & 0x7fffffff;
            rd_toggle = rd_ptr_and_toggle >> 31;

        } while (this->cq_write_interface.fifo_wr_ptr < rd_ptr and
                this->cq_write_interface.fifo_wr_ptr + cmd_size_16B > rd_ptr or

                // This is the special case where we wrapped our wr ptr and our rd ptr
                // has not yet moved
                (rd_toggle != this->cq_write_interface.fifo_wr_toggle and this->cq_write_interface.fifo_wr_ptr == rd_ptr));
    }

    // Ideally, data should be an array or pointer, but vector for time-being
    void cq_write(Device* device, const uint32_t* data, uint32_t size, uint32_t write_ptr) const {

        // There is a 50% overhead if hugepage_start is not made static.
        // Eventually when we want to have multiple hugepages, we may need to template
        // the sysmem writer to get this optimization.
        static char* hugepage_start = this->hugepage_start;
        void* user_scratchspace = hugepage_start + write_ptr;
        memcpy(user_scratchspace, data, size);
    }

    void send_write_ptr(Device* device) const {
        static CoreCoord dispatch_core = device->worker_core_from_logical_core(*device->dispatch_cores().begin());

        uint32_t write_ptr_and_toggle = this->cq_write_interface.fifo_wr_ptr | (this->cq_write_interface.fifo_wr_toggle << 31);
        // tt::Cluster::instance().write_dram_vec(&write_ptr_and_toggle, 1, tt_cxy_pair(device->id(), dispatch_core), CQ_WRITE_PTR, true);
        auto dram_core = tt_cxy_pair(device->id(), dispatch_core);
        tt_cxy_pair virtual_dram_core = tt::Cluster::instance().get_soc_desc(device->id()).convert_to_umd_coordinates(dram_core);
        tt::Cluster::instance().device_->write_to_device(&write_ptr_and_toggle, 1, virtual_dram_core, CQ_WRITE_PTR, "LARGE_WRITE_TLB");
        tt_driver_atomics::sfence();
    }

    void cq_push_back(Device* device, uint32_t push_size_B) {
        // All data needs to be 32B aligned
        uint32_t push_size_16B = (((push_size_B - 1) | 31) + 1) >> 4; // Terse way to find next multiple of 32 in 16B words

        this->cq_write_interface.fifo_wr_ptr += push_size_16B;

        if (this->cq_write_interface.fifo_wr_ptr > this->cq_write_interface.fifo_limit) {
            this->cq_write_interface.fifo_wr_ptr = CQ_START >> 4;

            // Flip the toggle
            this->cq_write_interface.fifo_wr_toggle = not this->cq_write_interface.fifo_wr_toggle;
        }

        // Notify dispatch core
        this->send_write_ptr(device);
    }
};
