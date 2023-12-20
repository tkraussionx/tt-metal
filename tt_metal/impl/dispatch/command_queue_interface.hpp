// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/base.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/llrt/llrt.hpp"

using namespace tt::tt_metal;

inline uint32_t get_cq_rd_ptr(chip_id_t chip_id) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    tt::Cluster::instance().read_sysmem(&recv, sizeof(uint32_t), HOST_CQ_READ_PTR, mmio_device_id, channel);
    return recv;
}

struct CQWriteInterface {
    // Equation for fifo size is
    // | fifo_wr_ptr + command size B - fifo_rd_ptr |
    // Space available would just be fifo_limit - fifo_size
    const uint32_t fifo_size;
    const uint32_t fifo_limit;  // Last possible FIFO address
    uint32_t fifo_wr_ptr;
    bool fifo_wr_toggle;

    CQWriteInterface(const uint32_t fifo_size, const uint32_t fifo_limit, const uint32_t fifo_wr_ptr): fifo_size(fifo_size), fifo_limit(fifo_limit) {
        this->fifo_wr_ptr = fifo_wr_ptr;  // In 16B words
        this->fifo_wr_toggle =
            0;  // This is used for the edge case where we wrap and our read pointer has not yet moved
    }
};


class CommandQueueWriter {
   private:
    chip_id_t device_id;
    const uint32_t m_dma_buf_size;
    const std::function<void(uint32_t, uint32_t, const uint8_t*, uint32_t)> fast_write_callable;
    const std::function<CoreCoord (CoreCoord)> worker_from_logical_callable;
    vector<uint32_t> producer_core_wr_ptr_addrs;
    char* hugepage_start;
    vector<CQWriteInterface> cq_write_interfaces;

   public:
    CommandQueueWriter(chip_id_t device_id, const std::set<CoreCoord>& producer_cores, const std::function<CoreCoord (CoreCoord)> &worker_from_logical) :
        device_id(device_id),
        m_dma_buf_size(tt::Cluster::instance().get_m_dma_buf_size(device_id)),
        hugepage_start(
            (char*) tt::Cluster::instance().host_dma_address(0, tt::Cluster::instance().get_associated_mmio_device(device_id), tt::Cluster::instance().get_assigned_channel_for_device(device_id))),
        fast_write_callable(
            tt::Cluster::instance().get_fast_pcie_static_tlb_write_callable(device_id)),
        worker_from_logical_callable(worker_from_logical) {

        TT_ASSERT(producer_cores.size() < 3, "Can have at most 2 hardware CQs");
        TT_ASSERT(producer_cores.size() > 0, "Need to have at least one producer");
        size_t num_hw_cqs = producer_cores.size();

        const uint32_t fifo_size = DeviceCommand::HUGE_PAGE_SIZE / num_hw_cqs;
        auto producer_core_iterator = producer_cores.begin();
        for (uint32_t i = 0; i < num_hw_cqs; i++) {
            const std::tuple<uint32_t, uint32_t> tlb_data = tt::Cluster::instance().get_tlb_data(tt_cxy_pair(device_id, this->worker_from_logical_callable(*producer_core_iterator))).value();
            auto [tlb_offset, tlb_size] = tlb_data;
            this->producer_core_wr_ptr_addrs.push_back(tlb_offset + CQ_WRITE_PTR % tlb_size);
            uint32_t fifo_limit = (i + 1) * fifo_size - 1;
            uint32_t fifo_wr_ptr = (i * fifo_size + CQ_WRITE_PTR) >> 4;
            cq_write_interfaces.push_back(CQWriteInterface(fifo_size, fifo_limit, fifo_wr_ptr));
            producer_core_iterator++;
        }
    }

    void cq_reserve_back(const uint32_t cq_channel, uint32_t cmd_size_B) const {
        uint32_t cmd_size_16B = align(cmd_size_B, 32);

        uint32_t rd_ptr_and_toggle;
        uint32_t rd_ptr;
        uint32_t rd_toggle;
        const CQWriteInterface& cq_write_interface = this->cq_write_interfaces[cq_channel];
        do {
            rd_ptr_and_toggle = get_cq_rd_ptr(this->device_id);
            rd_ptr = rd_ptr_and_toggle & 0x7fffffff;
            rd_toggle = rd_ptr_and_toggle >> 31;

        } while (
            cq_write_interface
                .fifo_wr_ptr < rd_ptr and cq_write_interface.fifo_wr_ptr + cmd_size_16B> rd_ptr or

            // This is the special case where we wrapped our wr ptr and our rd ptr
            // has not yet moved
            (rd_toggle != cq_write_interface.fifo_wr_toggle and cq_write_interface.fifo_wr_ptr == rd_ptr));
    }

    void cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr) const {
        void* user_scratchspace = this->hugepage_start + write_ptr;
        memcpy(user_scratchspace, data, size_in_bytes);
    }

    uint32_t get_wr_ptr(const uint32_t cq_channel) const {
        return this->cq_write_interfaces[cq_channel].fifo_wr_ptr << 4;
    }

    void send_write_ptr(const uint32_t cq_channel) const {
        const CQWriteInterface& cq_write_interface = this->cq_write_interfaces[cq_channel];
        uint32_t write_ptr_and_toggle =
            cq_write_interface.fifo_wr_ptr | (cq_write_interface.fifo_wr_toggle << 31);
        this->fast_write_callable(this->producer_core_wr_ptr_addrs[cq_channel], 4, (uint8_t*)&write_ptr_and_toggle, this->m_dma_buf_size);
        tt_driver_atomics::sfence();
    }

    void cq_push_back(const uint32_t cq_channel, uint32_t push_size_B) {
        // All data needs to be 32B aligned
        uint32_t push_size_16B = align(push_size_B, 32);

        CQWriteInterface& cq_write_interface = this->cq_write_interfaces[cq_channel];
        cq_write_interface.fifo_wr_ptr += push_size_16B;
        if (cq_write_interface.fifo_wr_ptr > cq_write_interface.fifo_limit) {
            cq_write_interface.fifo_wr_ptr = CQ_START >> 4;

            // Flip the toggle
            cq_write_interface.fifo_wr_toggle = not cq_write_interface.fifo_wr_toggle;
        }

        // Notify dispatch core
        this->send_write_ptr(cq_channel);
    }
};
