// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <mutex>

#include "tt_metal/common/base.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/dispatch_address_map.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/common/math.hpp"

using namespace tt::tt_metal;

// todo consider moving these to dispatch_addr_map
static constexpr uint32_t PCIE_ALIGNMENT = 32;
static constexpr uint32_t MAX_HUGEPAGE_SIZE = 1 << 30; // 1GB;

struct dispatch_constants {
   public:
    dispatch_constants &operator=(const dispatch_constants &) = delete;
    dispatch_constants &operator=(dispatch_constants &&other) noexcept = delete;
    dispatch_constants(const dispatch_constants &) = delete;
    dispatch_constants(dispatch_constants &&other) noexcept = delete;

    static const dispatch_constants &get(const CoreType &core_type) {
        static dispatch_constants inst = dispatch_constants(core_type);
        return inst;
    }

    typedef uint32_t prefetch_q_entry_type;
    static constexpr uint32_t PREFETCH_Q_LOG_MINSIZE = 4;
    static constexpr uint32_t PREFETCH_Q_ENTRIES = 128;
    static constexpr uint32_t PREFETCH_Q_SIZE = PREFETCH_Q_ENTRIES * sizeof(prefetch_q_entry_type);
    static constexpr uint32_t PREFETCH_Q_BASE = DISPATCH_L1_UNRESERVED_BASE;

    static constexpr uint32_t CMDDAT_Q_BASE = PREFETCH_Q_BASE + ((PREFETCH_Q_SIZE + PCIE_ALIGNMENT - 1) / PCIE_ALIGNMENT * PCIE_ALIGNMENT);

    static constexpr uint32_t LOG_TRANSFER_PAGE_SIZE = 12;
    static constexpr uint32_t TRANSFER_PAGE_SIZE = 1 << LOG_TRANSFER_PAGE_SIZE;

    static constexpr uint32_t DISPATCH_BUFFER_LOG_PAGE_SIZE = 12;
    static constexpr uint32_t DISPATCH_BUFFER_SIZE_BLOCKS = 4;
    static constexpr uint32_t DISPATCH_BUFFER_BASE = ((DISPATCH_L1_UNRESERVED_BASE - 1) | ((1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) - 1)) + 1;

    static constexpr uint32_t PREFETCH_D_BUFFER_SIZE = 256 * 1024;
    static constexpr uint32_t PREFETCH_D_BUFFER_LOG_PAGE_SIZE = 12;
    static constexpr uint32_t PREFETCH_D_BUFFER_BLOCKS = 4;
    static constexpr uint32_t PREFETCH_D_BUFFER_PAGES = PREFETCH_D_BUFFER_SIZE >> PREFETCH_D_BUFFER_LOG_PAGE_SIZE;

    static constexpr uint32_t EVENT_PADDED_SIZE = 16;
    // When page size of buffer to write/read exceeds MAX_PREFETCH_COMMAND_SIZE, the PCIe aligned page size is broken down into equal sized partial pages
    // BASE_PARTIAL_PAGE_SIZE denotes the initial partial page size to use, it is incremented by PCIe alignment until page size can be evenly split
    static constexpr uint32_t BASE_PARTIAL_PAGE_SIZE = 4096;

    uint32_t max_prefetch_command_size() const { return max_prefetch_command_size_; }

    uint32_t cmddat_q_size() const { return cmddat_q_size_; }

    uint32_t scratch_db_base() const { return scratch_db_base_; }

    uint32_t scratch_db_size() const { return scratch_db_size_; }

    uint32_t dispatch_buffer_block_size_pages() const { return dispatch_buffer_block_size_pages_; }

    uint32_t dispatch_buffer_pages() const { return dispatch_buffer_pages_; }

   private:
    dispatch_constants(const CoreType &core_type) {
        TT_ASSERT(core_type == CoreType::WORKER or core_type == CoreType::ETH);
        // make this 2^N as required by the packetized stages
        uint32_t dispatch_buffer_block_size;
        if (core_type == CoreType::WORKER) {
            max_prefetch_command_size_ = 64 * 1024;
            cmddat_q_size_ = 128 * 1024;
            scratch_db_size_ = 128 * 1024;
            dispatch_buffer_block_size = 512 * 1024;
        } else {
            max_prefetch_command_size_ = 32 * 1024;
            cmddat_q_size_ = 64 * 1024;
            scratch_db_size_ = 64 * 1024;
            dispatch_buffer_block_size = 128 * 1024;
        }
        TT_ASSERT(cmddat_q_size_ >= 2 * max_prefetch_command_size_);
        TT_ASSERT(scratch_db_size_ % 2 == 0);
        TT_ASSERT((dispatch_buffer_block_size & (dispatch_buffer_block_size - 1)) == 0);
        scratch_db_base_ = CMDDAT_Q_BASE + ((cmddat_q_size_ + PCIE_ALIGNMENT - 1) / PCIE_ALIGNMENT * PCIE_ALIGNMENT);
        uint32_t l1_size = core_type == CoreType::WORKER ? MEM_L1_SIZE : MEM_ETH_SIZE;
        TT_ASSERT(scratch_db_base_ + scratch_db_size_ < l1_size);
        dispatch_buffer_block_size_pages_ = dispatch_buffer_block_size / (1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) / DISPATCH_BUFFER_SIZE_BLOCKS;
        dispatch_buffer_pages_ = dispatch_buffer_block_size_pages_ * DISPATCH_BUFFER_SIZE_BLOCKS;
        uint32_t dispatch_cb_end = DISPATCH_BUFFER_BASE + (1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) * dispatch_buffer_pages_;
        TT_ASSERT(dispatch_cb_end < l1_size);
    }

    uint32_t max_prefetch_command_size_;
    uint32_t cmddat_q_size_;
    uint32_t scratch_db_base_;
    uint32_t scratch_db_size_;
    uint32_t dispatch_buffer_block_size_pages_;
    uint32_t dispatch_buffer_pages_;
};

/// @brief Get offset of the command queue relative to its channel
/// @param cq_id uint8_t ID the command queue
/// @param cq_size uint32_t size of the command queue
/// @return uint32_t relative offset
inline uint32_t get_relative_cq_offset(uint8_t cq_id, uint32_t cq_size) {
    return cq_id * cq_size;
}

/// @brief Get absolute offset of the command queue
/// @param channel uint16_t channel ID (hugepage)
/// @param cq_id uint8_t ID the command queue
/// @param cq_size uint32_t size of the command queue
/// @return uint32_t absolute offset
inline uint32_t get_absolute_cq_offset(uint16_t channel, uint8_t cq_id, uint32_t cq_size) {
    return (MAX_HUGEPAGE_SIZE * channel) + get_relative_cq_offset(cq_id, cq_size);
}

template <bool addr_16B>
inline uint32_t get_cq_issue_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    tt::Cluster::instance().read_sysmem(&recv, sizeof(uint32_t), HOST_CQ_ISSUE_READ_PTR + get_relative_cq_offset(cq_id, cq_size), mmio_device_id, channel);
    if (not addr_16B) {
        return recv << 4;
    }
    return recv;
}

template <bool addr_16B>
inline uint32_t get_cq_completion_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    tt::Cluster::instance().read_sysmem(&recv, sizeof(uint32_t), HOST_CQ_COMPLETION_WRITE_PTR + get_relative_cq_offset(cq_id, cq_size), mmio_device_id, channel);
    if (not addr_16B) {
        return recv << 4;
    }
    return recv;
}

struct SystemMemoryCQInterface {
    // CQ is split into issue and completion regions
    // Host writes commands and data for H2D transfers in the issue region, device reads from the issue region
    // Device signals completion and writes data for D2H transfers in the completion region, host reads from the completion region
    // Equation for issue fifo size is
    // | issue_fifo_wr_ptr + command size B - issue_fifo_rd_ptr |
    // Space available would just be issue_fifo_limit - issue_fifo_size
    SystemMemoryCQInterface(uint16_t channel, uint8_t cq_id, uint32_t cq_size):
      command_completion_region_size((((cq_size - CQ_START) / dispatch_constants::TRANSFER_PAGE_SIZE) / 4) * dispatch_constants::TRANSFER_PAGE_SIZE),
      command_issue_region_size((cq_size - CQ_START) - this->command_completion_region_size),
      issue_fifo_size(command_issue_region_size >> 4),
      issue_fifo_limit(((CQ_START + this->command_issue_region_size) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4),
      completion_fifo_size(command_completion_region_size >> 4),
      completion_fifo_limit(issue_fifo_limit + completion_fifo_size),
      offset(get_absolute_cq_offset(channel, cq_id, cq_size))
     {
        TT_ASSERT(this->command_completion_region_size % PCIE_ALIGNMENT == 0 and this->command_issue_region_size % PCIE_ALIGNMENT == 0, "Issue queue and completion queue need to be {}B aligned!", PCIE_ALIGNMENT);
        TT_ASSERT(this->issue_fifo_limit != 0, "Cannot have a 0 fifo limit");
        // Currently read / write pointers on host and device assumes contiguous ranges for each channel
        // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command queue
        //  but on host, we access a region of sysmem using addresses relative to a particular channel
        this->issue_fifo_wr_ptr = (CQ_START + this->offset) >> 4;  // In 16B words
        this->issue_fifo_wr_toggle = 0;

        this->completion_fifo_rd_ptr = this->issue_fifo_limit;
        this->completion_fifo_rd_toggle = 0;
    }

    // Percentage of the command queue that is dedicated for issuing commands. Issue queue size is rounded to be 32B aligned and remaining space is dedicated for completion queue
    // Smaller issue queues can lead to more stalls for applications that send more work to device than readback data.
    static constexpr float default_issue_queue_split = 0.75;
    const uint32_t command_completion_region_size;
    const uint32_t command_issue_region_size;

    uint32_t issue_fifo_size;
    uint32_t issue_fifo_limit;  // Last possible FIFO address
    const uint32_t offset;
    uint32_t issue_fifo_wr_ptr;
    bool issue_fifo_wr_toggle;

    uint32_t completion_fifo_size;
    uint32_t completion_fifo_limit;  // Last possible FIFO address
    uint32_t completion_fifo_rd_ptr;
    bool completion_fifo_rd_toggle;
};

class SystemMemoryManager {
   private:
    chip_id_t device_id;
    uint8_t num_hw_cqs;
    const uint32_t m_dma_buf_size;
    const std::function<void(uint32_t, uint32_t, const uint8_t*, uint32_t)> fast_write_callable;
    vector<uint32_t> completion_byte_addrs;
    char* cq_sysmem_start;
    vector<SystemMemoryCQInterface> cq_interfaces;
    uint32_t cq_size;
    uint32_t channel_offset;
    vector<int> cq_to_event;
    vector<int> cq_to_last_completed_event;
    vector<std::mutex> cq_to_event_locks;
    vector<tt_cxy_pair> prefetcher_cores;
    vector<uint32_t> prefetch_q_dev_ptrs;
    vector<uint32_t> prefetch_q_dev_fences;

    bool bypass_enable;
    vector<uint32_t> bypass_buffer;
    uint32_t bypass_buffer_write_offset;

   public:
    SystemMemoryManager(chip_id_t device_id, uint8_t num_hw_cqs);

    uint32_t get_next_event(const uint8_t cq_id);

    void reset_event_id(const uint8_t cq_id);

    void increment_event_id(const uint8_t cq_id, const uint32_t val);

    void set_last_completed_event(const uint8_t cq_id, const uint32_t event_id);

    uint32_t get_last_completed_event(const uint8_t cq_id);

    void reset(const uint8_t cq_id);

    void set_issue_queue_size(const uint8_t cq_id, const uint32_t issue_queue_size);

    void set_bypass_mode(const bool enable, const bool clear);

    bool get_bypass_mode();

    std::vector<uint32_t> get_bypass_data();

    uint32_t get_issue_queue_size(const uint8_t cq_id) const;

    uint32_t get_issue_queue_limit(const uint8_t cq_id) const;

    uint32_t get_completion_queue_size(const uint8_t cq_id) const;

    uint32_t get_completion_queue_limit(const uint8_t cq_id) const;

    uint32_t get_issue_queue_write_ptr(const uint8_t cq_id) const;

    uint32_t get_completion_queue_read_ptr(const uint8_t cq_id) const;

    uint32_t get_completion_queue_read_toggle(const uint8_t cq_id) const;

    uint32_t get_cq_size() const;

    void *issue_queue_reserve(uint32_t cmd_size_B, const uint8_t cq_id);

    void cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr);

    // TODO: RENAME issue_queue_stride ?
    void issue_queue_push_back(uint32_t push_size_B, const uint8_t cq_id);

    void completion_queue_wait_front(const uint8_t cq_id, volatile bool& exit_condition) const;

    void send_completion_queue_read_ptr(const uint8_t cq_id) const;

    void wrap_issue_queue_wr_ptr(const uint8_t cq_id);

    void wrap_completion_queue_rd_ptr(const uint8_t cq_id);

    void completion_queue_pop_front(uint32_t num_pages_read, const uint8_t cq_id);

    void fetch_queue_reserve_back(const uint8_t cq_id);

    void fetch_queue_write(uint32_t command_size_B, const uint8_t cq_id);
};
