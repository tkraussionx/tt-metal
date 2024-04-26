// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/command_queue_interface.hpp"

SystemMemoryManager::SystemMemoryManager(chip_id_t device_id, uint8_t num_hw_cqs) :
    device_id(device_id),
    num_hw_cqs(num_hw_cqs),
    m_dma_buf_size(tt::Cluster::instance().get_m_dma_buf_size(device_id)),
    fast_write_callable(tt::Cluster::instance().get_fast_pcie_static_tlb_write_callable(device_id)),
    bypass_enable(false),
    bypass_buffer_write_offset(0) {
    this->completion_byte_addrs.resize(num_hw_cqs);
    this->prefetcher_cores.resize(num_hw_cqs);
    this->prefetch_q_dev_ptrs.resize(num_hw_cqs);
    this->prefetch_q_dev_fences.resize(num_hw_cqs);

    // Split hugepage into however many pieces as there are CQs
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
    char* hugepage_start = (char*) tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel);
    this->cq_sysmem_start = hugepage_start;

    // TODO(abhullar): Remove env var and expose sizing at the API level
    char* cq_size_override_env = std::getenv("TT_METAL_CQ_SIZE_OVERRIDE");
    if (cq_size_override_env != nullptr) {
        uint32_t cq_size_override = std::stoi(string(cq_size_override_env));
        this->cq_size = cq_size_override;
    } else {
        this->cq_size = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / num_hw_cqs;
    }
    this->channel_offset = MAX_HUGEPAGE_SIZE * channel;

    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        tt_cxy_pair prefetcher_core = dispatch_core_manager::get(num_hw_cqs).prefetcher_core(device_id, channel, cq_id);
        CoreType core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(device_id);
        tt_cxy_pair prefetcher_physical_core = tt_cxy_pair(prefetcher_core.chip, tt::get_physical_core_coordinate(prefetcher_core, core_type));
        this->prefetcher_cores[cq_id] = prefetcher_physical_core;

        tt_cxy_pair completion_queue_writer_core = dispatch_core_manager::get(num_hw_cqs).completion_queue_writer_core(device_id, channel, cq_id);
        const std::tuple<uint32_t, uint32_t> completion_interface_tlb_data = tt::Cluster::instance().get_tlb_data(tt_cxy_pair(completion_queue_writer_core.chip, tt::get_physical_core_coordinate(completion_queue_writer_core, core_type))).value();
        auto [completion_tlb_offset, completion_tlb_size] = completion_interface_tlb_data;
        this->completion_byte_addrs[cq_id] = completion_tlb_offset + CQ_COMPLETION_READ_PTR % completion_tlb_size;

        this->cq_interfaces.push_back(SystemMemoryCQInterface(channel, cq_id, this->cq_size));
        this->cq_to_event.push_back(0);
        this->cq_to_last_completed_event.push_back(0);
        this->prefetch_q_dev_ptrs[cq_id] = dispatch_constants::PREFETCH_Q_BASE;
        this->prefetch_q_dev_fences[cq_id] = dispatch_constants::PREFETCH_Q_BASE + dispatch_constants::PREFETCH_Q_ENTRIES * sizeof(dispatch_constants::prefetch_q_entry_type);
    }
    vector<std::mutex> temp_mutexes(num_hw_cqs);
    cq_to_event_locks.swap(temp_mutexes);
}

uint32_t SystemMemoryManager::get_next_event(const uint8_t cq_id) {
    cq_to_event_locks[cq_id].lock();
    uint32_t next_event = this->cq_to_event[cq_id]++;
    cq_to_event_locks[cq_id].unlock();
    return next_event;
}

void SystemMemoryManager::reset_event_id(const uint8_t cq_id) {
    cq_to_event_locks[cq_id].lock();
    this->cq_to_event[cq_id] = 0;
    cq_to_event_locks[cq_id].unlock();
}

void SystemMemoryManager::increment_event_id(const uint8_t cq_id, const uint32_t val) {
    cq_to_event_locks[cq_id].lock();
    this->cq_to_event[cq_id] += val;
    cq_to_event_locks[cq_id].unlock();
}

void SystemMemoryManager::set_last_completed_event(const uint8_t cq_id, const uint32_t event_id) {
    TT_ASSERT(event_id >= this->cq_to_last_completed_event[cq_id], "Event ID is expected to increase. Wrapping not supported for sync. Completed event {} but last recorded completed event is {}", event_id, this->cq_to_last_completed_event[cq_id]);
    cq_to_event_locks[cq_id].lock();
    this->cq_to_last_completed_event[cq_id] = event_id;
    cq_to_event_locks[cq_id].unlock();
}

uint32_t SystemMemoryManager::get_last_completed_event(const uint8_t cq_id) {
    cq_to_event_locks[cq_id].lock();
    uint32_t last_completed_event = this->cq_to_last_completed_event[cq_id];
    cq_to_event_locks[cq_id].unlock();
    return last_completed_event;
}

void SystemMemoryManager::reset(const uint8_t cq_id) {
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.issue_fifo_wr_ptr = (CQ_START + cq_interface.offset) >> 4;  // In 16B words
    cq_interface.issue_fifo_wr_toggle = 0;
    cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
    cq_interface.completion_fifo_rd_toggle = 0;
}

void SystemMemoryManager::set_issue_queue_size(const uint8_t cq_id, const uint32_t issue_queue_size) {
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.issue_fifo_size = (issue_queue_size >> 4);
    cq_interface.issue_fifo_limit = (CQ_START + cq_interface.offset + issue_queue_size) >> 4;
}

void SystemMemoryManager::set_bypass_mode(const bool enable, const bool clear) {
    this->bypass_enable = enable;
    if (clear) {
        this->bypass_buffer.clear();
        this->bypass_buffer_write_offset = 0;
    }
}

bool SystemMemoryManager::get_bypass_mode() {
    return this->bypass_enable;
}

std::vector<uint32_t> SystemMemoryManager::get_bypass_data() {
    return std::move(this->bypass_buffer);
}

uint32_t SystemMemoryManager::get_issue_queue_size(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].issue_fifo_size << 4;
}

uint32_t SystemMemoryManager::get_issue_queue_limit(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].issue_fifo_limit << 4;
}

uint32_t SystemMemoryManager::get_completion_queue_size(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].completion_fifo_size << 4;
}

uint32_t SystemMemoryManager::get_completion_queue_limit(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].completion_fifo_limit << 4;
}

uint32_t SystemMemoryManager::get_issue_queue_write_ptr(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].issue_fifo_wr_ptr << 4;
}

uint32_t SystemMemoryManager::get_completion_queue_read_ptr(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].completion_fifo_rd_ptr << 4;
}

uint32_t SystemMemoryManager::get_completion_queue_read_toggle(const uint8_t cq_id) const {
    return this->cq_interfaces[cq_id].completion_fifo_rd_toggle;
}

uint32_t SystemMemoryManager::get_cq_size() const {
    return this->cq_size;
}

void* SystemMemoryManager::issue_queue_reserve(uint32_t cmd_size_B, const uint8_t cq_id) {
    if (this->bypass_enable) {
        TT_FATAL(cmd_size_B % sizeof(uint32_t) == 0, "Data cmd_size_B={} is not {}-byte aligned", cmd_size_B, sizeof(uint32_t));
        uint32_t curr_size = this->bypass_buffer.size();
        uint32_t new_size = curr_size + (cmd_size_B / sizeof(uint32_t));
        this->bypass_buffer.resize(new_size);
        return (void *)((char *)this->bypass_buffer.data() + (curr_size * sizeof(uint32_t)));
    }

    uint32_t issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);

    const uint32_t command_issue_limit = this->get_issue_queue_limit(cq_id);
    if (issue_q_write_ptr + align(cmd_size_B, PCIE_ALIGNMENT) > command_issue_limit) {
        this->wrap_issue_queue_wr_ptr(cq_id);
        issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);
    }

    // Currently read / write pointers on host and device assumes contiguous ranges for each channel
    // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command queue
    //  but on host, we access a region of sysmem using addresses relative to a particular channel
    //  this->cq_sysmem_start gives start of hugepage for a given channel
    //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
    //  so channel offset needs to be subtracted to get address relative to channel
    // TODO: Reconsider offset sysmem offset calculations based on https://github.com/tenstorrent/tt-metal/issues/4757
    void* issue_q_region = this->cq_sysmem_start + (issue_q_write_ptr - this->channel_offset);

    return issue_q_region;
}

void SystemMemoryManager::cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr) {
    // Currently read / write pointers on host and device assumes contiguous ranges for each channel
    // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command queue
    //  but on host, we access a region of sysmem using addresses relative to a particular channel
    //  this->cq_sysmem_start gives start of hugepage for a given channel
    //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
    //  so channel offset needs to be subtracted to get address relative to channel
    // TODO: Reconsider offset sysmem offset calculations based on https://github.com/tenstorrent/tt-metal/issues/4757
    void* user_scratchspace = this->cq_sysmem_start + (write_ptr - this->channel_offset);

    if (this->bypass_enable) {
        TT_FATAL(size_in_bytes % sizeof(uint32_t) == 0, "Data size_in_bytes={} is not {}-byte aligned", size_in_bytes, sizeof(uint32_t));
        std::copy((uint32_t*)data, (uint32_t*)data + size_in_bytes / sizeof(uint32_t), this->bypass_buffer.begin() + this->bypass_buffer_write_offset);
    } else {
        memcpy(user_scratchspace, data, size_in_bytes);
    }
}

void SystemMemoryManager::issue_queue_push_back(uint32_t push_size_B, const uint8_t cq_id) {
    if (this->bypass_enable) {
        TT_FATAL(push_size_B % sizeof(uint32_t) == 0, "Data push_size_B={} is not {}-byte aligned", push_size_B, sizeof(uint32_t));
        this->bypass_buffer_write_offset += (push_size_B / sizeof(uint32_t));
        return;
    }

    // All data needs to be 32B aligned
    uint32_t push_size_16B = align(push_size_B, 32) >> 4;

    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

    if (cq_interface.issue_fifo_wr_ptr + push_size_16B >= cq_interface.issue_fifo_limit) {
        cq_interface.issue_fifo_wr_ptr = (CQ_START + cq_interface.offset) >> 4;  // In 16B words
        cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle; // Flip the toggle
    } else {
        cq_interface.issue_fifo_wr_ptr += push_size_16B;
    }
}

void SystemMemoryManager::completion_queue_wait_front(const uint8_t cq_id, volatile bool& exit_condition) const {
    uint32_t write_ptr_and_toggle;
    uint32_t write_ptr;
    uint32_t write_toggle;
    const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

    do {
        write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(this->device_id, cq_id, this->cq_size);
        write_ptr = write_ptr_and_toggle & 0x7fffffff;
        write_toggle = write_ptr_and_toggle >> 31;
    } while (cq_interface.completion_fifo_rd_ptr == write_ptr and cq_interface.completion_fifo_rd_toggle == write_toggle and not exit_condition);
}

void SystemMemoryManager::send_completion_queue_read_ptr(const uint8_t cq_id) const {
    const SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];

    uint32_t read_ptr_and_toggle =
        cq_interface.completion_fifo_rd_ptr | (cq_interface.completion_fifo_rd_toggle << 31);
    this->fast_write_callable(this->completion_byte_addrs[cq_id], 4, (uint8_t*)&read_ptr_and_toggle, this->m_dma_buf_size);
    tt_driver_atomics::sfence();
}

void SystemMemoryManager::wrap_issue_queue_wr_ptr(const uint8_t cq_id) {
    if (this->bypass_enable) return;
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.issue_fifo_wr_ptr = (CQ_START + cq_interface.offset) >> 4;
    cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;
}

void SystemMemoryManager::wrap_completion_queue_rd_ptr(const uint8_t cq_id) {
    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
    cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
}

void SystemMemoryManager::completion_queue_pop_front(uint32_t num_pages_read, const uint8_t cq_id) {
    uint32_t data_read_B = num_pages_read * dispatch_constants::TRANSFER_PAGE_SIZE;
    uint32_t data_read_16B = data_read_B >> 4;

    SystemMemoryCQInterface& cq_interface = this->cq_interfaces[cq_id];
    cq_interface.completion_fifo_rd_ptr += data_read_16B;
    if (cq_interface.completion_fifo_rd_ptr >= cq_interface.completion_fifo_limit) {
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
    }

    // Notify dispatch core
    this->send_completion_queue_read_ptr(cq_id);
}

void SystemMemoryManager::fetch_queue_reserve_back(const uint8_t cq_id) {
    if (this->bypass_enable) return;

    // Wait for space in the FetchQ
    uint32_t fence;
    while (this->prefetch_q_dev_ptrs[cq_id] == this->prefetch_q_dev_fences[cq_id]) {
        tt::Cluster::instance().read_core(&fence, sizeof(uint32_t), this->prefetcher_cores[cq_id], CQ_PREFETCH_Q_RD_PTR);
        this->prefetch_q_dev_fences[cq_id] = fence;
    }

    // Wrap FetchQ if possible
    uint32_t prefetch_q_base = DISPATCH_L1_UNRESERVED_BASE;
    uint32_t prefetch_q_limit = prefetch_q_base + dispatch_constants::PREFETCH_Q_ENTRIES * sizeof(dispatch_constants::prefetch_q_entry_type);
    if (this->prefetch_q_dev_ptrs[cq_id] == prefetch_q_limit) {
        this->prefetch_q_dev_ptrs[cq_id] = prefetch_q_base;

        while (this->prefetch_q_dev_ptrs[cq_id] == this->prefetch_q_dev_fences[cq_id]) {
            tt::Cluster::instance().read_core(&fence, sizeof(uint32_t), this->prefetcher_cores[cq_id], CQ_PREFETCH_Q_RD_PTR);
            this->prefetch_q_dev_fences[cq_id] = fence;
        }
    }
}

void SystemMemoryManager::fetch_queue_write(uint32_t command_size_B, const uint8_t cq_id) {
    CoreType dispatch_core_type = dispatch_core_manager::get(this->num_hw_cqs).get_dispatch_core_type(this->device_id);
    uint32_t max_command_size_B = dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();
    TT_FATAL(command_size_B <= max_command_size_B, "Generated prefetcher command of size {} B exceeds max command size {} B", command_size_B, max_command_size_B);
    TT_FATAL((command_size_B >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE) < 0xFFFF, "FetchQ command too large to represent");

    if (this->bypass_enable) return;
    uint32_t command_size_16B = command_size_B >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE;
    tt::Cluster::instance().write_reg(&command_size_16B, this->prefetcher_cores[cq_id], this->prefetch_q_dev_ptrs[cq_id]);
    this->prefetch_q_dev_ptrs[cq_id] += sizeof(dispatch_constants::prefetch_q_entry_type);
    tt_driver_atomics::sfence();
}
