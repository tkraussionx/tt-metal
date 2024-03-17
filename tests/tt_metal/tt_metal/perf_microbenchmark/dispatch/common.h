// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include "core_coord.h"
#include "logger.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_cmds.hpp"
#include "noc/noc_parameters.h"

extern bool debug_g;
extern bool use_coherent_data_g;
extern uint32_t dispatch_buffer_page_size_g;
extern CoreCoord first_worker_g;
extern CoreRange all_workers_g;
extern uint32_t min_xfer_size_bytes_g;
extern uint32_t max_xfer_size_bytes_g;
extern bool send_to_all_g;
extern bool perf_test_g;

struct one_worker_data_t {
    vector<bool> valid;
    vector<uint32_t> data;
};

typedef unordered_map<CoreCoord, one_worker_data_t> worker_data_t;

inline void reset_worker_data(worker_data_t& awd) {
    for (auto& wd : awd) {
        wd.second.valid.resize(0);
        wd.second.data.resize(0);
    }
}

inline uint32_t worker_data_size(worker_data_t& awd) {
    return awd[first_worker_g].data.size();

}

inline uint32_t padded_size(uint32_t size, uint32_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

// Return a vector of core coordinates that are used as interleaved paged banks (L1 or DRAM)
inline std::vector<CoreCoord> get_cores_per_bank_id(Device *device, bool is_dram){
    uint32_t num_banks = device->num_banks(is_dram ? BufferType::DRAM : BufferType::L1);
    std::vector<CoreCoord> bank_cores;

    for (int bank_id = 0; bank_id < num_banks; bank_id++) {
        auto core = is_dram ?
            device->core_from_dram_channel(device->dram_channel_from_bank_id(bank_id)) :
            device->logical_core_from_bank_id(bank_id);
        bank_cores.push_back(core);
    }

    return bank_cores;
}

inline void generate_random_payload(vector<uint32_t>& cmds,
                                    uint32_t length) {

    for (uint32_t i = 0; i < length; i++) {
        uint32_t datum = (use_coherent_data_g) ? i : std::rand();
        cmds.push_back(datum);
    }
}

inline void generate_random_payload(vector<uint32_t>& cmds,
                                    const CoreRange& workers,
                                    worker_data_t& data,
                                    uint32_t length_words) {

    static uint32_t coherent_count = 0;

    // Note: the dst address marches in unison regardless of weather or not a core is written to
    for (uint32_t i = 0; i < length_words; i++) {
        uint32_t datum = (use_coherent_data_g) ? coherent_count++ : std::rand();
        cmds.push_back(datum);
        for (uint32_t y = all_workers_g.start.y; y <= all_workers_g.end.y; y++) {
            for (uint32_t x = all_workers_g.start.x; x <= all_workers_g.end.x; x++) {
                CoreCoord core(x, y);
                data[core].data.push_back(datum);
                data[core].valid.push_back(workers.contains(core));
            }
        }
    }
}

// Generate a random payload for a paged write command. Note: Doesn't currently support using the base_addr here.
inline void generate_random_paged_payload(Device *device,
                                          CQDispatchCmd cmd,
                                          vector<uint32_t>& cmds,
                                          worker_data_t& data,
                                          uint32_t start_page,
                                          bool is_dram) {

    static uint32_t coherent_count = 0;
    uint32_t bank_id = 0;
    uint32_t bank_offset = 0;
    auto buf_type = is_dram ? BufferType::DRAM : BufferType::L1;
    uint32_t num_banks = device->num_banks(buf_type);
    uint32_t words_per_page = cmd.write_paged.page_size / sizeof(uint32_t);

    // Note: the dst address marches in unison regardless of weather or not a core is written to
    for (uint32_t page_id = start_page; page_id < start_page + cmd.write_paged.pages; page_id++) {

        CoreCoord bank_core;
        bank_id = page_id % num_banks;
        bank_offset = align(cmd.write_paged.page_size, 32) * (page_id / num_banks); // 32B alignment taken from InterleavedAddrGen

        if (is_dram) {
            auto dram_channel = device->dram_channel_from_bank_id(bank_id);
            bank_core = device->core_from_dram_channel(dram_channel);
        } else {
            bank_core = device->logical_core_from_bank_id(bank_id);
        }

        // Add zero-padding if needed to reach required alignment in worker data for comparison.
        uint32_t padding_words = bank_offset / sizeof(uint32_t) - data[bank_core].valid.size();
        if (padding_words > 0) {
            log_trace(tt::LogTest, "{} - Adding {} padding words for bank_core: {}", __FUNCTION__, padding_words, bank_core.str());
            for (uint32_t i = 0; i < padding_words; i++) {
                data[bank_core].data.push_back(0);
                data[bank_core].valid.push_back(false); // Skip checking padding words.
            }
        }

        // We could have filled data with all zeros and then used bank_offset to write specific data.
        TT_ASSERT(bank_offset / sizeof(uint32_t) == data[bank_core].valid.size());

        // Generate data and add to cmd for sending to device, and worker_data for correctness checking.
        for (uint32_t i = 0; i < words_per_page; i++) {
            uint32_t datum = (use_coherent_data_g) ? coherent_count++ : std::rand();
            log_debug(tt::LogTest, "{} - Setting {} page_id: {} word: {} on core: {} (bank_id: {} bank_offset: {}) => datum: 0x{:x}",
                __FUNCTION__, is_dram ? "DRAM" : "L1", page_id, i, bank_core.str(), bank_id, bank_offset, datum);
            cmds.push_back(datum);
            data[bank_core].data.push_back(datum);
            data[bank_core].valid.push_back(true);

        }
    }
}

inline void generate_random_packed_payload(vector<uint32_t>& cmds,
                                           vector<CoreCoord>& worker_cores,
                                           worker_data_t& data,
                                           uint32_t size_words) {

    static uint32_t coherent_count = 0;

    // Note: the dst address marches in unison regardless of weather or not a core is written to
    for (uint32_t y = all_workers_g.start.y; y <= all_workers_g.end.y; y++) {
        for (uint32_t x = all_workers_g.start.x; x <= all_workers_g.end.x; x++) {
            CoreCoord core(x, y);
            for (uint32_t i = 0; i < size_words; i++) {
                bool contains = false;
                for (CoreCoord worker_core : worker_cores) {
                    if (core == worker_core) {
                        contains = true;
                        break;
                    }
                }
                if (contains) {
                    uint32_t datum = (use_coherent_data_g) ? ((x << 16) | (y << 24) | coherent_count++) : std::rand();

                    cmds.push_back(datum);
                    data[core].data.push_back(datum);
                    data[core].valid.push_back(true);
                } else {
                    data[core].data.push_back(0xbaadf00d);
                    data[core].valid.push_back(false);
                }
            }

            cmds.resize(padded_size(cmds.size(), 4)); // XXXXX L1_ALIGNMENT16/sizeof(uint)
            data[core].data.resize(padded_size(data[core].data.size(), 4)); // XXXXX L1_ALIGNMENT16/sizeof(uint)
            data[core].valid.resize(padded_size(data[core].valid.size(), 4)); // XXXXX L1_ALIGNMENT16/sizeof(uint)
        }
    }
}

inline void add_bare_dispatcher_cmd(vector<uint32_t>& cmds,
                                    CQDispatchCmd cmd) {

    uint32_t *ptr = (uint32_t *)&cmd;
    for (int i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
        cmds.push_back(*ptr++);
    }
}

inline size_t debug_prologue(vector<uint32_t>& cmds) {
    size_t prior = cmds.size();

    if (debug_g) {
        CQDispatchCmd debug_cmd;
        debug_cmd.base.cmd_id = CQ_DISPATCH_CMD_DEBUG;
        add_bare_dispatcher_cmd(cmds, debug_cmd);
    }

    return prior;
}

inline void debug_epilogue(vector<uint32_t>& cmds,
                           size_t prior_end) {
    if (debug_g) {
        // Doing a checksum on the full command length is problematic in the kernel
        // as it requires the debug code to pull all the pages in before the actual
        // command is processed.  So, limit this to doing a checksum on the first page
        // (which is disappointing).  Any other value requires the checksum code to handle
        // buffer wrap which then messes up the routines w/ the embedded insn - not worth it
        CQDispatchCmd* debug_cmd_ptr;
        debug_cmd_ptr = (CQDispatchCmd *)&cmds[prior_end];
        uint32_t full_size = (cmds.size() - prior_end) * sizeof(uint32_t) - sizeof(CQDispatchCmd);
        uint32_t max_size = dispatch_buffer_page_size_g - sizeof(CQDispatchCmd);
        uint32_t size = (full_size > max_size) ? max_size : full_size;
        debug_cmd_ptr->debug.size = size;
        debug_cmd_ptr->debug.stride = sizeof(CQDispatchCmd);
        uint32_t checksum = 0;
        uint32_t start = prior_end + sizeof(CQDispatchCmd) / sizeof(uint32_t);
        for (uint32_t i = start; i < start + size / sizeof(uint32_t); i++) {
            checksum += cmds[i];
        }
        debug_cmd_ptr->debug.checksum = checksum;
    }
}

inline void add_dispatcher_cmd(vector<uint32_t>& cmds,
                               const CoreRange& workers,
                               worker_data_t& worker_data,
                               CQDispatchCmd cmd,
                               uint32_t length) {

    size_t prior_end = debug_prologue(cmds);

    add_bare_dispatcher_cmd(cmds, cmd);
    uint32_t length_words = length / sizeof(uint32_t);
    generate_random_payload(cmds, workers, worker_data, length_words);

    debug_epilogue(cmds, prior_end);
}

inline void add_dispatcher_paged_cmd(Device *device,
                                     vector<uint32_t>& cmds,
                                     worker_data_t& worker_data,
                                     CQDispatchCmd cmd,
                                     uint32_t start_page,
                                     bool is_dram) {

    size_t prior_end = debug_prologue(cmds);
    add_bare_dispatcher_cmd(cmds, cmd);
    generate_random_paged_payload(device, cmd, cmds, worker_data, start_page, is_dram);
    debug_epilogue(cmds, prior_end);
}

inline void add_dispatcher_packed_cmd(Device *device,
                                      vector<uint32_t>& cmds,
                                      vector<CoreCoord>& worker_cores,
                                      worker_data_t& worker_data,
                                      CQDispatchCmd cmd,
                                      uint32_t size_words) {

    size_t prior_end = debug_prologue(cmds);

    add_bare_dispatcher_cmd(cmds, cmd);
    for (CoreCoord core : worker_cores) {
        CoreCoord phys_worker_core = device->worker_core_from_logical_core(core);
        cmds.push_back(NOC_XY_ENCODING(phys_worker_core.x, phys_worker_core.y));
    }
    cmds.resize(padded_size(cmds.size(), 4)); // XXXXX L1_ALIGNMENT16/sizeof(uint)

    generate_random_packed_payload(cmds, worker_cores, worker_data, size_words);

    debug_epilogue(cmds, prior_end);
}

// bare: doesn't generate random payload data, for use w/ eg, dram reads
inline void gen_bare_dispatcher_unicast_write_cmd(Device *device,
                                                  vector<uint32_t>& cmds,
                                                  CoreCoord worker_core,
                                                  worker_data_t& worker_data,
                                                  uint32_t dst_addr,
                                                  uint32_t length) {

    CQDispatchCmd cmd;

    CoreCoord phys_worker_core = device->worker_core_from_logical_core(worker_core);

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
    cmd.write_linear.noc_xy_addr = NOC_XY_ENCODING(phys_worker_core.x, phys_worker_core.y);
    cmd.write_linear.addr = dst_addr + worker_data_size(worker_data) * sizeof(uint32_t);
    cmd.write_linear.length = length;
    cmd.write_linear.num_mcast_dests = 0;

    add_bare_dispatcher_cmd(cmds, cmd);
}

inline void gen_dispatcher_unicast_write_cmd(Device *device,
                                             vector<uint32_t>& cmds,
                                             CoreCoord worker_core,
                                             worker_data_t& worker_data,
                                             uint32_t dst_addr,
                                             uint32_t length) {

    CQDispatchCmd cmd;

    CoreCoord phys_worker_core = device->worker_core_from_logical_core(worker_core);

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
    cmd.write_linear.noc_xy_addr = NOC_XY_ENCODING(phys_worker_core.x, phys_worker_core.y);
    cmd.write_linear.addr = dst_addr + worker_data[worker_core].data.size() * sizeof(uint32_t);
    cmd.write_linear.length = length;
    cmd.write_linear.num_mcast_dests = 0;

    add_dispatcher_cmd(cmds, worker_core, worker_data, cmd, length);
}

inline void gen_dispatcher_multicast_write_cmd(Device *device,
                                             vector<uint32_t>& cmds,
                                             CoreRange worker_core_range,
                                             worker_data_t& worker_data,
                                             uint32_t dst_addr,
                                             uint32_t length) {

    CQDispatchCmd cmd;

    CoreCoord physical_start = device->physical_core_from_logical_core(worker_core_range.start, CoreType::WORKER);
    CoreCoord physical_end = device->physical_core_from_logical_core(worker_core_range.end, CoreType::WORKER);

    // Sanity check that worker_data covers all cores being targeted.
    for (uint32_t y = worker_core_range.start.y; y <= worker_core_range.end.y; y++) {
        for (uint32_t x = worker_core_range.start.x; x <= worker_core_range.end.x; x++) {
            CoreCoord worker(x, y);
            TT_ASSERT(worker_data.find(worker) != worker_data.end(), "Worker core x={},y={} missing in worker_data", x, y);
        }
    }

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
    cmd.write_linear.noc_xy_addr = NOC_MULTICAST_ENCODING(physical_start.x, physical_start.y, physical_end.x, physical_end.y);
    cmd.write_linear.addr = dst_addr + worker_data[worker_core_range.start].data.size() * sizeof(uint32_t);
    cmd.write_linear.length = length;
    cmd.write_linear.num_mcast_dests = worker_core_range.size();
    tt::log_debug(tt::LogTest, "{} Setting addr: {} from dst_addr: {} and worker_data[{}].data.size(): {}",
        __FUNCTION__, (uint32_t) cmd.write_linear.addr, dst_addr, worker_core_range.start.str(),
        worker_data[worker_core_range.start].data.size() * sizeof(uint32_t));

    add_dispatcher_cmd(cmds, worker_core_range, worker_data, cmd, length);
}

inline void gen_dispatcher_paged_write_cmd(Device *device,
                                             vector<uint32_t>& cmds,
                                             worker_data_t& worker_data,
                                             bool is_dram,
                                             uint32_t start_page,
                                             uint32_t dst_addr,
                                             uint32_t page_size,
                                             uint32_t pages) {

    // To account for previous paged writes, go through all cores and find largest amt of data written so far.
    uint32_t max_data_size = 0;
    for (auto& [core, data] : worker_data) {
        if (data.data.size() > max_data_size) {
            max_data_size = data.data.size();
        }
    }

    CQDispatchCmd cmd;
    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_PAGED;
    cmd.write_paged.is_dram = is_dram;
    cmd.write_paged.start_page = start_page;
    cmd.write_paged.base_addr = dst_addr + (max_data_size * sizeof(uint32_t));
    cmd.write_paged.page_size = page_size;
    cmd.write_paged.pages = pages;

    log_info(tt::LogTest, "Adding CQ_DISPATCH_CMD_WRITE_PAGED, is_dram: {} start_page: {} base_addr: {:x} page_size: {} pages: {} (max_data_size: {})",
        is_dram, start_page, dst_addr + max_data_size * sizeof(uint32_t), page_size, pages, max_data_size);

    add_dispatcher_paged_cmd(device, cmds, worker_data, cmd, start_page, is_dram);
}


inline void gen_dispatcher_packed_write_cmd(Device *device,
                                            vector<uint32_t>& cmds,
                                            vector<CoreCoord>& worker_cores,
                                            worker_data_t& worker_data,
                                            uint32_t dst_addr,
                                            uint32_t size_words) {

    CQDispatchCmd cmd;

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_PACKED;
    cmd.write_packed.is_multicast = 0;
    cmd.write_packed.count = worker_cores.size();
    cmd.write_packed.addr = dst_addr + worker_data_size(worker_data) * sizeof(uint32_t);
    cmd.write_packed.size = size_words * sizeof(uint32_t);

    add_dispatcher_packed_cmd(device, cmds, worker_cores, worker_data, cmd, size_words);
}

inline uint32_t gen_rnd_dispatcher_packed_write_cmd(Device *device,
                                                    vector<uint32_t>& cmds,
                                                    worker_data_t& worker_data,
                                                    uint32_t dst_addr) {

    // Note: this cmd doesn't clamp to a max size which means it can overflow L1 buffer
    // However, this cmd doesn't send much data and the L1 buffer is < L1 limit, so...

    uint32_t xfer_size_words = (std::rand() % (dispatch_buffer_page_size_g / sizeof(uint32_t))) + 1;
    uint32_t xfer_size_bytes = xfer_size_words * sizeof(uint32_t);
    if (perf_test_g) {
        TT_ASSERT(max_xfer_size_bytes_g < dispatch_buffer_page_size_g);
        if (xfer_size_bytes > max_xfer_size_bytes_g) xfer_size_bytes = max_xfer_size_bytes_g;
        if (xfer_size_bytes < min_xfer_size_bytes_g) xfer_size_bytes = min_xfer_size_bytes_g;
    }

    vector<CoreCoord> gets_data;
    for (uint32_t y = all_workers_g.start.y; y <= all_workers_g.end.y; y++) {
        for (uint32_t x = all_workers_g.start.x; x <= all_workers_g.end.x; x++) {
            if (send_to_all_g || std::rand() % 2) {
                gets_data.push_back({x, y});
            }
        }
    }
    if (gets_data.size() == 0) {
        gets_data.push_back({all_workers_g.start.x, all_workers_g.start.y});
    }

    gen_dispatcher_packed_write_cmd(device, cmds, gets_data, worker_data,
                                    dst_addr, xfer_size_bytes / sizeof(uint32_t));

    return xfer_size_bytes;
}

inline void gen_dispatcher_terminate_cmd(vector<uint32_t>& cmds) {

    worker_data_t dummy_data;
    CoreCoord worker_dummy;
    CQDispatchCmd cmd;
    cmd.base.cmd_id = CQ_DISPATCH_CMD_TERMINATE;
    add_dispatcher_cmd(cmds, worker_dummy, dummy_data, cmd, 0);
}


// Validate a single core's worth of results vs expected. bank_id is purely informational, can pass -1 for non-paged testing.
inline bool validate_core_data(std::unordered_set<CoreCoord> &validated_cores, Device *device, const worker_data_t& worker_data, CoreCoord core, uint32_t base_addr, uint32_t bank_id, bool is_dram) {

    int fail_count = 0;
    const vector<uint32_t>& dev_data = worker_data.at(core).data;
    const vector<bool>& dev_valid = worker_data.at(core).valid;
    uint32_t size_bytes =  dev_data.size() * sizeof(uint32_t);

    // If we had SOC desc, could get core type this way. Revisit this.
    // soc_desc.cores.at(core).type
    auto core_type = is_dram ? CoreType::DRAM : CoreType::WORKER;
    CoreCoord phys_core;

    // Paged tests set bank_id to valid value.
    if (bank_id != -1){
        if (is_dram) {
            auto channel = device->dram_channel_from_bank_id(bank_id);
            phys_core = device->core_from_dram_channel(channel);
            base_addr += device->dram_bank_offset_from_bank_id(bank_id);
        } else {
            phys_core = device->physical_core_from_logical_core(core, core_type);
            base_addr += device->l1_bank_offset_from_bank_id(bank_id);
        }
    } else {
        TT_ASSERT(core_type == CoreType::WORKER, "Non-Paged write tests expected to target L1 only for now and not DRAM.");
        phys_core = device->physical_core_from_logical_core(core, core_type);
    }

    // Read results from device and compare to expected for this core.
    vector<uint32_t> results = tt::llrt::read_hex_vec_from_core(device->id(), phys_core, base_addr, size_bytes);

    log_info(tt::LogTest, "Validating {} bytes from {} (paged bank_id: {}) from logical_core: {} / physical: {} at addr: 0x{:x}",
        size_bytes, is_dram ? "DRAM" : "L1", bank_id, core.str(), phys_core.str(), base_addr);

    for (int i = 0; i < dev_data.size(); i++) {
        if (!dev_valid[i]) continue;
        validated_cores.insert(core);

        if (results[i] != dev_data[i]) {
            if (!fail_count) {
                log_fatal(tt::LogTest, "Data mismatch - First 20 failures for logical_core: {} (physical: {})", core.str(), phys_core.str());
            }
            log_fatal(tt::LogTest, "[{:02d}] (Fail) Expected: 0x{:08x} Observed: 0x{:08x}", i, (unsigned int)dev_data[i], (unsigned int)results[i]);
            if (fail_count++ > 20) {
                break;
            }
        } else {
            log_debug(tt::LogTest, "[{:02d}] (Pass) Expected: 0x{:08x} Observed: 0x{:08x}", i, (unsigned int)dev_data[i], (unsigned int)results[i]);
        }
    }
    return fail_count;
}

inline bool validate_results(Device *device, CoreRange workers, const worker_data_t& worker_data, uint64_t l1_buf_base) {

    bool failed = false;
    std::unordered_set<CoreCoord> validated_cores;
    for (uint32_t y = workers.start.y; y <= workers.end.y; y++) {
        for (uint32_t x = workers.start.x; x <= workers.end.x; x++) {
            CoreCoord worker(x, y);
            failed |= validate_core_data(validated_cores, device, worker_data, worker, l1_buf_base, -1, false);
        }
    }

    log_info(tt::LogTest, "Validated {} cores total.", validated_cores.size());
    return !failed;
}

// Validate paged writes to DRAM or L1 by iterating over just the banks/cores that are valid for the worker date based on prior writes.
inline bool validate_results_paged(Device *device, const worker_data_t& worker_data, uint64_t l1_buf_base, bool is_dram) {

    // Get banks, to be able to iterate over them in order here, since worker_data is umap.
    auto bank_cores = get_cores_per_bank_id(device, is_dram);
    std::unordered_set<CoreCoord> validated_cores;
    bool failed = false;

    for (int bank_id = 0; bank_id < bank_cores.size(); bank_id++) {
        auto bank_core = bank_cores.at(bank_id);
        if (worker_data.find(bank_core) == worker_data.end()) {
            log_debug(tt::LogTest, "Skipping bank_id: {} bank_core: {} as it has no data", bank_id, bank_core.str());
            continue;
        }
        failed |= validate_core_data(validated_cores, device, worker_data, bank_core, l1_buf_base, bank_id, is_dram);
    }

    log_info(tt::LogTest, "{} - Validated {} cores total.", __FUNCTION__, validated_cores.size());
    return !failed;
}
