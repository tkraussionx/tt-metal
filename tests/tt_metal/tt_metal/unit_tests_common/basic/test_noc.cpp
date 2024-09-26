// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/common/logger.hpp"


using namespace tt;

namespace unit_tests_common::noc {

std::vector<uint32_t> generate_data(uint32_t data_size_bytes, uint32_t transaction_size_bytes, uint32_t aligned_transaction_size, std::optional<uint32_t> reset_index = std::nullopt) {
    std::vector<uint32_t> data_vec(data_size_bytes/sizeof(uint32_t));
    uint32_t val = 0;
    uint32_t num_val_is_expected = transaction_size_bytes / sizeof(uint32_t);
    uint32_t num_gaps_expected = (aligned_transaction_size - transaction_size_bytes) / sizeof(uint32_t);
    uint32_t indices_per_val = num_val_is_expected + num_gaps_expected;
    uint32_t num_val_seen = 0;
    for (int i = 0; i < data_vec.size(); i++) {
        uint32_t curr_val = val;
        if (num_val_seen >= num_val_is_expected and num_val_seen < indices_per_val) {
            curr_val = 0;
        }
        data_vec[i] = curr_val;
        num_val_seen = (num_val_seen + 1) % indices_per_val;
        if (num_val_seen == 0) {
            val++;
        }

        if (reset_index.has_value() and i == reset_index.value()) {
            val = 0;
            num_val_seen = 0;
        }
    }
    return data_vec;
}

bool validate_results(const CoreCoord &logical_core, const std::vector<uint32_t> result_vec, uint32_t transaction_size_bytes, uint32_t aligned_transaction_size, std::optional<uint32_t> reset_index = std::nullopt) {
    bool pass = true;
    uint32_t expected_val = 0;
    uint32_t num_val_is_expected = transaction_size_bytes / sizeof(uint32_t);
    uint32_t num_gaps_expected = (aligned_transaction_size - transaction_size_bytes) / sizeof(uint32_t);
    uint32_t indices_per_val = num_val_is_expected + num_gaps_expected;
    uint32_t num_val_seen = 0;
    for (int i = 0; i < result_vec.size(); i++) {
        uint32_t curr_expected_val = expected_val;
        if (num_val_seen >= num_val_is_expected and num_val_seen < indices_per_val) {
            curr_expected_val = 0;
        }
        pass &= result_vec[i] == curr_expected_val;
        if (result_vec[i] != curr_expected_val) {
            std::cout << "Expected " << curr_expected_val << " but got " << result_vec[i] << std::endl;
        }
        if (result_vec[i] == 0xBADC00DE) {
            uint32_t misordered_noc = (i == 0) ? 0 : 1; // in the mcast test, NoC 0 populates L1 from L1_UNRESERVED_BASE
            std::cout << "Logical core " << logical_core.str() << " misordered writes on NoC " << misordered_noc << std::endl;
        }
        num_val_seen = (num_val_seen + 1) % indices_per_val;
        if (num_val_seen == 0) {
            expected_val++;
        }

        if (reset_index.has_value() and i == reset_index.value()) {
            expected_val = 0;
            num_val_seen = 0;
        }
    }
    return pass;
}

bool run_cmd_buffer_ordering_test(CommonFixture *fixture, tt_metal::Device *device, uint32_t transaction_size_bytes, uint32_t second_write_cmd_buf, bool add_noc_traffic) {
    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord sender_core(0, 0);

    uint32_t num_cols = device->compute_with_storage_grid_size().x;
    uint32_t num_rows = device->compute_with_storage_grid_size().y;

    CoreCoord receiver_core(3, 0);

    CoreRange traffic_core_range(CoreCoord(1, 0), CoreCoord(2, 0));

    CoreCoord phys_sender_core = device->physical_core_from_logical_core(sender_core, CoreType::WORKER);
    CoreCoord phys_receiver_core = device->physical_core_from_logical_core(receiver_core, CoreType::WORKER);

    uint32_t available_l1_size = device->l1_size_per_core() - L1_UNRESERVED_BASE;
    uint32_t aligned_transaction_size = align(transaction_size_bytes, L1_ALIGNMENT);
    uint32_t num_writes_per_cmd_buff = (available_l1_size / aligned_transaction_size) / 2;
    uint32_t total_bytes_written = num_writes_per_cmd_buff * 2 * aligned_transaction_size;
    uint32_t addr_inc_per_cmd_buff = aligned_transaction_size * 2;

    auto sender_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cmd_buf_ucast_sender.cpp",
        sender_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {num_writes_per_cmd_buff, addr_inc_per_cmd_buff, transaction_size_bytes}
        });

    auto receiver_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cmd_buf_receiver.cpp",
        receiver_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {num_writes_per_cmd_buff, addr_inc_per_cmd_buff}
        });

    KernelHandle traffic_kernel;
    if (add_noc_traffic) {
        traffic_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_l1.cpp",
            traffic_core_range,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::NOC_0});
    }

    uint32_t first_receiver_dst_address = L1_UNRESERVED_BASE;
    uint32_t second_receiver_dst_address = L1_UNRESERVED_BASE + aligned_transaction_size;
    uint32_t first_sender_address = L1_UNRESERVED_BASE;
    uint32_t second_sender_address = first_sender_address + aligned_transaction_size;
    std::vector<uint32_t> sender_rt_args = {
        (uint32_t)phys_receiver_core.x,
        (uint32_t)phys_receiver_core.y,
        first_receiver_dst_address,
        second_receiver_dst_address,
        first_sender_address,
        second_sender_address,
        second_write_cmd_buf
    };

    uint32_t first_send_val = 0;
    uint32_t second_send_val = 1;
    uint32_t receiver_result_address = L1_UNRESERVED_BASE;
    std::vector<uint32_t> receiver_rt_args = {
        (uint32_t)phys_sender_core.x,
        (uint32_t)phys_sender_core.y,
        first_send_val,
        second_send_val,
        first_receiver_dst_address,
        second_receiver_dst_address,
        receiver_result_address
    };

    tt_metal::SetRuntimeArgs(program, sender_kernel, sender_core, sender_rt_args);
    tt_metal::SetRuntimeArgs(program, receiver_kernel, receiver_core, receiver_rt_args);
    if (add_noc_traffic) {
        uint32_t traffic_num_to_send = 10;
        uint32_t traffic_single_send_bytes = 2048;
        uint32_t traffic_total_to_send_bytes = traffic_num_to_send * traffic_single_send_bytes;
        std::vector<uint32_t> traffic_core_rt_args(10, 0);
        for (uint32_t x = traffic_core_range.start_coord.x; x <= traffic_core_range.end_coord.x; x++) {
            for (uint32_t y = traffic_core_range.start_coord.y; y <= traffic_core_range.end_coord.y; y++) {
                CoreCoord curr_traffic_core(x, y);
                CoreCoord phys_curr_traffic_core = device->physical_core_from_logical_core(curr_traffic_core, CoreType::WORKER);
                uint32_t rcv_x = x == 1 ? 2 : 1;
                CoreCoord receiver_traffic_core(rcv_x, y);
                CoreCoord phys_rcv_traffic_core = device->physical_core_from_logical_core(receiver_traffic_core, CoreType::WORKER);
                traffic_core_rt_args[0] = L1_UNRESERVED_BASE;
                traffic_core_rt_args[1] = (uint32_t)phys_curr_traffic_core.x;
                traffic_core_rt_args[2] = (uint32_t)phys_curr_traffic_core.y;
                traffic_core_rt_args[3] = L1_UNRESERVED_BASE;
                traffic_core_rt_args[4] = L1_UNRESERVED_BASE;
                traffic_core_rt_args[5] = (uint32_t)phys_rcv_traffic_core.x;
                traffic_core_rt_args[6] = (uint32_t)phys_rcv_traffic_core.y;
                traffic_core_rt_args[7] = traffic_num_to_send;
                traffic_core_rt_args[8] = traffic_single_send_bytes;
                traffic_core_rt_args[9] = traffic_total_to_send_bytes;

                tt_metal::SetRuntimeArgs(program, traffic_kernel, curr_traffic_core, traffic_core_rt_args);
            }
        }
    }

    CreateSemaphore(program, CoreRange(sender_core, sender_core), 0, CoreType::WORKER);

    std::vector<uint32_t> src_vec = generate_data(total_bytes_written, transaction_size_bytes, aligned_transaction_size);
    detail::WriteToDeviceL1(device, sender_core, first_sender_address, src_vec, CoreType::WORKER);

    // std::vector<uint32_t> second_sender_vals(transaction_size_bytes / sizeof(uint32_t), second_send_val);
    // detail::WriteToDeviceL1(device, sender_core, second_sender_address, second_sender_vals, CoreType::WORKER);

    std::vector<uint32_t> zero_vec(available_l1_size/sizeof(uint32_t), 0);
    detail::WriteToDeviceL1(device, receiver_core, L1_UNRESERVED_BASE, zero_vec, CoreType::WORKER);

    tt::Cluster::instance().l1_barrier(device->id());

    fixture->RunProgram(device, program);

    bool pass = true;
    std::vector<uint32_t> result_vec(total_bytes_written/sizeof(uint32_t));
    detail::ReadFromDeviceL1(device, receiver_core, L1_UNRESERVED_BASE, total_bytes_written, result_vec);
    pass &= validate_results(receiver_core, result_vec, transaction_size_bytes, aligned_transaction_size);

    return pass;
}

CoreCoord get_noc1_physical_coordinates(const CoreCoord &noc_grid_size, const CoreCoord &noc0_physical_coordinates) {
    CoreCoord noc1_coordinates(noc_grid_size.x - 1 - noc0_physical_coordinates.x, noc_grid_size.y - 1 - noc0_physical_coordinates.y);
    return noc1_coordinates;
}

bool run_multicast_test(CommonFixture *fixture, tt_metal::Device *device, uint32_t transaction_size_bytes, uint32_t second_write_cmd_buf) {
    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord noc_grid_size = device->grid_size();

    uint32_t num_cols = device->compute_with_storage_grid_size().x;
    uint32_t num_rows = device->compute_with_storage_grid_size().y;

    CoreRange left_column(CoreCoord(0, 0), CoreCoord(0, num_rows - 2));
    CoreRange bottom_row_minus_left_column(CoreCoord(1, num_rows - 1), CoreCoord(num_cols - 1, num_rows - 1));

    CoreRange receivers(CoreCoord(1, 0), CoreCoord(num_cols - 1, num_rows - 2));

    uint32_t available_l1_size = device->l1_size_per_core() - L1_UNRESERVED_BASE;
    uint32_t aligned_transaction_size = align(transaction_size_bytes, L1_ALIGNMENT);
    uint32_t l1_bytes_per_noc = available_l1_size / 2;
    uint32_t num_writes_per_cmd_buff = (l1_bytes_per_noc / aligned_transaction_size) / 2;
    uint32_t addr_inc_per_cmd_buff = aligned_transaction_size * 2;

    auto sender_kernel_noc0 = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cmd_buf_mcast_sender.cpp",
        left_column,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {num_writes_per_cmd_buff, addr_inc_per_cmd_buff, transaction_size_bytes}
        });

    auto sender_kernel_noc1 = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cmd_buf_mcast_sender.cpp",
        bottom_row_minus_left_column,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::NOC_1,
            .compile_args = {num_writes_per_cmd_buff, addr_inc_per_cmd_buff, transaction_size_bytes}
        });

    auto receiver_kernel_dm0 = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cmd_buf_receiver.cpp",
        receivers,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {num_writes_per_cmd_buff, addr_inc_per_cmd_buff}
        });

    auto receiver_kernel_dm1 = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cmd_buf_receiver.cpp",
        receivers,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::NOC_1,
            .compile_args = {num_writes_per_cmd_buff, addr_inc_per_cmd_buff}
        });


    CreateSemaphore(program, left_column, 0, CoreType::WORKER);
    CreateSemaphore(program, bottom_row_minus_left_column, 0, CoreType::WORKER);

    uint32_t first_send_val = 0;
    uint32_t second_send_val = 1;

    std::vector<uint32_t> zero_vec(available_l1_size/sizeof(uint32_t), 0);

    std::vector<uint32_t> sender_rt_args(10, 0);
    std::vector<uint32_t> receiver_dm0_rt_args(7, 0);
    std::vector<uint32_t> receiver_dm1_rt_args(7, 0);

    uint32_t sender_noc0_first_addr = L1_UNRESERVED_BASE;
    uint32_t sender_noc0_second_addr = sender_noc0_first_addr + aligned_transaction_size;
    uint32_t sender_noc1_first_addr = sender_noc0_first_addr + l1_bytes_per_noc;
    uint32_t sender_noc1_second_addr = sender_noc1_first_addr + aligned_transaction_size;
    uint32_t rcv_from_noc0_first_addr = L1_UNRESERVED_BASE;
    uint32_t rcv_from_noc0_second_addr = rcv_from_noc0_first_addr + aligned_transaction_size;
    uint32_t rcv_from_noc1_first_addr = rcv_from_noc0_first_addr + l1_bytes_per_noc;
    uint32_t rcv_from_noc1_second_addr = rcv_from_noc1_first_addr + aligned_transaction_size;

    for(int core_idx_x = 0; core_idx_x < num_cols; core_idx_x++) {
        for(int core_idx_y = 0; core_idx_y < num_rows; core_idx_y++) {
            CoreCoord core(core_idx_x, core_idx_y);
            CoreCoord phys_noc0 = device->physical_core_from_logical_core(core, CoreType::WORKER);
            CoreCoord phys_noc1 = get_noc1_physical_coordinates(noc_grid_size, phys_noc0);

            if (left_column.contains(core)) {
                // Only sender noc0
                CoreCoord start_mcast(1, core_idx_y);
                CoreCoord end_mcast(num_cols - 1, core_idx_y);
                CoreCoord phys_start_mcast = device->physical_core_from_logical_core(start_mcast, CoreType::WORKER);
                CoreCoord phys_end_mcast = device->physical_core_from_logical_core(end_mcast, CoreType::WORKER);
                sender_rt_args[0] = (uint32_t)phys_start_mcast.x;
                sender_rt_args[1] = (uint32_t)phys_start_mcast.y;
                sender_rt_args[2] = (uint32_t)phys_end_mcast.x;
                sender_rt_args[3] = (uint32_t)phys_end_mcast.y;
                sender_rt_args[4] = rcv_from_noc0_first_addr;
                sender_rt_args[5] = rcv_from_noc0_second_addr;
                sender_rt_args[6] = num_cols - 1;
                sender_rt_args[7] = sender_noc0_first_addr;
                sender_rt_args[8] = sender_noc0_second_addr;
                sender_rt_args[9] = second_write_cmd_buf;

                tt_metal::SetRuntimeArgs(program, sender_kernel_noc0, core, sender_rt_args);

                std::vector<uint32_t> first_noc0_sender_vals(transaction_size_bytes / sizeof(uint32_t), first_send_val);
                detail::WriteToDeviceL1(device, core, sender_noc0_first_addr, first_noc0_sender_vals, CoreType::WORKER);

                std::vector<uint32_t> second_noc0_sender_vals(transaction_size_bytes / sizeof(uint32_t), second_send_val);
                detail::WriteToDeviceL1(device, core, sender_noc0_second_addr, second_noc0_sender_vals, CoreType::WORKER);

            } else if (bottom_row_minus_left_column.contains(core)) {
                // Only sender noc1
                CoreCoord start_mcast(core_idx_x, 0);
                CoreCoord end_mcast(core_idx_x, num_rows - 2);
                CoreCoord phys_start_mcast = device->physical_core_from_logical_core(start_mcast, CoreType::WORKER);
                CoreCoord phys_start_mcast_noc1 = get_noc1_physical_coordinates(noc_grid_size, phys_start_mcast);
                CoreCoord phys_end_mcast = device->physical_core_from_logical_core(end_mcast, CoreType::WORKER);
                CoreCoord phys_end_mcast_noc1 = get_noc1_physical_coordinates(noc_grid_size, phys_end_mcast);
                sender_rt_args[0] = (uint32_t)phys_end_mcast.x;
                sender_rt_args[1] = (uint32_t)phys_end_mcast.y;
                sender_rt_args[2] = (uint32_t)phys_start_mcast.x;
                sender_rt_args[3] = (uint32_t)phys_start_mcast.y;
                sender_rt_args[4] = rcv_from_noc1_first_addr;
                sender_rt_args[5] = rcv_from_noc1_second_addr;
                sender_rt_args[6] = num_rows - 1;
                sender_rt_args[7] = sender_noc1_first_addr;
                sender_rt_args[8] = sender_noc1_second_addr;
                sender_rt_args[9] = second_write_cmd_buf;

                tt_metal::SetRuntimeArgs(program, sender_kernel_noc1, core, sender_rt_args);

                std::vector<uint32_t> first_noc1_sender_vals(transaction_size_bytes / sizeof(uint32_t), first_send_val);
                detail::WriteToDeviceL1(device, core, sender_noc1_first_addr, first_noc1_sender_vals, CoreType::WORKER);

                std::vector<uint32_t> second_noc1_sender_vals(transaction_size_bytes / sizeof(uint32_t), second_send_val);
                detail::WriteToDeviceL1(device, core, sender_noc1_second_addr, second_noc1_sender_vals, CoreType::WORKER);
            } else if (receivers.contains(core)) {
                CoreCoord dm0_sender_core(0, core_idx_y);
                CoreCoord phys_dm0_sender_core = device->physical_core_from_logical_core(dm0_sender_core, CoreType::WORKER);

                CoreCoord dm1_sender_core(core_idx_x, num_rows - 1);
                CoreCoord phys_dm1_sender_core_noc0 = device->physical_core_from_logical_core(dm1_sender_core, CoreType::WORKER);
                CoreCoord phys_dm1_sender_core_noc1 = get_noc1_physical_coordinates(noc_grid_size, phys_dm1_sender_core_noc0);

                receiver_dm0_rt_args[0] = (uint32_t)phys_dm0_sender_core.x;
                receiver_dm0_rt_args[1] = (uint32_t)phys_dm0_sender_core.y;
                receiver_dm0_rt_args[2] = first_send_val;
                receiver_dm0_rt_args[3] = second_send_val;
                receiver_dm0_rt_args[4] = rcv_from_noc0_first_addr;
                receiver_dm0_rt_args[5] = rcv_from_noc0_second_addr;
                receiver_dm0_rt_args[6] = rcv_from_noc0_first_addr;

                receiver_dm1_rt_args[0] = (uint32_t)phys_dm1_sender_core_noc1.x;
                receiver_dm1_rt_args[1] = (uint32_t)phys_dm1_sender_core_noc1.y;
                receiver_dm1_rt_args[2] = first_send_val;
                receiver_dm1_rt_args[3] = second_send_val;
                receiver_dm1_rt_args[4] = rcv_from_noc1_first_addr;
                receiver_dm1_rt_args[5] = rcv_from_noc1_second_addr;
                receiver_dm1_rt_args[6] = rcv_from_noc1_first_addr;

                tt_metal::SetRuntimeArgs(program, receiver_kernel_dm0, core, receiver_dm0_rt_args);
                tt_metal::SetRuntimeArgs(program, receiver_kernel_dm1, core, receiver_dm1_rt_args);

                detail::WriteToDeviceL1(device, core, L1_UNRESERVED_BASE, zero_vec, CoreType::WORKER);
            }
        }
    }

    tt::Cluster::instance().l1_barrier(device->id());

    fixture->RunProgram(device, program);

    std::vector<uint32_t> result_vec(available_l1_size/sizeof(uint32_t), 0);
    bool pass = true;
    for (uint32_t x = receivers.start_coord.x; x <= receivers.end_coord.x; x++) {
        for (uint32_t y = receivers.start_coord.y; y <= receivers.end_coord.y; y++) {
            CoreCoord receiver_core(x, y);
            detail::ReadFromDeviceL1(device, receiver_core, L1_UNRESERVED_BASE, available_l1_size, result_vec);
            pass &= validate_results(receiver_core, result_vec, transaction_size_bytes, aligned_transaction_size, (l1_bytes_per_noc/sizeof(uint32_t) - 1));
        }
    }

    return pass;
}


}

// This test tries to validate that order of issuing commands is preserved when different command buffers are used with the same noc and static VC
// Only one sender and one receiver core are used
TEST_F(CommonFixture, CommandBufferOrdering) {
    log_info (LogTest, "Running unicast command buffer ordering test: 4B transaction, same command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_cmd_buffer_ordering_test(this, devices_.at(0), 4, 0, false));
    log_info (LogTest, "Running unicast command buffer ordering test: 16B transaction, same command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_cmd_buffer_ordering_test(this, devices_.at(0), 16, 0, false));
    log_info (LogTest, "Running unicast command buffer ordering test: 32B transaction, diff command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_cmd_buffer_ordering_test(this, devices_.at(0), 32, 0, false));
    log_info (LogTest, "Running unicast command buffer ordering test: 4B transaction, diff command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_cmd_buffer_ordering_test(this, devices_.at(0), 4, 2, false));
    log_info (LogTest, "Running unicast command buffer ordering test: 16B transaction, diff command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_cmd_buffer_ordering_test(this, devices_.at(0), 16, 2, false));
    log_info (LogTest, "Running unicast command buffer ordering test: 32B transaction, diff command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_cmd_buffer_ordering_test(this, devices_.at(0), 32, 2, false));
}

// This test tries to validate that order of issuing commands is preserved when different command buffers are used with the same noc and static VC
// There is one sender and one receiver but an additional two cores that read from their local L1 and send to each other's L1
TEST_F(CommonFixture, CommandBufferOrderingAddNocTraffic) {
    log_info (LogTest, "Running unicast command buffer ordering test with NoC traffic: 4B transaction, same command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_cmd_buffer_ordering_test(this, devices_.at(0), 4, 0, true));
    log_info (LogTest, "Running unicast command buffer ordering test with NoC traffic: 16B transaction, same command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_cmd_buffer_ordering_test(this, devices_.at(0), 16, 0, true));
    log_info (LogTest, "Running unicast command buffer ordering test with NoC traffic: 32B transaction, same command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_cmd_buffer_ordering_test(this, devices_.at(0), 32, 0, true));
    log_info (LogTest, "Running unicast command buffer ordering test with NoC traffic: 4B transaction, diff command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_cmd_buffer_ordering_test(this, devices_.at(0), 4, 2, true));
    log_info (LogTest, "Running unicast command buffer ordering test with NoC traffic: 16B transaction, diff command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_cmd_buffer_ordering_test(this, devices_.at(0), 16, 2, true));
    log_info (LogTest, "Running unicast command buffer ordering test with NoC traffic: 32B transaction, diff command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_cmd_buffer_ordering_test(this, devices_.at(0), 32, 2, true));
}

// This test has:
//  1. sender kernels on the left-most column that mcast data across same row of the sender kernel on NoC 0
//  2. sender kernels on the bottom-most row that mcast data across same column of the sender kernel on NoC 1
TEST_F(CommonFixture, MulticastNoc0RowsNoc1Cols) {
    log_info (LogTest, "Running multicast command buffer ordering test: 4B transaction, same command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_multicast_test(this, devices_.at(0), 4, 0));
    log_info (LogTest, "Running multicast command buffer ordering test: 16B transaction, same command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_multicast_test(this, devices_.at(0), 16, 0));
    log_info (LogTest, "Running multicast command buffer ordering test: 32B transaction, same command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_multicast_test(this, devices_.at(0), 32, 0));
    log_info (LogTest, "Running multicast command buffer ordering test: 4B transaction, diff command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_multicast_test(this, devices_.at(0), 4, 2));
    log_info (LogTest, "Running multicast command buffer ordering test: 16B transaction, diff command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_multicast_test(this, devices_.at(0), 16, 2));
    log_info (LogTest, "Running multicast command buffer ordering test: 32B transaction, diff command buffer for both writes");
    EXPECT_TRUE(unit_tests_common::noc::run_multicast_test(this, devices_.at(0), 32, 2));
}
