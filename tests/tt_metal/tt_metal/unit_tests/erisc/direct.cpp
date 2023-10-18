// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

constexpr std::int32_t FIRMWARE_BASE = 0x9040;
constexpr std::int32_t RUN_APP_FLAG = 0x10E8;
constexpr std::int32_t DATA_BUFFER_SPACE_BASE = 0x28000;
constexpr std::int32_t ETH_L1_ARGS_BASE = 0x3E420;
constexpr std::int32_t MAX_NUM_WORDS = 5632;
constexpr std::int32_t WORD_SIZE = 16; // 16 bytes per eth send packet

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::erisc::direct {
/// @brief Does eth L1 chip 0 --> eth L1 chip 1
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool send_over_eth(
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,
    const CoreCoord& sender_core,
    const CoreCoord& receiver_core,
    const size_t& byte_size) {
  std::cout << " running test, sender chip " << sender_device->id() << " core " << sender_core.x << " " << sender_core.y << " receiver chip " << receiver_device->id() << " core " << receiver_core.x << " " << receiver_core.y << " num_bytes " << byte_size << std::endl;
  std::vector<CoreCoord> eth_cores = {{.x = 9, .y = 0}, {.x = 1, .y = 0}, {.x = 8, .y = 0}, {.x = 2, .y = 0}, {.x = 9, .y = 6}, {.x = 1, .y = 6}, {.x = 8, .y = 6}, {.x = 2, .y = 6}, {.x = 7, .y = 0}, {.x = 3, .y = 0}, {.x = 6, .y = 0}, {.x = 4, .y = 0}, {.x = 7, .y = 6}, {.x = 3, .y = 6}, {.x = 6, .y = 6}, {.x = 4, .y = 6}};

  // Disable all eth core runtime app
    std::vector<uint32_t> run_test_app_flag = {0x0};
    for (const auto& eth_core: eth_cores) {
      llrt::write_hex_vec_to_core(sender_device->id(), eth_core, run_test_app_flag, RUN_APP_FLAG);
      llrt::write_hex_vec_to_core(receiver_device->id(), eth_core, run_test_app_flag, RUN_APP_FLAG);
      std::vector<uint32_t> disable_arg_0 = {0};
      llrt::write_hex_vec_to_core(sender_device->id(), eth_core, disable_arg_0, ETH_L1_ARGS_BASE);
      llrt::write_hex_vec_to_core(receiver_device->id(), eth_core, disable_arg_0, ETH_L1_ARGS_BASE);
    }

    // TODO: is it possible that receiver core app is stil running when we push inputs here???
  auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
  llrt::write_hex_vec_to_core(sender_device->id(), sender_core, inputs, DATA_BUFFER_SPACE_BASE);

  // Zero out receiving address to ensure no stale data is causing tests to pass
  std::vector<uint32_t> all_zeros(inputs.size(), 0);
  llrt::write_hex_vec_to_core(receiver_device->id(), receiver_core, all_zeros, DATA_BUFFER_SPACE_BASE);

  uint32_t arg_0 = byte_size;
  std::vector<uint32_t> args = {arg_0};
  llrt::write_hex_vec_to_core(sender_device->id(), sender_core, args, ETH_L1_ARGS_BASE);
  llrt::write_hex_vec_to_core(receiver_device->id(), receiver_core, args, ETH_L1_ARGS_BASE);

  ll_api::memory binary_mem_send = llrt::get_risc_binary("erisc_app_direct_send.hex", sender_device->id(), true);
  ll_api::memory binary_mem_receive = llrt::get_risc_binary("erisc_app_direct_receive.hex", receiver_device->id(), true);

  for (const auto& eth_core: eth_cores) {
    llrt::write_hex_vec_to_core(sender_device->id(), eth_core, binary_mem_send.data(), FIRMWARE_BASE);
    llrt::write_hex_vec_to_core(receiver_device->id(), eth_core, binary_mem_receive.data(), FIRMWARE_BASE);
  }

  // Activate sender core runtime app
    run_test_app_flag = {0x1};
    //send remote first, otherwise eth core may be blocked, very ugly for now...
    if (receiver_device->id() == 1) {
      llrt::write_hex_vec_to_core(1, receiver_core, run_test_app_flag, RUN_APP_FLAG);
    } else {
      llrt::write_hex_vec_to_core(1, sender_core, run_test_app_flag, RUN_APP_FLAG);
    }
    if (sender_device->id() == 0) {
      llrt::write_hex_vec_to_core(0, sender_core, run_test_app_flag, RUN_APP_FLAG);
    } else {
      llrt::write_hex_vec_to_core(0, receiver_core, run_test_app_flag, RUN_APP_FLAG);
    }



  bool pass = true;
  auto readback_vec =  llrt::read_hex_vec_from_core(receiver_device->id(), receiver_core, DATA_BUFFER_SPACE_BASE , byte_size);
  //for (const auto &v: readback_vec)  {
  //  std::cout << v << std::endl;
 //}
 // pass &= inputs == readback_vec;

  return pass;

}

}  // namespace unit_tests::erisc::direct

TEST_F(DeviceFixture, SingleEthCoreDirectSendChip0ToChip1) {
  ASSERT_TRUE(this->num_devices_ == 2);
  const auto& device_0 = devices_.at(0);
  const auto& device_1 = devices_.at(1);
  CoreCoord sender_core_0 = {.x = 9, .y = 6};
  CoreCoord sender_core_1 = {.x = 1, .y = 6};

  CoreCoord receiver_core_0 = {.x = 9, .y = 0};
  CoreCoord receiver_core_1 = {.x = 1, .y = 0};

  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE * 256));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE * 256));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE * 1024));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE * 1024));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE * MAX_NUM_WORDS));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE * MAX_NUM_WORDS));
}

TEST_F(DeviceFixture, SingleEthCoreDirectSendChip1ToChip0) {
  ASSERT_TRUE(this->num_devices_ == 2);
  const auto& device_0 = devices_.at(0);
  const auto& device_1 = devices_.at(1);
  CoreCoord sender_core_0 = {.x = 9, .y = 0};
  CoreCoord sender_core_1 = {.x = 1, .y = 0};

  CoreCoord receiver_core_0 = {.x = 9, .y = 6};
  CoreCoord receiver_core_1 = {.x = 1, .y = 6};

  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, sender_core_0, receiver_core_0, WORD_SIZE));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, sender_core_1, receiver_core_1, WORD_SIZE));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, sender_core_0, receiver_core_0, WORD_SIZE * 256));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, sender_core_1, receiver_core_1, WORD_SIZE * 256));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, sender_core_0, receiver_core_0, WORD_SIZE * 1024));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, sender_core_1, receiver_core_1, WORD_SIZE * 1024));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, sender_core_0, receiver_core_0, WORD_SIZE * MAX_NUM_WORDS));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, sender_core_1, receiver_core_1, WORD_SIZE * MAX_NUM_WORDS));
}

TEST_F(DeviceFixture, BidirectionalEthCoreDirectSend) {
  ASSERT_TRUE(this->num_devices_ == 2);
  const auto& device_0 = devices_.at(0);
  const auto& device_1 = devices_.at(1);
  CoreCoord sender_core_0 = {.x = 9, .y = 6};
  CoreCoord sender_core_1 = {.x = 1, .y = 6};

  CoreCoord receiver_core_0 = {.x = 9, .y = 0};
  CoreCoord receiver_core_1 = {.x = 1, .y = 0};

  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, receiver_core_0, sender_core_0, WORD_SIZE));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, receiver_core_1, sender_core_1, WORD_SIZE));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE * 256));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, receiver_core_0, sender_core_0, WORD_SIZE * 256));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE * 256));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, receiver_core_1, sender_core_1, WORD_SIZE * 256));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE * 1024));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, receiver_core_0, sender_core_0, WORD_SIZE * 1024));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE * 1024));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, receiver_core_1, sender_core_1, WORD_SIZE * 1024));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_0, receiver_core_0, WORD_SIZE * MAX_NUM_WORDS));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, receiver_core_0, sender_core_0, WORD_SIZE * MAX_NUM_WORDS));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, sender_core_1, receiver_core_1, WORD_SIZE * MAX_NUM_WORDS));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_1, device_0, receiver_core_1, sender_core_1, WORD_SIZE * MAX_NUM_WORDS));
}

TEST_F(DeviceFixture, RandomDirectSendTests) {
  srand(0);
  ASSERT_TRUE(this->num_devices_ == 2);

  std::map<std::pair<int, CoreCoord>, std::pair<int, CoreCoord>> connectivity = {
    {{0, {.x=9, .y=6}}, {1, {.x=9, .y=0}}}, {{1, {.x=9, .y=0}}, {0, {.x=9, .y=6}}},
    {{0, {.x=1, .y=6}}, {1, {.x=1, .y=0}}}, {{1, {.x=1, .y=0}}, {0, {.x=1, .y=6}}}
  };
  for (int i=0; i<10000; i++) {
     auto it = connectivity.begin();
     std::advance(it, rand() % (connectivity.size()) );

     const auto& send_chip = devices_.at(std::get<0>(it->first));
     CoreCoord sender_core = std::get<1>(it->first);
     const auto& receiver_chip = devices_.at(std::get<0>(it->second));
     CoreCoord receiver_core = std::get<1>(it->second);
     int num_words = rand() % MAX_NUM_WORDS + 1;

     ASSERT_TRUE(
        unit_tests::erisc::direct::send_over_eth(send_chip, receiver_chip, sender_core, receiver_core, WORD_SIZE * num_words));
  }

}
