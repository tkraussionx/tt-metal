// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/types.hpp"
#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_impl.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "common/bfloat16.hpp"
#include "common/constants.hpp"

#include "tt_numpy/functions.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;

bool test_tensor_ownership_sanity(Device* device) {
    // Sanity test tensor read, write and update paths with synchronous
    // Ensure that tensor data is copied and owned as expected
    log_info(LogTest, "Running {}", __FUNCTION__);
    bool pass = true;
    Tensor host_tensor = tt::numpy::arange<float>(0, 32 * 32 * 4, 1);
    auto input_tensor_0 = host_tensor.reshape(1, 1, 32, 128).to(Layout::TILE).to(device);
    auto input_tensor_1 =  host_tensor.reshape(1, 1, 32, 128).to(Layout::TILE).to(device);
    auto output = tt::tt_metal::add(input_tensor_0, input_tensor_1).cpu();
    // auto readback_tensor = device_tensor.cpu();

    // std::this_thread::sleep_for(std::chrono::seconds(5));
    input_tensor_1.print();
    input_tensor_0.print();
    output.print();
    // std::cout << device_tensor.get_dtype() << std::endl;
    std::cout << output.get_shape() << std::endl;
    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    int device_id = 0;
    tt_metal::Device *device = tt_metal::CreateDevice(device_id);
    pass &= test_tensor_ownership_sanity(device);
    // pass &= test_tensor_async_data_movement(device);
    pass &= CloseDevice(device);
    // TT_ASSERT(pass, "Tests failed");
    // log_info(LogTest, "Tests Passed");
}
