// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <random>
#include <tt_numpy/functions.hpp>

#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tt_dnn/op_library/embeddings/embeddings_op.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace constants;

void run_embeddings_bw(Device *device, std::vector<uint8_t> indexes) {
    Shape shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};
    Tensor weight = tt::numpy::random::random(shape, DataType::BFLOAT16, Layout::TILE).to(device);
    weight.print();

    Tensor index = tt::numpy::from_vector(shape, std::move(indexes), Layout::TILE, device);

    Tensor device_output_tensor = embeddings_bw_test(weight, index);
    Tensor output_tensor = device_output_tensor.cpu();

    output_tensor.print();
}

int main(int argc, char **argv) {
    int device_id = 0;
    tt_metal::Device *device = tt_metal::CreateDevice(device_id);


    // 1. Indexes that cause no reshuffling
    std::vector<uint8_t> no_op_indexes(1024, 0);
    std::iota(no_op_indexes.begin(), no_op_indexes.begin() + 32, 0);
    run_embeddings_bw(device, std::move(no_op_indexes));

    // 2. Unique reshfulling
    std::vector<uint8_t> unique_indexes(1024, 0);
    std::vector<uint8_t> initial_indexes(32);
    std::iota(initial_indexes.begin(), initial_indexes.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(initial_indexes.begin(), initial_indexes.end(), g);
    std::copy(initial_indexes.begin(), initial_indexes.end(), unique_indexes.begin());
    run_embeddings_bw(device, std::move(unique_indexes));

    // 3. Reshuffling with out-of-range indexes (>= 32) and no repeating indexes
    std::vector<uint8_t> out_of_range_indexes(1024, 0);
    std::unordered_set<uint8_t> unique_elements;
    std::uniform_int_distribution<uint8_t> d(0, 64);
    while (unique_elements.size() < 32) {
        unique_elements.insert(d(g));
    }
    std::copy(unique_elements.begin(), unique_elements.end(), out_of_range_indexes.begin());
    run_embeddings_bw(device, std::move(out_of_range_indexes));

    // 4. Reshuffling with reduction (repeating indexes)
    std::vector<uint8_t> repeating_indexes(1024, 0);
    std::uniform_int_distribution<uint8_t> d2(0, 31);
    for (int i = 0; i < 32; i++) {
        repeating_indexes[i] = d2(g);
    }
    run_embeddings_bw(device, std::move(repeating_indexes));

    bool pass = CloseDevice(device);

    if (CloseDevice(device)) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);

    return 0;
}
