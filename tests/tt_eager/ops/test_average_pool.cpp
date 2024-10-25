// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/global_avg_pool/global_avg_pool.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/operations/numpy/functions.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "common/constants.hpp"

using tt::tt_metal::Device;
using tt::tt_metal::Tensor;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::LegacyShape;

Tensor run_avg_pool_2d_resnet(tt::tt_metal::LegacyShape& tensor_shape, Device* device) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    auto input_tensor = ttnn::numpy::random::random(tensor_shape, DataType::BFLOAT16);
    auto padded_input_shape = AutoFormat::pad_to_tile_shape(tensor_shape, false, false);
    Tensor padded_input_tensor = input_tensor;
    if (!AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape)) {
        padded_input_tensor = AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, 0, Layout::TILE);    // pad with 0s
    }
    auto device_output = global_avg_pool2d(padded_input_tensor);
    return device_output.cpu();
};

int main () {
    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);

    tt::tt_metal::LegacyShape resnet18_shape = {1, 1, 7 * 7, 2048};
    auto result = run_avg_pool_2d_resnet(resnet18_shape, device);

    TT_FATAL(result.get_padded_shape() == ttnn::SimpleShape({1, 1, tt::constants::TILE_HEIGHT, 2048}), "Incorrect shape {}.", result.get_padded_shape());
    TT_FATAL(result.get_logical_shape() == ttnn::SimpleShape({1, 1, 1, 2048}), "Incorrect shape {}.", result.get_logical_shape());

    TT_FATAL(tt::tt_metal::CloseDevice(device), "Error");
    return 0;
}
