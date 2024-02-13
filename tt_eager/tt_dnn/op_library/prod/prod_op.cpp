// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/prod/prod_op.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

#include <algorithm>
#include <optional>

#include "tt_metal/common/constants.hpp"
#include <tt_eager/tt_numpy/functions.hpp>
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

namespace tt {
using namespace constants;
namespace operations {
namespace primary {


inline bool is_1d_tensor(const Tensor& tensor) {
    const auto& shape = tensor.shape().without_padding();
    // because TT Tensor only support 4d shape, so if the first 3 dims are 1, assume it's 1d.
    return shape[0] == 1 && shape[1] == 1 && shape[2] == 1;
}

void Prod_op::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    // const auto& input_tensor_b = input_tensors.at(1);

    //TT_ASSERT(is_1d_tensor(input_tensor_a));
    // TT_ASSERT(is_1d_tensor(input_tensor_b));

    const auto& a_shape_wo_padding = input_tensor_a.shape().without_padding();
    // const auto& b_shape_wo_padding = input_tensor_b.shape().without_padding();
    // TT_ASSERT(a_shape_wo_padding[3] == b_shape_wo_padding[3]);

    TT_ASSERT(
        input_tensor_a.dtype() == DataType::BFLOAT16 || input_tensor_a.dtype() == DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_ASSERT(
        input_tensor_a.storage_type() == StorageType::DEVICE /*and input_tensor_b.storage_type() == StorageType::DEVICE*/,
        "Operands to matmul need to be on device!");
    TT_ASSERT(input_tensor_a.device() /*== input_tensor_b.device()*/, "Operands to matmul need to be on the same device!");
    TT_ASSERT(
        input_tensor_a.buffer() != nullptr /*and input_tensor_b.buffer() != nullptr*/,
        "Operands to matmul need to be allocated in buffers on device!");
}

std::vector<Shape> Prod_op::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto output_shape = input_tensor.shape();
    auto padding = output_shape.padding();
    output_shape[3] = TILE_WIDTH;
    padding[3] = Padding::PadDimension{0, 31};
    return {Shape(output_shape, padding)};
}

std::vector<Tensor> Prod_op::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks Prod_op::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return prod_single_core(input_tensor_a, output_tensor);
}

Tensor prod(const Tensor& input, const MemoryConfig& output_mem_config ) {
    // return operation::run(Prod_op{.output_mem_config = output_mem_config}, {input}).at(0);
    Tensor result =  tiled_prod( operation::run(Prod_op{.output_mem_config = output_mem_config}, {input}).at(0), output_mem_config);
    // return result;
    return tt::numpy::prod_result_computation<bfloat16>(result, result.dtype(), result.layout(), result.device(), output_mem_config);
}


}
}
}
