// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "slice_op.hpp"
#include "slice_program_factory.hpp"

namespace ttnn::operations::data_movement {

inline __attribute__((always_inline)) uint32_t get_upper_dims_compressed(const ttnn::Shape& shape) {
    return std::accumulate(shape.begin(), shape.end() - 2, 1, std::multiplies<uint32_t>{});
}

inline __attribute__((always_inline)) uint32_t get_upper_start_offset(const Tensor &tensor, const Shape &slice_start) {
    // offset for every dim except last 2
    uint32_t start_offset = 0;
    const auto& shape = tensor.get_shape().with_tile_padding();

    uint32_t num_pages = tensor.volume();
    if (tensor.get_layout() == Layout::TILE) {
        num_pages /= tt::constants::TILE_HW;
    } else {
        uint32_t page_width = shape[-1];
        num_pages /= page_width;
    }

    for (uint32_t dim_outer = 0; dim_outer < shape.rank() - 2; dim_outer++) {
        uint32_t compressed_dims = 1;
        for (uint32_t dim_inner = 0; dim_inner <= dim_outer; dim_inner++) {
            compressed_dims *= shape[dim_inner];
        }
        start_offset += (num_pages / compressed_dims) * slice_start[dim_outer];
    }
    return start_offset;
}

uint32_t get_tiled_start_offset(const Tensor &input_tensor, const Shape &slice_start) {
    using namespace tt::constants;
    uint32_t num_input_pages = input_tensor.volume() / (TILE_HW);
    const auto& shape = input_tensor.get_shape().with_tile_padding();
    uint32_t upper_dims_compressed = get_upper_dims_compressed(shape);
    uint32_t num_pages_width =
        num_input_pages / (upper_dims_compressed * (shape[-2] / TILE_HEIGHT));

    // offset for every dim except last 2
    uint32_t start_offset = get_upper_start_offset(input_tensor, slice_start);

    start_offset += slice_start[-2] / TILE_HEIGHT * num_pages_width + slice_start[-1] / TILE_WIDTH;
    return start_offset;
}

uint32_t get_rm_start_offset(const Tensor &tensor, const Shape &slice_start) {
    uint32_t start_offset = 0;

    if (tensor.get_shape().with_tile_padding().rank() >= 2) {
        const auto& shape = tensor.get_shape().with_tile_padding();
        uint32_t num_pages = tensor.volume() / shape[-1];
        uint32_t upper_dims_compressed = get_upper_dims_compressed(shape);
        start_offset = get_upper_start_offset(tensor, slice_start);
        start_offset += slice_start[-2];
    }

    return start_offset;
}



void SliceDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    using namespace tt::constants;
    const auto &input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to unpad need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE || input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Error");
    TT_FATAL(input_tensor_a.get_shape().with_tile_padding().rank() == this->slice_start.rank() && this->slice_start.rank() == this->slice_end.rank(), "Error");
    for (uint32_t i = 0; i < input_tensor_a.get_shape().with_tile_padding().rank(); i++) {
        TT_FATAL(this->slice_start[i] < input_tensor_a.get_shape().with_tile_padding()[i], "Error");
        TT_FATAL(this->slice_end[i] <= input_tensor_a.get_shape().with_tile_padding()[i], "Ends {} must be less than or equal to the shape of the tensor {}", this->slice_end[i], input_tensor_a.get_shape().with_tile_padding()[i]);
        // Check if start shape is <= end shape
        TT_FATAL(this->slice_start[i] <= this->slice_end[i], "Error");
    }
    if(!output_tensors.empty() && output_tensors[0].has_value()){
        const auto output_shape_required = this->compute_output_shapes(input_tensors)[0];
        const auto& out_tensor = output_tensors[0].value();
        TT_FATAL(out_tensor.get_shape().with_tile_padding() == output_shape_required, "The input tensors need a shape of {}, however the output tensor is only {}", output_shape_required,  out_tensor.get_shape().with_tile_padding());
    }
    auto output_tensor_shape = this->compute_output_shapes(input_tensors)[0];
    if (step.has_value()) { // if all ones modify before passing in to function
        TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Strided slice is only supported for row major layout");
        TT_FATAL(!input_tensor_a.is_sharded(), "Strided slice is not supported for sharded tensor");
        TT_FATAL(input_tensor_a.get_dtype() == DataType::BFLOAT16, "Strided slice is only supported for BFLOAT16");
        TT_FATAL(this->step.value().size() == this->slice_end.rank(), "Number of steps {} must match number of ends/starts {}", this->step.value().size(), this->slice_end.rank());
    }
    if (input_tensor_a.get_layout() == Layout::TILE) {
        TT_FATAL(input_tensor_a.volume() % TILE_HW == 0, "Error");
        TT_FATAL(
            (output_tensor_shape[-2] % TILE_HEIGHT == 0) && (this->slice_start[-2] % TILE_HEIGHT == 0),
            "Can only unpad tilized tensor with full tiles");
        TT_FATAL(
            (output_tensor_shape[-1] % TILE_WIDTH == 0) && (this->slice_start[-1] % TILE_WIDTH == 0),
            "Can only unpad tilized tensor with full tiles");
    } else if (input_tensor_a.get_layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            (output_tensor_shape[-1] * input_tensor_a.element_size() % sizeof(uint32_t) == 0),
            "An unpadding slice operations for a RowMajor layout on the output tensor requires the last dimension to be on a 32 bit boundary. For example, the final dimension needs to be divisible by 2 for bfloat16. The resulting tensor shape is {}, which is not 4B aligned as the last dimension is {}",
                        output_tensor_shape[-1], input_tensor_a.element_size());
        if (this->step.has_value()) {
            for (uint32_t i = 0; i < input_tensor_a.get_shape().with_tile_padding().rank(); i++) {
                TT_FATAL(this->step.value()[i] > 0, "Step({}) = {} should be positive", i, this->step.value()[i]);
            }
        }
        else {
            TT_FATAL(this->slice_start[-1] * input_tensor_a.element_size() % sizeof(uint32_t) == 0, "Slice needs to start at an aligned position");
        }
    }
}

std::vector<ttnn::Shape> SliceDeviceOperation::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    std::vector<uint32_t> out_shape;
    auto rank = input_tensors[0].get_shape().with_tile_padding().rank();
    out_shape.reserve(rank);
    if (!step.has_value()) {
        for (uint32_t i = 0; i < rank; i++) {
            out_shape.push_back(this->slice_end[i] - this->slice_start[i]);
        }
    }
    else {

        auto output_dim_i = [this] (size_t i) {
            return (this->slice_end[i] - this->slice_start[i] + this->step.value()[i] - 1) / this->step.value()[i];
        };
        for (uint32_t i = 0; i < rank; i++) {
            out_shape.push_back(output_dim_i(i));
        }
    }
    ttnn::Shape output_tensor_shape(out_shape);
    return {output_tensor_shape};
}

std::vector<Tensor> SliceDeviceOperation::create_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    if (!output_tensors.empty() && output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    }
    const auto &input_tensor_a = input_tensors.at(0);
    const auto shapes = compute_output_shapes(input_tensors);

    if (input_tensor_a.is_sharded()) {
        return {create_device_tensor(
            shapes[0],
            input_tensor_a.get_dtype(),
            input_tensor_a.get_layout(),
            input_tensor_a.device(),
            this->output_mem_config)};
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, input_tensor_a.get_dtype(), input_tensor_a.get_layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks SliceDeviceOperation::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input_tensor_a = input_tensors.at(0);
    auto &output_tensor = output_tensors.at(0);

    return detail::slice_multi_core(input_tensor_a, output_tensor, this->slice_start, this->slice_end, this->step);
}


}  // namespace ttnn::operations::reduction
