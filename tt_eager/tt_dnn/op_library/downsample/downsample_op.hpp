// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization

struct Downsample {
    std::array<uint32_t, 5> downsample_params;
    DataType output_dtype;
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("downsample_params", "output_dtype");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->downsample_params), std::cref(this->output_dtype));
    }
};

//operation::ProgramWithCallbacks downsample_multi_core(const Tensor &a, Tensor& output);
operation::ProgramWithCallbacks downsample_multi_core(const Tensor &a, std::array<uint32_t, 5> downsample_params, Tensor& output);

Tensor downsample (const Tensor &a, std::array<uint32_t, 5> downsample_params, std::optional<DataType> output_dtype=std::nullopt);

struct DownsampleV2 {
    std::array<uint32_t, 5> downsample_params_;
    const uint32_t ncores_nhw_;
    const uint32_t max_resharded_untilized_nsticks_per_core_;
    const std::vector<std::pair<int32_t, int32_t>>& l_data_start_and_size_;
    const std::vector<std::pair<int32_t, int32_t>>& local_data_start_and_size_;
    const std::vector<std::pair<int32_t, int32_t>>& r_data_start_and_size_;
    const std::vector<int32_t> l_data_src_start_offset_;
    const std::vector<int32_t> local_data_src_start_offset_;
    const std::vector<int32_t> r_data_src_start_offset_;
    const MemoryConfig out_mem_config_;
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        "downsample_params",
        "ncores_nhw_",
        "max_resharded_untilized_nsticks_per_core_",
        "out_mem_config_");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(downsample_params_),
            std::cref(ncores_nhw_),
            std::cref(max_resharded_untilized_nsticks_per_core_),
            std::cref(out_mem_config_));
    }
};

Tensor downsample_v2(const Tensor& a,
                             std::array<uint32_t, 5> downsample_params,
                            const uint32_t ncores_nhw,
                             const uint32_t max_resharded_untilized_nsticks_per_core,
                             const std::vector<std::pair<int32_t, int32_t>>& l_data_start_and_size,
                             const std::vector<std::pair<int32_t, int32_t>>& local_data_start_and_size,
                             const std::vector<std::pair<int32_t, int32_t>>& r_data_start_and_size,
                             const std::vector<int32_t>& l_data_src_start_offsets_per_core,
                             const std::vector<int32_t>& local_data_src_start_offsets_per_core,
                             const std::vector<int32_t>& r_data_src_start_offsets_per_core,
                             const MemoryConfig& mem_config);

// namespace downsample_helpers {
// uint32_t get_num_cores(CoreCoord grid_size, uint32_t nblocks);
// }

}  // namespace tt_metal

}  // namespace tt
