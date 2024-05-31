// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/conv/optimized_conv_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/sharding_utilities.hpp"
#include "tt_dnn/op_library/sliding_window_op_infra/sliding_window.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_stl/reflection.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

    const uint32_t act_cb = CB::c_in0;
    const uint32_t weight_cb = CB::c_in1;
    const uint32_t bias_cb = CB::c_in2;
    const uint32_t sharded_act_cb = CB::c_in3;
    const uint32_t cb_for_reader_indices = CB::c_in4;
    const uint32_t cb_for_l1_array = CB::c_in5;
    const uint32_t act_cb_row_major_bfloat16 = CB::c_in6;
    const uint32_t act_cb_second_reader = CB::c_in7;
    const uint32_t matmul_partials_cb = CB::c_intermed0;
    const uint32_t tilize_mode_tilized_act_cb = CB::c_intermed1;
    const uint32_t untilize_mode_reblock_cb = CB::c_intermed2;
    const uint32_t out0_cb = CB::c_out0;


/*tuple<CBHandle, CBHandle> create_CBs_for_sharded_input_v2(
    tt_metal::Program& program,
    const Tensor& input,
    CoreRange core,
    uint32_t num_cb0_tiles,
    uint32_t num_cb1_tiles,
    uint32_t num_cb0_tilized_tiles,
    uint32_t num_output_tiles,
    uint32_t num_reblock_cb_tiles,
    uint32_t num_writer_output_tiles,
    bool untilize_out,
    DataFormat act_df,
    DataFormat weight_df,
    DataFormat tilized_act_df,
    DataFormat out_df,
    DataFormat bias_df,
    bool weight_width_sliced,
    const Tensor& output,
    uint32_t bias_ntiles,
    bool with_bias,
    bool split_reader,
    bool fp32_dest_acc_en,
    bool packer_l1_acc_en);*/

    operation::ProgramWithCallbacks multi_core_optimized_conv_sharded_v2_impl_width_sharded(
    tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const Shape& ashape,
    std::optional<const Tensor> bias,
    const std::optional<const Tensor> conv_reader_indices,
    vector<int> conv_params,
    uint32_t output_channels,
    bool untilize_out,
    bool has_bias,
    bool fuse_relu,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    uint32_t extra_padding_for_32B_alignment,
    bool use_shallow_conv_variant,
    bool transpose_mcast,
    Tensor& output,
    DeviceComputeKernelConfig compute_kernel_config);

    }
}
