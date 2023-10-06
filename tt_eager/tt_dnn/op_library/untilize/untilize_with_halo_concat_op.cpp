// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>


#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {
namespace tt_metal {

operation::ProgramWithCallbacks untilize_with_halo_concat_multi_core(const Tensor& a, Tensor& output) {
    Program program = Program();

    Device *device = a.device();
    Buffer *src_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    Shape input_shape = a.shape();
    Shape output_shape = output.shape();

    DataFormat in_df = datatype_to_dataformat_converter(a.dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);

    DataFormat cb_df = in_df;
    uint32_t tile_size = tt_metal::detail::TileSize(cb_df);

    uint32_t ntiles = a.volume() / TILE_HW;
    uint32_t ntiles_per_block = a.shape()[3] / TILE_WIDTH;
    uint32_t nblocks = ceil((float) ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = a.shape()[3] * a.element_size();

    {
        log_debug(LogOp, "ntiles: {}", ntiles);
        log_debug(LogOp, "ntiles_per_block: {}", ntiles_per_block);
        log_debug(LogOp, "nblocks: {}", nblocks);
    }

    auto grid_size = device->compute_with_storage_grid_size();
    // auto [ncores, ncores_x, ncores_x_cliff, ncores_y, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] = split_blocks_across_cores(grid_size, nblocks);

    int32_t ncores_x = device->compute_with_storage_grid_size().x;
    int32_t ncores_y = device->compute_with_storage_grid_size().y;
    CoreRangeSet all_cores = a.shard_spec().value().shard_grid;
    int32_t ncores = 0;
    for (const auto& core_range : all_cores.ranges()) {
        ncores += core_range.size();
    }
    CoreRangeSet core_range = all_cores;
    CoreRangeSet core_range_cliff = CoreRangeSet({});
    uint32_t nblocks_per_core = a.shard_spec().value().shard_shape.first / TILE_HEIGHT;
    uint32_t nblocks_per_core_cliff = 0;

    {
        log_debug(LogOp, "shard_shape: {},{}", a.shard_spec().value().shard_shape.first, a.shard_spec().value().shard_shape.second);
        log_debug(LogOp, "ncores: {}", ncores);
        log_debug(LogOp, "ncores_x: {}", ncores_x);
        log_debug(LogOp, "ncores_y: {}", ncores_y);
        log_debug(LogOp, "nblocks_per_core: {}", nblocks_per_core);
        log_debug(LogOp, "nblocks_per_core_cliff: {}", nblocks_per_core_cliff);
    }

    uint32_t src_cb_id = CB::c_in0;
    uint32_t num_input_tiles = ntiles_per_block * nblocks_per_core;
    auto cb_src0 = CreateCircularBuffers(
        program,
        src_cb_id,
        all_cores,
        num_input_tiles,
        num_input_tiles * tile_size,
        cb_df,
        a.buffer()->address(),
        true
    );

    // output of untilize from compute kernel goes into this CB
    uint32_t untilize_out_cb_id = CB::c_out0;
    uint32_t num_output_tiles = ntiles_per_block * nblocks_per_core;
    auto cb_untilize = CreateCircularBuffers(
        program,
        untilize_out_cb_id,
        all_cores,
        num_output_tiles,
        num_output_tiles * tile_size,
        cb_df,
        output.buffer()->address(),
        true
    );

    // output after concatenating halo and padding goes into this CB, as input to next op.
    uint32_t out_cb_id = CB::c_out1;
    uint32_t out_cb_pagesize = ;    // one row per page + padding
    uint32_t out_cb_npages = local_npages + halo_npages;
    auto cb_out = CreateCircularBuffers(
        program,
        out_cb_id,
        all_cores,
        out_cb_npages,
        out_cb_pagesize * out_cb_npages,
        out_df
    );

    /** reader
     */

    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t) src_cb_id
    };

    KernelID reader_kernel_id = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    /** writer
     */
    std::vector<uint32_t> writer_ct_args = {
        (std::uint32_t) output_cb_id
    };
    KernelID writer_kernel_id = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_sharded_with_halo.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    /** compute
     */
    TT_ASSERT(core_range_cliff.ranges().size() == 0);
    vector<uint32_t> compute_args = {
        (uint32_t) nblocks_per_core,    // per_core_block_cnt
        (uint32_t) ntiles_per_block,    // per_block_ntiles
    };
    KernelID untilize_kernel_id = CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/untilize.cpp",
        core_range,
        ComputeConfig{
            .compile_args = compute_args});

    // 1D distribution of blocks across all cores
    uint32_t ncores_full = ncores;
    // cliff core not yet supported
    TT_ASSERT(nblocks_per_core_cliff == 0);

    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    for (uint32_t i = 0; i < ncores_full; ++ i) {
        CoreCoord core = {i % ncores_x, i / ncores_x};

        // reader runtime args
        vector<uint32_t> reader_rt_args = {
            ntiles_per_block * nblocks_per_core // ntiles
        };

        // writer runtime args
        vector<uint32_t> writer_rt_args = {
            ntiles_per_block * nblocks_per_core // ntiles
        };

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            reader_rt_args
        );

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            writer_rt_args
        );

        tile_start_id += ntiles_per_block * nblocks_per_core;
        row_start_id += TILE_HEIGHT * nblocks_per_core;
    }
    if (ncores_full < ncores) {
        // last core is the cliff core with nblocks_per_core_cliff blocks
        // TODO: support cliff core
        TT_ASSERT(false);
    }

    auto override_runtime_args_callback = [
        reader_kernel_id=reader_kernel_id,
        writer_kernel_id=writer_kernel_id,
        ncores=ncores,
        ncores_x=ncores_x
    ](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < ncores; ++ i) {
            CoreCoord core = {i % ncores_x, i / ncores_x};
            // in and out are sharded
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void UntilizeWithHaloConcat::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands to untilize need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16, "Only bloat16 dataformat supported");
    TT_ASSERT(input_tensor_a.layout() == Layout::TILE, "Input tensor is not TILE for untilize");
    TT_ASSERT(input_tensor_a.memory_config().is_sharded());
    TT_ASSERT(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Only works for sharded input");
    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);
}

std::vector<Shape> UntilizeWithHaloConcat::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_shape = input_tensors.at(0).shape();
    Shape output_shape = input_shape;
    output_shape[3] = output_shape[3] ... // TODO
    return {output_shape};
}

std::vector<Tensor> UntilizeWithHaloConcat::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensor.dtype(), Layout::ROW_MAJOR, input_tensor.device(), this->output_mem_config, input_tensor.shard_spec().value())};
}

operation::ProgramWithCallbacks UntilizeWithHaloConcat::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return {untilize_with_halo_concat_multi_core(input_tensor_a, output_tensor)};
}

tt::stl::reflection::Attributes UntilizeWithHaloConcat::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor untilize_with_halo_concat(const Tensor &input_tensor_a, const MemoryConfig& mem_config) {
    return operation::run_without_autoformat(UntilizeWithHaloConcat{mem_config}, {input_tensor_a}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
