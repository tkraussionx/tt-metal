// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/pool/avgpool/device/avg_pool2d_device_op.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

namespace ttnn::operations::experimental::pool {

namespace {

AvgPool2D::MultiCore::cached_program_t avg_pool_2d_multi_cire_sharded_with_halo(
    Program& program,
    const Tensor& input,
    const Tensor& reader_indices,
    Tensor& output,
    uint32_t kernel_size_h,
    uint32_t kernel_size_w,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t nblocks) {
    // This should allocate a DRAM buffer on the device
    Device* device = input.device();
    tt::tt_metal::Buffer* src_dram_buffer = input.buffer();
    tt::tt_metal::Buffer* reader_indices_buffer = reader_indices.buffer();
    tt::tt_metal::Buffer* dst_dram_buffer = output.buffer();

    const tt::tt_metal::LegacyShape input_shape = input.get_legacy_shape();
    const tt::tt_metal::LegacyShape output_shape = output.get_legacy_shape();

    tt::DataFormat in_df = datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat out_df = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);
    uint32_t in_nbytes_c = input_shape[3] * in_nbytes;                                      // row of input (channels)
    uint32_t out_nbytes_c = output_shape[3] * out_nbytes;                                   // row of output (channels)
    TT_ASSERT((in_nbytes_c & (in_nbytes_c - 1)) == 0, "in_nbytes_c should be power of 2");  // in_nbytes_c is power of 2
    TT_ASSERT(
        (out_nbytes_c & (out_nbytes_c - 1)) == 0, "out_nbytes_c should be power of 2");  // out_nbytes_c is power of 2

    tt::DataFormat indices_df =
        tt::DataFormat::RawUInt16;  // datatype_to_dataformat_converter(reader_indices.get_dtype());
    uint32_t indices_nbytes = datum_size(indices_df);

    const uint32_t kernel_size_hw = kernel_size_w * kernel_size_h;  // number of valid rows, to read
    const uint32_t kernel_size_hw_padded = tt::round_up(kernel_size_hw, tt::constants::TILE_HEIGHT);
    const uint32_t in_ntiles_hw = (uint32_t)std::ceil((float)kernel_size_hw_padded / tt::constants::TILE_HEIGHT);
    const uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[3] / tt::constants::TILE_WIDTH);
    const uint32_t out_ntiles_c = (uint32_t)std::ceil((float)output_shape[3] / tt::constants::TILE_WIDTH);

    // Hardware can do reduction of 8 tiles at a time.
    // CB sizes can be restricted to this in case input channels are more than 256 to perform reduction iteratively.
    constexpr uint32_t MAX_SMALL_KERNEL_SIZE_HW = 16;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    const bool is_large_kernel = kernel_size_hw > MAX_SMALL_KERNEL_SIZE_HW;
    const bool is_wide_reduction = in_ntiles_c > MAX_TILES_PER_REDUCTION;

    TT_ASSERT(nblocks == 1, "Multiple blocks not yet supported");

    uint32_t tile_w = tt::constants::TILE_WIDTH;
    if (input_shape[3] < tt::constants::TILE_WIDTH) {
        TT_FATAL(input_shape[3] == 16, "Error");
        tile_w = tt::constants::FACE_WIDTH;
    }
    const uint32_t out_w_loop_count = std::ceil((float)out_w / nblocks);

    // distributing out_hw across the grid
    const auto grid_size = device->compute_with_storage_grid_size();
    const auto all_cores = input.shard_spec().value().grid;
    const uint32_t ncores = all_cores.num_cores();
    const auto core_range = all_cores;
    const auto core_range_cliff = CoreRangeSet({});
    const uint32_t in_nhw_per_core = input.shard_spec()->shape[0];
    const uint32_t in_nhw_per_core_cliff = 0;
    const uint32_t out_nhw_per_core = output.shard_spec()->shape[0];

    uint32_t ncores_w = grid_size.x;

    tt::log_info("out_nhw_per_core {}", out_nhw_per_core);

    // TODO: support generic nblocks
    TT_ASSERT(
        out_nhw_per_core % nblocks == 0,
        "number of sticks per core ({}) should be divisible by nblocks ({})",
        out_nhw_per_core,
        nblocks);

    // CBs
    const uint32_t multi_buffering_factor = 2;

    const uint32_t split_reader = 0;

    // scalar CB as coefficient of reduce
    const uint32_t in_scalar_cb_id = tt::CB::c_in4;
    const uint32_t in_scalar_cb_pagesize = tile_size(in_df);
    const uint32_t in_scalar_cb_npages = 1;
    const CircularBufferConfig in_scalar_cb_config =
        CircularBufferConfig(in_scalar_cb_npages * in_scalar_cb_pagesize, {{in_scalar_cb_id, in_df}})
            .set_page_size(in_scalar_cb_id, in_scalar_cb_pagesize);
    auto in_scalar_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, in_scalar_cb_config);
    tt::log_info("CB {} :: PS = {}, NP = {}", in_scalar_cb_id, in_scalar_cb_pagesize, in_scalar_cb_npages);

    // incoming data is the input cb instead of raw l1/dram addr
    // this input shard has halo and padding inserted.
    const auto raw_in_cb_id = tt::CB::c_in2;
    const uint32_t raw_in_cb_npages = input.shard_spec().value().shape[0];
    const uint32_t raw_in_cb_pagesize = in_nbytes_c;
    CircularBufferConfig raw_in_cb_config =
        CircularBufferConfig(raw_in_cb_npages * raw_in_cb_pagesize, {{raw_in_cb_id, in_df}})
            .set_page_size(raw_in_cb_id, raw_in_cb_pagesize)
            .set_globally_allocated_address(*input.buffer());
    auto raw_in_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, raw_in_cb_config);
    tt::log_info("CB {} :: PS = {}, NP = {}", raw_in_cb_id, raw_in_cb_pagesize, raw_in_cb_npages);

    // reader indices
    const auto in_reader_indices_cb_id = tt::CB::c_in3;
    const uint32_t in_reader_indices_cb_pagesize =
        tt::round_up(out_nhw_per_core * indices_nbytes, 4);  // pagesize needs to be multiple of 4
    const uint32_t in_reader_indices_cb_npages = 1;
    CircularBufferConfig in_reader_indices_cb_config =
        CircularBufferConfig(
            in_reader_indices_cb_npages * in_reader_indices_cb_pagesize, {{in_reader_indices_cb_id, indices_df}})
            .set_page_size(in_reader_indices_cb_id, in_reader_indices_cb_pagesize)
            .set_globally_allocated_address(*reader_indices_buffer);
    auto in_reader_indices_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, in_reader_indices_cb_config);
    tt::log_info(
        "CB {} :: PS = {}, NP = {}",
        in_reader_indices_cb_id,
        in_reader_indices_cb_pagesize,
        in_reader_indices_cb_npages);

    uint32_t in_cb_sz = input_shape[3] * kernel_size_hw_padded;
    uint32_t in_nblocks_c = 1;
    // if (is_large_kernel) {
    //     TT_ASSERT(false, "TODO: large kernel");
    //     // in_cb_sz = (input_shape[3] * kernel_size_hw_padded) > (tt::constants::TILE_HW * MAX_TILES_PER_REDUCTION)
    //     //                ? (tt::constants::TILE_HW * MAX_TILES_PER_REDUCTION)
    //     //                : input_shape[3] * kernel_size_hw_padded;
    // } else {
    //     if (is_wide_reduction) {
    //         TT_ASSERT(false, "TODO: wide reduction");
    //         // in_cb_sz = MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH * kernel_size_hw_padded;
    //         // TT_FATAL(
    //         //     in_ntiles_c % MAX_TILES_PER_REDUCTION == 0,
    //         //     "input channels should be multiple of {} tiles. General case TODO.",
    //         //     MAX_TILES_PER_REDUCTION);
    //         // in_nblocks_c = in_ntiles_c / MAX_TILES_PER_REDUCTION;
    //     } else {
    //     }
    // }
    // reader output == input to tilize
    const uint32_t in_cb_id_0 = tt::CB::c_in0;  // input rows for "multiple (out_nelems)" output pixels
    const uint32_t in_cb_id_1 = tt::CB::c_in1;  // input rows for "multiple (out_nelems)" output pixels
    const uint32_t in_cb_page_padded = tt::round_up(
        in_cb_sz,
        tt::constants::TILE_HW);  // NOTE: ceil to tile size since triscs work with tilesize instead of pagesize
    const uint32_t in_cb_pagesize = in_nbytes * in_cb_page_padded;
    const uint32_t in_cb_npages = multi_buffering_factor * nblocks;

    CircularBufferConfig in_cb_config_0 = CircularBufferConfig(in_cb_npages * in_cb_pagesize, {{in_cb_id_0, in_df}})
                                              .set_page_size(in_cb_id_0, in_cb_pagesize);
    auto in_cb_0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config_0);
    tt::log_info("CB {} :: PS = {}, NP = {}", in_cb_id_0, in_cb_pagesize, in_cb_npages);

    if (split_reader) {
        CircularBufferConfig in_cb_config_1 = CircularBufferConfig(in_cb_npages * in_cb_pagesize, {{in_cb_id_1, in_df}})
                                                  .set_page_size(in_cb_id_1, in_cb_pagesize);
        auto in_cb_1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config_1);
        tt::log_info("CB {} :: PS = {}, NP = {}", in_cb_id_1, in_cb_pagesize, in_cb_npages);
    }

    if (is_large_kernel) {
        TT_ASSERT(false, "TODO: large kernel");
        // uint32_t max_pool_partials_cb_id = tt::CB::c_intermed1;  // max_pool partials
        // uint32_t max_pool_partials_cb_pagesize = in_cb_sz;
        // uint32_t max_pool_partials_cb_npages = nblocks;
        // CircularBufferConfig max_pool_partials_cb_config =
        //     CircularBufferConfig(
        //         max_pool_partials_cb_npages * max_pool_partials_cb_pagesize, {{max_pool_partials_cb_id, in_df}})
        //         .set_page_size(max_pool_partials_cb_id, max_pool_partials_cb_pagesize);
        // auto max_pool_partials_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores,
        // max_pool_partials_cb_config); log_debug(
        //     tt::LogOp,
        //     "CB {} :: PS = {}, NP = {}",
        //     max_pool_partials_cb_id,
        //     max_pool_partials_cb_pagesize,
        //     max_pool_partials_cb_npages);
    }

    // output of reduce == writer to write output rows in RM after reduction
    uint32_t out_cb_id = tt::CB::c_out0;
    // there is just one row of channels after each reduction (or 1 block of c if its greater than 8 tiles)
    const uint32_t out_cb_pagesize = output.shard_spec().value().shape[1] * out_nbytes / in_nblocks_c;
    const uint32_t out_cb_npages = output.shard_spec().value().shape[0] * in_nblocks_c;
    const CircularBufferConfig cb_out_config =
        CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, out_df}})
            .set_page_size(out_cb_id, out_cb_pagesize)
            .set_globally_allocated_address(*output.buffer());
    auto cb_out = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);
    tt::log_info("CB {} :: PS = {}, NP = {}", out_cb_id, out_cb_pagesize, out_cb_npages);

    TT_FATAL(output.memory_config().is_sharded(), "Output memory config needs to be sharded");

    /**
     * Reader Kernel: input rows -> input cb
     */
    float one = 1.0 / ((float)kernel_size_hw);
    uint32_t bf16_one_u32 = *reinterpret_cast<uint32_t*>(&one);
    uint32_t in_nbytes_c_log2 = (uint32_t)std::log2((float)in_nbytes_c);
    std::vector<uint32_t> reader0_ct_args = {
        out_nhw_per_core,
        kernel_size_h,
        kernel_size_w,
        pad_w,
        in_nbytes_c,
        in_nbytes_c_log2,
        in_w,
        in_cb_page_padded * in_cb_npages / tile_w,
        input_shape[3],
        nblocks,
        split_reader,  // enable split reader
        0,             // split reader id
        bf16_one_u32,
        in_nblocks_c};

    std::vector<uint32_t> reader1_ct_args = {
        out_nhw_per_core,
        kernel_size_h,
        kernel_size_w,
        pad_w,
        in_nbytes_c,
        in_nbytes_c_log2,
        in_w,
        in_cb_page_padded * in_cb_npages / tile_w,
        input_shape[3],
        nblocks,
        split_reader,  // enable split reader
        1,             // split reader id
        bf16_one_u32,
        in_nblocks_c};

    std::string reader_kernel_fname;
    if (is_large_kernel) {
        reader_kernel_fname =
            "ttnn/cpp/ttnn/operations/pool/maxpool/device/kernels/dataflow/"
            "reader_max_pool_2d_multi_core_sharded_with_halo_large_kernel_v2.cpp";
    } else if (is_wide_reduction) {
        reader_kernel_fname =
            "ttnn/cpp/ttnn/operations/pool/maxpool/device/kernels/dataflow/"
            "reader_max_pool_2d_multi_core_sharded_with_halo_wide.cpp";
    } else {
        reader_kernel_fname =
            "ttnn/cpp/ttnn/operations/experimental/pool/avgpool/device/kernels/dataflow/"
            "reader_max_pool_2d_multi_core_sharded_with_halo_v2.cpp";
    }

    auto reader0_config = DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader0_ct_args};
    auto reader0_kernel = CreateKernel(program, reader_kernel_fname, all_cores, reader0_config);

    auto reader1_config = DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader1_ct_args};
    auto reader1_kernel = split_reader ? CreateKernel(program, reader_kernel_fname, all_cores, reader1_config) : 0;

    /**
     * Compute Kernel: input cb -> tilize_block -> input tiles -> reduce_h max -> output tiles -> untilize_block ->
     * output cb
     */
    std::vector<uint32_t> compute_ct_args = {
        in_ntiles_hw,
        in_ntiles_c,
        in_ntiles_hw * in_ntiles_c,
        kernel_size_hw,
        out_h,
        out_w,
        tt::div_up(output_shape[2], tt::constants::TILE_HEIGHT),
        tt::div_up(output_shape[3], tt::constants::TILE_WIDTH),
        nblocks,
        out_w_loop_count,
        1,
        out_nhw_per_core,
        split_reader,                // enable split reader
        out_nhw_per_core / nblocks,  // loop count with blocks
        input_shape[3],
        in_nblocks_c};

    auto reduce_op = tt::tt_metal::ReduceOpMath::SUM;
    auto reduce_dim = tt::tt_metal::ReduceOpDim::H;
    auto compute_config = ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
        .compile_args = compute_ct_args,
        .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};
    std::string compute_kernel_fname;
    if (is_large_kernel) {
        compute_kernel_fname =
            "ttnn/cpp/ttnn/operations/pool/maxpool/device/kernels/compute/max_pool_multi_core_large_kernel.cpp";
    } else {
        // both regular and wide reductions
        compute_kernel_fname =
            "ttnn/cpp/ttnn/operations/experimental/pool/avgpool/device/kernels/compute/"
            "avg_pool2d_multi_core_sharded.cpp";
    }

    auto compute_kernel = CreateKernel(program, compute_kernel_fname, core_range, compute_config);

    return {
        std::move(program),
        {.reader0_kernel = reader0_kernel,
         .reader1_kernel = reader1_kernel,
         .raw_in_cb = raw_in_cb,
         .cb_out = cb_out,
         .ncores = ncores,
         .ncores_w = ncores_w}};
}

}  // namespace

AvgPool2D::MultiCore::cached_program_t AvgPool2D::MultiCore::create(
    const AvgPool2D::operation_attributes_t& op_attrs,
    const AvgPool2D::tensor_args_t& inputs,
    AvgPool2D::tensor_return_value_t& output_tensor) {
    const auto& input = inputs.input_tensor_;
    const auto& out_mem_config = output_tensor.memory_config();
    const auto& sliding_window_config = op_attrs.sliding_window_config_;

    tt::tt_metal::Program program{};

    const auto parallel_config = sliding_window::ParallelConfig{
        .grid = input.shard_spec().value().grid,
        .shard_scheme = input.memory_config().memory_layout,
        .shard_orientation = input.shard_spec().value().orientation,
    };

    const auto out_shape = sliding_window_config.get_output_shape();
    const uint32_t out_h = out_shape[1];
    const uint32_t out_w = out_shape[2];

    bool is_block_sharded = input.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED;

    const auto pad_metadata = sliding_window::generate_pad_metadata(sliding_window_config);
    const auto op_trace_metadata = sliding_window::generate_op_trace_metadata(sliding_window_config);
    const auto shard_boundaries = sliding_window::generate_shard_boundaries(sliding_window_config, op_trace_metadata);
    const auto top_left_indices =
        sliding_window::generate_sliding_window_op_config(op_trace_metadata, shard_boundaries, false, false);
    const auto reader_indices =
        sliding_window::construct_on_host_config_tensor(top_left_indices, sliding_window_config, parallel_config);
    const auto reader_indices_on_device =
        sliding_window::move_config_tensor_to_device(reader_indices, parallel_config, is_block_sharded, input.device());
    tt::tt_metal::detail::AddConfigBuffer(program, reader_indices_on_device.device_buffer());

    tt::log_info("host reader_indices {}", top_left_indices);

    tt::log_info("reader_indices: {}", reader_indices_on_device);

    return avg_pool_2d_multi_cire_sharded_with_halo(
        program,
        input,
        reader_indices_on_device,
        output_tensor,
        sliding_window_config.window_hw.first,
        sliding_window_config.window_hw.second,
        sliding_window_config.input_hw.first,
        sliding_window_config.input_hw.second,
        out_h,
        out_w,
        sliding_window_config.pad_hw.first,
        sliding_window_config.pad_hw.second,
        /*nblocks=*/1);
}

void AvgPool2D::MultiCore::override_runtime_arguments(
    AvgPool2D::MultiCore::cached_program_t& cached_program,
    const AvgPool2D::operation_attributes_t& op_attrs,
    const AvgPool2D::tensor_args_t& inputs,
    AvgPool2D::tensor_return_value_t& output_tensor) {}

}  // namespace ttnn::operations::experimental::pool
