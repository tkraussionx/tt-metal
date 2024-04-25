// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/logger.hpp"
#include "impl/buffers/buffer.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_eager/tt_dnn/op_library/sdpa/sdpa_op.hpp"
#include "tt_eager/tt_dnn/op_library/math.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/util.hpp"

#include <optional>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace tt {
namespace operations {
namespace primary {

// implementation of softmax with optional scale/mask (see the header for input_tensor more detailed description)
operation::ProgramWithCallbacks sdpa_multi_core(
    const Tensor &input_tensor_q,
    const Tensor &input_tensor_k,
    const Tensor &input_tensor_v,
    const Tensor &output_tensor,
    const std::optional<const Tensor> attn_mask,
    std::optional<float> scale,
    bool is_causal,
    std::size_t q_chunk_size,
    std::size_t k_chunk_size,
    DeviceComputeKernelConfig compute_kernel_config,
    transformers::SDPAProgramConfig program_config
) {

    /*
    Q: B x NQH x S x DH
    K: B x NKH x DH x S
    V: B x NKH x S x DH
    attn_mask: B x NQH x S x S
    */

    // TT_FATAL(q_chunk_size == k_chunk_size);

    const auto q_shape = input_tensor_q.get_legacy_shape();
    uint32_t B = q_shape[0], NQH = q_shape[1], S = q_shape[2], DH = q_shape[3];
    uint32_t St = S/TILE_HEIGHT;
    uint32_t DHt = DH/TILE_WIDTH;

    uint32_t Sq_chunk_t = q_chunk_size / TILE_HEIGHT;
    uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;
    uint32_t q_num_chunks = S / q_chunk_size;
    uint32_t k_num_chunks = S / k_chunk_size;

    const auto k_shape = input_tensor_k.get_legacy_shape();
    uint32_t NKH = k_shape[1];

    // log_info all of the above
    log_info("B: {}", B);
    log_info("NQH: {}", NQH);

    log_info("S: {}", S);
    log_info("DH: {}", DH);
    log_info("St: {}", St);
    log_info("DHt: {}", DHt);
    log_info("Sq_chunk_t: {}", Sq_chunk_t);
    log_info("Sk_chunk_t: {}", Sk_chunk_t);
    log_info("q_num_chunks: {}", q_num_chunks);
    log_info("k_num_chunks: {}", k_num_chunks);
    log_info("NKH: {}", NKH);


    Program program = CreateProgram();

    // This should allocate input_tensor DRAM buffer on the device
    Device *device = input_tensor_q.device();

    MathFidelity math_fidelity = MathFidelity::HiFi2;
    bool math_approx_mode = true;
    bool fp32_dest_acc_en;


    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
            // math_fidelity = compute_kernel_config.math_fidelity;
            // math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = false;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::WORMHOLE_B0, "kernel config is not for wormhole_b0");
            // math_fidelity = compute_kernel_config.math_fidelity;
            // math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
        } else {
            TT_FATAL("arch not supported");
        }

    }, compute_kernel_config);

    auto q_buffer = input_tensor_q.buffer();
    auto k_buffer = input_tensor_k.buffer();
    auto v_buffer = input_tensor_v.buffer();
    auto mask_buffer = attn_mask.has_value() ? attn_mask.value().buffer() : nullptr;
    TT_ASSERT(mask_buffer != nullptr);

    auto out0_buffer = output_tensor.buffer();

    CoreCoord grid_size;

    std::visit([&](auto&& program_config) {
        using T = std::decay_t<decltype(program_config)>;
        if constexpr (std::is_same_v<T, transformers::SDPAMultiCoreProgramConfig>) {
            grid_size = program_config.compute_with_storage_grid_size;
        } else {
            log_info("Using default grid size");
            grid_size = device->compute_with_storage_grid_size();

        }
    }, program_config);

    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    uint32_t num_cores = grid_size.x * grid_size.y;

    // TT_FATAL(num_cores == 64); // For now, we only support 64 cores
    TT_FATAL(num_cores <= device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y);

    // Parallelization scheme
    // We will choose parallelization factors for batch, num_heads, and q_seq_len in that order
    uint32_t batch_parallel_factor = std::min(B, num_cores);
    uint32_t nh_parallel_factor = std::min(num_cores / batch_parallel_factor, NQH);
    uint32_t q_parallel_factor = std::min(num_cores / (batch_parallel_factor * nh_parallel_factor), q_num_chunks);

    TT_FATAL( batch_parallel_factor * nh_parallel_factor * q_parallel_factor == num_cores );

    log_info("Parallelization scheme:");
    log_info("batch_parallel_factor: {}", batch_parallel_factor);
    log_info("nh_parallel_factor: {}", nh_parallel_factor);
    log_info("q_parallel_factor: {}", q_parallel_factor);


    // Ceiling divide to allow for non-perfect divisions
    const uint32_t batch_per_core = (B + batch_parallel_factor-1) / batch_parallel_factor;
    const uint32_t nh_per_core = (NQH + nh_parallel_factor-1) / nh_parallel_factor;
    const uint32_t q_per_core = (q_num_chunks + q_parallel_factor-1) / q_parallel_factor;

    const uint32_t q_buffer_factor = (q_per_core > 1) ? 2 : 1;



    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t q_tiles  = Sq_chunk_t * DHt * q_buffer_factor;
    uint32_t k_tiles  = Sk_chunk_t * DHt * 2; // double buffer
    uint32_t v_tiles  = Sk_chunk_t * DHt * 2; // double buffer
    uint32_t mask_tiles = Sq_chunk_t * Sk_chunk_t * 2; // double buffer
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * DHt;
    uint32_t out0_t = Sq_chunk_t * DHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t; // Single column of values in each iteration

    // log all values
    log_info("q_tiles: {}", q_tiles);
    log_info("k_tiles: {}", k_tiles);
    log_info("v_tiles: {}", v_tiles);
    log_info("mask_tiles: {}", mask_tiles);
    log_info("qk_tiles: {}", qk_tiles);
    log_info("out0_t: {}", out0_t);
    log_info("scale_tiles: {}", scale_tiles);
    log_info("statistics_tiles: {}", statistics_tiles);




    // Host code is responsible for determining matmul configuration
    const uint32_t dst_size = fp32_dest_acc_en ? 4: 8;
    const uint32_t qk_in0_block_w = DHt;
    // max of Sk_chunk_t and dst_size
    const uint32_t qk_out_subblock_w = std::min(Sk_chunk_t, dst_size);
    // If qk_out_subblock_w is full row of output, scale subblock_h so volume = dst_size. Otherwise it's 1 to maintain row-major intermediate buffer.
    const uint32_t qk_out_subblock_h = (qk_out_subblock_w == Sk_chunk_t) ? (std::min(Sq_chunk_t, dst_size / qk_out_subblock_w)) : 1;

    const uint32_t qk_in0_num_subblocks = Sq_chunk_t / qk_out_subblock_h;
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    const uint32_t qk_num_blocks = DHt / qk_in0_block_w;

    // now for out0
    const uint32_t out_in0_block_w = Sk_chunk_t;
    const uint32_t out_out_subblock_w = std::min(DHt, dst_size);
    const uint32_t out_out_subblock_h = (out_out_subblock_w == DHt) ? (std::min(Sq_chunk_t, dst_size / out_out_subblock_w)) : 1;

    const uint32_t out_in0_num_subblocks = Sq_chunk_t / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = DHt / out_out_subblock_w;
    const uint32_t out_num_blocks = Sk_chunk_t / out_in0_block_w;

    // log all values
    log_info("dst_size: {}", dst_size);
    log_info("qk_in0_block_w: {}", qk_in0_block_w);
    log_info("qk_out_subblock_w: {}", qk_out_subblock_w);
    log_info("qk_out_subblock_h: {}", qk_out_subblock_h);
    log_info("qk_in0_num_subblocks: {}", qk_in0_num_subblocks);
    log_info("qk_in1_num_subblocks: {}", qk_in1_num_subblocks);
    log_info("qk_num_blocks: {}", qk_num_blocks);
    log_info("out_in0_block_w: {}", out_in0_block_w);
    log_info("out_out_subblock_w: {}", out_out_subblock_w);
    log_info("out_out_subblock_h: {}", out_out_subblock_h);
    log_info("out_in0_num_subblocks: {}", out_in0_num_subblocks);
    log_info("out_in1_num_subblocks: {}", out_in1_num_subblocks);
    log_info("out_num_blocks: {}", out_num_blocks);

    // Determine granularity for statistics computation
    const uint32_t stats_granularity = std::min(Sq_chunk_t, dst_size);
    // Find log2 of stats_granularity using std
    const uint32_t log2_stats_granularity = std::log2(stats_granularity);
    // Assert that this is a power of 2
    TT_ASSERT(stats_granularity == (1 << log2_stats_granularity));

    const uint32_t sub_exp_granularity = std::min(Sk_chunk_t, dst_size);
    const uint32_t log2_sub_exp_granularity = std::log2(sub_exp_granularity);
    TT_ASSERT(sub_exp_granularity == (1 << log2_sub_exp_granularity));

    const uint32_t mul_bcast_granularity = std::min(Sq_chunk_t * Sk_chunk_t, dst_size);
    const uint32_t log2_mul_bcast_granularity = std::log2(mul_bcast_granularity);
    TT_ASSERT(mul_bcast_granularity == (1 << log2_mul_bcast_granularity));

    const uint32_t dht_granularity = std::min(DHt, dst_size);
    const uint32_t log2_dht_granularity = std::log2(dht_granularity);

    // Log these
    log_info("stats_granularity: {}", stats_granularity);
    log_info("log2_stats_granularity: {}", log2_stats_granularity);
    log_info("sub_exp_granularity: {}", sub_exp_granularity);
    log_info("log2_sub_exp_granularity: {}", log2_sub_exp_granularity);
    log_info("mul_bcast_granularity: {}", mul_bcast_granularity);
    log_info("log2_mul_bcast_granularity: {}", log2_mul_bcast_granularity);
    log_info("dht_granularity: {}", dht_granularity);
    log_info("log2_dht_granularity: {}", log2_dht_granularity);



    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    bfloat16 bfloat_identity_scalar = bfloat16(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    union {float f; uint32_t u;} scale_union; scale_union.f = scale.value_or(1.0f);

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        B, NQH, NKH, St, DHt, Sq_chunk_t, q_num_chunks, Sk_chunk_t, k_num_chunks,

    };

    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        B, NQH, NKH, St, DHt, Sq_chunk_t, q_num_chunks, Sk_chunk_t, k_num_chunks,
        packed_identity_scalar,
        scale_union.u,

    };

    std::vector<uint32_t> compute_compile_time_args = {
        // matmul args
        B, NQH, NKH, St, DHt, Sq_chunk_t, q_num_chunks, Sk_chunk_t, k_num_chunks,
        qk_in0_block_w, qk_out_subblock_w, qk_out_subblock_h, qk_in0_num_subblocks, qk_in1_num_subblocks, qk_num_blocks,
        out_in0_block_w, out_out_subblock_w, out_out_subblock_h, out_in0_num_subblocks, out_in1_num_subblocks, out_num_blocks
    };

    std::map<string, string> defines;
    defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
    defines["LOG2_STATS_GRANULARITY"] = std::to_string(log2_stats_granularity);
    defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    defines["LOG2_SUB_EXP_GRANULARITY"] = std::to_string(log2_sub_exp_granularity);
    defines["MUL_BCAST_GRANULARITY"] = std::to_string(mul_bcast_granularity);
    defines["LOG2_MUL_BCAST_GRANULARITY"] = std::to_string(log2_mul_bcast_granularity);
    defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    defines["LOG2_DHT_GRANULARITY"] = std::to_string(log2_dht_granularity);

    auto reader_kernels_id = CreateKernel(
        program, "tt_eager/tt_dnn/op_library/sdpa/kernels/dataflow/reader_interleaved.cpp", core_grid,
        tt_metal::ReaderDataMovementConfig(
            reader_compile_time_args
    ));

    auto writer_kernels_id = CreateKernel(
        program, "tt_eager/tt_dnn/op_library/sdpa/kernels/dataflow/writer_interleaved.cpp", core_grid,
        tt_metal::WriterDataMovementConfig(
            writer_compile_time_args
    ));

    auto compute_kernels_id = CreateKernel(
        program, "tt_eager/tt_dnn/op_library/sdpa/kernels/compute/sdpa.cpp", core_grid,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = defines
    });

    // Create circular buffers

    tt::DataFormat q_df = tt_metal::datatype_to_dataformat_converter(input_tensor_q.get_dtype());
    tt::DataFormat k_df = tt_metal::datatype_to_dataformat_converter(input_tensor_k.get_dtype());
    tt::DataFormat v_df = tt_metal::datatype_to_dataformat_converter(input_tensor_v.get_dtype());
    tt::DataFormat mask_df = attn_mask.has_value() ? tt_metal::datatype_to_dataformat_converter(attn_mask.value().get_dtype()) : tt::DataFormat::Float16_b;
    tt::DataFormat out_df = tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    tt::DataFormat im_df = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = tt::DataFormat::Float16_b;

    uint32_t q_tile_size = tt_metal::detail::TileSize(q_df);
    uint32_t k_tile_size = tt_metal::detail::TileSize(k_df);
    uint32_t v_tile_size = tt_metal::detail::TileSize(v_df);
    uint32_t mask_tile_size = attn_mask.has_value() ? tt_metal::detail::TileSize(mask_df) : 0;
    uint32_t out_tile_size = tt_metal::detail::TileSize(out_df);
    uint32_t scalar_tile_size = tt_metal::detail::TileSize(scalar_df);
    uint32_t im_tile_size = tt_metal::detail::TileSize(im_df);
    uint32_t stats_tile_size = tt_metal::detail::TileSize(stats_df);

    log_info("q_data_format: {}", q_df);
    log_info("k_data_format: {}", k_df);
    log_info("v_data_format: {}", v_df);
    log_info("mask_data_format: {}", mask_df);
    log_info("out_data_format: {}", out_df);
    log_info("scalar_data_format: {}", scalar_df);
    log_info("intermediate_data_format: {}", im_df);
    log_info("statistics_data_format: {}", stats_df);


    // Q input
    auto c_in0_config = CircularBufferConfig(q_tiles * q_tile_size, {{CB::c_in0, q_df}}).set_page_size(CB::c_in0, q_tile_size);
    auto cb_in0_id = CreateCircularBuffer(program, core_grid, c_in0_config);
    // K input
    auto c_in1_config = CircularBufferConfig(k_tiles * k_tile_size, {{CB::c_in1, k_df}}).set_page_size(CB::c_in1, k_tile_size);
    auto cb_in1_id = CreateCircularBuffer(program, core_grid, c_in1_config);
    // V input
    auto c_in2_config = CircularBufferConfig(v_tiles * v_tile_size, {{CB::c_in2, v_df}}).set_page_size(CB::c_in2, v_tile_size);
    auto cb_in2_id = CreateCircularBuffer(program, core_grid, c_in2_config);

    // attn_mask input
    auto c_in3_config = CircularBufferConfig(mask_tiles * mask_tile_size, {{CB::c_in3, mask_df}}).set_page_size(CB::c_in3, mask_tile_size);
    auto cb_in3_id = CreateCircularBuffer(program, core_grid, c_in3_config);

    // scale input
    auto c_in4_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{CB::c_in4, scalar_df}}).set_page_size(CB::c_in4, scalar_tile_size);
    auto cb_in4_id = CreateCircularBuffer(program, core_grid, c_in4_config);

    // identity scale input
    auto c_in5_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{CB::c_in5, scalar_df}}).set_page_size(CB::c_in5, scalar_tile_size);
    auto cb_in5_id = CreateCircularBuffer(program, core_grid, c_in5_config);

    // cb_qk_im
    auto c_intermed0_config = CircularBufferConfig(qk_tiles * im_tile_size, {{CB::c_intermed0, im_df}}).set_page_size(CB::c_intermed0, im_tile_size);
    auto cb_intermed0_id = CreateCircularBuffer(program, core_grid, c_intermed0_config);

    // cb_out_im
    auto c_intermed1_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{CB::c_intermed1, im_df}}).set_page_size(CB::c_intermed1, im_tile_size);
    auto cb_intermed1_id = CreateCircularBuffer(program, core_grid, c_intermed1_config);

    // cb_out_accumulate_im
    auto c_intermed2_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{CB::c_intermed2, im_df}}).set_page_size(CB::c_intermed2, im_tile_size);
    auto cb_intermed2_id = CreateCircularBuffer(program, core_grid, c_intermed2_config);

    // cb_cur_max
    auto c_intermed3_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CB::c_intermed3, stats_df}}).set_page_size(CB::c_intermed3, stats_tile_size);
    auto cb_intermed3_id = CreateCircularBuffer(program, core_grid, c_intermed3_config);

    // cb_prev_max
    auto c_intermed4_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CB::c_intermed4, stats_df}}).set_page_size(CB::c_intermed4, stats_tile_size);
    auto cb_intermed4_id = CreateCircularBuffer(program, core_grid, c_intermed4_config);

    // cb_cur_sum
    auto c_intermed5_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CB::c_intermed5, stats_df}}).set_page_size(CB::c_intermed5, stats_tile_size);
    auto cb_intermed5_id = CreateCircularBuffer(program, core_grid, c_intermed5_config);

    // cb_prev_sum
    auto c_intermed6_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CB::c_intermed6, stats_df}}).set_page_size(CB::c_intermed6, stats_tile_size);
    auto cb_intermed6_id = CreateCircularBuffer(program, core_grid, c_intermed6_config);

    // cb_exp_max_diff
    auto c_intermed7_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CB::c_intermed7, stats_df}}).set_page_size(CB::c_intermed7, stats_tile_size);
    auto cb_intermed7_id = CreateCircularBuffer(program, core_grid, c_intermed7_config);

    // Output
    auto c_out0_config = CircularBufferConfig(out0_t * out_tile_size, {{CB::c_out0, out_df}}).set_page_size(CB::c_out0, out_tile_size);
    auto cb_out0_id = CreateCircularBuffer( program, core_grid, c_out0_config );

    // auto c_intermed1_config = CircularBufferConfig(im1_t * im_tile_size, {{CB::c_intermed1, im_cb_data_format}}).set_page_size(CB::c_intermed1, im_tile_size);
    // auto cb_intermed1_id = CreateCircularBuffer( program, all_device_cores, c_intermed1_config );
    // auto c_in2_config = CircularBufferConfig(in2_t * scalar_tile_size, {{CB::c_in2, scalar_cb_data_format}}).set_page_size(CB::c_in2, scalar_tile_size);
    // auto cb_in2_id = CreateCircularBuffer( program, all_device_cores, c_in2_config );
    // auto c_intermed0_config = CircularBufferConfig(im0_t * im_tile_size, {{CB::c_intermed0, im_cb_data_format}}).set_page_size(CB::c_intermed0, im_tile_size);
    // auto cb_intermed0_id = CreateCircularBuffer( program, all_device_cores, c_intermed0_config );
    // std::optional<CBHandle> cb_intermed3_id;
    // std::optional<CBHandle> cb_in3_id;
    // std::optional<CBHandle> cb_in4_id;
    // if (mask.has_value()) {
    //     CircularBufferConfig c_intermed3_config = CircularBufferConfig(im3_t * im_tile_size, {{CB::c_intermed3, im_cb_data_format}}).set_page_size(CB::c_intermed3, im_tile_size);
    //     cb_intermed3_id = CreateCircularBuffer( program, all_device_cores, c_intermed3_config );
    //     CircularBufferConfig c_in3_config = CircularBufferConfig(in3_t * scalar_tile_size, {{CB::c_in3, scalar_cb_data_format}}).set_page_size(CB::c_in3, scalar_tile_size);
    //     cb_in3_id = CreateCircularBuffer( program, all_device_cores, c_in3_config );
    //     CircularBufferConfig c_in4_config = CircularBufferConfig(in4_t * mask_tile_size, {{CB::c_in4, mask_cb_data_format}}).set_page_size(CB::c_in4, mask_tile_size);
    //     cb_in4_id = CreateCircularBuffer( program, all_device_cores, c_in4_config);
    // }
    uint32_t q_addr = q_buffer->address();
    uint32_t k_addr = k_buffer->address();
    uint32_t v_addr = v_buffer->address();
    uint32_t mask_addr = mask_buffer->address();
    uint32_t out_addr = out0_buffer->address();


    // Set reader rt args
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        // log_info("core: {} getting runtime args for idx {i}", core, i);
        uint32_t local_batch_start = (i / (nh_parallel_factor * q_parallel_factor)) * batch_per_core;
        uint32_t local_batch_end = local_batch_start + batch_per_core;
        uint32_t local_nh_start = ((i / q_parallel_factor) % nh_parallel_factor) * nh_per_core;
        uint32_t local_nh_end = local_nh_start + nh_per_core;
        uint32_t local_q_start = (i % q_parallel_factor) * q_per_core;
        uint32_t local_q_end = local_q_start + q_per_core;

        // clamp all to max values for non-even partitioning
        local_batch_start = std::min(local_batch_start, B);
        local_batch_end = std::min(local_batch_end, B);
        local_nh_start = std::min(local_nh_start, NQH);
        local_nh_end = std::min(local_nh_end, NQH);
        local_q_start = std::min(local_q_start, q_num_chunks);
        local_q_end = std::min(local_q_end, q_num_chunks);


        // log the above
        log_debug("core: {}", i);
        log_debug("local_batch_start: {}", local_batch_start);
        log_debug("local_batch_end: {}", local_batch_end);
        log_debug("local_nh_start: {}", local_nh_start);
        log_debug("local_nh_end: {}", local_nh_end);
        log_debug("local_q_start: {}", local_q_start);
        log_debug("local_q_end: {}", local_q_end);


        SetRuntimeArgs(program, reader_kernels_id, core, { q_addr, k_addr, v_addr, mask_addr, i, num_cores, local_batch_start, local_batch_end, local_nh_start, local_nh_end, local_q_start, local_q_end });
        SetRuntimeArgs(program, writer_kernels_id, core, { out_addr, i, num_cores, local_batch_start, local_batch_end, local_nh_start, local_nh_end, local_q_start, local_q_end });
        SetRuntimeArgs(program, compute_kernels_id, core, { i, num_cores, local_batch_start, local_batch_end, local_nh_start, local_nh_end, local_q_start, local_q_end });
    }


    // uint32_t curr_row = 0;
    // union { float f; uint32_t u; } s; s.f = scale.value_or(1.0f); // scale for fused scale-mask-softmax
    // for (uint32_t i = 0; i < grid_size.x * grid_size.y; ++i) {
    //     CoreCoord core = {i % grid_size.x, i / grid_size.x};
    //     if (i >= num_cores) {
    //         SetRuntimeArgs(program, reader_kernels_id, core, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }); // [8]=1.0f is scaler
    //         SetRuntimeArgs(program, softmax_kernels_id, core, { 0, 0, 0, 0, 0 });
    //         SetRuntimeArgs(program, writer_kernels_id, core, { 0, 0, 0, 0 });
    //         continue;
    //     }
    //     uint32_t num_tile_rows_per_core = 0;
    //     if (core_group_1.core_coord_in_core_ranges(core)) {
    //         num_tile_rows_per_core = num_tile_rows_per_core_group_1;
    //     } else if (core_group_2.core_coord_in_core_ranges(core)) {
    //         num_tile_rows_per_core = num_tile_rows_per_core_group_2;
    //     } else {
    //         TT_ASSERT(false, "Core not in specified core ranges");
    //     }

    //     uint32_t tile_offset = curr_row * Wt;
    //     uint32_t curr_ht = curr_row % Ht;
    //     uint32_t mask_curr_ht = curr_ht % mask_Ht;   // the start offset for causal mask
    //     uint32_t mask_offset = curr_row / Ht * mask_Ht * Wt; // causal mask batch offset
    //     uint32_t mask_id = causal_mask ? (mask_curr_ht * Wt + mask_offset) : (curr_row / Ht * Wt); // causal mask start offset + causal mask batch offset

    //     if (causal_mask) {
    //         SetRuntimeArgs(program, reader_kernels_id, core, { src_addr, block_size, s.u, num_tile_rows_per_core, tile_offset, Wt, Ht, mask_addr, curr_ht, mask_id, 0x3f803f80, mask_curr_ht, mask_offset }); // [10]=1.0f is scaler
    //     } else {
    //         SetRuntimeArgs(program, reader_kernels_id, core, { src_addr, block_size, s.u, num_tile_rows_per_core, tile_offset, Wt, Ht, mask_addr, curr_ht, mask_id, 0x3f803f80 }); // [10]=1.0f is scaler
    //     }

    //     SetRuntimeArgs(program, softmax_kernels_id, core, { num_tile_rows_per_core, Ht, Wt, block_size, curr_ht });
    //     SetRuntimeArgs(program, writer_kernels_id, core, { out_addr, num_tile_rows_per_core * Wt, tile_offset, block_size });
    //     curr_row += num_tile_rows_per_core;
    // }

    auto override_runtime_arguments_callback = [&]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {

        auto q_buffer = input_tensors.at(0).buffer();
        auto k_buffer = input_tensors.at(1).buffer();
        auto v_buffer = input_tensors.at(2).buffer();
        auto mask_buffer = optional_input_tensors.at(0).has_value() ? optional_input_tensors.at(0).value().buffer() : nullptr;
        TT_ASSERT(mask_buffer != nullptr);

        auto out0_buffer = output_tensor.buffer();
        uint32_t q_addr = q_buffer->address();
        uint32_t k_addr = k_buffer->address();
        uint32_t v_addr = v_buffer->address();
        uint32_t mask_addr = mask_buffer->address();
        uint32_t out_addr = out0_buffer->address();

        // Set reader rt args
        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};

            // log_info("core: {} getting runtime args for idx {i}", core, i);
            const uint32_t local_batch_start = (i / (nh_parallel_factor * q_parallel_factor)) * batch_per_core;
            const uint32_t local_batch_end = local_batch_start + batch_per_core;
            const uint32_t local_nh_start = ((i / q_parallel_factor) % nh_parallel_factor) * nh_per_core;
            const uint32_t local_nh_end = local_nh_start + nh_per_core;
            const uint32_t local_q_start = (i % q_parallel_factor) * q_per_core;
            const uint32_t local_q_end = local_q_start + q_per_core;

            // log the above
            log_debug("core: {}", i);
            log_debug("local_batch_start: {}", local_batch_start);
            log_debug("local_batch_end: {}", local_batch_end);
            log_debug("local_nh_start: {}", local_nh_start);
            log_debug("local_nh_end: {}", local_nh_end);
            log_debug("local_q_start: {}", local_q_start);
            log_debug("local_q_end: {}", local_q_end);


            SetRuntimeArgs(program, reader_kernels_id, core, { q_addr, k_addr, v_addr, mask_addr, i, num_cores, local_batch_start, local_batch_end, local_nh_start, local_nh_end, local_q_start, local_q_end });
            SetRuntimeArgs(program, writer_kernels_id, core, { out_addr, i, num_cores, local_batch_start, local_batch_end, local_nh_start, local_nh_end, local_q_start, local_q_end });
            SetRuntimeArgs(program, compute_kernels_id, core, { i, num_cores, local_batch_start, local_batch_end, local_nh_start, local_nh_end, local_q_start, local_q_end });
        }


    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
} // scale_mask_softmax_multi_core

}  // namespace tt_metal
}  // namespace tt_metal
}  // namespace tt
