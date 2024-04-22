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

    tt::DataFormat input_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_q.get_dtype());
    uint32_t input_tile_size = tt_metal::detail::TileSize(input_data_format);

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
            fp32_dest_acc_en = input_data_format == tt::DataFormat::Float32 ? true : compute_kernel_config.fp32_dest_acc_en;
        } else {
            TT_FATAL("arch not supported");
        }

    }, compute_kernel_config);



    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scalar_tile_size = tt_metal::detail::TileSize(scalar_cb_data_format);

    tt::DataFormat out0_cb_data_format = tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t out0_tile_size = tt_metal::detail::TileSize(out0_cb_data_format);

    tt::DataFormat mask_cb_data_format = attn_mask.has_value() ? tt_metal::datatype_to_dataformat_converter(attn_mask.value().get_dtype()) : tt::DataFormat::Float16_b;
    uint32_t mask_tile_size = tt_metal::detail::TileSize(mask_cb_data_format);

    tt::DataFormat im_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t im_tile_size = tt_metal::detail::TileSize(im_cb_data_format);

    log_info("in0_cb_data_format: {}", input_data_format);
    log_info("out0_cb_data_format: {}", out0_cb_data_format);
    log_info("mask_cb_data_format: {}", mask_cb_data_format);
    log_info("im_cb_data_format: {}", im_cb_data_format);
    log_info("math_fidelity: {}", math_fidelity);
    log_info("math_approx_mode: {}", math_approx_mode);
    log_info("fp32_dest_acc_en: {}", fp32_dest_acc_en);

    auto q_buffer = input_tensor_q.buffer();
    auto k_buffer = input_tensor_k.buffer();
    auto v_buffer = input_tensor_v.buffer();
    auto mask_buffer = attn_mask.has_value() ? attn_mask.value().buffer() : nullptr;
    TT_ASSERT(mask_buffer != nullptr);

    auto out0_buffer = output_tensor.buffer();


    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t q_tiles  = Sq_chunk_t * DHt;
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

    // Parallelization scheme
    // We parallelize over batch, q_heads, and q_seq_len. We always run one batch and one head per core,
    // so the question is how much of the q_seq_len does each core handle.
    uint32_t q_parallel_factor = num_cores / (B * NQH);
    // q_parallel_factor should max out at q_num_chunks
    q_parallel_factor = std::min(q_parallel_factor, q_num_chunks);
    TT_FATAL(q_parallel_factor * B * NQH == num_cores);


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
    const uint32_t half_dst = 8; // Always 8 since stats don't use fp32 dst
    const uint32_t stats_granularity = std::min(Sq_chunk_t, half_dst);
    // Find log2 of stats_granularity using std
    const uint32_t log2_stats_granularity = std::log2(stats_granularity);
    // Assert that this is a power of 2
    TT_ASSERT(stats_granularity == (1 << log2_stats_granularity));

    const uint32_t sub_exp_granularity = std::min(Sk_chunk_t, half_dst);
    const uint32_t log2_sub_exp_granularity = std::log2(sub_exp_granularity);
    TT_ASSERT(sub_exp_granularity == (1 << log2_sub_exp_granularity));

    const uint32_t mul_bcast_granularity = std::min(Sq_chunk_t * Sk_chunk_t, half_dst);
    const uint32_t log2_mul_bcast_granularity = std::log2(mul_bcast_granularity);
    TT_ASSERT(mul_bcast_granularity == (1 << log2_mul_bcast_granularity));

    // Log these
    log_info("stats_granularity: {}", stats_granularity);
    log_info("log2_stats_granularity: {}", log2_stats_granularity);
    log_info("sub_exp_granularity: {}", sub_exp_granularity);
    log_info("log2_sub_exp_granularity: {}", log2_sub_exp_granularity);
    log_info("mul_bcast_granularity: {}", mul_bcast_granularity);
    log_info("log2_mul_bcast_granularity: {}", log2_mul_bcast_granularity);



    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    bfloat16 bfloat_identity_scalar = bfloat16(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        packed_identity_scalar,
    };

    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
    };

    std::vector<uint32_t> compute_compile_time_args = {
        // matmul args
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
    // Q input
    auto c_in0_config = CircularBufferConfig(q_tiles * input_tile_size, {{CB::c_in0, input_data_format}}).set_page_size(CB::c_in0, input_tile_size);
    auto cb_in0_id = CreateCircularBuffer(program, core_grid, c_in0_config);
    // K input
    auto c_in1_config = CircularBufferConfig(k_tiles * input_tile_size, {{CB::c_in1, input_data_format}}).set_page_size(CB::c_in1, input_tile_size);
    auto cb_in1_id = CreateCircularBuffer(program, core_grid, c_in1_config);
    // V input
    auto c_in2_config = CircularBufferConfig(v_tiles * input_tile_size, {{CB::c_in2, input_data_format}}).set_page_size(CB::c_in2, input_tile_size);
    auto cb_in2_id = CreateCircularBuffer(program, core_grid, c_in2_config);

    // attn_mask input
    auto c_in3_config = CircularBufferConfig(mask_tiles * input_tile_size, {{CB::c_in3, input_data_format}}).set_page_size(CB::c_in3, input_tile_size);
    auto cb_in3_id = CreateCircularBuffer(program, core_grid, c_in3_config);

    // scale input
    auto c_in4_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{CB::c_in4, input_data_format}}).set_page_size(CB::c_in4, input_tile_size);
    auto cb_in4_id = CreateCircularBuffer(program, core_grid, c_in4_config);

    // identity scale input
    auto c_in5_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{CB::c_in5, input_data_format}}).set_page_size(CB::c_in5, input_tile_size);
    auto cb_in5_id = CreateCircularBuffer(program, core_grid, c_in5_config);

    // cb_qk_im
    auto c_intermed0_config = CircularBufferConfig(qk_tiles * im_tile_size, {{CB::c_intermed0, im_cb_data_format}}).set_page_size(CB::c_intermed0, im_tile_size);
    auto cb_intermed0_id = CreateCircularBuffer(program, core_grid, c_intermed0_config);

    // cb_out_im
    auto c_intermed1_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{CB::c_intermed1, im_cb_data_format}}).set_page_size(CB::c_intermed1, im_tile_size);
    auto cb_intermed1_id = CreateCircularBuffer(program, core_grid, c_intermed1_config);

    // cb_out_accumulate_im
    auto c_intermed2_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{CB::c_intermed2, im_cb_data_format}}).set_page_size(CB::c_intermed2, im_tile_size);
    auto cb_intermed2_id = CreateCircularBuffer(program, core_grid, c_intermed2_config);

    // cb_cur_max
    auto c_intermed3_config = CircularBufferConfig(statistics_tiles * input_tile_size, {{CB::c_intermed3, input_data_format}}).set_page_size(CB::c_intermed3, input_tile_size);
    auto cb_intermed3_id = CreateCircularBuffer(program, core_grid, c_intermed3_config);

    // cb_prev_max
    auto c_intermed4_config = CircularBufferConfig(statistics_tiles * input_tile_size, {{CB::c_intermed4, input_data_format}}).set_page_size(CB::c_intermed4, input_tile_size);
    auto cb_intermed4_id = CreateCircularBuffer(program, core_grid, c_intermed4_config);

    // cb_cur_sum
    auto c_intermed5_config = CircularBufferConfig(statistics_tiles * input_tile_size, {{CB::c_intermed5, input_data_format}}).set_page_size(CB::c_intermed5, input_tile_size);
    auto cb_intermed5_id = CreateCircularBuffer(program, core_grid, c_intermed5_config);

    // cb_prev_sum
    auto c_intermed6_config = CircularBufferConfig(statistics_tiles * input_tile_size, {{CB::c_intermed6, input_data_format}}).set_page_size(CB::c_intermed6, input_tile_size);
    auto cb_intermed6_id = CreateCircularBuffer(program, core_grid, c_intermed6_config);

    // cb_exp_max_diff
    auto c_intermed7_config = CircularBufferConfig(statistics_tiles * input_tile_size, {{CB::c_intermed7, input_data_format}}).set_page_size(CB::c_intermed7, input_tile_size);
    auto cb_intermed7_id = CreateCircularBuffer(program, core_grid, c_intermed7_config);

    // Output
    auto c_out0_config = CircularBufferConfig(out0_t * out0_tile_size, {{CB::c_out0, out0_cb_data_format}}).set_page_size(CB::c_out0, out0_tile_size);
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
    union {float f; uint32_t u;} scale_union; scale_union.f = scale.value_or(1.0f);




    log_info("Parallelization scheme:");
    log_info("num_cores: {}", num_cores);
    log_info("batch size: {}", B);
    log_info("num q_heads: {}", NQH);
    log_info("q_parallel_factor: {}", q_parallel_factor);


    // Set reader rt args
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        // log_info("core: {} getting runtime args for idx {i}", core, i);

        SetRuntimeArgs(program, reader_kernels_id, core, { q_addr, k_addr, v_addr, mask_addr, B, NQH, NKH, St, DHt, Sq_chunk_t, q_num_chunks, Sk_chunk_t, k_num_chunks, scale_union.u, i, num_cores, q_parallel_factor });
        SetRuntimeArgs(program, writer_kernels_id, core, { out_addr, B, NQH, St, DHt, Sq_chunk_t, q_num_chunks, Sk_chunk_t, k_num_chunks, i, num_cores, q_parallel_factor });
        SetRuntimeArgs(program, compute_kernels_id, core, { B, NQH, NKH, St, DHt, Sq_chunk_t, q_num_chunks, Sk_chunk_t, k_num_chunks, i, num_cores, q_parallel_factor });
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

    auto override_runtime_arguments_callback = [
            // reader_kernels_id,
            // writer_kernels_id,
            // softmax_kernels_id,
            // grid_size,
            // scalar_tile_size,
            // // in0_tile_size,
            // im_tile_size,
            // out0_tile_size,
            // mask_tile_size,
            // cb_in0_id,
            // cb_out0_id,
            // // cb_intermed1_id,
            // cb_in2_id,
            // cb_intermed0_id,
            // cb_intermed3_id,
            // cb_in3_id,
            // cb_in4_id,
            // causal_mask
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {

        // const auto scale = static_cast<const Softmax*>(operation)->scale;

        // auto src_buffer_address = input_tensors.at(0).buffer()->address();
        // auto mask_buffer_address = optional_input_tensors.at(0).has_value() ? optional_input_tensors.at(0).value().buffer()->address() : 0;
        // auto dst_buffer_address = output_tensors.size() == 1 ? output_tensors.at(0).buffer()->address() : src_buffer_address;

        // const auto shape = input_tensors.at(0).get_legacy_shape();
        // uint32_t W = shape[-1], H = (input_tensors.at(0).volume() / (shape[0] * shape[-1])), NC = shape[0];
        // uint32_t HW = H*W;

        // uint32_t Wt = W/TILE_WIDTH;
        // uint32_t Ht = H/TILE_HEIGHT;

        // int32_t num_tiles = input_tensors.at(0).volume()/TILE_HW;
        // uint32_t block_size = find_max_divisor(Wt, 8);

        // // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
        // uint32_t in0_t  = block_size*2;
        // uint32_t out0_t = block_size*2;
        // uint32_t im1_t  = 1; // 1/sum(exp(x))
        // uint32_t in2_t  = 1; // scaler for reduce coming from reader
        // uint32_t in3_t  = 1; // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
        // uint32_t in4_t  = div_up(Wt, block_size)*block_size; // attention mask (N,C,32,W) - Wt is reused for each Ht, NC is cycled

        // // cb_exps - keeps exps in CB in L1 to avoid recomputing
        // uint32_t im0_t  = block_size*div_up(Wt, block_size);
        // TT_ASSERT(im0_t == Wt);

        // // used for buffering scale-mask
        // // can't easily reuse im0_t because cumulative wait for Wt needs to have Wt tiles contiguous free
        // uint32_t im3_t  = block_size*(div_up(Wt, block_size)+1);
        // TT_ASSERT(im3_t == Wt+block_size);

        // TT_ASSERT(Wt % block_size == 0);
        // TT_ASSERT((block_size != -1) && "Wt must be divisible by one of the numbers in the range from 8 to 1.");
        // TT_ASSERT(im0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
        // TT_ASSERT(out0_t % block_size == 0 && "Size of cb must be divisible by the size of block used by the reader and compute kernel.");
        // TT_ASSERT(in4_t % block_size == 0);
        // TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");

        // uint32_t NCHt = NC*Ht;
        // uint32_t num_tile_rows = NC * Ht;
        // auto all_device_cores = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
        // auto [num_cores, all_cores, core_group_1, core_group_2, num_tile_rows_per_core_group_1, num_tile_rows_per_core_group_2] = split_work_to_cores(grid_size, num_tile_rows, true);

        // UpdateCircularBufferTotalSize(program, cb_in0_id, in0_t * in0_tile_size);
        // UpdateCircularBufferTotalSize(program, cb_out0_id, out0_t * out0_tile_size);
        // UpdateCircularBufferTotalSize(program, cb_intermed1_id, im1_t * im_tile_size);
        // UpdateCircularBufferTotalSize(program, cb_in2_id, in2_t * scalar_tile_size);
        // UpdateCircularBufferTotalSize(program, cb_intermed0_id, im0_t * im_tile_size);

        // if (optional_input_tensors.at(0).has_value()) {
        //     UpdateCircularBufferTotalSize(program, cb_intermed3_id.value(), im3_t * im_tile_size);
        //     UpdateCircularBufferTotalSize(program, cb_in3_id.value(), in3_t * scalar_tile_size);
        //     UpdateCircularBufferTotalSize(program, cb_in4_id.value(), in4_t * mask_tile_size);
        // }

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
        //     uint32_t mask_curr_ht = curr_ht % Wt;   // the start offset for causal mask
        //     uint32_t mask_offset = curr_row / Ht * Wt * Wt; // causal mask batch offset
        //     uint32_t mask_id = causal_mask ? (mask_curr_ht * Wt + mask_offset) : (curr_row / Ht * Wt); // causal mask start offset + causal mask batch offset

        //     if (causal_mask) {
        //         SetRuntimeArgs(program, reader_kernels_id, core, { src_buffer_address, block_size, s.u, num_tile_rows_per_core, tile_offset, Wt, Ht, mask_buffer_address, curr_ht, mask_id, 0x3f803f80, mask_curr_ht, mask_offset }); // [10]=1.0f is scaler
        //     } else {
        //         SetRuntimeArgs(program, reader_kernels_id, core, { src_buffer_address, block_size, s.u, num_tile_rows_per_core, tile_offset, Wt, Ht, mask_buffer_address, curr_ht, mask_id, 0x3f803f80 }); // [10]=1.0f is scaler
        //     }

        //     SetRuntimeArgs(program, softmax_kernels_id, core, { num_tile_rows_per_core, Ht, Wt, block_size, curr_ht });
        //     SetRuntimeArgs(program, writer_kernels_id, core, { dst_buffer_address, num_tile_rows_per_core * Wt, tile_offset, block_size });
        //     curr_row += num_tile_rows_per_core;
        // }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
} // scale_mask_softmax_multi_core

}  // namespace tt_metal
}  // namespace tt_metal
}  // namespace tt
