// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/mm_multi_device/mm_multi_device.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

namespace tt::tt_metal {

Tensor mm_multi_device(const Tensor& a, const Tensor& b, std::optional<const Tensor> bias) {
    TT_ASSERT(not bias.has_value());
    return operation::run_without_autoformat(MatmulMultiDevice{}, {a, b}, {}).at(0);
}

void MatmulMultiDevice::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto shape_a = input_tensor_a.get_legacy_shape();
    auto shape_b = input_tensor_b.get_legacy_shape();
    TT_FATAL(shape_a[2] == shape_b[3]);
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE,
        "Operands to matmul multi device need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
        "Operands to matmul multi device need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor_a.device() == input_tensor_b.device(),
        "Operands to matmul multi device need to be on the same device!");
    TT_FATAL(
        (input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE),
        "Inputs to matmul multi device must be tilized");

    TT_FATAL(
        (input_tensor_a.is_sharded() && input_tensor_b.is_sharded()), "Inputs to matmul multi device must be sharded");

    TT_FATAL(input_tensor_a.shard_spec().value().shape[0] % TILE_HEIGHT == 0);
    TT_FATAL(input_tensor_a.shard_spec().value().shape[1] % TILE_WIDTH == 0);
    TT_FATAL(input_tensor_b.shard_spec().value().shape[0] % TILE_HEIGHT == 0);
    TT_FATAL(input_tensor_b.shard_spec().value().shape[1] % TILE_WIDTH == 0);

    TT_FATAL(input_tensor_a.shard_spec().value().shape[1] == input_tensor_b.shard_spec().value().shape[0]);

    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
    TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
}

std::vector<Shape> MatmulMultiDevice::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto shape_a = input_tensor_a.get_legacy_shape();
    auto shape_b = input_tensor_b.get_legacy_shape();
    auto output_shape = shape_a;
    output_shape[3] = shape_b[3];
    return {output_shape};
}

std::vector<Tensor> MatmulMultiDevice::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    ShardSpec shard_spec_a = input_tensor_a.shard_spec().value();
    ShardSpec shard_spec_b = input_tensor_b.shard_spec().value();

    TT_ASSERT(shard_spec_a.grid.ranges().size() == 1);
    TT_ASSERT(shard_spec_b.grid.ranges().size() == 1);
    CoreRange core_range_a = *shard_spec_a.grid.ranges().begin();
    CoreRange core_range_b = *shard_spec_b.grid.ranges().begin();
    CoreRange out_core_range(
        CoreCoord(core_range_b.start.x, core_range_b.end.x), CoreCoord(core_range_a.start.y, core_range_a.end.y));
    CoreRangeSet out_core_range_set({out_core_range});

    std::array<uint32_t, 2> out_shard_shape = {shard_spec_a.shape[0], shard_spec_b.shape[1]};

    ShardSpec shard_spec(out_core_range_set, out_shard_shape, shard_spec_a.orientation);
    MemoryConfig output_mem_config = MemoryConfig{
        .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
        .buffer_type = BufferType::L1,
        .shard_spec = shard_spec,
    };
    return {create_sharded_device_tensor(
        this->compute_output_shapes(input_tensors).at(0),
        input_tensor_a.get_dtype(),
        Layout::TILE,
        input_tensor_a.device(),
        output_mem_config)};
}

operation::ProgramWithCallbacks MatmulMultiDevice::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    Program program;

    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& output_tensor = output_tensors.at(0);

    ShardSpec shard_spec_a = input_tensor_a.shard_spec().value();
    ShardSpec shard_spec_b = input_tensor_b.shard_spec().value();

    auto m_tiles = shard_spec_a.shape[0] / TILE_HEIGHT;
    auto k_tiles = shard_spec_a.shape[1] / TILE_WIDTH;
    auto n_tiles = shard_spec_b.shape[1] / TILE_WIDTH;

    auto block_size_tiles = 1;

    tt::DataFormat df_a = tt_metal::datatype_to_dataformat_converter(input_tensor_a.get_dtype());
    uint32_t tile_size_a = tt_metal::detail::TileSize(df_a);
    tt::DataFormat df_b = tt_metal::datatype_to_dataformat_converter(input_tensor_b.get_dtype());
    uint32_t tile_size_b = tt_metal::detail::TileSize(df_b);
    tt::DataFormat out_df = tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t out_tile_size = tt_metal::detail::TileSize(out_df);

    auto compute_with_storage_grid_size = input_tensor_a.device()->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(m_tiles * k_tiles * tile_size_a, {{src0_cb_index, df_a}})
            .set_page_size(src0_cb_index, block_size_tiles * tile_size_a)
            .set_globally_allocated_address(*input_tensor_a.buffer());
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src0_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(k_tiles * n_tiles * tile_size_b, {{src1_cb_index, df_b}})
            .set_page_size(src1_cb_index, block_size_tiles * tile_size_b)
            .set_globally_allocated_address(*input_tensor_b.buffer());
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src1_config);

    uint32_t block0_cb_index = 2;
    tt_metal::CircularBufferConfig cb_block0_config =
        tt_metal::CircularBufferConfig(block_size_tiles * tile_size_a * 2, {{block0_cb_index, df_a}})
            .set_page_size(block0_cb_index, block_size_tiles * tile_size_a)
            .set_globally_allocated_address(*input_tensor_a.buffer());
    auto cb_block0 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_block0_config);

    uint32_t block1_cb_index = 3;
    tt_metal::CircularBufferConfig cb_block1_config =
        tt_metal::CircularBufferConfig(block_size_tiles * tile_size_b * 2, {{block1_cb_index, df_b}})
            .set_page_size(block1_cb_index, block_size_tiles * tile_size_b)
            .set_globally_allocated_address(*input_tensor_b.buffer());
    auto cb_block1 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_block1_config);

    std::vector<uint32_t> dataflow_ct_args;
    KernelHandle lhs_dataflow_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/mm_multi_device/kernels/dataflow.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig(dataflow_ct_args));

    KernelHandle rhs_dataflow_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/mm_multi_device/kernels/dataflow.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig(dataflow_ct_args));

    std::vector<uint32_t> compute_ct_args;
    KernelHandle compute_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/mm_multi_device/kernels/compute.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig(compute_ct_args));

    auto override_runtime_arguments_callback = [](const void* operation,
                                                  Program& program,
                                                  const std::vector<Tensor>& input_tensors,
                                                  const std::vector<std::optional<const Tensor>>&,
                                                  const std::vector<Tensor>& output_tensors) {};
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::OpPerformanceModel MatmulMultiDevice::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto &in_a_shape = input_tensors.at(0).get_shape();
    const auto &in_b_shape = input_tensors.at(1).get_shape();
    const auto &out_shape = output_tensors.at(0).get_shape();

    const auto &t = output_tensors.at(0);
    if (t.storage_type() != StorageType::DEVICE) {
        tt::log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }

    auto arch =
        t.storage_type() == StorageType::DEVICE ? t.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    const int num_cores = (arch == ARCH::WORMHOLE_B0) ? 8 * 8 : 9 * 12;
    const int tensix_mul_adds_per_cycle_lofi = (arch == ARCH::WORMHOLE_B0) ? 4096 : 2048;

    // Calculate number of mul/add operations
    // TODO: add bias modeling
    int64_t num_mul_adds_per_elem = in_a_shape[3] * 2; // 1 multiply and 1 add per element
    int64_t num_mul_adds = num_mul_adds_per_elem * out_shape[2] * out_shape[3] * out_shape[1] * out_shape[0];

    MathFidelity math_fidelity = MathFidelity::Invalid;

    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            math_fidelity = compute_kernel_config.math_fidelity;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            math_fidelity = compute_kernel_config.math_fidelity;
        } else {
            TT_FATAL("arch not supported");
        }
    }, compute_kernel_config);


    int ideal_dev_clock_cycles = std::ceil(((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi)) * (float)operation::OpPerformanceModel::fidelity_multiplier(math_fidelity));

    return operation::OpPerformanceModel(input_tensors, output_tensors, ideal_dev_clock_cycles);
}

const operation::Hash MatmulMultiDevice::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    operation::Hash hash = tt::stl::hash::hash_objects(
        0,
        typeid(*this).hash_code(),
        input_tensor_a.get_dtype(),
        input_tensor_a.memory_config(),
        input_tensor_b.get_dtype(),
        input_tensor_b.memory_config());

    return hash;
}

}  // namespace tt::tt_metal
