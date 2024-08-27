// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/constants.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor_utils.hpp"
#include "matmul_validate_parameters.hpp"

namespace ttnn::operations::primary {

std::optional<tt::tt_metal::StorageType> MatmulValidateParameters::str_to_storage_type(const string& storage_type_str)
{
    if (storage_type_str == "OWNED") return StorageType::OWNED;
    if (storage_type_str == "DEVICE") return StorageType::DEVICE;
    if (storage_type_str == "BORROWED") return StorageType::BORROWED;
    if (storage_type_str == "MULTI_DEVICE") return StorageType::MULTI_DEVICE;
    if (storage_type_str == "MULTI_DEVICE_HOST") return StorageType::MULTI_DEVICE_HOST;
    return std::nullopt;
}

std::optional<tt::tt_metal::Layout> MatmulValidateParameters::str_to_layout(const string& layout_str)
{
    if (layout_str == "ROW_MAJOR") return Layout::ROW_MAJOR;
    if (layout_str == "TILE") return Layout::TILE;
    if (layout_str == "INVALID") return Layout::INVALID;
    return std::nullopt;
}

std::optional<tt::tt_metal::DataType> MatmulValidateParameters::str_to_data_type(const string& data_type_str)
{
    if (data_type_str == "BFLOAT16") return DataType::BFLOAT16;
    if (data_type_str == "FLOAT32") return DataType::FLOAT32;
    if (data_type_str == "UINT32") return DataType::UINT32;
    if (data_type_str == "BFLOAT8_B") return DataType::BFLOAT8_B;
    if (data_type_str == "BFLOAT4_B") return DataType::BFLOAT4_B;
    if (data_type_str == "UINT8") return DataType::UINT8;
    if (data_type_str == "UINT16") return DataType::UINT16;
    if (data_type_str == "INT32") return DataType::INT32;
    if (data_type_str == "INVALID") return DataType::INVALID;
    return std::nullopt;
}

const auto volume(const tt::tt_metal::Shape& shape) {
    auto rank = shape.rank();
    auto volume = 1;
    for (auto index = 0; index < rank; index++) {
        volume *= shape.without_padding()[index];
    }
    return volume;
}

MatmulProgramConfig MatmulValidateParameters::create_matmul_program_config() {
    auto a_shape = input_shapes.at(0);
    auto b_shape = input_shapes.at(1);
    auto a_padded_shape = input_shapes_with_padding.at(0);
    auto b_padded_shape = input_shapes_with_padding.at(1);
    auto a_layout = input_memory_layouts.at(0);
    auto inteneded_k_size_of_a = a_shape[-1];
    auto inteneded_k_size_of_b = b_shape[-2];
    auto k_size = a_padded_shape[-1];
    auto m_size = a_padded_shape[-2];
    auto n_size = b_padded_shape[-1];
    uint32_t batch_size_a = get_batch_size(a_padded_shape);
    uint32_t batch_size_b = get_batch_size(b_padded_shape);
    bool input_b_is_batched = batch_size_b > 1;
    bool any_size_within_tile = k_size <= ttnn::TILE_SIZE || m_size <= ttnn::TILE_SIZE || n_size <= ttnn::TILE_SIZE;
    bool fp32_dest_acc_en = bmm_op_utils::get_fp32_dest_acc_en(compute_kernel_config);
    bool a_is_sharded = input_sharded.at(0);
    TT_FATAL(inteneded_k_size_of_a == inteneded_k_size_of_b, "The k dimension does not match between tensors");
    TT_FATAL(
        (batch_size_a * m_size) % ttnn::TILE_SIZE == 0 && k_size % ttnn::TILE_SIZE == 0 &&
            n_size % ttnn::TILE_SIZE == 0,
        "The last two dimensions of the first tensor and the last dimension of the second tensor must be a multiple of "
        "tile size");
    auto core_coord = input_tensor_a.device()->compute_with_storage_grid_size();
    bool has_user_core_coord = user_core_coord.has_value();
    if (has_user_core_coord) {
        auto x = user_core_coord.value().x;
        auto y = user_core_coord.value().y;
        if (x <= core_coord.x && y <= core_coord.y) {
            core_coord = user_core_coord.value();
        }
    }

    uint32_t m_tiles_per_core;
    uint32_t n_tiles_per_core;
    uint32_t k_tiles_per_core;
    if (input_b_is_batched) {
        TT_FATAL(!fused_activation.has_value(), "Cannot use activation with batched input b");
        if (!a_is_sharded && !input_tensor_b.is_sharded()) {
            m_tiles_per_core = div_up(m_size, ttnn::TILE_SIZE);
            n_tiles_per_core = div_up(n_size, ttnn::TILE_SIZE);
            k_tiles_per_core = 1;  // TODO(arakhmati): Can it be more than 1 without running out of memory?
        } else if (a_is_sharded) {
            TT_FATAL(
                a_layout != TensorMemoryLayout::WIDTH_SHARDED,
                "MatmulMultiCoreReuseProgramConfig: Cannot be width sharded");
            auto shard_shape = input_tensor_a_memory_config.shard_spec.value().shape;
            uint32_t n = b_shape[-1] / ttnn::TILE_SIZE;
            m_tiles_per_core = shard_shape[0] / ttnn::TILE_SIZE;
            n_tiles_per_core = n;
            k_tiles_per_core = shard_shape[1] / ttnn::TILE_SIZE;
        } else {
            TT_FATAL(
                input_tensor_b_memory_config.memory_layout != TensorMemoryLayout::WIDTH_SHARDED,
                "MatmulMultiCoreReuseProgramConfig: Cannot be width sharded");
            auto shard_shape = input_tensor_b_memory_config.shard_spec.value().shape;
            m_tiles_per_core = div_up(m_size, ttnn::TILE_SIZE);
            n_tiles_per_core = shard_shape[1] / ttnn::TILE_SIZE;
            k_tiles_per_core = 1;
        }

        auto matmul_params = bmm_op_utils::get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, fp32_dest_acc_en);
        uint32_t out_subblock_h = std::get<0>(matmul_params);
        uint32_t out_subblock_w = std::get<1>(matmul_params);

        return MatmulMultiCoreReuseProgramConfig{
            .compute_with_storage_grid_size = {core_coord.x, core_coord.y},
            .in0_block_w = k_tiles_per_core,
            .out_subblock_h = out_subblock_h,
            .out_subblock_w = out_subblock_w,
            .per_core_M = m_tiles_per_core,
            .per_core_N = n_tiles_per_core,
        };
    }

    auto height = batch_size_a * m_size;
    auto width = n_size;
    auto height_width_ratio = (height > width) ? height / width : width / height;
    bool a_is_block_sharded = a_layout == TensorMemoryLayout::BLOCK_SHARDED;
    if (height_width_ratio > 8 || any_size_within_tile) {
        if (!a_is_block_sharded) {
            return create_matmul_1d_systolic_array_program_config(
                a_shape, b_shape, core_coord, fused_activation, fp32_dest_acc_en, a_layout);
        }
    }
    if (!a_is_sharded) {
        m_tiles_per_core = (uint32_t)std::ceil((((double)batch_size_a * m_size) / ttnn::TILE_SIZE) / core_coord.y);
        n_tiles_per_core = (uint32_t)std::ceil((double)n_size / ttnn::TILE_SIZE / core_coord.x);
        k_tiles_per_core = 4;  // TODO(arakhmati): What is a good starting point?
        while ((k_size / ttnn::TILE_SIZE) % k_tiles_per_core != 0) {
            k_tiles_per_core -= 1;
        }
    } else {
        if (!a_is_block_sharded) {
            return create_matmul_1d_systolic_array_program_config(
                a_shape, b_shape, core_coord, fused_activation, fp32_dest_acc_en, a_layout);
        }
        uint32_t k = a_shape[-1] / ttnn::TILE_SIZE;
        uint32_t n = b_shape[-1] / ttnn::TILE_SIZE;
        auto shard_shape = input_tensor_a_memory_config.shard_spec.value().shape;
        m_tiles_per_core = shard_shape[0] / ttnn::TILE_SIZE;
        n_tiles_per_core = (n * shard_shape[1]) / (k * ttnn::TILE_SIZE);
        k_tiles_per_core = shard_shape[1] / ttnn::TILE_SIZE;
    }

    auto matmul_params = bmm_op_utils::get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, fp32_dest_acc_en);
    uint32_t out_subblock_h = std::get<0>(matmul_params);
    uint32_t out_subblock_w = std::get<1>(matmul_params);
    bool transpose_mcast =
        a_is_block_sharded && input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR;
    if (out_subblock_w != n_tiles_per_core) {
        out_subblock_h = 1;
    }

    return MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {core_coord.x, core_coord.y},
        .in0_block_w = k_tiles_per_core,
        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
        .per_core_M = m_tiles_per_core,
        .per_core_N = n_tiles_per_core,
        .transpose_mcast = transpose_mcast,
        .fused_activation = fused_activation,
    };
}

void MatmulValidateParameters::validate()
{
    TT_FATAL(input_shapes.size() == 2);
    const auto& a_shape = input_shapes.at(0);
    const auto& b_shape = input_shapes.at(1);

    TT_FATAL(
        (input_layouts.at(0) == Layout::TILE && input_layouts.at(1) == Layout::TILE),
        "Inputs to matmul must be tilized");
    TT_FATAL(
        a_shape[-1] == b_shape[-2],
        "The width of the first tensor must be equal to the height of the second tensor. Mismatch: width={} height={}",
        a_shape[-1],
        b_shape[-2]);

    TT_FATAL(this->bcast_batch.has_value());
    if (this->bcast_batch.value()) {
        TT_FATAL(
            get_batch_size(b_shape) == 1 &&
            "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN or equivalent");
    } else {
        // same condition as above, different message
        TT_FATAL(a_shape.rank() == b_shape.rank() && "bmm (non-bcast matmul) expects input tensors of the same rank");
        for (auto i = 0; i < a_shape.rank() - 2; i++) {
            TT_FATAL(
                a_shape[i] == b_shape[i] &&
                "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN or equivalent");
        }
    }

    TT_FATAL(is_floating_point(input_data_types.at(0)), "Unsupported data format");
    TT_FATAL(
        input_storage_types.at(0) == StorageType::DEVICE and input_storage_types.at(1) == StorageType::DEVICE,
        "Operands to matmul need to be on device!");
    TT_FATAL(has_buffers.at(0) and has_buffers.at(1), "Operands to matmul need to be allocated in buffers on device!");
    TT_FATAL(input_devices.at(0) == input_devices.at(1), "Operands to matmul need to be on the same device!");

    // MatmulProgramConfig chosen_program_config = get_program_config(input_tensor_a, input_tensor_b, this);

    MatmulProgramConfig chosen_program_config = MatmulMultiCoreProgramConfig();

    if (has_bias) {
        TT_FATAL(input_layouts.at(2) == Layout::TILE, "Unsupported input layout");
        const auto& bias_shape = input_shapes.at(2);
        uint32_t bias_batch_size = get_batch_size(bias_shape);
        TT_FATAL(bias_batch_size == 1, "Unsupported bias shape: batch size not equal to 1.");
        TT_FATAL(
            bias_shape[-2] == tt::constants::TILE_HEIGHT, "Unsupported bias shape: second last dimension not equal to tile height");
        TT_FATAL(
            bias_shape[-1] == b_shape[-1],
            "Unsupported bias shape: last dimension not equal to second input's last dimension.");
    }

    if (this->untilize_out) {
        TT_FATAL(
            (this->output_data_type == DataType::BFLOAT16) || (this->output_data_type == DataType::FLOAT32));
    }

    std::visit(
        [this](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            // TODO: For 1D and 2D mcasts, we don't check if tensor is single core or single row/col
            // We can uplift these variants to skip mcasting to support single core (1D) or single row/col (2D)
            if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                if (program_config.mcast_in0) {
                    if (input_sharded.at(0)) {
                        TT_FATAL(program_config.fuse_batch);
                        TT_FATAL(input_memory_layouts.at(0) == TensorMemoryLayout::WIDTH_SHARDED);
                        if (this->output_sharded) {
                            TT_FATAL(input_buffer_types.at(0) == input_buffer_types.at(1));
                            TT_FATAL(
                                input_memory_layouts.at(0) == output_layout);
                        }
                        TT_FATAL(input_orientations.at(0) == ShardOrientation::ROW_MAJOR);
                        uint32_t M =
                            (program_config.fuse_batch ? volume(input_shapes.at(0)) / input_shapes.at(0)[-1]
                                                       : input_shapes.at(0)[-2]) /
                            tt::constants::TILE_HEIGHT;
                        uint32_t N =  input_shapes.at(1)[-1] / tt::constants::TILE_WIDTH;
                        uint32_t K =  input_shapes.at(0)[-1] / tt::constants::TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;
                        auto shard_shape = input_shard_shapes.at(0);

                        // No padding
                        TT_FATAL(M == per_core_M);
                        TT_FATAL(per_core_M == (shard_shape[0] / tt::constants::TILE_HEIGHT));
                        TT_FATAL(K % program_config.in0_block_w == 0);
                        TT_FATAL((shard_shape[1] / tt::constants::TILE_WIDTH) % program_config.in0_block_w == 0);
                    }
                    if (output_sharded) {
                        TT_FATAL(output_layout == TensorMemoryLayout::WIDTH_SHARDED);
                        uint32_t M =
                            (program_config.fuse_batch ? volume(input_shapes.at(0)) / input_shapes.at(0)[-1]
                                                       : input_shapes.at(0)[-2]) /
                            tt::constants::TILE_HEIGHT;
                        uint32_t N = input_shapes.at(1)[-1] / tt::constants::TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        // No padding
                        TT_FATAL(M == per_core_M);

                        TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                    }
                } else {
                    if (input_sharded.at(0)) {
                        TT_FATAL(program_config.fuse_batch);
                        TT_FATAL(input_memory_layouts.at(0) == TensorMemoryLayout::HEIGHT_SHARDED);
                        if (output_sharded) {
                            TT_FATAL(input_buffer_types.at(0) == output_buffer_type);
                            TT_FATAL(
                                input_memory_layouts.at(0) == output_layout);
                        }
                        TT_FATAL(input_orientations.at(0) == ShardOrientation::ROW_MAJOR);
                        uint32_t M =
                            (program_config.fuse_batch ? volume(input_shapes.at(0)) / input_shapes.at(0)[-1]
                                                       : input_shapes.at(0)[-2]) /
                            tt::constants::TILE_HEIGHT;
                        uint32_t K = input_shapes.at(0)[-1] / tt::constants::TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        auto shard_shape = input_shard_shapes.at(0);

                        TT_FATAL(tt::div_up(M, per_core_M) == input_num_cores.at(0));
                        TT_FATAL(per_core_M == (shard_shape[0] / tt::constants::TILE_HEIGHT));
                        TT_FATAL(K % program_config.in0_block_w == 0);
                        TT_FATAL(K == (shard_shape[1] / tt::constants::TILE_WIDTH));
                    }
                    if (output_sharded) {
                        TT_FATAL(output_layout == TensorMemoryLayout::HEIGHT_SHARDED);
                        uint32_t M =
                            (program_config.fuse_batch ? volume(input_shapes.at(0)) / input_shapes.at(0)[-1]
                                                       : input_shapes.at(0)[-2]) /
                             tt::constants::TILE_HEIGHT;
                        uint32_t N = input_shapes.at(1)[-1] / tt::constants::TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        TT_FATAL(N == per_core_N);
                        TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                    }
                }
                TT_FATAL(input_memory_layouts.at(1) == TensorMemoryLayout::INTERLEAVED);
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                TT_FATAL(input_sharded.at(0));
                TT_FATAL(output_sharded);
                TT_FATAL(input_memory_layouts.at(0) == TensorMemoryLayout::WIDTH_SHARDED);
                TT_FATAL(input_buffer_types.at(0) == output_buffer_type);
                TT_FATAL(input_memory_layouts.at(0) == output_layout);
                TT_FATAL(input_orientations.at(0) == ShardOrientation::ROW_MAJOR);
                uint32_t M = volume(input_shapes.at(0)) / input_shapes.at(0)[-1] / tt::constants::TILE_HEIGHT;
                uint32_t N = input_shapes.at(1)[-1] / tt::constants::TILE_WIDTH;
                uint32_t K = input_shapes.at(0)[-1] / tt::constants::TILE_WIDTH;
                uint32_t per_core_M = program_config.per_core_M;
                uint32_t per_core_N = program_config.per_core_N;
                auto shard_shape = input_shard_shapes.at(0);

                // No padding
                TT_FATAL(M == per_core_M);
                TT_FATAL(per_core_M == (shard_shape[0] / tt::constants::TILE_HEIGHT));
                TT_FATAL(K % program_config.in0_block_w == 0);
                TT_FATAL((shard_shape[1] / tt::constants::TILE_WIDTH) % program_config.in0_block_w == 0);

                // tensor in1
                TT_FATAL(input_memory_layouts.at(1) == TensorMemoryLayout::WIDTH_SHARDED);
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                if (input_sharded.at(0)) {
                    auto tensor_a_memory_layout = input_memory_layouts.at(0);
                    uint32_t M = volume(input_shapes.at(0)) / input_shapes.at(0)[-1] / tt::constants::TILE_HEIGHT;
                    uint32_t K = input_shapes.at(0)[-1] / tt::constants::TILE_WIDTH;
                    uint32_t N = input_shapes.at(1)[-1] / tt::constants::TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    auto shard_shape = input_shard_shapes.at(0);

                    TT_FATAL(
                        tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
                        tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);

                    if (tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                        if (program_config.transpose_mcast) {
                            TT_FATAL(input_orientations.at(0) == ShardOrientation::COL_MAJOR);
                        } else {
                            TT_FATAL(input_orientations.at(0) == ShardOrientation::ROW_MAJOR);
                        }
                        if (output_sharded) {
                            TT_FATAL(input_buffer_types.at(0) == output_buffer_type);
                            TT_FATAL(
                                input_memory_layouts.at(0) == output_layout);
                        }

                    } else if (tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                        TT_FATAL(!program_config.transpose_mcast);
                        TT_FATAL(K == program_config.in0_block_w);
                        TT_FATAL(program_config.in0_block_w == (shard_shape[1] / tt::constants::TILE_WIDTH));
                        TT_FATAL(
                            input_start_coordinates.at(0).x ==
                            input_end_coordinates.at(0).x);
                    }

                    TT_FATAL(per_core_M == (shard_shape[0] / tt::constants::TILE_HEIGHT));
                    TT_FATAL((shard_shape[1] / tt::constants::TILE_WIDTH) % program_config.in0_block_w == 0);
                }

                if (input_sharded.at(1)) {
                    TT_FATAL(!program_config.transpose_mcast);
                    auto tensor_b_memory_layout = input_memory_layouts.at(1);
                    TT_FATAL(tensor_b_memory_layout == TensorMemoryLayout::WIDTH_SHARDED);
                    if (input_buffer_types.at(1) != tt::tt_metal::BufferType::DRAM) {
                        TT_FATAL(
                            program_config.per_core_N == (input_shard_shapes.at(1)[1] / tt::constants::TILE_WIDTH));
                    }
                    TT_FATAL(
                        input_start_coordinates.at(1).y ==
                        input_end_coordinates.at(1).y);
                }

                if (output_sharded) {
                    TT_FATAL(output_layout == TensorMemoryLayout::BLOCK_SHARDED);
                    uint32_t M = volume(input_shapes.at(0)) / input_shapes.at(0)[-1] / tt::constants::TILE_HEIGHT;
                    uint32_t N = input_shapes.at(1)[-1] / tt::constants::TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                }
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                uint32_t M = input_shapes.at(0)[-2] / tt::constants::TILE_HEIGHT;
                uint32_t total_M = volume(input_shapes.at(0)) / input_shapes.at(0)[-1] / tt::constants::TILE_HEIGHT;
                uint32_t N = input_shapes.at(1)[-1] / tt::constants::TILE_WIDTH;
                uint32_t K = input_shapes.at(0)[-1];
                uint32_t per_core_M = program_config.per_core_M;
                uint32_t per_core_N = program_config.per_core_N;
                if (per_core_M > M) {
                    TT_FATAL(per_core_M % M == 0, "per_core_M must be a multiple of M if per_core_M > M!");
                    TT_FATAL(total_M % per_core_M == 0, "input a total height must be divisible by per_core_M!");
                } else {
                    TT_FATAL(M % per_core_M == 0, "per_core_M must divide M if per_core_M < M!");
                }
                TT_FATAL(N == per_core_N);
                if (input_sharded.at(0)) {
                    TT_FATAL(input_memory_layouts.at(0) != TensorMemoryLayout::WIDTH_SHARDED);
                    auto in0_shard_shape = input_shard_shapes.at(0);

                    TT_FATAL(K == in0_shard_shape[1]);
                    TT_FATAL(in0_shard_shape[1] == program_config.in0_block_w * tt::constants::TILE_WIDTH);
                    TT_FATAL(per_core_M * tt::constants::TILE_HEIGHT == in0_shard_shape[0]);

                    if (input_sharded.at(1)) {
                        TT_FATAL(
                            input_buffer_types.at(0) == input_buffer_types.at(1));
                        TT_FATAL(
                            input_memory_layouts.at(0) ==
                            input_memory_layouts.at(1));
                        TT_FATAL(input_grids.at(0) == input_grids.at(1));
                        TT_FATAL(
                            input_orientations.at(0) ==
                            input_orientations.at(1));
                    }
                    if (output_sharded) {
                        TT_FATAL(input_buffer_types.at(0) == output_buffer_type);
                        TT_FATAL(input_memory_layouts.at(0) == output_layout);
                    }
                }

                uint32_t batch_size_a = tt::tt_metal::get_batch_size(input_shapes.at(0));
                uint32_t batch_size_b = tt::tt_metal::get_batch_size(input_shapes.at(1));
                bool broadcast_batch = batch_size_a > 1 and batch_size_b == 1;
                TT_FATAL(!broadcast_batch);

                if (input_sharded.at(1)) {
                    TT_FATAL(per_core_M % M == 0, "per_core_M must be a multiple of M if input b is sharded!");
                    TT_FATAL(input_memory_layouts.at(1) != TensorMemoryLayout::WIDTH_SHARDED);
                    auto in1_shard_shape = input_shard_shapes.at(1);
                    TT_FATAL(in1_shard_shape[1] == input_shapes.at(1)[-1]);
                    TT_FATAL(per_core_N * tt::constants::TILE_HEIGHT == in1_shard_shape[1]);
                    TT_FATAL(in1_shard_shape[0] % K == 0);
                }
                if (output_sharded) {
                    TT_FATAL(output_layout != TensorMemoryLayout::WIDTH_SHARDED);
                    TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                }
            } else {
                TT_FATAL(input_memory_layouts.at(0) == TensorMemoryLayout::INTERLEAVED);
                TT_FATAL(input_memory_layouts.at(0) == TensorMemoryLayout::INTERLEAVED);
                TT_FATAL(output_layout == TensorMemoryLayout::INTERLEAVED);
            }
            if constexpr (
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig> ||
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig> ||
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                TT_FATAL(
                    (input_shapes.at(0)[-1] / tt::constants::TILE_WIDTH) % program_config.in0_block_w == 0,
                    "Kt must be divisible by in0_block_w");
                TT_FATAL(
                    program_config.per_core_M % program_config.out_subblock_h == 0,
                    "per_core_M must be divisible by out_subblock_h");
                TT_FATAL(
                    program_config.per_core_N % program_config.out_subblock_w == 0,
                    "per_core_N must be divisible by out_subblock_w");
            }
        },
        chosen_program_config);
}

}
