// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/tensor/tensor_utils.hpp"
#include "matmul_validate_parameters.hpp"
#include <stdexcept>


namespace ttnn::operations::matmul {

bool MatmulValidateParameters::validateOutput(
    const std::vector<std::string>& output_layouts_str,
    const std::vector<std::string>& output_data_types_str,
    const std::vector<std::vector<uint32_t>>& output_shapes_vectors,
    const std::vector<bool>& output_sharded,
    const std::vector<std::string>& output_orientations_str,
    const std::vector<std::string>& output_memory_layouts,
    const std::vector<std::string>& output_buffer_types,
    OperationProgramConfig& program_config_parameters,
    const std::optional<std::vector<std::vector<uint32_t>>>& output_shard_shapes,
    const std::optional<std::vector<CoreRangeSet>>& output_core_range_sets)
{
    return true;
}

bool MatmulValidateParameters::validateInputAndOutput(
    const std::vector<std::string>& input_layouts_str,
    const std::vector<std::string>& input_data_types_str,
    const std::vector<std::vector<uint32_t>>& input_shapes_vectors,
    const std::vector<bool>& input_sharded,
    const std::vector<std::string>& input_orientations_str,
    const std::vector<std::string>& input_memory_layouts,
    const std::vector<std::string>& input_buffer_types,
    const std::vector<std::string>& output_layouts_str,
    const std::vector<std::string>& ouput_data_types_str,
    const std::vector<std::vector<uint32_t>>& output_shapes_vectors,
    const std::vector<bool>& output_sharded,
    const std::vector<std::string>& output_orientations_str,
    const std::vector<std::string>& output_memory_layouts,
    const std::vector<std::string>& output_buffer_types,
    OperationProgramConfig& program_config_parameters,
    const std::optional<std::vector<std::vector<uint32_t>>>& input_shard_shapes,
    const std::optional<std::vector<std::vector<uint32_t>>>& output_shard_shapes,
    const std::optional<std::vector<CoreRangeSet>>& input_core_range_sets,
    const std::optional<std::vector<CoreRangeSet>>& output_core_range_sets)
{
    try
    {
        TT_FATAL(input_shapes_vectors.size() == 2);
        const auto& a_shape = shape_by_index(0, input_shapes_vectors);
        const auto& b_shape = shape_by_index(1, input_shapes_vectors);

        TT_FATAL(
            (layout_by_index(0, input_layouts_str) == Layout::TILE && layout_by_index(1, input_layouts_str) == Layout::TILE),
            "Inputs to matmul must be tilized");
        TT_FATAL(
            a_shape[-1] == b_shape[-2],
            "The width of the first tensor must be equal to the height of the second tensor. Mismatch: width={} height={}",
            a_shape[-1],
            b_shape[-2]);
        TT_FATAL(a_shape.rank() == b_shape.rank() && "bmm (non-bcast matmul) expects input tensors of the same rank");
        for (auto i = 0; i < a_shape.rank() - 2; i++) {
            TT_FATAL(
                a_shape[i] == b_shape[i] &&
                "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN or equivalent");
        }
        TT_FATAL(is_floating_point(datatype_by_index(0, input_data_types_str)), "Unsupported data format");
        std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            // TODO: For 1D and 2D mcasts, we don't check if tensor is single core or single row/col
            // We can uplift these variants to skip mcasting to support single core (1D) or single row/col (2D)
            if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                if (program_config.mcast_in0) {
                    TT_FATAL(false, "Currently not supported");
                } else {
                    if (is_sharded_by_index(0, input_sharded)) {
                        TT_FATAL(program_config.fuse_batch);
                        TT_FATAL(memory_layout_by_index(0, input_memory_layouts) == TensorMemoryLayout::HEIGHT_SHARDED);
                        if (is_sharded_by_index(0, output_sharded)) {
                            TT_FATAL(buffer_type_by_index(0, input_buffer_types) == buffer_type_by_index(0, output_buffer_types));
                            TT_FATAL(
                                memory_layout_by_index(0, input_memory_layouts) == memory_layout_by_index(0, output_memory_layouts));
                        }
                        TT_FATAL(shard_orientation_by_index(0, input_orientations_str) == ShardOrientation::ROW_MAJOR);
                        tt::tt_metal::Shape input_a_shape = shape_by_index(0, input_shapes_vectors);
                        uint32_t M =
                            (program_config.fuse_batch ? volume(input_a_shape) / input_a_shape[-1]
                                                       : input_a_shape[-2]) /
                            tt::constants::TILE_HEIGHT;
                        uint32_t K = input_a_shape[-1] / tt::constants::TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        auto shard_shape = shard_shape_by_index(0, input_shard_shapes.value());

                        TT_FATAL(tt::div_up(M, per_core_M) == get_core_range_set_by_index(0, input_core_range_sets.value()).num_cores());
                        TT_FATAL(per_core_M == (shard_shape[0] / tt::constants::TILE_HEIGHT));
                        TT_FATAL(K % program_config.in0_block_w == 0);
                        TT_FATAL(K == (shard_shape[1] / tt::constants::TILE_WIDTH));
                    }
                    if (is_sharded_by_index(0, output_sharded)) {
                        TT_FATAL(memory_layout_by_index(0, output_memory_layouts) == TensorMemoryLayout::HEIGHT_SHARDED);
                        tt::tt_metal::Shape input_a_shape = shape_by_index(0, input_shapes_vectors);
                        uint32_t M =
                            (program_config.fuse_batch ? volume(input_a_shape) / input_a_shape[-1]
                                                       : input_a_shape[-2]) /
                            tt::constants::TILE_HEIGHT;
                        uint32_t N = shape_by_index(0, output_shapes_vectors)[-1] / tt::constants::TILE_WIDTH;
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        TT_FATAL(N == per_core_N);
                        TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                    }
                }
                TT_FATAL(memory_layout_by_index(1, input_memory_layouts) == TensorMemoryLayout::INTERLEAVED);
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                TT_FATAL(false, "Program config is not supported.");
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                if (is_sharded_by_index(0, input_sharded)) {
                    tt::tt_metal::TensorMemoryLayout tensor_a_memory_layout = memory_layout_by_index(0, input_memory_layouts);
                    tt::tt_metal::Shape input_a_shape = shape_by_index(0, input_shapes_vectors);
                    uint32_t M = volume(input_a_shape) / shape_by_index(0, input_shapes_vectors)[-1] / tt::constants::TILE_HEIGHT;
                    uint32_t K = shape_by_index(0, input_shapes_vectors)[-1] / tt::constants::TILE_WIDTH;
                    uint32_t N = shape_by_index(1, input_shapes_vectors)[-1] / tt::constants::TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    auto shard_shape = shard_shape_by_index(0, input_shard_shapes.value());

                    TT_FATAL(
                        tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
                        tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);

                    if (tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                        if (program_config.transpose_mcast) {
                            TT_FATAL(shard_orientation_by_index(0, input_orientations_str) == ShardOrientation::COL_MAJOR);
                        } else {
                            TT_FATAL(shard_orientation_by_index(0, input_orientations_str) == ShardOrientation::ROW_MAJOR);
                        }
                        if (is_sharded_by_index(0, output_sharded)) {
                            TT_FATAL(buffer_type_by_index(0, input_buffer_types) == buffer_type_by_index(0, output_buffer_types));
                            TT_FATAL(
                                memory_layout_by_index(0, input_memory_layouts) == memory_layout_by_index(0, output_memory_layouts));
                        }

                    } else if (tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                        TT_FATAL(!program_config.transpose_mcast);
                        TT_FATAL(K == program_config.in0_block_w);
                        TT_FATAL(program_config.in0_block_w == (shard_shape[1] / tt::constants::TILE_WIDTH));
                        TT_FATAL(
                            get_core_range_set_by_index(0, input_core_range_sets.value()).bounding_box().start_coord.x ==
                            get_core_range_set_by_index(0, input_core_range_sets.value()).bounding_box().end_coord.x);
                    }

                    TT_FATAL(per_core_M == (shard_shape[0] / tt::constants::TILE_HEIGHT));
                    TT_FATAL((shard_shape[1] / tt::constants::TILE_WIDTH) % program_config.in0_block_w == 0);
                }

                if (is_sharded_by_index(1, input_sharded)) {
                    TT_FATAL(!program_config.transpose_mcast);
                    auto tensor_b_memory_layout = memory_layout_by_index(1, input_memory_layouts);
                    TT_FATAL(tensor_b_memory_layout == TensorMemoryLayout::WIDTH_SHARDED);
                    if (buffer_type_by_index(1, input_buffer_types) != tt::tt_metal::BufferType::DRAM) {
                        auto shard_shape_b = shard_shape_by_index(1, input_shard_shapes.value());
                        TT_FATAL(
                            program_config.per_core_N == (shard_shape_b[1] / tt::constants::TILE_WIDTH));
                    }
                    TT_FATAL(
                        get_core_range_set_by_index(1, input_core_range_sets.value()).bounding_box().start_coord.y ==
                        get_core_range_set_by_index(1, input_core_range_sets.value()).bounding_box().end_coord.y);
                }

                if (is_sharded_by_index(0, input_sharded)) {
                    TT_FATAL(memory_layout_by_index(0, output_memory_layouts) == TensorMemoryLayout::BLOCK_SHARDED);
                    tt::tt_metal::Shape input_a_shape = shape_by_index(0, input_shapes_vectors);
                    uint32_t M = volume(input_a_shape) / shape_by_index(0, input_shapes_vectors)[-1] / tt::constants::TILE_HEIGHT;
                    uint32_t N = shape_by_index(1, input_shapes_vectors)[-1] / tt::constants::TILE_WIDTH;
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                }
            } else if constexpr (std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseProgramConfig>) {
                uint32_t M = shape_by_index(0, input_shapes_vectors)[-2] / tt::constants::TILE_HEIGHT;
                tt::tt_metal::Shape input_a_shape = shape_by_index(0, input_shapes_vectors);
                uint32_t total_M = volume(input_a_shape) / shape_by_index(0, input_shapes_vectors)[-1] / tt::constants::TILE_HEIGHT;
                uint32_t N = shape_by_index(1, input_shapes_vectors)[-1] / tt::constants::TILE_WIDTH;
                uint32_t K = shape_by_index(0, input_shapes_vectors)[-1];
                uint32_t per_core_M = program_config.per_core_M;
                uint32_t per_core_N = program_config.per_core_N;
                if (per_core_M > M) {
                    TT_FATAL(per_core_M % M == 0, "per_core_M must be a multiple of M if per_core_M > M!");
                    TT_FATAL(total_M % per_core_M == 0, "input a total height must be divisible by per_core_M!");
                } else {
                    TT_FATAL(M % per_core_M == 0, "per_core_M must divide M if per_core_M < M!");
                }
                TT_FATAL(N == per_core_N);
                if (is_sharded_by_index(0, input_sharded)) {
                    TT_FATAL(memory_layout_by_index(0, input_memory_layouts) != TensorMemoryLayout::WIDTH_SHARDED);

                    auto in0_shard_shape = shard_shape_by_index(0, input_shard_shapes.value());

                    TT_FATAL(K == in0_shard_shape[1]);
                    TT_FATAL(in0_shard_shape[1] == program_config.in0_block_w * tt::constants::TILE_WIDTH);
                    TT_FATAL(per_core_M * tt::constants::TILE_HEIGHT == in0_shard_shape[0]);

                    if (is_sharded_by_index(1, input_sharded)) {
                        TT_FATAL(buffer_type_by_index(0, input_buffer_types) == buffer_type_by_index(1, input_buffer_types));
                        TT_FATAL(
                            memory_layout_by_index(0, input_memory_layouts) ==
                            memory_layout_by_index(1, input_memory_layouts));
                        TT_FATAL(get_core_range_set_by_index(0, input_core_range_sets.value()) == get_core_range_set_by_index(1, input_core_range_sets.value()));
                        TT_FATAL(
                            shard_orientation_by_index(0, input_orientations_str) ==
                            shard_orientation_by_index(1, input_orientations_str));
                    }
                    if (is_sharded_by_index(0, output_sharded)) {
                        TT_FATAL(buffer_type_by_index(0, input_buffer_types) == buffer_type_by_index(0, output_buffer_types));
                        TT_FATAL(memory_layout_by_index(0, input_memory_layouts) == memory_layout_by_index(1, input_memory_layouts));
                    }
                }

                uint32_t batch_size_a = get_batch_size(shape_by_index(0, input_shapes_vectors));
                uint32_t batch_size_b = get_batch_size(shape_by_index(1, input_shapes_vectors));
                bool broadcast_batch = batch_size_a > 1 and batch_size_b == 1;
                TT_FATAL(!broadcast_batch);

                if (is_sharded_by_index(1, input_sharded)) {
                    TT_FATAL(per_core_M % M == 0, "per_core_M must be a multiple of M if input b is sharded!");
                    TT_FATAL(memory_layout_by_index(1, input_memory_layouts) != TensorMemoryLayout::WIDTH_SHARDED);
                    auto in1_shard_shape = shard_shape_by_index(0, input_shard_shapes.value());
                    TT_FATAL(in1_shard_shape[1] == shape_by_index(1, input_shapes_vectors)[-1]);
                    TT_FATAL(per_core_N * tt::constants::TILE_HEIGHT == in1_shard_shape[1]);
                    TT_FATAL(in1_shard_shape[0] % K == 0);
                }
                if (is_sharded_by_index(0, output_sharded)) {
                    TT_FATAL(memory_layout_by_index(1, output_memory_layouts) != TensorMemoryLayout::WIDTH_SHARDED);
                    TT_FATAL(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1);
                }
            } else {
                TT_FATAL(false, "Program config not supported.");
            }
            if constexpr (
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastProgramConfig> ||
                std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                TT_FATAL(
                    (shape_by_index(0, input_shapes_vectors)[-1] / tt::constants::TILE_WIDTH) % program_config.in0_block_w == 0,
                    "Kt must be divisible by in0_block_w");
                TT_FATAL(
                    program_config.per_core_M % program_config.out_subblock_h == 0,
                    "per_core_M must be divisible by out_subblock_h");
                TT_FATAL(
                    program_config.per_core_N % program_config.out_subblock_w == 0,
                    "per_core_N must be divisible by out_subblock_w");
            }
        },
        program_config_parameters);
    }
    catch(const std::runtime_error& e)
    {
        return false;
    }
    return true;
}

}
