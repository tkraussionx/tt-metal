// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "softmax_validate_parameters.hpp"
#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/constants.hpp"

namespace ttnn::operations::normalization {

std::optional<tt::tt_metal::StorageType> SoftmaxValidateParameters::str_to_storage_type(const string& storage_type_str)
{
    if (storage_type_str == "OWNED") return StorageType::OWNED;
    if (storage_type_str == "DEVICE") return StorageType::DEVICE;
    if (storage_type_str == "BORROWED") return StorageType::BORROWED;
    if (storage_type_str == "MULTI_DEVICE") return StorageType::MULTI_DEVICE;
    if (storage_type_str == "MULTI_DEVICE_HOST") return StorageType::MULTI_DEVICE_HOST;
    return std::nullopt;
}

std::optional<tt::tt_metal::Layout> SoftmaxValidateParameters::str_to_layout(const string& layout_str)
{
    if (layout_str == "ROW_MAJOR") return Layout::ROW_MAJOR;
    if (layout_str == "TILE") return Layout::TILE;
    if (layout_str == "INVALID") return Layout::INVALID;
    return std::nullopt;
}

std::optional<tt::tt_metal::DataType> SoftmaxValidateParameters::str_to_data_type(const string& data_type_str)
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


tt::tt_metal::Shape SoftmaxValidateParameters::vector_to_shape(const std::vector<uint32_t>& shape_vector)
{
    return tt::tt_metal::Shape(shape_vector);
}


std::optional<tt::tt_metal::ShardOrientation> SoftmaxValidateParameters::str_to_shard_orientation(const string& shard_str)
{
    if (shard_str == "ROW_MAJOR") return ShardOrientation::ROW_MAJOR;
    if (shard_str == "COL_MAJOR") return ShardOrientation::COL_MAJOR;
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

bool SoftmaxValidateParameters::validate_parameters()
{
    try
    {
        validate();
    }
    catch(const std::runtime_error& e)
    {
        return false;
    }
    return true;
}

void SoftmaxValidateParameters::validate()
{
    TT_FATAL(input_shapes.size() <= 2, "Must have 1 or 2 input tensors");
    TT_FATAL(input_storage_types.at(0) == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_FATAL(has_buffers.at(0), "Operands to softmax need to be allocated in buffers on device!");
    TT_FATAL((input_layouts.at(0) == Layout::TILE), "Inputs to softmax must be tilized");
    TT_FATAL(input_data_types.at(0) == DataType::FLOAT32 || input_data_types.at(0) == DataType::BFLOAT16 || input_data_types.at(0) == DataType::BFLOAT8_B);
    if (input_shapes.size() == 2)
    {
        if (true)
        {
            TT_FATAL(input_storage_types.at(1) == StorageType::DEVICE, "Operands to softmax need to be on device!");
            TT_FATAL(input_devices.at(0) == input_devices.at(1));
            if (input_sharded.at(1))
            { // sharded mask
                TT_FATAL(input_layouts.at(1) == Layout::TILE);
                TT_FATAL(input_shapes.at(0) == input_shapes.at(1));
            }
            else
            {
                if (input_layouts.at(1) == Layout::ROW_MAJOR)
                {
                    tt::tt_metal::Shape expected_shape = {input_shapes.at(1)[0], 1, input_shapes.at(0)[-1] / tt::constants::TILE_WIDTH, tt::constants::TILE_WIDTH};
                    TT_FATAL(input_shapes.at(1) == expected_shape);
                }
                for (uint32_t i = 1; i < input_shapes.at(0).rank() - 2; i++)
                {
                    TT_FATAL(input_shapes.at(1)[i] == 1);
                }
            }

            std::visit(
                [&](const auto& program_config) {
                    using ProgramConfigType = std::decay_t<decltype(program_config)>;
                    if constexpr (
                        std::is_same_v<ProgramConfigType, SoftmaxDefaultProgramConfig>
                    ) {
                        TT_FATAL(input_shapes.at(0)[0] == input_shapes.at(1)[0]);
                        TT_FATAL(!this->is_scale_causal_mask_hw_dims_softmax);
                    } else if constexpr (
                        std::is_same_v<ProgramConfigType, SoftmaxShardedMultiCoreProgramConfig>
                    ) {
                        const auto shape = input_shapes.at(0);
                        uint32_t M = volume(shape) / shape[-1];
                        uint32_t K = shape[-1];

                        TT_FATAL(M % tt::constants::TILE_HEIGHT == 0, "M must be divisible by tile height.");
                        TT_FATAL(K % tt::constants::TILE_WIDTH == 0, "K must be divisible by tile width.");
                        TT_FATAL(program_config.block_w % program_config.subblock_w == 0, "block_w must be divisible by subblock_w.");
                        TT_FATAL(program_config.block_w * tt::constants::TILE_WIDTH == shape[3], "shard width must equal to input tensor shape[3]!");
                        TT_FATAL(this->inplace);
                        if (!this->is_scale_causal_mask_hw_dims_softmax) {
                            // grid
                            auto num_cores_c = program_config.compute_with_storage_grid_size.x;
                            auto num_cores_r = program_config.compute_with_storage_grid_size.y;
                            // check dims
                            TT_FATAL(M * K / ((program_config.block_w * program_config.block_h) * tt::constants::TILE_HW) == num_cores_r * num_cores_c, "number of shards must equal to number of cores. M = {}, K = {}, block_w = {}, block_h = {}, num_cores = {}", M, K, program_config.block_w, program_config.block_h, num_cores_r * num_cores_c);
                        } else {
                            TT_FATAL(this->is_causal_mask);
                            TT_FATAL(input_layouts.at(1) == Layout::TILE);
                            TT_FATAL(input_sharded.at(1) == false);
                            TT_FATAL(input_layouts.at(0) == Layout::TILE);
                            TT_FATAL(input_sharded.at(0));
                            TT_FATAL(input_orientations.at(0) == ShardOrientation::ROW_MAJOR);
                            TT_FATAL(this->scale.has_value());
                        }
                    }
                },
                this->program_config
            );
        }
        else {
            TT_FATAL(not this->scale.has_value());
        }
    } else {
        TT_FATAL(not this->scale.has_value());
        TT_FATAL(not this->is_scale_causal_mask_hw_dims_softmax);
    }
}

SoftmaxValidateParameters::SoftmaxValidateParameters(
    const std::vector<std::string>& input_storage_types_str,
    const std::vector<std::string>& input_layouts_str,
    const std::vector<std::string>& input_data_types_str,
    const std::vector<std::vector<uint32_t>>& input_shapes_vectors,
    const std::vector<bool>& input_sharded,
    const std::vector<std::string>& input_orientations_str,
    std::vector<std::string> input_devices,
    std::vector<bool> has_buffers,
    bool is_scale_causal_mask_hw_dims_softmax,
    bool inplace,
    bool is_causal_mask,
    const std::string& program_config,
    std::optional<int> block_w,
    std::optional<int> block_h,
    std::optional<int> subblock_w,
    std::optional<int> subblock_h,
    std::optional<int> grid_size_x,
    std::optional<int> grid_size_y,
    std::optional<int> scale
)
{
    for (const string& input_storage_type_str : input_storage_types_str)
    {
        input_storage_types.push_back(str_to_storage_type(input_storage_type_str).value());
    }
    for (const string& input_layout_str : input_layouts_str)
    {
        input_layouts.push_back(str_to_layout(input_layout_str).value());
    }
    for (const string& input_data_type_str : input_data_types_str)
    {
        input_data_types.push_back(str_to_data_type(input_data_type_str).value());
    }
    for (const vector<uint32_t>& input_shapes_vector : input_shapes_vectors)
    {
        input_shapes.push_back(vector_to_shape(input_shapes_vector));
    }
    this->input_sharded = std::vector<bool>(input_sharded.begin(), input_sharded.end());
    for (const string& input_orientation_str : input_orientations_str)
    {
        input_orientations.push_back(str_to_shard_orientation(input_orientation_str).value());
    }
    this->input_devices = std::vector<std::string>(input_devices.begin(), input_devices.end());
    this->has_buffers = std::vector<bool>(has_buffers.begin(), has_buffers.end());
    this->is_scale_causal_mask_hw_dims_softmax = is_scale_causal_mask_hw_dims_softmax;
    this->inplace = inplace;
    this->is_causal_mask = is_causal_mask;
    if (program_config == "SoftmaxDefaultProgramConfig")
    {
        this->program_config = SoftmaxDefaultProgramConfig();
    }
    if (program_config == "SoftmaxShardedMultiCoreProgramConfig")
    {
        this->program_config = SoftmaxShardedMultiCoreProgramConfig(CoreCoord(grid_size_x.value(), grid_size_y.value()), subblock_w.value(), block_h.value(), block_w.value());
    }
    this->scale = scale;
}

SoftmaxValidateParameters::SoftmaxValidateParameters(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const SoftmaxProgramConfig& program_config,
    bool is_scale_causal_mask_hw_dims_softmax,
    bool inplace,
    bool is_causal_mask,
    std::optional<int> scale
) : program_config(program_config)
{
    this->is_scale_causal_mask_hw_dims_softmax = is_scale_causal_mask_hw_dims_softmax;
    this->inplace = inplace;
    this->is_causal_mask = is_causal_mask;
    this->scale = scale;
    TT_FATAL(input_tensors.size() == 1, "Must have one obligatory input tensor.");
    TT_FATAL(optional_input_tensors.size() < 2, "Must have at most one input tensor.");
    const Tensor& input_tensor = input_tensors.at(0);
    input_storage_types.push_back(input_tensor.storage_type());
    input_layouts.push_back(input_tensor.get_layout());
    input_data_types.push_back(input_tensor.get_dtype());
    input_shapes.push_back(input_tensor.get_legacy_shape());
    input_sharded.push_back(input_tensor.is_sharded());
    input_orientations.push_back(input_tensor.shard_spec()->orientation);
    input_devices.push_back(std::to_string(input_tensor.device()->id()));
    has_buffers.push_back(input_tensor.buffer() != nullptr);
    for (const std::optional<const Tensor>& other_input_tensor : optional_input_tensors)
    {
        if (other_input_tensor.has_value())
        {
            const Tensor& other_input_tensor_value = other_input_tensor.value();
            input_storage_types.push_back(other_input_tensor_value.storage_type());
            input_layouts.push_back(other_input_tensor_value.get_layout());
            input_data_types.push_back(other_input_tensor_value.get_dtype());
            input_shapes.push_back(other_input_tensor_value.get_legacy_shape());
            input_sharded.push_back(other_input_tensor_value.is_sharded());
            input_orientations.push_back(other_input_tensor_value.shard_spec()->orientation);
            input_devices.push_back(std::to_string(other_input_tensor_value.device()->id()));
            has_buffers.push_back(other_input_tensor_value.buffer() != nullptr);
        }
    }
}

}
