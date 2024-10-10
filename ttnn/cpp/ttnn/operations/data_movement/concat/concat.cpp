// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/constants.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "tt_metal/common/math.hpp"

#include "ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"

#include <ranges>


namespace ttnn {
namespace operations {
namespace data_movement {
    ttnn::Tensor pad_to_tile_vol(uint8_t queue_id,
                                 const ttnn::Tensor& tensor,
                                 const float value,
                                 const bool use_multicore,
                                 const std::optional<MemoryConfig>& memory_config) {
        auto logical_shape = tensor.get_logical_shape();
        auto padded_shape = tensor.get_padded_shape();
        auto rank = tensor.get_shape().rank();
        if (padded_shape.volume() % tt::constants::TILE_HW != 0) {
            TT_ASSERT(rank >= 2, "rank of tensor to pad to tile must be at least 2.");

            auto padded_height = tt::round_up(padded_shape[-2], tt::constants::TILE_HEIGHT);
            auto padded_width = tt::round_up(padded_shape[-1], tt::constants::TILE_WIDTH);
            uint32_t num_non_hw_dims = rank - 2u;
            auto padding_vec = std::vector<std::pair<uint32_t, uint32_t>>(num_non_hw_dims, {0,0});
            padding_vec.reserve(rank);
            padding_vec.emplace_back(0, padded_height - padded_shape[-2]);
            padding_vec.emplace_back(0, padded_width - padded_shape[-1]);

            constexpr bool pad_use_multicore = true;
            auto padded_output = ttnn::pad(queue_id,
                                            tensor,
                                            padding_vec,
                                            value,
                                            use_multicore,
                                            memory_config);
            return padded_output;
        }
        return tensor;
    }

    template<typename OpOutputType, typename... OpInputTypes>
    class MassagedOperation {
    public:
        using ArgsType = std::tuple<OpInputTypes...>;
        using PredicateFunc = std::function<bool(OpInputTypes...)>;
        using PreTransformFunc = std::function<ArgsType(OpInputTypes...)>;
        using PostTransformFunc = std::function<OpOutputType(const OpOutputType&)>;
        using PrecompFunc = std::function<ArgsType(OpInputTypes...)>;
        using OpType = std::function<OpOutputType(OpInputTypes...)>;

        MassagedOperation(PredicateFunc predicate, PreTransformFunc pre_transform, PostTransformFunc post_transform, PrecompFunc precomp, OpType operation)
            : predicate_(predicate)
            , pre_transform_(pre_transform)
            , post_transform_(post_transform)
            , precomp_(precomp)
            , operation_(operation) {}

        bool should_format(OpInputTypes... args) const {
            return predicate_(args...);
        }

        ArgsType pre_format(OpInputTypes... args) const {
            return pre_transform_(args...);
        }

        OpOutputType post_format(OpOutputType output) const {
            return post_transform_(output);
        }

        ArgsType precomp(OpInputTypes... args) const {
            return precomp_(args...);
        }

        OpOutputType operator()(OpInputTypes... args) const {
            if (should_format(args...)) {
                auto formatted_input = pre_format(args...);
                auto precomped = std::apply(precomp_,formatted_input);
                return post_format(std::apply(operation_, precomped));
            }
            return operation_(args...);
        }

        MassagedOperation merge(const MassagedOperation& other) {
            auto merged_predicate = [p1 = this->predicate_,
                                     p2 = other.predicate_](OpInputTypes... args) -> bool {
                return p1(args...) or p2(args...);
            };
            auto merged_pre_transform = [t1 = this->pre_transform_, t2 = other.pre_transform_](OpInputTypes... args) -> ArgsType {
                return std::apply(t2, (t1(args...)));
            };

            auto merged_post_transform = [p1 = this->post_transform_, p2 = other.post_transform_](OpOutputType output) -> OpOutputType {
                return p2(p1(output));
            };

            auto merged_precomp = [pc1 = this->precomp_, pc2 = other.precomp_](OpInputTypes... args) -> ArgsType {
                return std::apply(pc2, (pc1(args...)));
            };

            return MassagedOperation(
                merged_predicate,
                merged_pre_transform,
                merged_post_transform,
                merged_precomp,
                this->operation_
            );
        }

        // getters for all private members
        PredicateFunc get_predicate() const { return predicate_; }
        PreTransformFunc get_pre_transform() const { return pre_transform_; }
        PostTransformFunc get_post_transform() const { return post_transform_; }
        PrecompFunc get_precomp() const { return precomp_; }
        OpType get_operation() const { return operation_; }

    private:
        PredicateFunc predicate_;
        PreTransformFunc pre_transform_;
        PostTransformFunc post_transform_;
        PrecompFunc precomp_;
        OpType operation_;
    };

    using ConcatArgs = std::tuple<const std::vector<ttnn::Tensor>&, int>;
    MassagedOperation<ttnn::Tensor,
                      const std::vector<ttnn::Tensor>&,
                      int> build_unsqueeze_concat(int input_rank, ttnn::MemoryConfig& output_memory_config) {
        return MassagedOperation<ttnn::Tensor,
                                 const std::vector<ttnn::Tensor>&,
                                 int>(
            [input_rank](const std::vector<ttnn::Tensor>& tensors, int dim) -> bool {
                return input_rank < 4;
            },
            [](const std::vector<ttnn::Tensor>& tensors, int dim) -> ConcatArgs {
                std::vector<ttnn::Tensor> itensor;
                std::transform(
                    tensors.begin(),
                    tensors.end(),
                    std::back_inserter(itensor),
                    [](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                        return ttnn::unsqueeze_to_4D(input_tensor);
                    }
                );
                return std::make_tuple(itensor, dim);
            },
            [input_rank](const ttnn::Tensor& output) -> ttnn::Tensor {
                ttnn::Tensor res = output;
                while (output.get_shape().rank() > input_rank) {
                    const auto shape = output.get_shape();
                    const auto full_shape = output.get_shape().with_tile_padding();
                    std::vector<uint32_t> shape_vec{};
                    std::vector<uint32_t> full_shape_vec{};
                    for (int i = 1; i < shape.rank(); i++) {
                        shape_vec.push_back(shape[i]);
                        full_shape_vec.push_back(full_shape[i]);
                    }
                    res = ttnn::reshape(output, ttnn::Shape(shape_vec, full_shape_vec));
                }
                return res;
            },
            [input_rank](const std::vector<ttnn::Tensor>& tensors, int dim) -> ConcatArgs {
                return std::make_tuple(tensors, dim + 4 - input_rank);
            },
            [output_memory_config](const std::vector<ttnn::Tensor>& tensors, int dim) -> ttnn::Tensor {
                std::vector<ttnn::Tensor> itensors(tensors);
                return concat_impl(itensors, dim, output_memory_config);
            }
        );
    }

    MassagedOperation<ttnn::Tensor,
                      const std::vector<ttnn::Tensor>&,
                      int> build_untilize_rm_retilize_concat(uint8_t queue_id, MemoryConfig &output_memory_config) {
        return MassagedOperation<ttnn::Tensor,
                                 const std::vector<ttnn::Tensor>&,
                                 int>(
            [](const std::vector<ttnn::Tensor>& tensors, int dim) -> bool {
                // untilize_rm_retilize if the concat dim is padded for tilized tensors
                auto first = tensors.front();
                return first.get_logical_shape()[dim] != first.get_padded_shape()[dim];
            },
            [](const std::vector<ttnn::Tensor>& tensors, int dim) -> ConcatArgs {
                std::vector<ttnn::Tensor> itensors;
                std::transform(
                    tensors.begin(),
                    tensors.end(),
                    std::back_inserter(itensors),
                    [](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                        auto untilized_tensor = ttnn::untilize(input_tensor);
                        // untilized, so now we have a padded rm tensor
                        untilized_tensor.set_shape(ttnn::Shape {input_tensor.get_logical_shape().as_vector(),
                                                                untilized_tensor.get_padded_shape().as_vector()});
                        return untilized_tensor;
                    }
                );
                return std::make_tuple(itensors, dim);
            },
            // post-processor
            [queue_id](const ttnn::Tensor& output) -> ttnn::Tensor {
                // now we have a rm tensor, so we need ensure it's padded to tile size and re-tilize it
                return ttnn::tilize(pad_to_tile_vol(queue_id,
                                                    output,
                                                    0.0f,
                                                    true,
                                                    output.memory_config()));
            },
            [](const std::vector<ttnn::Tensor>& tensors, int dim) -> ConcatArgs {
                return std::make_tuple(tensors, dim);
            },
            [output_memory_config](const std::vector<ttnn::Tensor>& tensors, int dim) -> ttnn::Tensor {
                std::vector<ttnn::Tensor> itensors(tensors);
                return concat_impl(itensors, dim, output_memory_config);
            }
        );
    }

    // Wrapper for TTDNN
    ttnn::Tensor ConcatOperation::invoke(
        uint8_t queue_id,
        const std::vector<ttnn::Tensor>& input_tensors,
        int dim,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<ttnn::Tensor> optional_output_tensor) {
        TT_FATAL(input_tensors.size() > 0, "ttnn.concat: expected a non-empty list of Tensors!");
        TT_FATAL(!optional_output_tensor.has_value(), "optional output tensor currently unsupported!");
        const auto mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG); // should match input tensor memory config when unpopulated but causes CI errors for now

        if (input_tensors.size() == 1) {
            return ttnn::to_memory_config(input_tensors.at(0), mem_config, std::nullopt);
        }

        // TODO: Issue #8426: Add validation for ttnn.concat for sharded inputs
        // const bool all_tensors_are_tile_layout_without_padding = std::all_of(input_tensors.begin(), input_tensors.end(),
        // [dim](const ttnn::Tensor& input_tensor){
        //    return input_tensor.get_layout() == ttnn::TILE_LAYOUT and not has_tile_padding(input_tensor, dim);
        //});
        // TT_FATAL(all_tensors_are_tile_layout_without_padding, "Not Implemented");

        const ttnn::Tensor& first_tensor = input_tensors.front();
        const int rank = first_tensor.get_shape().rank();

        dim = first_tensor.get_legacy_shape().get_normalized_index(dim);

        TT_FATAL(
            dim >= 0 and dim < rank,
            "ttnn: Dimension out of range: dim {} cannot be used for tensors of rank {}",
            dim,
            rank);

        const bool shapes_match =
            std::all_of(input_tensors.begin(), input_tensors.end(), [first_tensor, dim](const ttnn::Tensor& t) {
                const auto& ft_shape = first_tensor.get_shape();
                const auto& t_shape = t.get_shape();

                const bool ranks_match = ft_shape.rank() == t_shape.rank();
                bool non_concat_dims_match = true;
                for (int i = 0; i < ft_shape.rank(); i++) {
                    non_concat_dims_match &= dim == i or t_shape[i] == ft_shape[i];
                }
                // bool non_concat_padded_dims_match = true;
                // for(int i = 0; i < ft_shape.rank(); i++) {
                //     non_concat_padded_dims_match &= dim == i or t_shape.with_tile_padding()[i] ==
                //     ft_shape.with_tile_padding()[i];
                // }
                return ranks_match and non_concat_dims_match;  // and non_concat_padded_dims_match;
            });

        TT_FATAL(
            shapes_match,
            "All dimensions must be the same size except for the dimension along which the contenation is taking place.");

        auto output_memory_config = memory_config.value_or(first_tensor.memory_config());
        auto unsqueeze_concat = build_unsqueeze_concat(rank, output_memory_config);
        auto untilize_rm_retilize_concat = build_untilize_rm_retilize_concat(queue_id, output_memory_config);
        auto degankifier = unsqueeze_concat.merge(untilize_rm_retilize_concat);
        return degankifier(input_tensors, dim);
    }

    ttnn::Tensor ConcatOperation::invoke (
        const std::vector<ttnn::Tensor>& input_tensors,
        int dim,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<ttnn::Tensor> optional_output_tensor) {
        return invoke(DefaultQueueId, input_tensors, dim, memory_config, optional_output_tensor);
    }
};

}
}
