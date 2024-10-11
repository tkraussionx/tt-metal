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
    struct MassagedOperationParams {
        using OwnedArgsType = std::tuple<std::decay_t<OpInputTypes>...>;
        using PredicateFunc = std::function<bool(OpInputTypes...)>;
        using PreTransformFunc = std::function<OwnedArgsType(OpInputTypes...)>;
        using PostTransformFuncWithArgs = std::function<OpOutputType(const OpOutputType&, OpInputTypes...)>;
        using PostTransformFunctWithoutArgs = std::function<OpOutputType(const OpOutputType&)>;
        using PostTransformFunc = std::variant<PostTransformFuncWithArgs, PostTransformFunctWithoutArgs>;
        using PrecompFunc = std::function<OwnedArgsType(OpInputTypes...)>;
        using OpType = std::function<OpOutputType(OpInputTypes...)>;

        PredicateFunc predicate;      // Function to determine if formatting should be applied
        PreTransformFunc pre_transform;  // Function to pre-process input arguments
        PostTransformFunc post_transform;  // Function to post-process the operation output
        PrecompFunc precomp;           // Function for fixing up input arguments after preprocessing has been applied
        OpType operation;              // The main operation to be performed
    };

    template<typename OpOutputType, typename... OpInputTypes>
    class MassagedOperation {
    public:
        using OwnedArgsType = std::tuple<std::decay_t<OpInputTypes>...>;
        using PredicateFunc = std::function<bool(OpInputTypes...)>;
        using PreTransformFunc = std::function<OwnedArgsType(OpInputTypes...)>;
        // post transform takes the output and optionally the args; it may use
        // the args in order to know if it needs to post process the output.
        using PostTransformFuncWithArgs = std::function<OpOutputType(const OpOutputType&, OpInputTypes...)>;
        using PostTransformFunctWithoutArgs = std::function<OpOutputType(const OpOutputType&)>;
        using PostTransformFunc = std::variant<PostTransformFuncWithArgs, PostTransformFunctWithoutArgs>;
        using PrecompFunc = std::function<OwnedArgsType(OpInputTypes...)>;
        using OpType = std::function<OpOutputType(OpInputTypes...)>;

        MassagedOperation(MassagedOperationParams<OpOutputType, OpInputTypes...> params)
            : predicate_(params.predicate)
            , pre_transform_(params.pre_transform)
            , post_transform_(params.post_transform)
            , precomp_(params.precomp)
            , operation_(params.operation) {}

        inline bool should_format(OpInputTypes... args) const {
            return predicate_(args...);
        }

        inline OwnedArgsType pre_format(OpInputTypes... args) const {
            return pre_transform_(args...);
        }

        inline OpOutputType post_format(OpOutputType output, OpInputTypes... args) const {
            return std::visit([&output, &args...](auto&& f) -> OpOutputType {
                if constexpr (std::is_same_v<std::decay_t<decltype(f)>, PostTransformFuncWithArgs>) {
                    return f(output, args...);
                } else {
                    return f(output);
                }
            }, post_transform_);
        }

        inline OwnedArgsType precomp(OpInputTypes... args) const {
            return precomp_(args...);
        }

        inline OpOutputType operator()(OpInputTypes... args) const {
            if (should_format(args...)) {
                auto formatted_input = pre_format(args...);
                auto precomped = std::apply(precomp_,formatted_input);
                return post_format(std::apply(operation_, precomped), args...);
            }
            return operation_(args...);
        }

        MassagedOperation sequence(const MassagedOperation& other) {
            auto merged_predicate = [p1 = this->predicate_,
                                     p2 = other.predicate_](OpInputTypes... args) -> bool {
                return p1(args...) or p2(args...);
            };
            auto merged_pre_transform = [t1 = this->pre_transform_,
                                         t2 = other.pre_transform_,
                                         p1 = this->predicate_,
                                         p2 = other.predicate_](OpInputTypes... args) -> OwnedArgsType {
                // this is ugly, but I couldn't find a way around it since
                // the OpInputTypes may contain consts/refs.
                if (p1(args...) && std::apply(p2, t1(args...))) {
                    return std::apply(t2, t1(args...));
                } else if (p1(args...)) {
                    return t1(args...);
                } else if (p2(args...)) {
                    return t2(args...);
                } else {
                    return std::make_tuple(args...);
                }
            };

            auto merged_post_transform = [t1 = this->post_transform_,
                                          t2 = other.post_transform_,
                                          p1 = this->predicate_,
                                          p2 = other.predicate_,
                                          this_pretransform = this->pre_transform_](OpOutputType output, OpInputTypes... args) -> OpOutputType {

                OpOutputType transformed_output = output;
                if (p1(args...)) {
                    // for post-transformation, we need to go in reverse order
                    auto pretransformed_args = this_pretransform(args...);
                    if (std::apply(p2, pretransformed_args)) {
                        std::cout << "both pretransforms needed" << std::endl;
                        transformed_output = std::visit([&transformed_output, &pretransformed_args](auto&& f) -> OpOutputType {
                            if constexpr (std::is_same_v<std::decay_t<decltype(f)>, PostTransformFuncWithArgs>) {
                                return std::apply(f, std::tuple_cat(std::make_tuple(transformed_output), pretransformed_args));
                            } else {
                                return f(transformed_output);
                            }
                        }, t2);
                    }
                    std::cout << "only need first pretransform" << std::endl;
                    transformed_output = std::visit([&transformed_output, &args...](auto&& f) -> OpOutputType {
                        if constexpr (std::is_same_v<std::decay_t<decltype(f)>, PostTransformFuncWithArgs>) {
                            return f(transformed_output, args...);
                        } else {
                            return f(transformed_output);
                        }
                    }, t1);
                }
                else if (p2(args...)) {
                    std::cout << "only need second pretransform" << std::endl;
                    transformed_output = std::visit([&transformed_output, &args...](auto&& f) -> OpOutputType {
                        if constexpr (std::is_same_v<std::decay_t<decltype(f)>, PostTransformFuncWithArgs>) {
                            return f(transformed_output, args...);
                        } else {
                            return f(transformed_output);
                        }
                    }, t2);
                }
                else {
                    std::cout << "no pretransforms needed" << std::endl;
                }
                return transformed_output;
            };

            auto merged_precomp = [pc1 = this->precomp_,
                                   pc2 = other.precomp_,
                                   p1 = this->predicate_,
                                   p2 = other.predicate_,
                                   t1 = this->pre_transform_](OpInputTypes... args) -> OwnedArgsType {
                if (p1(args...) && std::apply(p2, t1(args...))) {
                    return std::apply(pc2, pc1(args...));
                }
                else if (p1(args...)) {
                    return pc1(args...);
                }
                else if (p2(args...)) {
                    return pc2(args...);
                }
                return std::make_tuple(args...);
            };

            return MassagedOperation(
                MassagedOperationParams<OpOutputType, OpInputTypes...>{
                    .predicate = merged_predicate,
                    .pre_transform = merged_pre_transform,
                    .post_transform = merged_post_transform,
                    .precomp = merged_precomp,
                    .operation = this->operation_
                }
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
    using OwnedConcatArgs = std::tuple<std::vector<ttnn::Tensor>, int>;
    MassagedOperation<ttnn::Tensor,
                      const std::vector<ttnn::Tensor>&,
                      int> build_unsqueeze_concat(int input_rank, ttnn::MemoryConfig& output_memory_config) {
        return MassagedOperation<ttnn::Tensor,
                                 const std::vector<ttnn::Tensor>&,
                                 int>(
            MassagedOperationParams<ttnn::Tensor,
                                    const std::vector<ttnn::Tensor>&,
                                    int> {
                .predicate = [input_rank](const std::vector<ttnn::Tensor>& tensors, int dim) -> bool {
                    return input_rank < 4;
                },
                .pre_transform = [input_rank](const std::vector<ttnn::Tensor>& tensors, int dim) -> OwnedConcatArgs {
                    std::cout << "performing unsqueeze concat pre-transform" << std::endl;
                    std::vector<ttnn::Tensor> itensor;
                    itensor.reserve(tensors.size());
                    std::transform(
                        tensors.begin(),
                        tensors.end(),
                        std::back_inserter(itensor),
                        [](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                            return ttnn::unsqueeze_to_4D(input_tensor);
                        }
                    );
                    return std::make_tuple(itensor, dim + 4 - input_rank);
                },
                .post_transform = [input_rank](const ttnn::Tensor& output, const std::vector<ttnn::Tensor>& tensors, int dim) -> ttnn::Tensor {
                    std::cout << "performing unsqueeze concat post-transform" << std::endl;
                    ttnn::Tensor res = output;
                    while (res.get_shape().rank() > input_rank) {
                        const auto shape = res.get_shape();
                        const auto full_shape = res.get_shape().with_tile_padding();
                        std::vector<uint32_t> shape_vec{};
                        std::vector<uint32_t> full_shape_vec{};
                        for (int i = 1; i < shape.rank(); i++) {
                            shape_vec.push_back(shape[i]);
                            full_shape_vec.push_back(full_shape[i]);
                        }
                        res = ttnn::reshape(res, ttnn::Shape(shape_vec, full_shape_vec));
                    }
                    return res;
                },
                .precomp = [input_rank](const std::vector<ttnn::Tensor>& tensors, int dim) -> OwnedConcatArgs {
                    auto owned_tensors = std::vector<ttnn::Tensor>(tensors);
                    return std::make_tuple(owned_tensors, dim + 4 - input_rank);
                },
                .operation = [output_memory_config](const std::vector<ttnn::Tensor>& tensors, int dim) -> ttnn::Tensor {
                    std::vector<ttnn::Tensor> itensors(tensors);
                    return concat_impl(itensors, dim, output_memory_config);
                }
            }
        );
    }

    MassagedOperation<ttnn::Tensor,
                      const std::vector<ttnn::Tensor>&,
                      int> build_untilize_rm_retilize_concat(uint8_t queue_id, MemoryConfig &output_memory_config) {
        return MassagedOperation<ttnn::Tensor,
                                 const std::vector<ttnn::Tensor>&,
                                 int>(
            MassagedOperationParams<ttnn::Tensor, const std::vector<ttnn::Tensor>&, int> {
                .predicate = [](const std::vector<ttnn::Tensor>& tensors, int dim) -> bool {
                    // untilize_rm_retilize if the concat dim is padded for tilized tensors
                    auto first = tensors.front();
                    return first.get_layout() == ttnn::TILE_LAYOUT and first.get_logical_shape()[dim] != first.get_padded_shape()[dim];
                },
                .pre_transform = [](const std::vector<ttnn::Tensor>& tensors, int dim) -> OwnedConcatArgs {
                    std::cout << "performing untilize_rm_retilize concat pre-transform" << std::endl;
                    std::vector<ttnn::Tensor> itensors;
                    itensors.reserve(tensors.size());
                    std::transform(
                        tensors.begin(),
                        tensors.end(),
                        std::back_inserter(itensors),
                        [](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                            TT_FATAL(input_tensor.get_layout() == ttnn::TILE_LAYOUT, "ttnn.concat: expected all input tensors to be in tile layout");
                            auto untilized_tensor = ttnn::untilize(input_tensor);
                            // untilized, so now we have a padded rm tensor
                            untilized_tensor.set_shape(ttnn::Shape {input_tensor.get_logical_shape().as_vector(),
                                                                    untilized_tensor.get_padded_shape().as_vector()});
                            return untilized_tensor;
                        }
                    );
                    return std::make_tuple(itensors, dim);
                },
                .post_transform = [queue_id](const ttnn::Tensor& output, const std::vector<ttnn::Tensor>& tensors, int dim) -> ttnn::Tensor {
                    std::cout << "performing untilize_rm_retilize concat post-transform" << std::endl;
                    // now we have a rm tensor, so we need ensure it's padded to tile size and re-tilize it
                    return ttnn::tilize(pad_to_tile_vol(queue_id,
                                                        output,
                                                        0.0f,
                                                        true,
                                                        output.memory_config()));
                },
                .precomp = [](const std::vector<ttnn::Tensor>& tensors, int dim) -> OwnedConcatArgs {
                    auto owned_tensors = std::vector<ttnn::Tensor>(tensors);
                    return std::make_tuple(owned_tensors, dim);
                },
                .operation = [output_memory_config](const std::vector<ttnn::Tensor>& tensors, int dim) -> ttnn::Tensor {
                    std::vector<ttnn::Tensor> itensors(tensors);
                    return concat_impl(itensors, dim, output_memory_config);
                }
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
        auto massaged_concat = unsqueeze_concat.sequence(untilize_rm_retilize_concat);
        return massaged_concat(input_tensors, dim);
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
