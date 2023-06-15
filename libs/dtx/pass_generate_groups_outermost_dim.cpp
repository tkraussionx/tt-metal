#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"


bool generate_groups_outermost_dim(DataTransformations * dtx) {
    bool DEBUG = false;

    if (DEBUG) cout << "\n\nPASS: Generate groups on outermost dimension" << endl;

    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    assert(producer->groups.size() == 1);
    TensorPairGroup * producer_group = producer->groups[0];

    auto producer_shape = producer_group->shape;
    uint rank = producer_shape.size();
    assert(rank == 3); // TODO: generalize for rank != 3
    uint32_t num_consumer_groups = producer_shape[0];
    if (DEBUG) std::cout << "Number of consumer groups - " << num_consumer_groups << std::endl;
    TransformationNode * consumer = new TransformationNode("generate_groups", num_consumer_groups);
    dtx->transformations.push_back(consumer);

    for (int producer_tp_idx=0; producer_tp_idx<producer_group->tensor_pairs.size(); producer_tp_idx++) {
        TensorPair * producer_tp = producer_group->tensor_pairs[producer_tp_idx];
        assert(producer_tp->dst_tensor->str.size() == 3);
        int g = producer_tp->dst_tensor->str[0];
        assert(g < consumer->groups.size());
        TensorPairGroup * consumer_group = consumer->groups[g];
        consumer_group->shape = {1, producer_shape[1], producer_shape[2]};
        vector<int> producer_str = producer_tp->dst_tensor->str;
        vector<int> producer_end = producer_tp->dst_tensor->end;
        vector<int> consumer_str = {0, producer_tp->dst_tensor->str[1], producer_tp->dst_tensor->str[2]};
        vector<int> consumer_end = {0, producer_tp->dst_tensor->end[1], producer_tp->dst_tensor->end[2]};

        TensorPair * tp = new TensorPair(new DTXTensor({producer_str}, {producer_end}),
                                        0,
                                        new DTXTensor({consumer_str}, {consumer_end}));
        if (DEBUG) cout << s(6) << g << ".  " << tp->get_string() << endl;

        consumer_group->tensor_pairs.push_back(tp);
    }

    return true;
}
