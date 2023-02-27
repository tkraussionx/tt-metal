#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

bool conv3d_to_channels_last_transformation(DataTransformations * dtx) {
    bool DEBUG = true;

    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    TransformationNode * consumer = new TransformationNode("conv3d_to_channels_last_transformation", producer->groups.size());  // TODO: generalize for groups>1
    dtx->transformations.push_back(consumer);
    assert(producer->groups.size() == 1); // TODO: generalize for group > 1

    TensorPairGroup * consumer_group = consumer->groups[0];
    TensorPairGroup * producer_group = producer->groups[0];
    inherit_group_attributes_from_producer(producer_group, consumer_group);
    vector<int> shape = producer_group->shape;
    int rank = producer_group->shape.size();
    assert(rank == 4);
    auto N = shape[0];
    auto C = shape[1];
    auto H = shape[2];
    auto W = shape[3];
    consumer_group->shape = {N*C*H*W};
    int i = 0;
    int count = 0;
    if(DEBUG) {
        cout << endl;
        cout << s(4) << "Producer shape: " << v2s(shape) << endl;
        cout << s(4) << "Consumer shape: " << v2s(consumer_group->shape) << endl;
        cout << s(4) << "Tensor Pairs: " << N*H*W << endl;
    }
    for(auto n = 0; n < N; n++) {
        for(auto h = 0; h < H; h++) {
            for(auto w = 0; w < W; w++) {
                vector<int> str = {n, 0, h, w};
                vector<int> end = {n, C, h, w};
                vector<int> consumer_str = {i};
                vector<int> consumer_end = {i+C-1};
                TensorPair * tp = new TensorPair(
                                        new Tensor({str}, {end}),
                                        0,
                                        new Tensor({consumer_str}, {consumer_end}));
                i+=C;
                count++;
                consumer_group->tensor_pairs.push_back(tp);
                if(DEBUG) {
                    cout << s(6) << count << ".  " << tp->get_string() << endl;
                }
            }
        }
    }
    if(DEBUG) {
        cout << endl;
    }
    return true;
}
