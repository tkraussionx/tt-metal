#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

bool conv3d_to_matrix_transformation(DataTransformations * dtx, std::array<int, 6> conv_params) {
    bool DEBUG = true;
    int R = conv_params[0];
    int S = conv_params[1];
    int U = conv_params[2];
    int V = conv_params[3];
    int PAD_H = conv_params[4];
    int PAD_W = conv_params[5];

    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    TransformationNode * consumer = new TransformationNode("conv_transformation", producer->groups.size());  // TODO: generalize for groups>1
    dtx->transformations.push_back(consumer);
    assert(producer->groups.size() == 1); // TODO: generalize for group > 1

    TensorPairGroup * consumer_group = consumer->groups[0];
    TensorPairGroup * producer_group = producer->groups[0];
    inherit_group_attributes_from_producer(producer_group, consumer_group);
    vector<int> shape = producer_group->shape;
    int rank = producer_group->shape.size();
    assert(rank == 4); // nchw
    auto N = shape[0];
    auto C = shape[1];
    auto H = shape[2];
    auto W = shape[3];
    auto PADDED_H = H + 2*PAD_H;
    auto PADDED_W = W + 2*PAD_W;

    int consumer_rows = (((PADDED_H - R) / U) + 1) * (((PADDED_W - S) / V) + 1);
    int consumer_cols = R * S * C;
    consumer_group->shape = {consumer_rows, consumer_cols};
    int i = 0;
    if(DEBUG) {
        cout << endl;
        cout << s(4) << "Producer shape: " << v2s(shape) << endl;
        cout << s(4) << "Consumer shape: " << v2s(consumer_group->shape) << endl;
        cout << s(4) << "Tensor Pairs: " << consumer_rows << endl;
    }
    for(auto n = 0; n < N; n++) {
        for(auto h_p = 0; h_p < PADDED_H - (R - 1); h_p=h_p+U) {
            for(auto w_p = 0; w_p < PADDED_W - (S - 1); w_p=w_p+V) {
                auto start_h = h_p;
                auto start_w = w_p;
                auto end_h = start_h + R - 1;
                auto end_w = start_w + S - 1;
                if(end_h < PAD_H || end_w < PAD_W || start_h >= H + PAD_H || start_w >= W + PAD_W) {
                    continue;
                }
                start_h = start_h - PAD_H;
                start_w = start_w - PAD_W;
                end_h = end_h - PAD_H;
                end_w = end_w - PAD_W;
                if(start_h < 0) {
                    start_h = 0;
                }
                if(start_w < 0) {
                    start_w = 0;
                }
                if(end_h >= H) {
                    end_h = H-1;
                }
                if(end_w >= W) {
                    end_w = W-1;
                }
                vector<int> str = {n, 0, start_h, start_w};
                vector<int> end = {n, C-1, end_h, end_w};
                assert(i < consumer_rows);
                vector<int> consumer_str = {i, 0};
                vector<int> consumer_end = {i, consumer_cols-1};
                TensorPair * tp = new TensorPair(
                                        new Tensor({str}, {end}),
                                        0,
                                        new Tensor({consumer_str}, {consumer_end}));
                i++;
                consumer_group->tensor_pairs.push_back(tp);
                if(DEBUG) {
                    cout << s(6) << i << ".  " << tp->get_string() << endl;
                }
            }
        }
    }
    if(DEBUG) {
        cout << endl;
    }
    return true;
}
