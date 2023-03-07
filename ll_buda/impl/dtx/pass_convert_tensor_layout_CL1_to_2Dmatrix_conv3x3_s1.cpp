#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"
#include "dtx_passes.hpp"


bool convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1(DataTransformations * dtx) {
    /*
    Notes:
    Should not support groups (I think). This transformation has to be applied before parallelizations.
    What if it's applied to a DRAM swizzled layout - need to think about that.

    Note: Input activatins need to already be padded. Padding not supported yet.
    // int pad = 1; // all around
    */

    bool DEBUG = true;
    bool pass = true;
    if (DEBUG) cout << "\n\nPASS: convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1 - START" << endl;

    // First add the 2 required transposes
    pass &= transpose_yz(dtx);
    pass &= transpose_xy(dtx);

    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    TransformationNode * consumer = new TransformationNode("convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1", producer->groups.size());
    dtx->transformations.push_back(consumer);

    // Calculate producer / consumer shapes
    vector<int> producer_shape = producer->groups[0]->shape;
    int rank = producer_shape.size();

    // Calculate the consumer shape
    int kernel_x = 3;
    int kernel_y = 3;
    int consumer_shape_x = kernel_x * kernel_y;
    int consumer_shape_y =  (producer_shape[Y(rank)] - 2)  *  (producer_shape[Z(rank)] - 2);
    vector<int> consumer_shape = {consumer_shape_y, consumer_shape_x};
    consumer->groups[0]->shape = {consumer_shape_y, consumer_shape_x};


    int consumer_y = 0;
    int consumer_x = 0;

    for (int producer_y=1; producer_y<producer_shape[Y(rank)]-1; producer_y++) {
    for (int producer_z=1; producer_z<producer_shape[Z(rank)]-1; producer_z++) {

        for (int kernel_x=-1; kernel_x<2; kernel_x++) {
        for (int kernel_y=-1; kernel_y<2; kernel_y++) {

            vector<int> sweep_zy = { producer_z + kernel_y, producer_y + kernel_x };

            // Producer str/end
            vector<int> producer_str = { producer_z + kernel_y, producer_y + kernel_x, 0};
            vector<int> producer_end = { producer_z + kernel_y, producer_y + kernel_x, producer_shape[X(rank)]};

            vector<int> consumer_str = {consumer_y, consumer_x};
            vector<int> consumer_end = {consumer_y, consumer_x + producer_shape[X(rank)]};

            TensorPair * tp = new TensorPair(new Tensor({producer_str}, {producer_end}),
                                            0,
                                            new Tensor({consumer_str}, {consumer_end}));
            consumer->groups[0]->tensor_pairs.push_back(tp);

            //if (DEBUG) cout << s(2) << "src = " << v2s(src_str) << "-" << v2s(src_end) << " ==> " << v2s(dst_str) << "-" << v2s(dst_end) << endl;

            consumer_x += producer_shape[X(rank)]; // length of channel

        }} // done with kernel_x/y sweep

        consumer_y++;
    }}




    if (DEBUG) cout << "\n\nPASS: convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1 - END\n\n" << endl;
    return true;
}
