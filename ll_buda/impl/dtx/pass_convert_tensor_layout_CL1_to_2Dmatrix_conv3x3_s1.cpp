#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"
#include "dtx_passes.hpp"


bool convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1(DataTransformations * dtx) {
    /*
    Notes:
    Should not support groups (I think). This transformation has to be applied before parallelizations.
    What if it's applied to a DRAM swizzled layout - need to think about that.
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






    // Setup
    vector<int> shape = producer->groups[0]->shape;
    int rank = shape.size();
    vector<int> channel_shape = vector_pad_on_left({shape[0]}, rank-1, 1);

    int x_size = shape[rank-1];
    int y_size = shape[rank-2];

    if (DEBUG) {
        cout << s(2) << "activation shape = " << v2s(shape) << endl;
        cout << s(2) << "channel shape = " << v2s(channel_shape) << endl;
        cout << s(2) << "x/y_size = " << x_size << ", " << y_size << endl;
    }

    // Kernel Window
    int kernel_size_x = 3;
    int kernel_size_y = 3;
    int activation_size_x = shape[rank-1];
    int activation_size_y = shape[rank-2];

    TensorPairGroup * consumer_group = consumer->groups[0];
    //consumer_group->shape = {, };


    // 2D Matrix destination:
    int matrix2d_x = 0;
    int matrix2d_y = 0;

    // Do the work

    // Sweep over the face of the activation
    for (int y=1; y<y_size-1; y++) {
        for (int x=1; x<x_size-1; x++) {
            if (DEBUG) cout << s(2) << "[y, x] = [" << y << ", " << x << "]" << endl;

            // Sweep over the kernel window size (hardcoded 3x3 right now)
            for (int kernel_y=-1; kernel_y<2; kernel_y++){
                for (int kernel_x=-1; kernel_x<2; kernel_x++){

                    int position_y = y + kernel_y;
                    int position_x = x + kernel_x;

                    if (DEBUG) {
                        if (DEBUG) cout << s(4) << "[pos_y, pos_x] = [" << position_y << ", " << position_x << "]" << endl;
                    }


                    //TensorPair * tp = new TensorPair(new Tensor({str}, {end}),
                    //                                            group_idx,
                    //                                            new Tensor({consumer_str}, {consumer_end}));
                    //consumer_group->tensor_pairs.push_back(tp);

                }
            }


        }
    }




    if (DEBUG) cout << "\n\nPASS: convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1 - END\n\n" << endl;
    return true;
}
