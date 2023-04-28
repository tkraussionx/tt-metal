import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from libs import tt_lib as ttl
from libs.tt_lib.utils import tilize_to_list, tilize, untilize, channels_last, _nearest_32, convert_weights_2d_matrix
from python_api_testing.models.utility_functions import print_diff_argmax, is_close, comp_pcc
import torch

# @pytest.mark.parametrize(
#     "K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w",
#     (
#         #(256, 128, 14, 14, 1, 1, 1, 1, 0, 0),
#         #(32, 1024, 5, 5, 1, 1, 1, 1, 0, 0),
#         # (64, 512, 5, 5, 1, 1, 1, 1, 0, 0),
#         #resnet 18 convs

#         (256, 128, 28, 28, 3, 3, 2, 2, 1, 1),

#         #(128, 128, 6, 6, 3, 3, 1, 1, 1, 1),

#         #lenet conv
#         #(16, 6, 5, 5, 1, 1, 1, 1, 0, 0),
#         #(32, 64, 5, 5, 5, 5, 1, 1, 1, 1,),
#         #(32, 256, 14, 14, 3, 3, 1, 1, 1, 1,),
#         #(32, 1024, 8, 4, 1, 1),
#         #(32, 32, 18, 18, 3, 3),
#         #(32, 32, 10, 10, 1, 1, 1, 1, 0, 0),
#         # (32, 32, 10, 10, 3, 3, 1, 1, 0, 0),
#         #(64, 64, 32, 16, 1, 1),
#         #(64, 64, 10, 10, 1, 1),
#         #(32, 64, 10, 10, 3, 3, 1, 1, 0, 0),
#     ),
# )
result = []
A_i = []
B_i = []
def test_run_conv_as_large_matmul(K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w, untilize_out):
    global result
    global A_i
    global B_i
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    print("Device Initialized")

    #torch.set_printoptions(threshold=10000)

    a_activation_shape = [1,C,H,W]
    b_weights_shape = [K,C,R,S]
    # check if params are valid
    assert (H - R + 2 * pad_h) >= 1 and (W - S + 2 * pad_w) >= 1
    OH = ((int) ((H - R + 2 * pad_h) / stride_h)) + 1
    OW = ((int) ((W - S + 2 * pad_w) / stride_w)) + 1
    mm_output_shape = [1,1,_nearest_32(OH*OW),_nearest_32(K)]
    torch.manual_seed(0)
    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    if (A_i == []):
        A_i = torch.flatten(A_pyt).tolist()
    elif(A_i != torch.flatten(A_pyt).tolist()):
        assert False
    A_ = ttl.tensor.Tensor(
        torch.flatten(A_pyt).tolist(),
        a_activation_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR)
    A_cl = A_.to(ttl.tensor.Layout.CHANNELS_LAST)
    A_cl_data = A_cl.data()
    A = A_cl.to(device, ttl.tensor.MemoryConfig(False, 0))

    # Prepare weights
    B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()
    if (B_i == []):
        B_i = torch.flatten(B_pyt).tolist()
    elif(B_i != torch.flatten(B_pyt).tolist()):
        assert False
    B_ = ttl.tensor.Tensor(
        torch.flatten(B_pyt).tolist(),
        b_weights_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR
    )
    B_tiled_ = ttl.tensor.convert_conv_weight_tensor_to_tiled_layout(B_)
    B_tiled = B_tiled_.to(device)

    # Calculate conv result with golden result. Run Pytorch conv
    out_golden = torch.nn.functional.conv2d(A_pyt, B_pyt, stride=(stride_h, stride_w), padding=(pad_h, pad_w))

    # Run TT metal OP
    out = ttl.tensor.conv_as_large_bmm_single_core(A, B_tiled, [R,S,stride_h,stride_w,pad_h,pad_w], untilize_out)
    out = out.to(host)
    assert(out.shape() == mm_output_shape)
    if not untilize_out:
        # untilize
        out = out.to(ttl.tensor.Layout.ROW_MAJOR)
    # Copy output to host and convert tt tensor to pytorch tensor
    out_pytorch_padded = torch.tensor(out.data()).reshape(mm_output_shape)

    if (result == []):
        result = torch.flatten(out_pytorch_padded).tolist()
    elif(result != torch.flatten(out_pytorch_padded).tolist()):
        assert False

    #Run pytorch matmul
    mm_input_shape = [1, 1, _nearest_32(OH*OW), _nearest_32(C*R*S)]
    mm_weight_shape = [1, 1, _nearest_32(C*R*S), _nearest_32(K)]
    mm_output_shape = [1,1,_nearest_32(OH*OW),_nearest_32(K)]
    # Call DTX pass to transform A
    A_transformed_data = ttl.dtx.evaluate(A_cl_data, ttl.dtx.conv_transform([C,H,W], [R,S,stride_h,stride_w,pad_h,pad_w], ([-1],[-1])))
    A_transformed_pytorch_tensor = torch.tensor(A_transformed_data).reshape(mm_input_shape)
    B_rm = B_tiled_.to(ttl.tensor.Layout.ROW_MAJOR)
    assert(B_rm.shape() == [1, 1, _nearest_32(C*R*S), _nearest_32(K)])
    B_data = B_rm.data()
    B_pytorch_tensor = torch.tensor(B_data).reshape(mm_weight_shape)
    out_mm_pytorch_padded = torch.matmul(A_transformed_pytorch_tensor, B_pytorch_tensor)
    assert(list(out_mm_pytorch_padded.shape) == mm_output_shape)
    out_mm_pytorch = out_mm_pytorch_padded[:, :, 0 : (OH * OW), 0 : K]

    # Convert matmul output layout to conv output layout
    out_mm_tr = torch.transpose(out_mm_pytorch, 2, 3)
    assert(list(out_mm_tr.shape) == [1,1,K,(OH*OW)])
    out_mm_result = out_mm_tr.reshape([1,K,OH,OW])
    # compare mm cpu with conv pytorch
    assert(out_mm_result.shape == out_golden.shape)
    passing_pcc, output_pcc = comp_pcc(out_golden, out_mm_result, 0.99)
    print("DTX Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc

    #Compare dtx + torch matmul output with tt conv op output
    #print(out_pytorch_padded)
    #print(out_mm_pytorch_padded)
    err = 0
    p = 0
    nzp = 0
    for h in range(0,mm_output_shape[2]):
        #print("at h=" + str(h))
        for w in range(0,mm_output_shape[3]):
            #if(out_pytorch_padded[0][0][h][w] != 0):
                #print("Non zero at w=" + str(w))
                #assert False
            if abs(out_pytorch_padded[0][0][h][w] - out_mm_pytorch_padded[0][0][h][w]) > 0.05:
                #print(" .  Fails at w=" + str(w))
                #print(out_pytorch_padded[0][0][h][w])
                #print(out_mm_pytorch_padded[0][0][h][w])
                err+=1
            else:
                if(out_pytorch_padded[0][0][h][w] != 0):
                    nzp+=1
                #print("Pases on w=" + str(w))
                p+=1
        #     if err == 1000:
        #         break
        # if err == 1000:
        #     break

    print("Total number of elements " + str(_nearest_32((OH * OW))*_nearest_32(K)))
    print("Number of errors=" + str(err))
    print("Number of correct elements="+str(p))
    print("Number of non zero correct elements="+ str(nzp))
    assert(out_pytorch_padded.shape == out_mm_pytorch_padded.shape)
    passing_pcc, output_pcc = comp_pcc(out_mm_pytorch_padded, out_pytorch_padded, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    #assert passing_pcc

    # remove padding
    out_pytorch = out_pytorch_padded[:, :, 0 : (OH * OW), 0 : K]

    # Convert matmul output layout to conv output layout
    out_tr = torch.transpose(out_pytorch, 2, 3)
    assert(list(out_tr.shape) == [1,1,K,(OH*OW)])
    out_result = out_tr.reshape([1,K,OH,OW])

    # Compare against golden
    assert(out_result.shape == out_golden.shape)
    passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    #assert passing_pcc
    print("Done")
    ttl.device.CloseDevice(device)
    print("Device closed")

if __name__ == "__main__":
    # Incorrect output
    #test_run_conv_as_large_matmul(256, 128, 28, 28, 3, 3, 2, 2, 1, 1)
    #test_run_conv_as_large_matmul(256, 128, 28, 28, 3, 3, 2, 2, 1, 1)
    #test_run_conv_as_large_matmul(256, 128, 28, 28, 3, 3, 2, 2, 1, 1, True)
    #test_run_conv_as_large_matmul(256, 128, 28, 28, 3, 3, 2, 2, 1, 1, False)
    # Hang
   #test_run_conv_as_large_matmul(32, 1024, 5, 5, 1, 1, 1, 1, 0, 0, True)

    # Crash
    test_run_conv_as_large_matmul(256, 256, 14, 14, 3, 3, 1, 1, 1, 1, True)

    # Works
    #test_run_conv_as_large_matmul(32, 32, 5, 5, 1, 1, 1, 1, 0, 0, True)
    #test_run_conv_as_large_matmul(32, 32, 5, 5, 1, 1, 1, 1, 0, 0, False)
    #test_run_conv_as_large_matmul(32, 32, 5, 5, 1, 1, 1, 1, 0, 0)
