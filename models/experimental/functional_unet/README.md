We have observed PCC drop after enc3_1 in the whole unet model. So, we kept whole model in torch except enc3_1 layer (i.e., in ttnn). tt_unet.py has the implementation as described. For this case, we got a PCC of ~0.52. Hence, we intended to create unit test for the same to narrow down the problem. However, in the unit test for enc3_1 gives PCC > 0.99.

Currently, the dtype and weight_dtype for above implementation for enc3_1 conv is bfloat8_b if we change the dtype and weight_dtype to bfloat16 then the pcc is ~0.86 and for float32 pcc is ~0.92 .
