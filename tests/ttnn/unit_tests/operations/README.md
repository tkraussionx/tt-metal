## This commit contains Conv2d unit test for VoVNet model.

1. `test_vovnet_convs` method in `tests/ttnn/unit_tests/operations/test_vovnet_fused_conv.py` file will contain unit test for ttnn.conv2d with batch_norm and relu fused.

    To run the test, use the following command : `pytest tests/ttnn/unit_tests/operations/test_vovnet_fused_conv.py::test_vovnet_convs`

## Expected Behaviour / Error(s):

### On GS:

`E           Statically allocated circular buffers in program 4 clash with L1 buffers on core range [(x=0,y=0) - (x=11,y=2)]. L1 buffer allocated at 913408 and static circular buffer region ends at 992096`

### On WH(n150):

`AssertionError: 0.9301251105478947`
