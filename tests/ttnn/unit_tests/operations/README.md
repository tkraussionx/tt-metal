## This commit contains Conv2d unit test for SqueezeBert model.

1. `test_squeezebert_conv1d` method in `tests/ttnn/unit_tests/operations/test_conv1d.py` file will contain unit test for ttnn.conv2d.

    To run the test, use the following command : `pytest tests/ttnn/unit_tests/operations/test_conv1d.py::test_squeezebert_conv1d`

## Expected Behaviour / Error(s):

 1. Input shape: [8, 384, 768] in (batch_size, input_length, in_channels) format and out_channels: 768, groups: 4, bias = True
 2. Input shape: [8, 384, 3072] in (batch_size, input_length, in_channels) format and out_channels: 768, groups: 4, bias = True
 3. Input shape: [8, 384, 768] in (batch_size, input_length, in_channels) format and out_channels: 768, groups: 4, bias = True

#### On GS and WH(n300):

    E       RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/tensor/types.cpp:172: normalized_index >= 0 and normalized_index < rank
    E       info:
    E       Index is out of bounds for the rank, should be between 0 and 2 however is 3
