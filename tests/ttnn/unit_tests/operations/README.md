## This commit contains Conv1d unit test for SqueezeBert model.

This commit adds a Conv1d unit test for the SqueezeBert model. This test references `tests/ttnn/unit_tests/operations/test_conv1d.py` .

1. `test_squeezebert_attention` method in `tests/ttnn/unit_tests/operations/test_squeezebert_conv1d.py` file will contain unit test for ttnn.conv1d.

    To run the test, use the following command : `pytest tests/ttnn/unit_tests/operations/test_squeezebert_conv1d.py::test_squeezebert_attention`


## Expected Behaviour:

 1. Input shape: [8, 384, 768] in (batch_size, input_length, in_channels) format and out_channels: 768, groups: 4, bias = True.

        Expected to pass the testcase with bias.

## Issue Details:

#### On WH(n300):

    E       RuntimeError: TT_THROW @ ../tt_metal/impl/allocator/allocator.cpp:143: tt::exception
    E       info:
    E       Out of Memory: Not enough space to allocate 192 B L1_SMALL buffer across 48 banks, where each bank needs to store 32 B
