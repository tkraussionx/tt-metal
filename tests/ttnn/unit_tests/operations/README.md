For the following test case ttnn.reshape is failing in mnist model.

To recreate the issue run the command:
`pytest tests/ttnn/unit_tests/operations/test_reshape.py`

For input shape [1, 1, 28, 28]

Cannot reshape the tensor if input tensor in MESH_DEVICE using ttnn.reshape
Can reshape using ttnn.reshape if the input tensor in HOST

RuntimeError: TT_THROW @ ../ttnn/cpp/ttnn/operations/core/core.cpp:60: tt::exception
E       info:
E       Unable to reshape given tensor!
E       backtrace:
E        --- /home/ubuntu/sabira/tt-metal/ttnn/ttnn/_ttnn.so(+0x4dcf59) [0x7fee0922df59]
