##  ttnn.conv2d fails in convnet mnist model.

To recreate the issue run the command:
`pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_conv_convnet_mnist`

Error:
```
E       RuntimeError: TT_THROW @ ../tt_metal/impl/program/program.cpp:511: tt::exception
E       info:
E       Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=0)] grow to 1504416 B which is beyond max L1 size of 1499136 B
```
