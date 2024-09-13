##  ttnn.max_pool2d fails in mnist model.

To recreate the issue run the command:
`pytest models/experimental/functional_convnet_mnist/test/test_convnet_mnist.py`

Error:
```
E       RuntimeError: TT_FATAL @ ../tt_metal/impl/buffers/buffer.cpp:38: valid_page_size
E       info:
E       For valid non-interleaved buffers page size 2156 must equal buffer size 107748. For interleaved-buffers page size should be divisible by buffer size
```
