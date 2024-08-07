There are 7 conv's in model_k model.

**model_k model: 256x256**
- To test the unit_test, run `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_k_256x256_failing_convs`
- Among 7 convs, 3 convs passed. 4 convs fail with Out of Memory issue.
```
E       RuntimeError: TT_THROW @ ../tt_metal/impl/allocator/allocator.cpp:143: tt::exception
E       info:
E       Out of Memory: Not enough space to allocate 25477120 B L1 buffer across 5 banks, where each bank needs to store 5095424 B
```

**model_k model: 128x128**
- The command to test the unit test: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_k_128x128_failing_convs`
- Among 7 convs, 4 convs passed. 3 convs fail with Out of Memory issue.
```
E       RuntimeError: TT_THROW @ ../tt_metal/impl/allocator/allocator.cpp:143: tt::exception
E       info:
E       Out of Memory: Not enough space to allocate 4636672 B L1 buffer across 1 banks, where each bank needs to store 4636672 B
```

Note: For conv checking purpose batch_size=1 is used even though 32 is used in the model.
