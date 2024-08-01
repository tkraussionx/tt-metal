For the following test cases ttnn.Maxpool2d is failing in blazepose model.

To recreate the issue run the command:
`pytest tests/ttnn/unit_tests/operations/test_blazepose_maxpool2d.py`


1. For input shape [1, 96, 32, 32]

```
E       RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/pool/maxpool/device/max_pool_program_factory.cpp:30: is_pow2
E       info:
E       Row size (nchannels * bytes = 192) should be power of 2 (false).
E       backtrace:
E        --- tt::tt_metal::MaxPool::validate(std::__1::vector<tt::tt_metal::Tensor, std::__1::allocator<tt::tt_metal::Tensor>> const&) const
```

2. For input shape [1, 48, 128, 128]

```
E       RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/pool/maxpool/device/max_pool_program_factory.cpp:30: is_pow2
E       info:
E       Row size (nchannels * bytes = 96) should be power of 2 (false).
E       backtrace:
E        --- tt::tt_metal::MaxPool::validate(std::__1::vector<tt::tt_metal::Tensor, std::__1::allocator<tt::tt_metal::Tensor>> const&) const
```
