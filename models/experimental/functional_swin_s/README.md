To recreate the issue, run the command: pytest tests/ttnn/integration_tests/swin_s/test_ttnn_shifted_window_attention.py

While running the command, the issue arised is here:
E           RuntimeError: TT_FATAL @ ../tt_metal/impl/buffers/buffer.cpp:39: valid_page_size
E           info:
E           For valid non-interleaved buffers page size 6 must equal buffer size 1012. For interleaved-buffers page size should be divisible by buffer size
