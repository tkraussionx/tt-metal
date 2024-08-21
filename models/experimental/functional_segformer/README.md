#Hang issue in efficient self_attention

To run the test use command `tests/ttnn/integration_tests/segformer/test_segformer_efficient_selfattention.py`

The hang is from API `ttnn.from_device` in `ttnn_to_torch` method.
