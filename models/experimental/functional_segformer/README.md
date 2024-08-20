To run the test, Run the folllowing command `pytest tests/ttnn/integration_tests/segformer/test_segformer_dwconv.py`.

The test_cases which is uncommented in file `test_segformer_dwconv.py` hangs while using bfloat8_b as dtype for conv and when passing bfloat8_b dtype input tensor  to the sub_module.

The hang issue is from the operation `ttnn.from_device`  in common.py file.
