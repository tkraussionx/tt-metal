By using ttnn Conv2d in Model_K implementation with dilation > 1, the output shape is different from the expected output shape.
The error is not reproducible.
Expected to have same shape for torch Conv2 output and ttnn Conv2 output.

Run the command to print the shapes of Conv2 outputs: `pytest tests/ttnn/unit_tests/operations/test_conv_dilation.py`
