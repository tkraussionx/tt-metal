To recreate the issue run the command:
`tests/ttnn/unit_tests/operations/test_avg_pool2d.py`

For input shape [1, 512, 7, 7] , Expected shape is [1, 512, 7, 7]

with the ttnn.global_avg_pool2d, the output shape is [1,512, 1, 1]

Need support for ttnn AdaptiveAvgPool2d.
