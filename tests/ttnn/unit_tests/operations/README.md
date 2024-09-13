##  ttnn.softmax affects PCC in convnet mnist model.

To recreate the issue run the command:
`pytest tests/ttnn/unit_tests/operations/test_softmax.py::test_softmax_convnet_mnist`

ttnn.softmax converts the tensor values to zeros resulting PCC = 0.0 (One tensor is all zero)
When ttnn.softmax is replaced with torch softmax PCC > 0.99 for ttnn convnet mnist model.
