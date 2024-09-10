For the following test cases ttnn.maxpool2d is failing in VGG11 and VGG16 model in n300.

To recreate the issue run the command:
`tests/ttnn/unit_tests/operations/test_maxpool2d.py`

1. For input shape [1, 512, 28, 28]

```
LOW PCC issue
PCC = 0.8291132190025535
```
2. For input shape [1, 512, 14, 14]
```
LOW PCC issue
PCC = 0.8859652206760608
```
