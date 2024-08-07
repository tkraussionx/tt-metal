**Unit tests of failing Maxpools of each resolution**

**When Maxpooling=True, Encoder res: 4094x510**
- To reproduce the issue, run the command: `pytest tests/ttnn/unit_tests/operations/test_maxpool2d.py::test_model_net_max_pool_4094x510`
- Among 4 maxpools, 2 maxpools fails with OOM issue, 2 maxpools fails with with valid_page_size error
- All 4 maxpools got skipped with Bfloat8_b

**When Maxpooling=True, Encoder res: 2047x255**
- To reproduce the issue, run the command: `pytest tests/ttnn/unit_tests/operations/test_maxpool2d.py::test_model_net_max_pool_2047x255`
- Among 4 maxpools, 3 maxpools fails with with valid_page_size error, 1 maxpools fails with OOM issue
- All 4 maxpools got skipped with Bfloat8_b
