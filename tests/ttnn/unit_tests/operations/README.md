For the following test case ttnn.reshape is failing in lenet model.

To recreate the issue run the command:
`pytest tests/ttnn/unit_tests/operations/test_maxpool.py::test_run_max_pool`

For input shape [1, 6, 28, 28] in NCHW

The test is skipped with "Current maxpool writer needs nchannels to be multiple of 16!"

# After commenting the skip statement
# E       RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp:23: (*this->output_mem_config.shard_spec).shape[1] * input_tensor.element_size() % hal.get_alignment(HalMemType::L1) == 0
# E       info:
# E       Shard page size must currently have L1 aligned page size
