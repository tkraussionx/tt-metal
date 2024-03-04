pytest models/demos/falcon40b/tests/test_falcon_decoder.py -k "prefill and BFLOAT8_B-SHARDED"
pytest models/demos/falcon40b/tests/test_falcon_end_to_end.py -k "prefill and BFLOAT8_B-SHARDED and layers_1"
