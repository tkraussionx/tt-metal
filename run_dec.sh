#!/bin/bash

for i in {1..50}
do
  echo "Iteration: $i"
  pytest -svv models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_decoder.py::test_falcon_decoder[wormhole_b0-True-False-16-BFLOAT16-DRAM-tiiuae/falcon-7b-instruct-0.98-decode_batch32]
done
