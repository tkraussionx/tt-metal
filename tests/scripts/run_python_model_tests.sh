#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi


# This must run in slow dispatch mode
# pytest -svv $TT_METAL_HOME/tests/python_api_testing/sweep_tests/pytests/test_sweep_conv_with_address_map.py

# For now, adding tests with fast dispatch and non-32B divisible page sizes here. Python/models people,
# you can move to where you'd like.

if [[ "$ARCH_NAME" != "wormhole_b0" ]]; then
  # TODO(arakhmati): Run ttnn tests only on graskull until the issue with ttnn.reshape on wormhole is resolved
  # Tests for tensors in L1


D0="$(date +%s)"
env pytest $TT_METAL_HOME/tests/ttnn/unit_tests
D1=$[$(date +%s)-${D0}]
echo "4697: $ARCH_NAME run_python_api_unit_tests unit_tests: ${D1}"

D0="$(date +%s)"
pytest $TT_METAL_HOME/models/experimental/bert_large_performant/unit_tests/test_bert_large*matmul* -k in0_L1-in1_L1-bias_L1-out_L1
pytest $TT_METAL_HOME/models/experimental/bert_large_performant/unit_tests/test_bert_large*bmm* -k in0_L1-in1_L1-out_L1
D1=$[$(date +%s)-${D0}]
echo "4697: $ARCH_NAME run_python_api_unit_tests tensors in L1 time: ${D1}"

# # Tests for mixed precision (sweeps combos of bfp8_b/bfloat16 dtypes for fused_qkv_bias and ff1_bias_gelu matmul and pre_softmax_bmm)
D0="$(date +%s)"
pytest $TT_METAL_HOME/models/experimental/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_matmul -k "fused_qkv_bias and batch_9 and L1"
pytest $TT_METAL_HOME/models/experimental/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_matmul -k "ff1_bias_gelu and batch_9 and DRAM"
pytest $TT_METAL_HOME/models/experimental/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_bmm -k "pre_softmax_bmm and batch_9"
D1=$[$(date +%s)-${D0}]
echo "4697: $ARCH_NAME run_python_api_unit_tests mixed precision time: ${D1}"

# # BERT TMs
D0="$(date +%s)"
pytest $TT_METAL_HOME/models/experimental/bert_large_performant/unit_tests/test_bert_large_split_query_key_value_and_split_heads.py -k "in0_L1-out_L1 and batch_9"
pytest $TT_METAL_HOME/models/experimental/bert_large_performant/unit_tests/test_bert_large_concatenate_heads.py -k "in0_L1-out_L1 and batch_9"
D1=$[$(date +%s)-${D0}]
echo "4697: $ARCH_NAME run_python_api_unit_tests BERT TMs time: ${D1}"

# # Test program cache
D0="$(date +%s)"
pytest $TT_METAL_HOME/models/experimental/bert_large_performant/unit_tests/ -k program_cache
D1=$[$(date +%s)-${D0}]
echo "4697: $ARCH_NAME run_python_api_unit_tests program cache time: ${D1}"

# Fused ops unit tests
D0="$(date +%s)"
pytest $TT_METAL_HOME/models/experimental/bert_large_performant/unit_tests/fused_ops/test_bert_large_fused_ln.py -k "in0_L1-out_L1 and batch_9"
pytest $TT_METAL_HOME/models/experimental/bert_large_performant/unit_tests/fused_ops/test_bert_large_fused_softmax.py -k "in0_L1 and batch_9"
D1=$[$(date +%s)-${D0}]
echo "4697: $ARCH_NAME run_python_api_unit_tests Fused ops unit tests time: ${D1}"

# # Resnet18 tests with conv on cpu and with conv on device
D0="$(date +%s)"
pytest $TT_METAL_HOME/models/demos/resnet/tests/test_resnet18.py
D1=$[$(date +%s)-${D0}]
echo "4697: $ARCH_NAME run_python_api_unit_tests Resnet18 tests time: ${D1}"

# # Falcon tests
D0="$(date +%s)"
pytest $TT_METAL_HOME/models/demos/falcon7b/tests/unit_tests/test_falcon_matmuls_and_bmms_with_mixed_precision.py -k "seq_len_128 and in0_BFLOAT16-in1_BFLOAT8_B-out_BFLOAT16-weights_DRAM"
pytest $TT_METAL_HOME/models/demos/falcon7b/tests/unit_tests/test_falcon_matmuls_and_bmms_with_mixed_precision.py -k "seq_len_512 and in0_BFLOAT16-in1_BFLOAT8_B-out_BFLOAT16-weights_DRAM"
pytest $TT_METAL_HOME/models/demos/falcon7b/tests/unit_tests/test_falcon_attn_matmul.py
D1=$[$(date +%s)-${D0}]
echo "4697: $ARCH_NAME run_python_api_unit_tests Falcon time: ${D1}"
fi
