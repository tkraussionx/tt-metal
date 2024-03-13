#!/bin/bash

# ——————————————
# SINGLE CORE
# ——————————————
# make programming_examples/matmul_single_core; ./build/programming_examples/matmul_single_core



# ——————————————
# MULTICORE
# ——————————————
# make programming_examples/matmul_multi_core; ./build/programming_examples/matmul_multi_core



# ——————————————
# MULTICORE_REUSE
# ——————————————
# make programming_examples/matmul_multicore_reuse; ./build/programming_examples/matmul_multicore_reuse



# ——————————————
# MULTICORE_REUSE_MCAST
# ——————————————
clear;
make programming_examples/matmul_multicore_reuse_mcast;
# make clean programming_examples/matmul_multicore_reuse_mcast;
./build/programming_examples/matmul_multicore_reuse_mcast
