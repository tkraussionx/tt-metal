#!/bin/bash

source build/python_env/bin/activate

destdir=$TT_METAL_HOME/resnet-loop.text

set -e
for i in {1..200}; do
    echo "$i" >> "$destdir"
    pytest models/demos/resnet/tests -m models_performance_bare_metal
done
