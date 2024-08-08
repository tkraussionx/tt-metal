#!/bin/bash

set +e
git config --global --add safe.directory /home/ppopovic/tt-metal
git config --global --add safe.directory /home/ppopovic/tt-metal/models/demos/t3000/llama2_70b/reference/llama

git checkout ppopovic/didt_fixes

 # On didt_fixes branch
git submodule update --recursive

export ARCH_NAME=wormhole_b0
export CONFIG=Release
export PYTHONPATH=/home/ppopovic/tt-metal
export TT_METAL_HOME=/home/ppopovic/tt-metal



source build_metal.sh
source create_venv.sh
source /opt/tt_metal_infra/tt-metal/python_env/bin/activate
