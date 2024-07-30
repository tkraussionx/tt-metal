#!/bin/bash

git checkout ppopovic/didt_fixes

 # On didt_fixes branch
git submodule update --recursive

export ARCH_NAME=wormhole_b0
export CONFIG=Release
export PYTHONPATH=/home/ppopovic/tt-metal
export TT_METAL_HOME=/home/ppopovic/tt-metal

source build_metal.sh
source create_venv.sh
source source /opt/tt_metal_infra/tt-metal/python_env/bin/activate
