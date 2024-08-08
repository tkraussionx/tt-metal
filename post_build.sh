#!/bin/bash

 # On didt_fixes branch

# Setup tt-topology
cd /home/ppopovic
rm -rf tt-topology
git clone https://github.com/tenstorrent/tt-topology.git
cd ./tt-topology
pip install .


# Setup tt-flash
cd /home/ppopovic
rm -rf tt-flash
git clone https://github.com/tenstorrent/tt-flash.git
cd ./tt-flash
pip install .

# Setup tt-smi
cd /home/ppopovic
rm -rf tt-smi
git clone https://github.com/tenstorrent/tt-smi.git
cd ./tt-smi
pip install .

# Setup tt-firmware
cd /home/ppopovic
rm -rf tt-firmware
git clone https://github.com/tenstorrent/tt-firmware.git

cd /home/ppopovic/tt-metal

tt-topology -l mesh
#tt-smi -r 0,1,2,3

export ARCH_NAME=wormhole_b0
export CONFIG=Release
export PYTHONPATH=/home/ppopovic/tt-metal
export TT_METAL_HOME=/home/ppopovic/tt-metal
