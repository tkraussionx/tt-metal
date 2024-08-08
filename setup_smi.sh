#!/bin/bash

cd /home/ppopovic/tt-smi
python3 -m venv .venv
source .venv/bin/activate
pip install .
cd /home/ppopovic/tt-metal
