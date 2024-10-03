#!/bin/bash
# export flag if needed
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml

# Remove build artifacts if needed
rm -rf build built

# Synchronize the repository
git submodule update --init --recursive
sudo apt-get install git-lfs
git submodule foreach 'git lfs fetch --all && git lfs pull'

# Build with profiler
./scripts/build_scripts/build_with_profiler_opt.sh

# Create env if not already created
# ./create_venv.sh

# Activate the environment
source python_env/bin/activate

# Set pytest script to run to do the git bisect
# Run pytest with the provided arguments
# TODO: If script hangs/crashes device, add timeout and reset device after it
pytest models/demos/t3000/falcon40b/tests/test_demo.py::test_demo[wormhole_b0-True-128]


# Check pytest exit code
if [ $? -eq 0 ]; then
    exit 0  # Test passed
else
    exit 1  # Test failed
fi
