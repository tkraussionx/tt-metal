#! /usr/bin/env bash

source scripts/tools_setup_common.sh

set -eo pipefail

run_profiling_test(){
    if [[ -z "$ARCH_NAME" ]]; then
      echo "Must provide ARCH_NAME in environment" 1>&2
      exit 1
    fi

    source build/python_env/bin/activate
    export PYTHONPATH=$TT_METAL_HOME

    TT_METAL_DEVICE_PROFILER=1 pytest $PROFILER_TEST_SCRIPTS_ROOT/test_device_profiler.py -vvv

    remove_default_log_locations

    $PROFILER_SCRIPTS_ROOT/profile_this.py -c "pytest -svvv $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_composite.py::test_run_eltwise_composite_test[lerp_binary-input_shapes0]"

    python -m tracy -r -v -p -m pytest -svvv $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_composite.py::test_run_eltwise_composite_test[lerp_binary-input_shapes0]

    remove_default_log_locations
}


cd $TT_METAL_HOME

counter=1
while [ $counter -le 100 ]
do
    echo $counter
    run_profiling_test
    ((counter++))
done
#run_tracy_test
