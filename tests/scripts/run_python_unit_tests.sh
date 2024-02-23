#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [[ -z "$ARCH_NAME" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

if [[ ! -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
  echo "Only Fast Dispatch mode allowed - Must have TT_METAL_SLOW_DISPATCH_MODE unset" 1>&2
  exit 1
fi

cd $TT_METAL_HOME
export PYTHONPATH=$TT_METAL_HOME

if [[ ! -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
    D0="$(date +%s)"
    env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/ -name 'test_*.py' -a ! -name 'test_untilize_with_halo_and_max_pool.py') -vvv
    env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/ -name 'test_*.py' -a ! -name 'test_sweep_conv_with_address_map.py') -vvv
    D1=$[$(date +%s)-${D0}]
    echo "4697: $ARCH_NAME run_python_api_unit_tests slow dispatch unit/sweep tests time: ${D1}"

else
    D0="$(date +%s)"
#   # Need to remove move for time being since failing
    env pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/ -vvv
    D1=$[$(date +%s)-${D0}]
    echo "4697: $ARCH_NAME run_python_api_unit_tests fast dispatch unit tests 1 time: ${D1}"

    D0="$(date +%s)"
    env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/ -name 'test_*.py' -a ! -name 'test_sweep_conv_with_address_map.py' -a ! -name 'test_move.py') -vvv
    D1=$[$(date +%s)-${D0}]
    echo "4697: $ARCH_NAME run_python_api_unit_tests fast dispatch sweep tests 2 time: ${D1}"

    D0="$(date +%s)"
    env pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_move.py -k input_L1 -vvv
    D1=$[$(date +%s)-${D0}]
    echo "4697: $ARCH_NAME run_python_api_unit_tests fast dispatch sweep tests 3 time: ${D1}"

fi
