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

START="$(date +%s)"
./tests/scripts/run_python_api_unit_tests.sh
D1=$[$(date +%s)-${START}]
echo "4697: run_python_api_unit_tests time: ${D1}"
env python tests/scripts/run_tt_metal.py --dispatch-mode fast
D2=$[$(date +%s)-${D1}]
echo "4697: run_python_api_unit_tests time: ${D2}"
env python tests/scripts/run_tt_eager.py --dispatch-mode fast
D3=$[$(date +%s)-${D2}]
echo "4697: run_python_api_unit_tests time: ${D3}"
./build/test/tt_metal/unit_tests_fast_dispatch
D4=$[$(date +%s)-${D3}]
echo "4697: run_python_api_unit_tests time: ${D4}"

echo "Checking docs build..."

cd $TT_METAL_HOME/docs
python -m pip install -r requirements-docs.txt
make clean
make html
