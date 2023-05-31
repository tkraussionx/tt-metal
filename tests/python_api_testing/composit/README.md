# Installation

From tt-metal top level:
```
cd tt_metal/third_party
git clone git@github.com:arakhmati/composit.git
cd -
```

# Test

```
export PYTHONPATH=$TT_METAL_HOME
pip install pytest pyrsistent graphviz networkx
pytest tests/python_api_testing/composit/test_composit.py -svv
```
