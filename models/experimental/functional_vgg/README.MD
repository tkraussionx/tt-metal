---

# ttnn vgg performance hang details

### Introduction

`jayasurya/ttnn_vgg_perf` branch contains ttnn implementation of vgg located at       `tt-metal/models/experimental/functional_vgg/tt/ttnn_vgg.py`,
end to end performance file for the same at `tt-metal/models/experimental/functional_vgg/tests/test_perf_vgg.py`


### Details
While running the performance test, VM is hanging in the second iteration of the for loop.
Upon stopping the process with  `ctrl + z` or killing the process, BM server is crashing.


## Steps to reproduce the issue

- Use the following command to run the e2e performance test for ttnn functional vgg:
 ```
  `git checkout jayasurya/ttnn_vgg_perf`
  ```
  ```
  `pytest models/experimental/functional_vgg/tests/test_perf_vgg.py`
  ```
