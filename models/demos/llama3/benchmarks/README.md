Steps:
1. pip install lmms-eval

2. add reference llama_models folder to path

3. export HF_DATASETS_CACHE="/proj_sw/user_dev/llama_vision_benchmarks"

4. Run: python models/demos/llama3/benchmarks/lm_harness_eval.py --tasks mmmu --model llama-cpu-reference
