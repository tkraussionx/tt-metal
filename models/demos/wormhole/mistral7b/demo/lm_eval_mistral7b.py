from unittest.mock import patch
import json

import torch

import lm_eval.evaluator as evaluator
import lm_eval.tasks as tasks
from lm_eval.loggers import EvaluationTracker
from lm_eval.evaluator_utils import print_writeout

from lm_eval.utils import (
    eval_logger,
    handle_non_serializable,
    hash_string,
    positional_deprecated,
    simple_parse_args_string,
)

from models.demos.wormhole.mistral7b.reference.tokenizer import Tokenizer

# from models.demos.t3000.llama2_70b.tt.llama_common import (
#     setup_llama_env,
# )
from models.demos.wormhole.mistral7b.tt.mistral_common import (
    prepare_inputs_ttnn,
    sample,
    precompute_freqs,
    freqs_to_rotation_matrix,
    cache_attention,
)
from models.demos.utils.tenstorrent_lm import TenstorrentLM
from models.demos.wormhole.mistral7b.demo.mistral7b_backend import PrefillDecodeBackend

from models.demos.wormhole.mistral7b.tt.mistral_model import TtTransformer
from models.demos.wormhole.mistral7b.tt.mistral_embedding import TtMistralEmbedding
from models.demos.wormhole.mistral7b.reference.tokenizer import Tokenizer
from models.demos.wormhole.mistral7b.tt.model_config import TtModelArgs
from models.demos.wormhole.mistral7b.demo.demo_with_prefill import Emb, preprocess_inputs_prefill
from models.demos.wormhole.mistral7b.demo.demo import preprocess_inputs

"""
This script uses the as a library (see https://github.com/EleutherAI/lm-evaluation-harness)
TenstorrentLM

"""


def get_env_var(var, msg):
    """Get an environment variable or raise an exception with helpful message."""
    value = os.getenv(var)
    if value is None:
        raise ValueError(f"Environment variable is required: {var}. {msg}")
    return value


def get_model_backend(mock_model=False):
    # llama_version = "llama3"
    # model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
    #     llama_version=llama_version,
    # )
    # if mock_model:
    #     with patch.object(PrefillDecodeBackend, "init_tt_metal_device", return_value=None):
    #         with patch(
    #             "models.demos.t3000.llama3_70b.demo.lm_backend.build_generator", new=mock_build_generator
    #         ):
    #             model_backend = PrefillDecodeBackend(
    #                 model_version="meta-llama/Meta-Llama-3.1-70B-Instruct",
    #                 batch_size=32,
    #                 num_layers=80,
    #                 max_seq_len=2048,
    #                 cache_root="/mnt/tt-metal-llama3_1-70b-t3000-api-fs",
    #             )
    # else:
    # model_backend = PrefillDecodeBackend(
    #     model_version="meta-llama/Meta-Llama-3.1-70B-Instruct",
    #     batch_size=32,
    #     num_layers=80,
    #     max_seq_len=2048,
    #     cache_root="/mnt/tt-metal-llama3_1-70b-t3000-api-fs",
    #     )
    model_backend = PrefillDecodeBackend(
        model_version="mistralai/Mistral-7B-Instruct-v0.2",
        batch_size=8,
        num_layers=32,
        max_seq_len=4096,
        cache_root=get_env_var("CACHE_ROOT", msg="Base path for all data caches."),
        verbose=verbose,
    )

    return model_backend, model_backend.tokenizer


def main():
    # -----------------------------------
    # configuration:
    # -----------------------------------
    # tasks = ["mmlu_econometrics", "mmlu_high_school_statistics"]
    # tasks = ["mmlu"]
    # tasks = ["gpqa_main_cot_zeroshot"]
    # tasks = ["ifeval"]
    tasks = ["ifeval", "gpqa_main_cot_zeroshot"]
    eval_output_fpath = "eval_output"
    limit = None        # limit the number of samples per task
    log_samples = True  # log samples and outputs to file
    mock_model = False   # use random logits model for testing
    num_fewshot = None  # number of fewshot samples (task dependent)
    # -----------------------------------
    if "ifeval" in tasks:
        # download nltk punkt tokenizer if not available
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    evaluation_tracker = EvaluationTracker(output_path=eval_output_fpath)
    model_backend, tokenizer = get_model_backend(mock_model=mock_model)
    # pretrained must be the hugginface pretrained model name
    # see: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct
    # mistralai/Mistral-7B-Instruct-v0.2
    pretrained = "mistralai/Mistral-7B-Instruct-v0.2"
    model = None
    lm = TenstorrentLM(
        model_backend=model_backend,
        pretrained=pretrained,
        tokenizer=tokenizer,
        eot_token_id=tokenizer.eos_id(),
        write_out=True,
    )
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        bootstrap_iters=0,
        batch_size=32,
        write_out=False,
        log_samples=log_samples,
        evaluation_tracker=evaluation_tracker,
        model_args={},
    )
    print(results["results"])
    if results is not None:
        if log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(
            results, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
        # print(dumped)
        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples if log_samples else None
        )
        if log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )


if __name__ == "__main__":
    main()