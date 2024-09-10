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

from models.demos.t3000.llama2_70b.reference.llama.llama.tokenizer3 import Tokenizer3

from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
)
from models.demos.utils.tenstorrent_lm import TenstorrentLM
from models.demos.t3000.llama3_70b.demo.lm_backend import PrefillDecodeBackend, build_generator

"""
This script uses the as a library (see https://github.com/EleutherAI/lm-evaluation-harness)
TenstorrentLM

"""


def mock_build_generator(model_args, tt_args):
    class MockModel:
        # mock implementation in TtLlamaModelForGeneration
        # see: tt-metal/models/demos/t3000/llama2_70b/tt/llama_generation.py
        def __init__(self, batch_size: int, vocab_size: int, max_seq_len: int):
            self.batch_size = batch_size
            self.vocab_size = vocab_size
            self.max_seq_len = max_seq_len

        def forward(self, tokens: torch.Tensor, start_pos: int):
            _, seq_len = tokens.shape
            if seq_len == 1:
                return self.decode_forward(tokens, start_pos)
            else:
                return self.prefill_forward(tokens, start_pos)

        def decode_forward(self, tokens: torch.Tensor, start_pos: int):
            batch, seq_len = tokens.shape
            assert seq_len == 1
            logits = torch.rand((batch, seq_len, self.vocab_size))
            # send the EOT token after 128 tokens for testing
            if start_pos == 128:
                logits[:, :, 128009] = 100.0
            return logits

        def prefill_forward(self, tokens: torch.Tensor, start_pos: int):
            batch, seq_len = tokens.shape
            assert seq_len <= 2048, f"Only prefill up to 2048 tokens is supported, got {seq_len}"

            prefill_seq_len = 128 if seq_len <= 128 else 2048

            batch, seq_len = tokens.shape
            output_logits = torch.zeros(batch, seq_len, self.vocab_size)
            padded_seq_len = 128 if seq_len <= 128 else 2048
            # pad tokens to 128 or 2048
            prefill_ids = torch.cat([tokens, torch.zeros(batch, padded_seq_len - seq_len).long()], dim=-1)
            return output_logits

    class MockGenerator:
        def __init__(self, tokenizer, model_args):
            self.tokenizer = tokenizer
            self.model = MockModel(
                batch_size=model_args.max_batch_size,
                vocab_size=tokenizer.n_words,
                max_seq_len=model_args.max_seq_len,
            )

    tokenizer = Tokenizer3(model_path=model_args.tokenizer_path)
    return MockGenerator(tokenizer=tokenizer, model_args=model_args)


def get_model_backend(mock_model=False):
    llama_version = "llama3"
    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
    )
    if mock_model:
        with patch.object(PrefillDecodeBackend, "init_tt_metal_device", return_value=None):
            with patch(
                "models.demos.t3000.llama3_70b.demo.lm_backend.build_generator", new=mock_build_generator
            ):
                model_backend = PrefillDecodeBackend(
                    model_version="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    batch_size=32,
                    num_layers=80,
                    max_seq_len=2048,
                    cache_root="/mnt/tt-metal-llama3_1-70b-t3000-api-fs",
                )
    else:
        model_backend = PrefillDecodeBackend(
            model_version="meta-llama/Meta-Llama-3.1-70B-Instruct",
            batch_size=32,
            num_layers=80,
            max_seq_len=2048,
            cache_root="/mnt/tt-metal-llama3_1-70b-t3000-api-fs",
        )

    return model_backend, model_backend.formatter


def main():
    # -----------------------------------
    # configuration:
    # -----------------------------------
    # tasks = ["mmlu_econometrics", "mmlu_high_school_statistics"]
    tasks = ["ifeval"]
    eval_output_fpath = "eval_output"
    limit = None        # limit the number of samples per task
    log_samples = True  # log samples and outputs to file
    mock_model = False   # use random logits model for testing
    num_fewshot = None  # number of fewshot samples (task dependent)
    # -----------------------------------
    evaluation_tracker = EvaluationTracker(output_path=eval_output_fpath)
    model_backend, tokenizer = get_model_backend(mock_model=mock_model)
    # pretrained must be the hugginface pretrained model name
    # see: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct
    pretrained = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    model = None
    lm = TenstorrentLM(
        model_backend=model_backend,
        pretrained=pretrained,
        tokenizer=tokenizer,
        eot_token_id=128009,
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
