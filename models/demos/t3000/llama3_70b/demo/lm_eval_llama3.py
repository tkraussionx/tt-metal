import json
import time
import torch
import os
from unittest.mock import patch

import lm_eval.evaluator as evaluator
from lm_eval.loggers import EvaluationTracker

from lm_eval.utils import (
    handle_non_serializable,
)

from models.demos.t3000.llama2_70b.reference.llama.llama.tokenizer3 import Tokenizer3

from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
)
from models.demos.utils.tenstorrent_lm import TenstorrentLM
from models.demos.t3000.llama3_70b.demo.lm_backend import PrefillDecodeBackend

"""
This script uses the as a library (see https://github.com/EleutherAI/lm-evaluation-harness)
TenstorrentLM

"""


def mock_build_generator(model_args, tt_args, *args, **kwargs):
    class MockModel:
        # mock implementation in TtLlamaModelForGeneration
        # see: tt-metal/models/demos/t3000/llama2_70b/tt/llama_generation.py
        def __init__(self, batch_size: int, vocab_size: int, max_seq_len: int):
            self.batch_size = batch_size
            self.vocab_size = vocab_size
            self.max_seq_len = max_seq_len

        def prefill_forward_single_user(
            self,
            tokens: torch.Tensor,
            start_pos: int,
            user_id: int,
            last_token_idx=None,
            page_table=None,
            kv_cache=None,
        ):
            return self.decode_forward(tokens=tokens, start_pos=start_pos)

        def decode_forward(
            self,
            tokens: torch.Tensor,
            start_pos: int,
            page_table=None,
            kv_cache=None,
        ):
            assert len(tokens.shape) == 2
            batch, seqlen = tokens.shape
            forward_start = time.time()
            simulated_tps = 10000.0
            simulated_duration = 1.0 / simulated_tps
            # update the new tokens generated to the input id
            # vocab_size = tokenizer.nwords
            # logits: [batch, seqlen, vocab_size]
            logits = torch.randn((batch, seqlen, 128256))
            # send a token every period loops
            EOT_ID = 128009
            # EOS_ID = 128001
            send_index = 200
            send_token = EOT_ID
            if start_pos is not None:
                if isinstance(start_pos, int):
                    cache_idxs = torch.tensor([start_pos for _ in range(batch)], dtype=torch.int64)
                else:
                    cache_idxs = start_pos.to(dtype=torch.int64)
                    send_token_mask = cache_idxs > send_index
                    batch_indices = torch.nonzero(send_token_mask).squeeze()
                    logits[batch_indices, 0, send_token] = 100.0

            actual_duration = time.time() - forward_start
            # simulate forward latency
            time.sleep(max(simulated_duration - actual_duration, 0))
            return logits

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
            with patch("models.demos.t3000.llama3_70b.demo.lm_backend.build_generator", new=mock_build_generator):
                model_backend = PrefillDecodeBackend(
                    model_version="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    batch_size=32,
                    num_layers=80,
                    max_seq_len=2048,
                    cache_root=cache_path,
                )
    else:
        model_backend = PrefillDecodeBackend(
            model_version="meta-llama/Meta-Llama-3.1-70B-Instruct",
            batch_size=32,
            num_layers=80,
            max_seq_len=2048,
            cache_root=cache_path,
        )

    return model_backend, model_backend.formatter


def main():
    # -----------------------------------
    # configuration:
    # -----------------------------------
    # tasks = ["mmlu_econometrics"]
    # tasks = ["mmlu"]
    # tasks = ["ifeval"]
    # tasks = ["gpqa_main_cot_zeroshot"]
    tasks = ["ifeval", "gpqa_main_cot_zeroshot"]
    eval_output_fname = "eval_output"
    limit = None  # limit the number of samples per task
    log_samples = True  # log samples and outputs to file
    mock_model = False  # use random logits model for testing
    num_fewshot = None  # number of fewshot samples (task dependent)
    # -----------------------------------
    cache_path = os.environ.get("CACHE_ROOT")
    assert cache_path is not None, "CACHE_ROOT environment variable must be set"
    eval_output_fpath = os.path.join(cache_path, eval_output_fname)
    if "ifeval" in tasks:
        # download nltk punkt tokenizer if not available
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    evaluation_tracker = EvaluationTracker(output_path=eval_output_fpath)
    model_backend, tokenizer = get_model_backend(mock_model=mock_model)
    # pretrained must be the hugginface pretrained model name
    # see: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct
    pretrained = "meta-llama/Meta-Llama-3.1-70B-Instruct"
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
        evaluation_tracker.save_results_aggregated(results=results, samples=samples if log_samples else None)
        if log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

        # dumped = json.dumps(results, indent=2, default=handle_non_serializable, ensure_ascii=False)
        # print(dumped)


if __name__ == "__main__":
    main()
