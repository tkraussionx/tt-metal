from unittest.mock import patch

import torch

import lm_eval.evaluator as evaluator
import lm_eval.tasks as tasks
from lm_eval.evaluator_utils import print_writeout

from models.demos.t3000.llama2_70b.reference.llama.llama.tokenizer3 import Tokenizer3
from models.demos.t3000.llama2_70b.reference.llama.llama.tokenizer import Tokenizer

from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
)
from models.demos.utils.tenstorrent_lm import TenstorrentLM
from models.demos.t3000.llama3_70b.demo.prefill_decode_backend import PrefillDecodeBackend, build_generator

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
                "models.demos.t3000.llama3_70b.demo.prefill_decode_backend.build_generator", new=mock_build_generator
            ):
                model_backend = PrefillDecodeBackend(
                    batch_size=32,
                    num_layers=80,
                    max_seq_len=2048,
                    n_devices=8,
                    model_config=model_config,
                    ckpt_dir=ckpt_dir,
                    tokenizer_path=tokenizer_path,
                    cache_path=cache_path,
                )
    else:
        model_backend = PrefillDecodeBackend(
            batch_size=32,
            num_layers=80,
            max_seq_len=2048,
            n_devices=8,
            model_config=model_config,
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            cache_path=cache_path,
        )

    return model_backend, model_backend.formatter


def main():
    model_backend, tokenizer = get_model_backend(mock_model=True)
    task = "mmlu_high_school_statistics"
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
    """
    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str, dict]
        String or dict arguments for each model class, see LM.create_from_arg_string and LM.create_from_arg_object.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param batch_size: int or str, optional
        Batch size for model
    
    """
    use_cache = "/home/user/cache_root/lm_eval_cache"
    eval_output = evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=0,
        limit=10,
        bootstrap_iters=0,
        batch_size=32,
    )
    print(eval_output["results"])


if __name__ == "__main__":
    main()
