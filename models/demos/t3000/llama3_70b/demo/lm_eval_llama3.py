from dataclasses import dataclass

import lm_eval.evaluator as evaluator
import lm_eval.tasks as tasks

from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
)
from models.demos.utils.tenstorrent_lm import TenstorrentLM
from models.demos.t3000.llama3_70b.demo.prefill_decode_backend import PrefillDecodeBackend

"""
This script uses the as a library (see https://github.com/EleutherAI/lm-evaluation-harness)
TenstorrentLM

"""


def get_dummy_backend():
    @dataclass
    class DummyBackend:
        model_backend: object
        tokenizer: object

    return DummyBackend(object, object), object


def get_model_backend():
    llama_version = "llama3"
    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
    )
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
    model_backend, tokenizer = get_model_backend()
    # model_backend, tokenizer = get_dummy_backend()
    # model_id = "llama3-70b"
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
        # use_cache=use_cache,
        num_fewshot=0,
        limit=10,
        bootstrap_iters=0,
        batch_size=32,
    )
    breakpoint()


if __name__ == "__main__":
    main()
