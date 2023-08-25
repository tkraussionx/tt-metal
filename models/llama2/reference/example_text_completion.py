# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from models.llama2.llama2_utils import model_location_generator

from models.llama2.reference.generation import Llama

import torch
device = torch.device("cpu")


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )


    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,

        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    llama2_path = str(model_location_generator("llama-2-13b", model_subdir="llama-2"))
    # llama2_tokenizer_path = str(model_location_generator("llama-2-13b", model_subdir="llama-2/tokenizer.model"))
    llama2_tokenizer_path = llama2_path + "/tokenizer.model"
    # print(llama2_path)
    # print(llama2_tokenizer_path)


    main(
    ckpt_dir = llama2_path,
    tokenizer_path= llama2_tokenizer_path,
    temperature= 0.6,
    top_p = 0.9,
    max_seq_len = 64,
    max_gen_len = 32,
    max_batch_size = 2,)
