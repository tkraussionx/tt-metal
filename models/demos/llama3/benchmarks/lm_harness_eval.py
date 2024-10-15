# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple
from tqdm import tqdm

import torch

from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.__main__ import cli_evaluate


from llama_models.models.llama3.api.datatypes import ImageMedia

from llama_models.models.llama3.reference_impl.generation import Llama

DEFAULT_IMAGE_TOKEN = "<|image|>"


@register_model("llama-cpu-reference")
class LLamaEvalWrapper(lmms):
    def __init__(
        self,
        ckpt_dir: str = "/proj_sw/user_dev/checkpoints/Llama3.2-11B-Vision-Instruct",
        tokenizer_path: str = "/proj_sw/user_dev/checkpoints/Llama3.2-11B-Vision-Instruct/tokenizer.model",
        max_seq_len=2048,
        max_batch_size=1,
        batch_size=1,
        device="cpu",
    ):
        lmms.__init__(self)

        self.model = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=None,
        )

        self.device = torch.device(device)

    def loglikelihood(self, requests: List[Instance]):
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:

            print("CONTEXTS", contexts)
            print("DOC_TO_TARGET", doc_to_target)
            print("DOC_TO_VISUAL", doc_to_visual)
            print("DOC_ID", doc_id)
            print("TASK", task)
            print("SPLIT", split)
            print("TASK DICT", self.task_dict[task][split][doc_id])
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            image_sizes = [[visual.size[0], visual.size[1]] for visual in visuals]
            print(visuals)
            if visuals:
                image = visuals #process_images(visuals, self._image_processor, self._config)
                if type(image) is list:
                    image = [_image.to(dtype=torch.float16, device=self.device) for _image in image]
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts
            labels = continuation


            print("Prompts Input: ", prompts_input)
            print("Labels: ", labels)
            print("Image: ", image)
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100
            with torch.inference_mode():
                dialog = [UserMessage(content=[ImageMedia(image=image), prompts_input])]
                greedy_tokens = self.model.chat_completion(dialog, max_gen_len=None, temperature=0.6, top_p=0.9)

            loss = outputs["loss"]
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests):
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            print("CONTEXTS", contexts)
            print("DOC_TO_TARGET", doc_to_target)
            print("DOC_TO_VISUAL", doc_to_visual)
            print("DOC_ID", doc_id)
            print("TASK", task)
            print("SPLIT", split)
            print("TASK DICT", self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            image_sizes = [[visual.size[0], visual.size[1]] for visual in visuals]
            if visuals:
                image = visuals
            else:
                image = None

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts
            with torch.inference_mode():
                dialog = [UserMessage(content=[ImageMedia(image=image), prompts_input])]
                greedy_tokens = self.model.chat_completion(dialog, max_gen_len=None, temperature=0.6, top_p=0.9)
            res.append(greedy_tokens)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError()


if __name__ == "__main__":
    cli_evaluate()
