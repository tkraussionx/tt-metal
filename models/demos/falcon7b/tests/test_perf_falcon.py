# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import numpy as np
from sklearn.metrics import top_k_accuracy_score

import tt_lib
from models.demos.falcon7b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.demos.falcon7b.tt.falcon_causallm import TtFalconCausalLM

# TODO: Remove this?
from models.demos.falcon7b.tt.falcon_common import (
    PytorchFalconCausalLM,
)

from models.demos.falcon7b.tt.model_config import (
    get_model_config,
    get_tt_cache_path,
)

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    disable_compilation_reports,
    is_e75,
    is_wormhole_b0,
)
from models.perf.perf_utils import prep_perf_report

from transformers import AutoTokenizer
import transformers
from datasets import load_dataset, load_from_disk
from langdetect import detect, LangDetectException
import os
import urllib.request
from typing import Dict, Sequence
import copy
from dataclasses import dataclass
from tqdm import tqdm


def set_random_seed(seed):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        os.environ[
            "CUBLAS_WORKSPACE_CONFIG"
        ] = ":4096:8"  # CuBLAS and CUDA >= 10.2. require this to use deterministic behaviour, see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility


def get_tokenizer(model_name, explicit_pad_token=False, hf_cache=None):
    # Get tokenizer from HF
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache)
    if explicit_pad_token:
        num_new_tokens = tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print(f"Num new tokens added : {num_new_tokens}")
    else:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.bos_token = tokenizer.eos_token
    return tokenizer


def extract_alpaca_dataset(example):
    ALPACA_PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: "
        ),
    }
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {"input": prompt_format.format(**example)}


def get_dataset(dataset_name, cached_dataset_dir=None, hf_cache=None):
    if dataset_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", cache_dir=hf_cache)
    elif dataset_name == "alpaca_eval":
        dataset = load_dataset("tatsu-lab/alpaca_eval", download_mode="force_redownload", cache_dir=hf_cache)
    elif dataset_name == "guanaco":
        dataset = load_dataset("timdettmers/openassistant-guanaco", cache_dir=hf_cache)
    elif dataset_name == "guanaco_en_sp_fr":
        ds_dir = os.path.join(cached_dataset_dir, "openassistant_guanaco_en_sp_fr")
        if os.path.exists(ds_dir):
            print(f"Loading prefiltered dataset from {ds_dir}")
            dataset = load_from_disk(ds_dir)
            print(f'Train: {len(dataset["train"])} samples')
            print(f'Test : {len(dataset["test"])} samples')
        else:
            print(f"Filtered guanaco_en_sp_fr dataset doesn't exist")
            dataset = load_dataset("timdettmers/openassistant-guanaco", cache_dir=hf_cache)
            languages = ["en", "es", "fr"]

            def check_lang(example):
                try:
                    return detect(example["text"]) in languages
                except LangDetectException:
                    return False

            orig_len_train = len(dataset["train"])
            orig_len_test = len(dataset["test"])
            dataset = dataset.filter(check_lang)
            print(f'Train: filtered {orig_len_train} -> {len(dataset["train"])} samples for languages: {languages}')
            print(f'Test : filtered {orig_len_test} -> {len(dataset["test"])} samples for languages: {languages}')
            dataset.save_to_disk(ds_dir)
            print(f"Saved prefiltered dataset to: {ds_dir}")

    elif dataset_name == "mmlu_eval":
        # Create the directory if it doesn't exist
        os.makedirs("mmlu", exist_ok=True)
        data_files = {"eval": "mmlu/five_shot_mmlu_val.json", "test": "mmlu/five_shot_mmlu_test.json"}
        url_map = {
            "eval": "https://raw.githubusercontent.com/artidoro/qlora/main/data/mmlu/five_shot_mmlu_val.json",
            "test": "https://raw.githubusercontent.com/artidoro/qlora/main/data/mmlu/five_shot_mmlu_test.json",
        }

        # Check and download JSON files if they don't exist
        for split, file_name in data_files.items():
            if not os.path.exists(file_name):
                print(f"'{file_name}' doesn't exist. Downloading...")
                try:
                    urllib.request.urlretrieve(url_map[split], file_name)
                    print(f"'{file_name}' downloaded successfully.")
                except Exception as e:
                    print(f"Error downloading '{file_name}': {e}")
        dataset = load_dataset("json", data_files=data_files, cache_dir=hf_cache)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    return dataset


def format_alpaca_eval_dataset(example):
    prompt_format = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n### Human: {instruction}\n### Assistant:"
    return {"input": prompt_format.format(**example)}


def format_dataset(dataset, dataset_name):
    if dataset_name == "alpaca":
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=["instruction"])
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names["train"] if col not in ["input", "output"]]
        )

    elif dataset_name == "alpaca_eval":
        dataset = dataset.map(format_alpaca_eval_dataset)

    elif dataset_name == "guanaco" or dataset_name == "guanaco_en_sp_fr":
        dataset = dataset.map(
            lambda x: {
                "input": "",
                "output": x["text"],
            }
        )
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names["train"] if col not in ["input", "output"]]
        )

    elif dataset_name == "mmlu_eval":
        # leave as is
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names["eval"] if col not in ["input", "output"]]
        )
    # Remove unused columns.

    return dataset


def preprocess_dataset(
    dataset,
    dataset_name,
    tokenizer,
    filter_longer_sequences=True,
    filter_longer_inputs=False,
    max_input_len=256,
    max_seq_len=512,
):
    def append_eos(text, dataset_name):
        if dataset_name == "guanaco" or dataset_name == "guanaco_en_sp_fr":
            """
            Add the eos_token after each response, so during inference the model can stop generating tokens after it completes its response. For example
            ### HUMAN:
            Hello
            ### Assistant:
            Hi, how are you?<eos_token>
            ### HUMAN:
            I'm fine.
            ### Assistant:
            How can I help you?<eos_token>
            """

            splitted_text = text.split("### Human:")
            for i in range(1, len(splitted_text)):
                splitted_text[i] = splitted_text[i] + tokenizer.eos_token
            processed_text = "### Human:".join(splitted_text)

        else:
            processed_text = text + tokenizer.eos_token

        return processed_text

    dataset = dataset.map(
        lambda x: {
            "output": append_eos(x["output"], dataset_name),
        }
    )

    def filter_long_sequences(sample):
        complete_sequence = sample["input"] + sample["output"]
        tokenized_complete_sequence = tokenizer.tokenize(complete_sequence)
        return len(tokenized_complete_sequence) <= max_seq_len

    def filter_long_inputs(sample, max_input_len=max_input_len):
        tokenized_input = tokenizer.tokenize(sample["input"])
        return len(tokenized_input) <= max_input_len

    if "eval" in dataset:
        orig_len_eval = len(dataset["eval"])
    if "train" in dataset:
        orig_len_train = len(dataset["train"])
    if filter_longer_sequences:
        dataset = dataset.filter(filter_long_sequences)
    if filter_longer_inputs:
        dataset = dataset.filter(filter_long_inputs)
    if "train" in dataset:
        print(f'Train: filtered {orig_len_train} -> {len(dataset["train"])} samples for max_seq_len: {max_seq_len}')
    if "eval" in dataset:
        print(f'Eval: filtered {orig_len_eval} -> {len(dataset["eval"])} samples for max_seq_len: {max_seq_len}')
    return dataset


@dataclass
class DataCollatorForCausalLM(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask")
        )

        return dict(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            hidden_states=None,
        )


def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    dataset_name,
    train=True,
    eval=True,
    max_train_samples=None,
    max_eval_samples=None,
    max_seq_len=512,
    data_seed=42,
    cached_dataset_dir=None,
    hf_cache=None,
    filter_longer_sequences=True,
    filter_longer_inputs=False,
    max_input_len=256,
    train_on_source=False,
) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    """
    raw_dataset = get_dataset(dataset_name, cached_dataset_dir, hf_cache)
    formatted_dataset = format_dataset(raw_dataset, dataset_name)
    dataset = preprocess_dataset(
        formatted_dataset,
        dataset_name,
        tokenizer,
        filter_longer_sequences,
        filter_longer_inputs,
        max_input_len,
        max_seq_len,
    )

    def tokenize(dataset):
        """Preprocess the data by tokenizing."""
        sources = dataset["input"]
        targets = dataset["output"]
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized = tokenizer(examples, truncation=True, padding="max_length", max_length=max_seq_len)
        sources_tokenized = tokenizer(sources, truncation=True, padding=False, max_length=max_seq_len)
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        # Don't train on source
        for label, source in zip(labels, sources_tokenized["input_ids"]):
            label[: len(source)] = [tokenizer.pad_token_id for _ in range(len(source))]
        return dict(input_ids=input_ids, labels=labels, attention_mask=examples_tokenized["attention_mask"])

    # Split train/eval, reduce size
    if eval:
        if "eval" in dataset:
            eval_dataset = dataset["eval"]
        elif "test" in dataset:
            eval_dataset = dataset["test"]
        else:
            print("Splitting train dataset in train and validation according to `90-10`")
            dataset = dataset["train"].train_test_split(test_size=0.10, shuffle=True, seed=data_seed)
            eval_dataset = dataset["test"]

        if max_eval_samples is not None and len(eval_dataset) > max_eval_samples:
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        # At the moment do not remove the columns
        # eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)
        eval_dataset = eval_dataset.map(tokenize, batched=True)

        print(f"Eval Dataset Length : {len(eval_dataset)}")

    if train:
        train_dataset = dataset["train"]
        if max_train_samples is not None and len(train_dataset) > max_train_samples:
            train_dataset = train_dataset.select(range(max_train_samples))

        if data_seed is not None:
            set_random_seed(data_seed)
            train_dataset = train_dataset.shuffle(seed=data_seed)
        train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)

        print(f"Train Dataset Length : {len(train_dataset)}")

    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset if train else None,
        eval_dataset=eval_dataset if eval else None,
        data_collator=data_collator,
    )


def dataloader_for_eval_metrics(
    tokenizer,
    dataset_name,
    dataset_split,
    batch_size,
    num_sequences=128,
    sequence_length=128,
    seed=42,
    prefiltered_dataset_dir=None,
):
    data_module = make_data_module(
        tokenizer,
        dataset_name,
        train=dataset_split == "train",
        eval=dataset_split == "eval",
        max_seq_len=sequence_length,
        data_seed=seed,
        cached_dataset_dir=prefiltered_dataset_dir,
        filter_longer_sequences=False,
        filter_longer_inputs=False,
    )

    dataset = data_module["train_dataset"] if dataset_split == "train" else data_module["eval_dataset"]
    dataset = dataset.filter(lambda x: tokenizer.eos_token_id not in x["input_ids"])
    assert num_sequences <= len(
        dataset
    ), f"num_sequences ({num_sequences}) must be less than or equal to the number of sequences in the dataset after filtering for sequences at least as long as {sequence_length} ({len(dataset)})"
    dataset = dataset.select(range(num_sequences))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_module["data_collator"], shuffle=False, drop_last=True
    )
    return dataloader


# TODO: Replace this with actual Falcon application-level tests
def run_test_FalconCausalLM_end_to_end(
    device,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    pcc,
    model_config,
    tt_cache_path,
    model_location_generator,
    expected_inference_time,
):
    # Clear global profiler state before starting measurements
    profiler.clear()

    model_name = model_location_generator(model_version, model_subdir="Falcon")

    profiler.start("hugging_face_model_setup")
    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()
    pytorch_FalconCausalLM = PytorchFalconCausalLM(hugging_face_reference_model, num_layers)
    profiler.end("hugging_face_model_setup")

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    base_url = ""
    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    use_cache = True

    if True:
        model_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        model_input = torch.stack([torch.arange(seq_len)] * batch).reshape(batch, seq_len)

    tokenizer = get_tokenizer("tiiuae/falcon-7b", explicit_pad_token=False)
    dataloader = dataloader_for_eval_metrics(
        tokenizer, "alpaca_eval", "eval", 1, batch, seq_len, 0, "/proj_sw/user_dev/falcon-ft/datasets"
    )
    inputs = []
    for sample in dataloader:
        inputs.append(sample["input_ids"])
    model_input = torch.cat(inputs)

    # Generate dummy kv_cache --------------------------------------------------------------
    if llm_mode == "prefill":
        kv_len = seq_len
        assert seq_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
        assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

        past_key_values = None
        tt_layer_past = ()
        k_cache = torch.zeros(batch, max_position_embeddings, head_dim).unsqueeze(1)
        v_cache = torch.zeros(batch, max_position_embeddings, head_dim).unsqueeze(1)
        for i in range(num_layers):
            tt_k_cache = torch2tt_tensor(k_cache, device)
            tt_v_cache = torch2tt_tensor(v_cache, device)
            tt_layer_past += ((tt_k_cache, tt_v_cache),)

    elif llm_mode == "decode":
        kv_len = kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert seq_len == 1, "For decode, seq_len must be 1!"

        past_key_values = ()
        tt_layer_past = ()
        for i in range(num_layers):
            k_cache = torch.rand(batch, 1, kv_cache_len, head_dim)
            v_cache = torch.rand(batch, 1, kv_cache_len, head_dim)
            past_key_values += ((k_cache, v_cache),)

            tt_k_cache = torch.zeros(batch, 1, max_position_embeddings, head_dim)
            tt_v_cache = torch.zeros(batch, 1, max_position_embeddings, head_dim)
            tt_k_cache[:, :, :kv_cache_len, :] = k_cache
            tt_v_cache[:, :, :kv_cache_len, :] = v_cache
            tt_k_cache = torch2tt_tensor(tt_k_cache, device)
            tt_v_cache = torch2tt_tensor(tt_v_cache, device)
            tt_layer_past += ((tt_k_cache, tt_v_cache),)

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

    # Prepare output -----------------------------------------------------------------------
    profiler.start("hugging_face_reference_model")
    pytorch_out, pytorch_layer_present = pytorch_FalconCausalLM(
        input_ids=model_input, past_key_values=past_key_values, use_cache=use_cache
    )
    profiler.end("hugging_face_reference_model")

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders

    profiler.start("TtFalcon_model_setup")
    tt_FalconCausalLM = TtFalconCausalLM(
        device,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    )
    profiler.end("TtFalcon_model_setup")

    profiler.start("processing_of_input")
    # TODO: Generate embeddings and attention_mask on device
    if llm_mode == "prefill":
        model_inputs = torch.split(model_input, 1)
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )
    elif llm_mode == "decode":
        tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )
    profiler.end("processing_of_input")

    # First run to fill compile cache ----------------------------------------------------
    logger.info(f"Running Falcon model once to fill caches -> disable profiler")
    profiler.disable()

    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run_with_compile", force_enable=True)
    if llm_mode == "prefill":
        tt_outs = []
        model_inputs = torch.split(model_input, 1)
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )
        for user_id in tqdm(range(batch)):
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_embeddings=tt_embeddings[user_id],
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask[user_id],
                user_id=user_id,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=use_cache,
            )
            tt_outs.append(tt_out)
        tt_out = tt_outs

    elif llm_mode == "decode":
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_embeddings=tt_embeddings,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
    tt_lib.device.Synchronize(device)
    profiler.end("first_model_run_with_compile", force_enable=True)
    del tt_out
    del tt_layer_past
    del tt_layer_present
    del tt_embeddings
    del tt_attention_mask

    # Second run for perf ----------------------------------------------------------------
    logger.info(f"Enable profiler and enable binary and compile cache")
    profiler.enable()
    enable_persistent_kernel_cache()
    if llm_mode == "prefill":
        tt_layer_past = ()
        for i in range(num_layers):
            tt_k_cache = torch2tt_tensor(k_cache, device)
            tt_v_cache = torch2tt_tensor(v_cache, device)
            tt_layer_past += ((tt_k_cache, tt_v_cache),)

    elif llm_mode == "decode":
        tt_layer_past = ()
        for i in range(num_layers):
            tt_k_cache = torch.zeros(batch, 1, max_position_embeddings, head_dim)
            tt_v_cache = torch.zeros(batch, 1, max_position_embeddings, head_dim)
            tt_k_cache[:, :, :kv_cache_len, :] = past_key_values[i][0]
            tt_v_cache[:, :, :kv_cache_len, :] = past_key_values[i][1]
            tt_k_cache = torch2tt_tensor(tt_k_cache, device)
            tt_v_cache = torch2tt_tensor(tt_v_cache, device)
            tt_layer_past += ((tt_k_cache, tt_v_cache),)

    if llm_mode == "prefill":
        model_inputs = torch.split(model_input, 1)
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )
    elif llm_mode == "decode":
        tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )

    profiler.start(f"model_run_for_inference")
    if llm_mode == "prefill":
        tt_outs = []
        model_inputs = torch.split(model_input, 1)
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )
        for user_id in tqdm(range(batch)):
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_embeddings=tt_embeddings[user_id],
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask[user_id],
                user_id=user_id,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=use_cache,
            )
            tt_outs.append(tt_out)

    elif llm_mode == "decode":
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_embeddings=tt_embeddings,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
    tt_lib.device.Synchronize(device)
    profiler.end(f"model_run_for_inference")

    if llm_mode == "prefill":
        tt_out = torch.vstack([tt2torch_tensor(tt_out).squeeze(1) for tt_out in tt_outs])
    elif llm_mode == "decode":
        tt_out = tt2torch_tensor(tt_out).squeeze(1)
        tt_out = tt_out.transpose(0, 1)

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output: {output_pcc}")

    reference_logits = pytorch_out.view(batch * seq_len, -1).float().detach().numpy()
    eval_logits = tt_out.view(batch * seq_len, -1).float().detach().numpy()
    reference_top1 = np.argmax(reference_logits, axis=-1)
    top1_acc = top_k_accuracy_score(reference_top1, eval_logits, k=1, labels=np.arange(eval_logits.shape[-1]))
    top5_acc = top_k_accuracy_score(reference_top1, eval_logits, k=5, labels=np.arange(eval_logits.shape[-1]))
    logger.info(f"Top-1 Accuracy: {top1_acc}")
    logger.info(f"Top-5 Accuracy: {top5_acc}")
    breakpoint()

    for i in range(num_layers):
        tt_layer_pres = (
            tt2torch_tensor(tt_layer_present[i][0]),
            tt2torch_tensor(tt_layer_present[i][1]),
        )
        if llm_mode == "prefill":
            pytorch_layer_pres = pytorch_layer_present[i]
            tt_layer_pres = (
                tt_layer_pres[0][:, :, :kv_len, :],
                tt_layer_pres[1][:, :, :kv_len, :],
            )
        elif llm_mode == "decode":
            pytorch_layer_pres = (
                pytorch_layer_present[i][0][:, :, kv_cache_len, :],
                pytorch_layer_present[i][1][:, :, kv_cache_len, :],
            )
            tt_layer_pres = (
                tt_layer_pres[0][:, :, kv_cache_len, :],
                tt_layer_pres[1][:, :, kv_cache_len, :],
            )

        does_pass2, output_pcc = comp_pcc(pytorch_layer_pres[0], tt_layer_pres[0], pcc)
        logger.info(f"K Cache Layer {i}: {output_pcc}")

        does_pass = does_pass and does_pass2

        does_pass2, output_pcc = comp_pcc(pytorch_layer_pres[1], tt_layer_pres[1], pcc)
        logger.info(f"V Cache Layer {i}: {output_pcc}")

        does_pass = does_pass and does_pass2

    profiler.print()

    comment = f"kv_cache_len={kv_cache_len}_seq_len={seq_len}_num_layers={num_layers}_config=L1-bf16"
    cpu_time = profiler.get("hugging_face_reference_model")
    first_iter_time = profiler.get("first_model_run_with_compile")
    second_iter_time = profiler.get("model_run_for_inference")
    expected_compile_time = 30
    prep_perf_report(
        model_name=f"Falcon_{llm_mode}_{comment}",
        batch_size=batch,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
        inference_time_cpu=cpu_time,
    )

    compile_time = first_iter_time - second_iter_time
    logger.info(f"falcon {comment} inference time: {second_iter_time}")
    logger.info(f"falcon {comment} compile time: {compile_time}")

    if does_pass:
        logger.info("Falcon PCC Check Passed!")
    else:
        logger.warning("Falcon PCC Check Failed!")
        if is_wormhole_b0():  # only assert for pcc on wormhole until grayskull pcc is fixed
            assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len, expected_inference_time",
    (
        ("prefill", 1, 128, 0, 0.30),
        ("prefill", 1, 256, 0, 0.44),
        ("prefill", 32, 256, 0, 0.44),
        ("prefill", 64, 256, 0, 0.44),
        ("decode", 32, 1, 128, 0.27),
        ("decode", 32, 1, 1024, 0.35),
        ("decode", 32, 1, 2047, 0.48),
    ),
    ids=[
        "prefill_seq128",
        "prefill_seq256",
        "prefill_seq256_batch32",
        "prefill_seq256_batch64",
        "decode_batch32",
        "decode_batch32_1024",
        "decode_batch32_2047",
    ],
)
@pytest.mark.parametrize(
    "num_layers, pcc",
    ((32, 0.89),),
    ids=["layers_32"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-L1",))
def test_perf_bare_metal(
    use_program_cache,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    expected_inference_time,
    num_layers,
    pcc,
    request,
    model_config_str,
    model_location_generator,
    device,
):
    if is_e75(device) and batch == 32:
        pytest.skip("Falcon batch 32 is not supported on E75")

    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    tt_lib.profiler.set_profiler_location(f"falcon-7b_{request.node.callspec.id}")

    run_test_FalconCausalLM_end_to_end(
        device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        num_layers,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
        expected_inference_time,
    )


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len, expected_inference_time",
    (
        ("prefill", 1, 128, 0, 0.4),
        ("decode", 32, 1, 128, 0.3),
        # ("prefill", 1, 256, 0, 0.40),
        # ("decode", 32, 1, 1024, 0.36),
        # ("decode", 32, 1, 2047, 0.47),
    ),
    ids=[
        "prefill_seq128",
        "decode_batch32",
    ],  # "prefill_seq256","decode_batch32_1024", "decode_batch32_2047"],
)
@pytest.mark.parametrize(
    "num_layers, pcc",
    ((32, 0.89),),
    ids=["layers_32"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-L1",))
def test_perf_virtual_machine(
    use_program_cache,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    expected_inference_time,
    num_layers,
    pcc,
    request,
    model_config_str,
    model_location_generator,
    device,
):
    if is_e75(device) and batch == 32:
        pytest.skip("Falcon batch 32 is not supported on E75")

    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    tt_lib.profiler.set_profiler_location(f"falcon-7b_{request.node.callspec.id}")

    run_test_FalconCausalLM_end_to_end(
        device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        num_layers,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
        expected_inference_time,
    )
