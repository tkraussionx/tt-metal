# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
from functools import partial
import tt_lib
import torch
from loguru import logger
import time
from pathlib import Path
from transformers import AutoTokenizer
import os
from tqdm import tqdm

from models.demos.falcon7b.tt.falcon_causallm import TtFalconCausalLM
from models.demos.falcon7b.reference.hf_modeling_falcon import FalconConfig, FalconForCausalLM
from models.demos.falcon7b.tt.model_config import get_model_config, get_tt_cache_path, model_config_entries
from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler,
    torch2tt_tensor,
    tt2torch_tensor,
    nearest_32,
)

END_OF_TEXT = 11
SPACE = 204


# load from jason, return as a list
def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    for i in range(batch):
        in_prompt.append(user_input[i]["question"])
    return in_prompt


def post_process(logits, index):
    next_token_logits = logits[:, index, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    ids = next_tokens[:, None]
    return ids


def preprocess_and_validate_inputs(input_prompts, tokenizer, max_seq_len):
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        input_prompts, padding="max_length", max_length=max_seq_len, add_special_tokens=False, return_tensors="pt"
    )
    prefill_ids = tokenized_inputs["input_ids"]

    tokenized_inputs_nopad = tokenizer(
        input_prompts, padding=False, max_length=max_seq_len, add_special_tokens=False, return_tensors="pt"
    )

    num_users = len(tokenized_inputs_nopad["input_ids"])
    num_input_tokens = len(tokenized_inputs_nopad["input_ids"][0])
    for input_prompt in tokenized_inputs_nopad["input_ids"]:
        assert len(input_prompt) == num_input_tokens
    logger.info(f"# of users: {num_users}")
    logger.info(f"# of input tokens per user: {num_input_tokens}")

    prefill_ids = prefill_ids[:, : nearest_32(num_input_tokens)]  # only pad up to nearest 32, not max seq len

    return prefill_ids, num_users, num_input_tokens


def initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, device):
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    kv_cache = ()
    for _ in range(num_layers):
        k_cache = torch.zeros(batch_size, 1, max_seq_len, head_dim)
        v_cache = torch.zeros(batch_size, 1, max_seq_len, head_dim)
        tt_k_cache = torch2tt_tensor(k_cache, device)
        tt_v_cache = torch2tt_tensor(v_cache, device)
        kv_cache += ([(tt_k_cache, tt_v_cache)],)
    return kv_cache


def print_output_prompts(generated_ids, tokenizer, num_users_to_display=None):
    output_prompts = tokenizer.batch_decode(generated_ids.tolist())
    for user_id, output_prompt in enumerate(output_prompts[:num_users_to_display]):
        logger.info(f"Output for user {user_id}:\n{output_prompt}")


def run_falcon_demo_kv(
    user_input, model_version, batch_size, num_layers, max_seq_len, model_config, model_location_generator, device
):
    torch.manual_seed(0)

    # device.enable_program_cache()

    tt_cache_path = get_tt_cache_path(model_version)

    configuration = FalconConfig(**model_config_entries)

    profiler.start(f"loading_inputs")
    if len(user_input) == 1:
        input_prompts = user_input
    else:
        input_prompts = load_inputs(user_input, batch_size)

    profiler.end(f"loading_inputs")

    # State dict is needed for embeddings
    logger.info("Loading weights...")
    profiler.start(f"loading_weights")
    if (tt_cache_path == Path(f"models/demos/falcon7b/datasets/{model_version}")) and (
        len(os.listdir(f"models/demos/falcon7b/datasets/{model_version}")) < 260
    ):
        logger.info("Weights not found on machine; downloading weights...")
        model_name = model_location_generator(model_version, model_subdir="Falcon")
        hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        hugging_face_reference_model.eval()
        state_dict = hugging_face_reference_model.state_dict()
        torch.save(state_dict["transformer.word_embeddings.weight"], tt_cache_path / "embedding.pt")
    else:
        state_dict = None

    logger.info("Loading weights finished!")
    profiler.end(f"loading_weights")

    tt_lib.device.Synchronize(device)

    logger.info("Moving weights to device; might take some time...")
    profiler.start(f"moving_to_device")

    base_url = ""
    num_layers = 5
    tt_FalconCausalLM_singlelayer = TtFalconCausalLM(
        [device],
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_seq_len,
        model_config,
        tt_cache_path,
    )  # single layer only used for compile

    logger.info("Moved weights to device!")
    profiler.end(f"moving_to_device")

    tt_lib.device.Synchronize(device)

    logger.info("Tokenizing inputs...")
    profiler.start(f"tokenizing_inputs")

    tokenizer = AutoTokenizer.from_pretrained(model_version)
    prefill_ids, num_users, num_input_tokens = preprocess_and_validate_inputs(input_prompts, tokenizer, max_seq_len)

    profiler.end(f"tokenizing_inputs")

    embs, masks, logs = [], [], []
    preselfout3 = []
    attn_out3 = []
    h4h_out3 = []
    h4h_in3 = []
    h4h_weight3 = []
    h4h_in23 = []
    h4h_weight23 = []
    mlp_out3 = []
    attnadd_out3 = []
    decode_out3 = []
    layers_out = []
    lnf_out = []
    b4log_out = []
    log_out = []
    for i in range(20):
        logger.info("Initializing KV cache...")
        # profiler.start(f"initializing_KV_cache")
        kv_cache_singlelayer = initialize_kv_cache(
            configuration, num_layers, batch_size, max_seq_len, device
        )  # only used for compile
        # kv_cache = initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, device)
        # profiler.end(f"initializing_KV_cache")
        # profiler.disable()

        ### First prefill run with compile ###
        logger.info("Running 1st run prefill stage with compile...")
        # post_processor = partial(post_process)
        use_cache = True
        output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
        time_prefill_compile = 0
        for user_id in tqdm(range(num_users)):
            # time_prefill_compile_start = time.time()
            (
                tt_prefill_embeddings,
                tt_prefill_attention_mask,
            ) = tt_FalconCausalLM_singlelayer.model_preprocessing(
                "prefill", prefill_ids[user_id : user_id + 1], 0, num_input_tokens=num_input_tokens
            )
            assert tt_prefill_attention_mask is not None
            # embs.append(tt2torch_tensor(tt_prefill_embeddings[0]))
            # masks.append(tt2torch_tensor(tt_prefill_attention_mask[0]))

            tt_logits, kv_cache_singlelayer = tt_FalconCausalLM_singlelayer(
                input_embeddings=tt_prefill_embeddings,
                llm_mode="prefill",
                attention_mask=tt_prefill_attention_mask,
                user_id=user_id,
                layer_past=kv_cache_singlelayer,
                layer_past_len=0,
                use_cache=use_cache,
            )

            # preselfout3.append([tt_FalconCausalLM_singlelayer.layers[j].self_attn.preselfout for j in range(num_layers)])
            attn_out3.append([tt_FalconCausalLM_singlelayer.layers[j].attn_out for j in range(num_layers)])
            h4h_in3.append([tt_FalconCausalLM_singlelayer.layers[j].mlp.h4hin for j in range(num_layers)])
            # h4h_weight3.append([tt_FalconCausalLM_singlelayer.layers[j].mlp.h4hweight for j in range(num_layers)])
            h4h_in23.append([tt_FalconCausalLM_singlelayer.layers[j].mlp.h4hin2 for j in range(num_layers)])
            # h4h_weight23.append([tt_FalconCausalLM_singlelayer.layers[j].mlp.h4hweight2 for j in range(num_layers)])
            h4h_out3.append([tt_FalconCausalLM_singlelayer.layers[j].mlp.h4hout for j in range(num_layers)])
            mlp_out3.append([tt_FalconCausalLM_singlelayer.layers[j].mlp_out for j in range(num_layers)])
            attnadd_out3.append([tt_FalconCausalLM_singlelayer.layers[j].attnadd_out for j in range(num_layers)])
            decode_out3.append([tt_FalconCausalLM_singlelayer.layers[j].decode_out for j in range(num_layers)])
            layers_out.append(tt_FalconCausalLM_singlelayer.layers_out)
            lnf_out.append(tt_FalconCausalLM_singlelayer.lnf_out)
            b4log_out.append(tt_FalconCausalLM_singlelayer.b4log_out)
            log_out.append(tt_FalconCausalLM_singlelayer.log_out)

            for j in range(num_layers):
                # assert (
                #    preselfout3[0][j] - preselfout3[i][j]
                # ).abs().sum() == 0, f"preselfout{j} mismatch \n {preselfout3[0][j]} \n {preselfout3[i][j]}"
                assert (
                    attn_out3[0][j] - attn_out3[i][j]
                ).abs().sum() == 0, f"attnout{j} mismatch \n {attn_out3[0][j]} \n {attn_out3[i][j]}"
                # assert (
                #     h4h_weight3[0][j] - h4h_weight3[i][j]
                # ).abs().sum() == 0, f"h4hweight{j} mismatch \n {h4h_weight3[0][j]} \n {h4h_weight3[i][j]}"
                assert (
                    h4h_in3[0][j] - h4h_in3[i][j]
                ).abs().sum() == 0, f"h4hin{j} mismatch \n {h4h_in3[0][j]} \n {h4h_in3[i][j]}"
                if (h4h_out3[0][j] - h4h_out3[i][j]).abs().sum() != 0:
                    print(f"h4h {j} mismatch")
                    print((h4h_in23[0][j] - h4h_in23[i][j]).abs().sum() != 0)
                    # print((h4h_weight23[0][j] - h4h_weight23[i][j]).abs().sum() != 0)
                    breakpoint()
                assert (
                    h4h_out3[0][j] - h4h_out3[i][j]
                ).abs().sum() == 0, f"h4h{j} mismatch \n {h4h_out3[0][j]} \n {h4h_out3[i][j]}"
                assert (
                    mlp_out3[0][j] - mlp_out3[i][j]
                ).abs().sum() == 0, f"mlp{j} mismatch \n {mlp_out3[0][j]} \n {mlp_out3[i][j]}"
                assert (
                    attnadd_out3[0][j] - attnadd_out3[i][j]
                ).abs().sum() == 0, f"attadd{j} mismatch \n {attnadd_out3[0][j]} \n {attnadd_out3[i][j]}"
                assert (
                    decode_out3[0][j] - decode_out3[i][j]
                ).abs().sum() == 0, f"decodeout{j} mismatch \n {decode_out3[0][j]} \n {decode_out3[i][j]}"
            assert (
                layers_out[0] - layers_out[i]
            ).abs().sum() == 0, f"layers_out mismatch \n {layers_out[0]} \n {layers_out[i]}"
            assert (lnf_out[0] - lnf_out[i]).abs().sum() == 0, f"lnf mismatch \n {lnf_out[0]} \n {lnf_out[i]}"
            assert (b4log_out[0] - b4log_out[i]).abs().sum() == 0, f"b4log mismatch \n {b4log_out[0]} \n {b4log_out[i]}"
            assert (log_out[0] - log_out[i]).abs().sum() == 0, f"log mismatch \n {log_out[0]} \n {log_out[i]}"

            # time_prefill_compile_end = time.time()
            # time_prefill_compile += time_prefill_compile_end - time_prefill_compile_start

            tt_prefill_embeddings[0].deallocate()
            if tt_prefill_attention_mask is not None:
                tt_prefill_attention_mask[0].deallocate()

            logits = tt2torch_tensor(tt_logits[0]).squeeze(1)
            tt_logits[0].deallocate()
            logs.append(logits)
            # assert (embs[0] - embs[i]).abs().sum() == 0, f"embs mismatch \n {embs[0]} \n {embs[i]}"
            # assert (masks[0] - masks[i]).abs().sum() == 0, f"masks mismatch \n {masks[0]} \n {masks[i]}"
            assert (logs[0] - logs[i]).abs().sum() == 0, f"logits mismatch \n {logs[0]} \n {logs[i]}"

        # user_output_ids = post_processor(logits=logits, index=num_input_tokens - 1)
        # output_ids[user_id] = user_output_ids

    return

    generated_ids = torch.concat((prefill_ids[..., :num_input_tokens], output_ids), dim=1)

    tt_lib.device.Synchronize(device)
    logger.info("Finished 1st run prefill stage with compile!")

    # print(output_ids)

    return

    ### First run decode stage with compile ###
    logger.info("Running 1st run decode stage with compile...")
    decode_ids = torch.zeros(batch_size, 1, dtype=torch.int64)

    for user_id, output_id in enumerate(output_ids):
        decode_ids[user_id] = output_id

    kv_cache_len = num_input_tokens  # This will increment by one after each decode
    prompt_is_done = [False for _ in range(num_users)]

    # decode_ids_tmp = [13, 40, 2070, 2070, 94, 193, 9677, 2070, 2070, 2070, 2070, 23, 2070, 2070, 2070, 2070, 2070, 2070, 3078, 2070, 2070, 273, 9677, 57513, 299, 13, 2070, 2070, 2070, 23, 2070, 2070]
    # for i in range(batch_size):
    #    decode_ids[i, 0] = decode_ids_tmp[i]

    time_decode_compile = 0
    for output_token_index in tqdm(range(max_seq_len - num_input_tokens)):
        # print(decode_ids)
        # print(decode_ids.squeeze(1).tolist())
        # print(decode_ids.min())
        # print(decode_ids.max())
        time_decode_compile_start = time.time()
        (
            tt_decode_embeddings,
            tt_decode_attention_mask,
        ) = tt_FalconCausalLM_singlelayer.model_preprocessing(
            "decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1
        )
        # print(tt_decode_embeddings)
        # print(tt2torch_tensor(tt_decode_embeddings[0]).min())
        # print(tt2torch_tensor(tt_decode_embeddings[0]).max())
        assert tt_decode_attention_mask is not None

        # logger.info("starting forward pass")
        tt_logits, kv_cache_singlelayer = tt_FalconCausalLM_singlelayer(
            input_embeddings=tt_decode_embeddings,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=kv_cache_singlelayer,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        tt_lib.device.Synchronize(device)
        # logger.info("finished forward pass")
        time_decode_compile_end = time.time()
        time_decode_compile += time_decode_compile_end - time_decode_compile_start

        tt_decode_embeddings[0].deallocate()
        if tt_decode_attention_mask is not None:
            tt_decode_attention_mask[0].deallocate()

        logits = tt2torch_tensor(tt_logits[0]).squeeze(1)
        tt_logits[0].deallocate()

        next_tokens = torch.argmax(logits, dim=-1).reshape(batch_size, 1)
        decode_ids = next_tokens
        # decode_ids_tmp = [13, 40, 2070, 2070, 94, 193, 9677, 2070, 2070, 2070, 2070, 23, 2070, 2070, 2070, 2070, 2070, 2070, 3078, 2070, 2070, 273, 9677, 57513, 299, 13, 2070, 2070, 2070, 23, 2070, 2070]
        # for i in range(batch_size):
        #  decode_ids[i, 0] = decode_ids_tmp[i]
        # time.sleep(0.001)

        # decode_ids = post_processor(logits=logits, index=...).reshape(batch_size, 1)
        # generated_ids = torch.concat((generated_ids, decode_ids[:num_users]), dim=1)
        kv_cache_len += 1

    logger.info("Finished 1st run decode stage with compile!")
    tt_lib.device.Synchronize(device)

    return

    del user_output_ids
    del output_ids
    del logits
    del tt_logits
    del tt_prefill_embeddings
    del tt_prefill_attention_mask
    del generated_ids
    del decode_ids
    del tt_decode_embeddings
    del tt_FalconCausalLM_singlelayer
    del kv_cache_singlelayer

    import gc

    gc.collect()
    logger.info("deleted outputs")

    tt_FalconCausalLM = TtFalconCausalLM(
        [device],
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_seq_len,
        model_config,
        tt_cache_path,
    )

    logger.info("loaded model")

    ### Second prefill run without compile ###
    profiler.enable()
    enable_persistent_kernel_cache()

    logger.info("enabled cache")

    post_processor = partial(post_process)
    use_cache = True
    output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
    logger.info("Running inference prefill stage...")
    time_prefill_inference = 0
    for user_id in tqdm(range(num_users)):
        time_prefill_inference_start = time.time()
        (
            tt_prefill_embeddings,
            tt_prefill_attention_mask,
        ) = tt_FalconCausalLM.model_preprocessing(
            "prefill", prefill_ids[user_id : user_id + 1], 0, num_input_tokens=num_input_tokens
        )
        assert tt_prefill_attention_mask is not None

        tt_logits, kv_cache = tt_FalconCausalLM(
            input_embeddings=tt_prefill_embeddings,
            llm_mode="prefill",
            attention_mask=tt_prefill_attention_mask,
            user_id=user_id,
            layer_past=kv_cache,
            layer_past_len=0,
            use_cache=use_cache,
        )
        tt_lib.device.Synchronize(device)
        time_prefill_inference_end = time.time()
        time_prefill_inference += time_prefill_inference_end - time_prefill_inference_start

        tt_prefill_embeddings[0].deallocate()
        if tt_prefill_attention_mask is not None:
            tt_prefill_attention_mask[0].deallocate()

        logits = tt2torch_tensor(tt_logits[0]).squeeze(1)
        tt_logits[0].deallocate()

        user_output_ids = post_processor(logits=logits, index=num_input_tokens - 1)
        output_ids[user_id] = user_output_ids

    logger.info("Finished inference prefill stage!")

    generated_ids = torch.concat((prefill_ids[..., :num_input_tokens], output_ids), dim=1)

    profiler.disable()

    ### Inference run decode ###
    logger.info("Running inference decode stage...")

    decode_ids = torch.zeros(batch_size, 1, dtype=torch.int64)
    for user_id, output_id in enumerate(output_ids):
        decode_ids[user_id] = output_id

    kv_cache_len = num_input_tokens  # This will increment by one after each decode
    prompt_is_done = [False for _ in range(num_users)]

    time_decode_inference = 0
    for output_token_index in range(max_seq_len - num_input_tokens):
        time_decode_inference_start = time.time()
        (
            tt_decode_embeddings,
            tt_decode_attention_mask,
        ) = tt_FalconCausalLM.model_preprocessing("decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1)
        assert tt_decode_attention_mask is not None

        tt_logits, kv_cache = tt_FalconCausalLM(
            input_embeddings=tt_decode_embeddings,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=kv_cache,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        tt_lib.device.Synchronize(device)
        time_decode_inference_end = time.time()
        time_decode_inference += time_decode_inference_end - time_decode_inference_start

        tt_decode_embeddings[0].deallocate()
        if tt_decode_attention_mask is not None:
            tt_decode_attention_mask[0].deallocate()

        logits = tt2torch_tensor(tt_logits[0]).squeeze(1)
        tt_logits[0].deallocate()

        decode_ids = post_processor(logits=logits, index=...).reshape(batch_size, 1)

        for user_id, user_decode_id in enumerate(decode_ids[:num_users]):
            if user_decode_id == END_OF_TEXT:
                prompt_is_done[user_id] = True
            if prompt_is_done[user_id]:
                decode_ids[user_id] = SPACE

        if all(prompt_is_done):
            break

        generated_ids = torch.concat((generated_ids, decode_ids[:num_users]), dim=1)
        kv_cache_len += 1

        # TODO: Remove if we don't want to print per generated token
        os.system("clear")
        print_output_prompts(generated_ids, tokenizer)

    logger.info("Finished inference decode stage!")
    num_tokens_generated_decode = batch_size * (output_token_index + 1)
    logger.info(f"Total number of tokens generated in decode: {num_tokens_generated_decode}")

    print_output_prompts(generated_ids, tokenizer)

    device.disable_and_clear_program_cache()

    generated_text = tokenizer.batch_decode(generated_ids.tolist())

    measurements = {
        "preprocessing": profiler.get("tokenizing_inputs"),
        "loading_weights": profiler.get("loading_weights"),
        "moving_to_device": profiler.get("moving_to_device"),
        "initializing_KV_cache": profiler.get("initializing_KV_cache"),
        "compile_prefill": time_prefill_compile,
        "compile_decode": time_decode_compile,
        "compile_total": time_prefill_compile + time_decode_compile,
        "inference_prefill": time_prefill_inference,
        "inference_decode": time_decode_inference,
        "inference_total": time_prefill_inference + time_decode_inference,
        "inference_throughput_prefill": num_users / time_prefill_inference,
        "inference_throughput_decode": num_tokens_generated_decode / time_decode_inference,
    }

    logger.info(f"pre processing: {round(measurements['preprocessing'], 5)} s")
    logger.info(f"loading weights (+downloading if not on machine): {round(measurements['loading_weights'], 5)} s")
    logger.info(
        f"conversion to TT (if downloaded) and moving weights to device: {round(measurements['moving_to_device'], 5)} s"
    )
    logger.info(f"initializing KV cache: {round(measurements['initializing_KV_cache'], 5)} s")
    logger.info(f"prefill compile time (single layer): {round(measurements['compile_prefill'],5)} s")
    logger.info(f"decode compile time (single layer): {round(measurements['compile_decode'], 5)} s")
    logger.info(f"total compile time (single layer): {round(measurements['compile_total'], 5)} s")
    logger.info(f"prefill inference time: {round(measurements['inference_prefill'], 5)} s")
    logger.info(f"decode inference time: {round(measurements['inference_decode'], 5)} s")
    logger.info(f"total inference time: {round(measurements['inference_total'], 5)} s")
    logger.info(f"inference throughput prefill: {round(measurements['inference_throughput_prefill'], 5)} users/s")
    logger.info(
        f"inference throughput prefill | seq_len={prefill_ids.shape[1]} : {round(measurements['inference_throughput_prefill']*prefill_ids.shape[1], 5)} tok/s"
    )
    logger.info(f"inference throughput decode: {round(measurements['inference_throughput_decode'], 5)} tok/s")
    logger.info(
        f"inference throughput decode (per user): {round(measurements['inference_throughput_decode']/batch_size, 5)} tok/s/user"
    )

    return generated_text, measurements


def test_demo(
    user_input,
    model_location_generator,
    device,
    use_program_cache,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_falcon_demo_kv(
        user_input=user_input,
        model_version="tiiuae/falcon-7b-instruct",
        batch_size=32,
        num_layers=32,
        max_seq_len=1024,
        model_config=get_model_config("BFLOAT16-DRAM"),
        model_location_generator=model_location_generator,
        device=device,
    )
