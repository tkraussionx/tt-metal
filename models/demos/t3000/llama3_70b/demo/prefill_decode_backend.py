import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Set

import torch

import ttnn
import tt_lib as ttl

from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from transformers.generation.utils import top_k_top_p_filtering
from models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration
from models.demos.t3000.llama2_70b.tt.llama_common import load_llama_state_dict
from models.demos.t3000.llama2_70b.reference.llama.llama.tokenizer3 import ChatFormat, Message
from models.demos.t3000.llama2_70b.tt.llama_common import (
    check_device_mesh,
    string_similarity_score,
)

from models.demos.t3000.llama2_70b.demo.demo import (
    ModelArgs,
    TTArgs,
    DataArgs,
    DemoArgs,
    construct_arg,
    build_generator,
    get_sampling_func,
    initialize_inputs,
    # prepare_next_input,
    top_pk_logits_efficient,
)
from conftest import get_dispatch_core_type

from inference_config import inference_config
from inference_logger import get_logger

logger = get_logger(__name__)
logger.info(f"importing {__name__}")


def get_t3k_device_mesh(num_devices_requested):
    logger.info("get_t3k_device_mesh ...")
    assert ttnn.get_num_devices() == 8
    device_ids = [0, 4, 5, 1, 2, 6, 7, 3]
    # device_params is empty dict in llama3 70B demo pytest execution
    device_params = {}
    device_mesh = ttnn.open_device_mesh(
        ttnn.DeviceGrid(1, num_devices_requested), device_ids[:num_devices_requested], dispatch_core_type=get_dispatch_core_type(), **device_params
    )
    logger.info(f"multidevice with {device_mesh.get_num_devices()} devices is created")
    try:
        yield device_mesh
    finally:
        logger.info("closing t3k device mesh ...")
        for device in device_mesh.get_devices():
            ttl.device.DumpDeviceProfiler(device)
        ttnn.close_device_mesh(device_mesh)
        del device_mesh


class UserInfo:
    def __init__(self, user_id: str, prompt_tokens: List[int], stop_tokens: Set[int], eos_token_id: int):
        self.user_id = user_id
        self.prompt_tokens = prompt_tokens
        self.num_prefill_tokens = len(prompt_tokens)
        self.num_tokens_decoded = 0
        self.num_tokens_prefilled = 0
        self.prefill_complete = False
        self.generated_tokens = []
        self.generation_completed = False
        # self.prefill_complete = False
        # self.generation_params = params
        # self.max_tokens = params["max_tokens"]
        # self.return_prompt = params["return_prompt"]
        # this may change for each tokenizer
        self.eos_token_id = eos_token_id
        self.stop_tokens = stop_tokens
        # stop sequence is multitoken stop condition
        self.stop_sequence = None
        # if params.get("stop_sequence"):
        #     self.stop_sequence = tokenizer.encode(params.get("stop_sequence"), bos=False, eos=False)
        # strip eos token from prompt
        # self.prompt_tokens = [tok for tok in self.prompt_tokens if tok not in self.stop_tokens]


class PrefillDecodeBackend:
    def __init__(
        self,
        batch_size,
        num_layers,
        max_seq_len,
        n_devices,
        model_config,
        ckpt_dir,
        tokenizer_path,
        cache_path,
        verbose=False,
    ) -> None:
        """
        Initialize pybuda model and all infracstructures to continuously run decode
        Maintain a cur_prompts for decode.
        """

        # inputs to model
        self.decode_ids = None
        self.prefill_ids = None
        # backend status
        self.time_last_status = time.time()
        self.update_period = 1  # status message period in seconds
        self.verbose = verbose  # enable conditional debug logging
        # new init:
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.n_devices = n_devices
        self.max_users = batch_size
        self.users = [None for _ in range(self.max_users)]
        self.num_users = None
        # default greedy
        self.default_top_p = 1.0
        self.default_top_k = 1
        self.default_temperature = 1.0
        self.sampling_func = get_sampling_func(self.default_top_k, self.default_top_p, self.default_temperature)
        # builtin basic profiler
        self.timestamps_start = {}
        self.timestamps_stop = {}
        self.enable_profile_logging = False
        self.batch_counter = 0
        self.forward_counter = 0
        self.prev_forward_counter = 0
        # for lm-evaluation-harness HFLM compatability
        self.device = "tt"
        self.config = {}
        # initialization
        self.model_config = model_config
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.cache_path = cache_path
        self.model = None
        self.tokenizer = None
        self.t3k_device_mesh = None
        self.chat = True
        self.skip_model_load = False
        self.decode_only = False
        self.init_model()

    def get_users(self):
        return [u for u in self.users if u]

    def get_user_param(self, param):
        return [user.generation_params[param] if user is not None else None for user in self.users]

    def tok_encode(self, string: str, add_special_tokens=True, **kwargs) -> List[int]:
        if self.chat and add_special_tokens:
            # encode as a single turn of dialog
            messages = [Message(role="user", content=string)]
            return self.formatter.encode_dialog_prompt(messages)
        else:
            return self.tokenizer.encode(string, bos=add_special_tokens, eos=False)

    def timer_start(self, name):
        self.timestamps_start[name] = time.time()

    def timer_stop(self, name, log=False):
        if name in self.timestamps_start.keys():
            self.timestamps_stop[name] = time.time()
            timedelta = self.timestamps_stop[name] - self.timestamps_start[name]
            if log or self.enable_profile_logging:
                print(f"timedelta: {name}: {timedelta} seconds")
                logger.info(f"timedelta: {name}: {timedelta} seconds")

    def teardown(self):
        logger.info("teardown ...")
        if self.t3k_device_mesh is not None:
            self.t3k_device_mesh.close()

    def init_tt_metal_device(self):
        logger.info("init_tt_metal_device ...")
        t3k_device_mesh = get_t3k_device_mesh(num_devices_requested=self.n_devices)
        for i in t3k_device_mesh.get_device_ids():
            device = t3k_device_mesh.get_device(i)
            device.enable_async(True)
            device.enable_program_cache()
        self.t3k_device_mesh = t3k_device_mesh
        check_device_mesh(self.t3k_device_mesh, self.model_config)
        logger.info("init_tt_metal_device finished.")

    def init_model(self):
        # set up variables for model init
        n_devices = inference_config.n_devices
        logger.info("init_model ...")
        # logger.info("todo ckpt vars ")

        self.init_tt_metal_device()

        # set unused vars to None to obviously break any code using them
        args = construct_arg(
            implementation="tt",
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            skip_model_load=self.skip_model_load,
            num_layers=self.num_layers,
            num_tokens=None,
            prompts_file=None,
            output_at_end=None,
            top_p=None,
            top_k=None,
            temperature=None,
            chat=self.chat,
            device_mesh=self.t3k_device_mesh,
            n_devices=self.n_devices,
            cache_path=self.cache_path,
            decode_only=self.decode_only,
            ground_truth=False,
        )
        model_args = args.model
        tt_args = args.tt

        generator = build_generator(model_args, tt_args)
        self.model = generator.model
        self.tokenizer = generator.tokenizer
        self.formatter = ChatFormat(self.tokenizer)

    def _get_user_by_id(self, user_id):
        for user in self.users:
            if user is not None and user.user_id == user_id:
                return user
        return None

    def _find_free_user_slot(self):
        """return the index of the first free user slot"""
        for i, user in enumerate(self.users):
            if user is None:
                return i

    def batch_stats(self):
        tokens_generated = self.forward_counter - self.prev_forward_counter
        batch_duration = time.time() - self.batch_start_time
        tps = tokens_generated / batch_duration
        logger.info(
            f"batch_counter:={self.batch_counter}, forward_counter:={self.forward_counter}, tokens_generated:={tokens_generated}, tps:={tps:.4f} tokens/sec (32 users)"
        )
        self.prev_forward_counter = self.forward_counter
        self.batch_start_time = time.time()

    def add_users_from_prompts(self, context_enc_list):
        assert len(context_enc_list) <= self.max_users
        # reset users
        for idx in range(len(self.get_users())):
            # reset memory
            self.decode_ids[idx, 0] = 0
            self.users[idx] = None

        for idx in range(len(context_enc_list)):
            self.users[idx] = UserInfo(
                user_id=1,
                prompt_tokens=context_enc_list[idx],
                stop_tokens=self.tokenizer.stop_tokens,
                eos_token_id=self.tokenizer.eos_id,
            )

    def batch_preprocessing(self):
        # TODO: investigate changing when continous batching supported
        # note: the cur_pos index if shared between all users
        # this may change for the continuous batching implementation
        self.prepare_batch_inputs()
        self.cur_pos = 1
        self.prev_pos = 0
        self.batch_counter += 1

    def prepare_batch_inputs(self):
        self.num_users = len(self.get_users())
        assert self.num_users <= self.max_users
        input_prompts = [user_info.prompt_tokens for user_info in self.get_users()]
        # initialize_inputs:
        # pad inputs, empty users get pad id
        prefill_tokens, input_text_mask, _ = initialize_inputs(
            tokenizer=self.tokenizer,
            prompt_tokens=input_prompts,
            bsz=len(input_prompts),
            total_len=self.max_seq_len,
        )
        # where does intput_text_mask get used?
        self.input_text_mask = input_text_mask
        self.prefill_ids = prefill_tokens
        # decode_ids are padded to batch_size
        decode_ids = torch.full((self.batch_size, 1), self.tokenizer.pad_id, dtype=torch.long, device="cpu")
        decode_ids[:self.num_users, :1] = prefill_tokens[:, :1].clone()
        self.decode_ids = decode_ids

    def prefill(self):
        if self.prefill_ids is None:
            return
        batch_size, seq_len = self.prefill_ids.shape
        # runs prefill for full batch
        if seq_len > 1:
            # prefill is defined in TtLlamaModelForGeneration by sending seq_len > 1
            # seq_len is tokens.shape[1]
            prefill_logits = self.model.forward(self.prefill_ids, self.prev_pos)
            self.num_tokens_prefilled = seq_len
        else:
            self.num_tokens_prefilled = 0
        self.prefill_complete = True
        for user in self.get_users():
            user.prefill_complete = True
        self.prefill_ids = None

    def get_logits(self):
        self.forward_counter += 1
        logits = self.model.forward(self.decode_ids, self.prev_pos)
        # remove pad users
        logits = logits[: self.num_users, -1, :]
        return logits

    def get_next_tokens(self):
        self.decode_ids = self.sampling_func(self.get_logits).reshape(self.batch_size, 1)
        self.update_user_tokens()

    def get_token_text(self):
        # Sentencepiece tokenizer doesn't handle spaces per token, must decode full text
        # then push new chars to output queue
        for user_info, user_decode_id in zip(self.users, self.decode_ids):
            if user_info is None:
                continue
            elif user_info.num_tokens_generated < 1:
                # still prefilling via decode
                continue
            last_token = user_decode_id.item()
            user_info.generated_tokens.append(last_token)
            full_text = self.tokenizer.decode(user_info.generated_tokens)
            return_text = full_text[user_info.num_generated_chars :]
            user_info.num_generated_chars = len(full_text)

    def decode(self):
        """
        self.cur_pos is the batch level position
        each user has a generation_pos
        """
        self.forward_counter += 1
        self.timer_stop("all_but_decode")
        self.timer_start("decode")
        logits = self.model.forward(self.decode_ids, self.prev_pos, decode_only=self.decode_only)
        self.timer_stop("decode")
        self.timer_start("token_selection")
        next_tokens = batch_top_pk_logits_efficient(
            logits,
            top_ps=self.get_user_param("top_p"),
            top_ks=self.get_user_param("top_k"),
            temperatures=self.get_user_param("temperature"),
        ).reshape(self.batch_size, 1)
        self.timer_stop("token_selection")
        self.timer_start("update_user_tokens")
        self.decode_ids = next_tokens
        self.update_user_tokens()
        self.cur_pos += 1
        self.prev_pos += 1
        self.timer_start("all_but_decode")

    def update_user_tokens(self):
        for idx, (user_info, user_decode_id) in enumerate(
            zip(self.users, self.decode_ids.reshape(self.batch_size).tolist())
        ):
            if user_info is None:
                continue
            user_info.num_tokens_generated += 1

            if user_decode_id in user_info.stop_tokens:
                user_info.decode_complete = True
            elif user_info.num_tokens_generated > user_info.max_tokens:
                user_info.decode_complete = True
            elif user_info.stop_sequence is not None:
                last_n_tokens = user_info.generated_tokens[-(len(user_info.stop_sequence) - 1) :]
                last_n_tokens.append(user_decode_id)
                if last_n_tokens == user_info.stop_sequence:
                    user_info.decode_complete = True
            if user_info.decode_complete:
                self.decode_ids[idx][0] = user_info.eos_token_id
