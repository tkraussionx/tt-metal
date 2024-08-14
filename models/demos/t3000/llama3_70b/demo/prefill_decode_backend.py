import os
import time
import traceback
import threading
from multiprocessing import Queue
from functools import partial
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F

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
    prepare_next_input,
    top_pk_logits_efficient,
)

# from model_weights_handler import get_model_weights_and_tt_cache_paths
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
        ttnn.DeviceGrid(1, num_devices_requested), device_ids[:num_devices_requested], **device_params
    )
    logger.info(f"multidevice with {device_mesh.get_num_devices()} devices is created")
    return device_mesh


def close_devices(device_mesh):
    logger.info("close_devices ...")
    for device in device_mesh.get_devices():
        ttl.device.DumpDeviceProfiler(device)
    ttnn.close_device_mesh(device_mesh)
    del device_mesh


class UserInfo:
    def __init__(self, user_id, prompt_tokens, position_id, params, tokenizer, formatter=None):
        self.user_id = user_id
        self.prompt_tokens = prompt_tokens
        self.position_id = position_id
        self.num_tokens_generated = 0
        self.generated_tokens = []
        self.num_generated_chars = 0
        self.num_tokens_prefilled = 0
        self.generation_params = params
        self.max_tokens = params["max_tokens"]
        self.return_prompt = params["return_prompt"]
        self.cancel = False
        self.prefill_complete = False
        self.decode_complete = False
        self.sent_stop = False
        self.chat_format = True
        # this may change for each tokenizer
        self.eos_token_id = tokenizer.eos_id
        self.stop_tokens = tokenizer.stop_tokens
        self.stop_sequence = None
        if params.get("stop_sequence"):
            self.stop_sequence = tokenizer.encode(params.get("stop_sequence"), bos=False, eos=False)
        # strip eos token from prompt
        self.prompt_tokens = [tok for tok in self.prompt_tokens if tok not in self.stop_tokens]
        self.num_prefill_tokens = len(self.prompt_tokens)


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
        self.max_users = 32
        self.users = [None for _ in range(self.max_users)]
        self.num_users = None
        # inputs to model
        self.decode_ids = None
        # backend status
        self.time_last_status = time.time()
        self.update_period = 1  # status message period in seconds
        self.verbose = verbose  # enable conditional debug logging
        # new init:
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.n_devices = n_devices
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
        close_devices(self.t3k_device_mesh)

    def init_tt_metal_device(self):
        logger.info("init_tt_metal_device ...")
        t3k_device_mesh = get_t3k_device_mesh(num_devices_requested=self.n_devices)
        for i in t3k_device_mesh.get_device_ids():
            device = t3k_device_mesh.get_device(i)
            device.enable_async(True)
            device.enable_program_cache()

        self.t3k_device_mesh = t3k_device_mesh
        logger.info("init_tt_metal_device finished.")

    def init_model(self):
        # set up variables for model init
        n_devices = inference_config.n_devices
        logger.info("init_model ...")
        # logger.info("todo ckpt vars ")

        self.init_tt_metal_device()
        check_device_mesh(self.t3k_device_mesh, self.model_config)

        for i in self.t3k_device_mesh.get_device_ids():
            device = self.t3k_device_mesh.get_device(i)
            device.enable_async(True)

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

    def _get_num_of_users(self):
        # find num of non None users
        return sum([1 for user in self.users if user is not None])

    def _find_free_user_slot(self):
        """return the index of the first free user slot"""
        for i, user in enumerate(self.users):
            if user is None:
                return i

    def _add_users_from_non_empty_queue(self, prompt_q):
        """add users from prompt_q to self.users"""
        while not prompt_q.empty() and self._get_num_of_users() < self.max_users:
            user_id, prompt, params = prompt_q.get()

            # Cancel on special stop token
            if prompt == "<|stop|>":
                if any((user is not None) and (user_id == user.user_id) for user in self.users):
                    logger.info(f"Cancelling input from user {user_id}")
                    self._get_user_by_id(user_id).cancel = True
                else:
                    logger.info(f"Unexpected cancelling for non-activte user {user_id}")
                continue

            # Don't accept a prompt from a user that's already being procesed
            if any((user is not None) and (user_id == user.user_id) for user in self.users):
                logger.warning(f"Ignoring duplicate input from user {user_id}")
                continue
            prompt_tokens = self.tok_encode(prompt)
            user_info = UserInfo(user_id, prompt_tokens, 0, params, self.tokenizer, formatter=self.formatter)
            idx = self._find_free_user_slot()
            self.users[idx] = user_info
            if self.verbose:
                logger.debug(f"Added user {user_id} to slot {idx} with prompt: {prompt}")

    def pick_prompts(self, prompt_q: Queue):
        if self._get_num_of_users() == self.max_users:
            return

        if self._get_num_of_users() == 0:
            # no users generating currently
            while prompt_q.empty():
                # wait for users
                time.sleep(0.02)
            # batch start delay
            time.sleep(0.5)
            self._add_users_from_non_empty_queue(prompt_q)
        else:
            if prompt_q.empty():
                # no users to add
                return
            else:
                self._add_users_from_non_empty_queue(prompt_q)

        # Check for duplicate user_ids and log it
        user_ids = [user.user_id for user in self.users if user is not None]
        if len(user_ids) != len(set(user_ids)):
            logger.warning(f"WARNING: Duplicate user ids: {user_ids}")

    def batch_stats(self):
        tokens_generated = self.forward_counter - self.prev_forward_counter
        batch_duration = time.time() - self.batch_start_time
        tps = tokens_generated / batch_duration
        logger.info(
            f"batch_counter:={self.batch_counter}, forward_counter:={self.forward_counter}, tokens_generated:={tokens_generated}, tps:={tps:.4f} tokens/sec (32 users)"
        )
        self.prev_forward_counter = self.forward_counter
        self.batch_start_time = time.time()

    def start_new_batch(context_enc_list):
        assert len(context_enc_list) <= self.batch_size
        # reset users
        for idx in range(len(self.users)):
            self.reset_user_memory(idx, self.users[idx])
            self.users[i] = None

        for idx in range(len(context_enc_list)):
            # TODO: simplify user info
            self.users[i] = UserInfo()

        # TODO: prefill
        self.prepare_inputs()

    def prepare_inputs(self):
        # empty users get pad id
        input_prompts = [user_info.prompt_tokens if user_info else [self.tokenizer.pad_id] for user_info in self.users]
        tokens, input_text_mask, eos_reached = initialize_inputs(
            tokenizer=self.tokenizer,
            prompt_tokens=input_prompts,
            bsz=self.batch_size,
            total_len=self.max_seq_len,
        )
        # TODO: when prefill is separate change
        self.cur_pos = 1
        self.prev_pos = 0
        self.input_text_mask = input_text_mask
        self.tokens = tokens
        self.decode_ids = tokens[:, :1]
        self.num_users = len(self.get_users())
        assert self.num_users < self.max_users
        self.batch_counter += 1

    def prefill(self):
        # TODO
        logger.info("Running prefill ...")
        logger.info("Done prefill")

    def get_logits(self):
        self.forward_counter += 1
        logits = self.model.forward(self.decode_ids, self.prev_pos, decode_only=self.decode_only)
        next_logits = logits[:, -1, :]
        self.decode_ids = self.sampling_func(next_logits)
        return next_logits

    def get_loglikelihood(self):
        next_logits = self.get_logits()
        loglikelihood = F.log_softmax(next_logits)
        return loglikelihood

    def get_next_tokens(self):
        # TODO: decode self.decode_ids
        pass

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
            if not user_info.prefill_complete:
                # take next token for prefill
                self.decode_ids[idx][0] = user_info.prompt_tokens[user_info.num_tokens_prefilled]
                user_info.num_tokens_prefilled += 1
                if user_info.num_tokens_prefilled >= user_info.num_prefill_tokens:
                    user_info.prefill_complete = True
            else:
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

    def push_outputs(self, output_q):
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
            # send special EOS string to frontend
            if (last_token in user_info.stop_tokens) or (user_info.decode_complete):
                return_text = inference_config.end_of_sequence_str
            output_q.put((user_info.user_id, return_text))
            if self.verbose:
                logger.debug(f"user_id:{user_info.user_id}, {return_text}")

    def reset_user_memory(self, user_idx, user):
        self.decode_ids[user_idx, 0] = 0

    def update_users(self):
        for i, token_id in enumerate(self.decode_ids.reshape(self.batch_size).tolist()):  # bc input_ids is 1x32
            if self.users[i] is None:
                continue

            if token_id in self.users[i].stop_tokens and self.users[i].decode_complete:
                self.reset_user_memory(i, self.users[i])
                if self.verbose:
                    logger.debug(f"Evicted user_id: {self.users[i].user_id} from index {i} in user list")
                self.users[i] = None
            elif token_id in self.users[i].stop_tokens and not self.users[i].decode_complete:
                logger.error(
                    f"user_id: {self.users[i].user_id} from index {i} had EOS token but decode_complete=False."
                )
                self.reset_user_memory(i, self.users[i])
                self.users[i] = None
            elif token_id not in self.users[i].stop_tokens and self.users[i].decode_complete:
                logger.error(
                    f"user_id: {self.users[i].user_id} from index {i} did not have EOS token but decode_complete=True."
                )
                self.reset_user_memory(i, self.users[i])
                self.users[i] = None

    def send_status(self, prompt_q, status_q):
        if time.time() - self.time_last_status > self.update_period:
            # send status queue which includes the (length of the prompt_q, the number of users being decoded rn, the user_ids being decoded)
            cur_status = (
                prompt_q.qsize(),
                self._get_num_of_users(),
                [user.user_id for user in self.users if user is not None],
            )
            status_q.put(cur_status)
            # udpate cur time
            self.time_last_status = time.time()

    def run_generate(self, prompt_q, output_q, status_q, loop_once):
        """
        Continuously pop prompt from prompt_q and push generated tokens to output_q
        while running decode. Automatically swap users from prompt_q
        prompt_q: {'user_id1': 'prompt1', 'user_id2': 'prompt2'...}
        output_q: {'user_id1': 'generated_1', 'user_id3': 'generated_1', 'user_id1': 'generated_2'...}
        stop_event: threading.Event, set to stop safely
        """
        logger.info("starting run_generate ...")
        LOOP_FORVEVER = True
        while LOOP_FORVEVER:
            self.pick_prompts(prompt_q)  # we update to self.users
            self.batch_start_time = time.time()
            self.prepare_inputs()
            logger.info("Running inference decode and pushing results ...")
            while not all([user.decode_complete for user in self.get_users()]):
                self.decode()
                self.push_outputs(output_q)
                self.update_users()
                self.send_status(prompt_q, status_q)
            self.batch_stats()
            if loop_once:
                break
