import os
import time
import traceback
from multiprocessing import Queue

# from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.generation.utils import top_k_top_p_filtering

import ttnn
from models.demos.wormhole.mistral7b.tt.mistral_common import (
    prepare_inputs_ttnn,
    sample,
    precompute_freqs,
    freqs_to_rotation_matrix,
    cache_attention,
    get_prefill_rot_mat,
    prepare_inputs_ttnn_prefill,
    get_rot_transformation_mat,
)
from models.demos.wormhole.mistral7b.tt.mistral_model import TtTransformer
from models.demos.wormhole.mistral7b.tt.mistral_embedding import TtMistralEmbedding
from models.demos.wormhole.mistral7b.reference.tokenizer import Tokenizer
from models.demos.wormhole.mistral7b.tt.model_config import TtModelArgs
from models.demos.wormhole.mistral7b.demo.demo_with_prefill import Emb, preprocess_inputs_prefill
from models.demos.wormhole.mistral7b.demo.demo import preprocess_inputs


# from inference_config import inference_config
import logging

logger = logging.getLogger(__name__)
logger.info(f"importing {__name__}")


def initialize_inputs(tokenizer, prompt_tokens, bsz, total_len):
    # pad the model to maximum length
    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cpu")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t[:total_len], dtype=torch.long, device="cpu").clone().detach()
    eos_reached = torch.tensor([False] * bsz, device="cpu")
    input_text_mask = tokens != pad_id  # use prefill token if that token is not masked
    return tokens, input_text_mask, eos_reached


class UserInfo:
    def __init__(self, user_id, prompt, rag_context, context_tokens, params, tokenizer):
        self.user_id = user_id
        self.prompt = prompt
        self.rag_context = None
        self.prompt_tokens = context_tokens
        self.num_prefill_tokens = len(self.prompt_tokens)
        self.num_tokens_decoded = 0
        self.position_id = 0
        self.batch_counter = 0
        # self.position_id = position_id
        self.num_tokens_generated = 0
        self.generation_params = params
        self.max_tokens = params["max_tokens"]
        self.return_prompt = params["return_prompt"]
        self.cancel = False
        self.prefill_complete = False
        self.decode_complete = False
        self.sent_stop = False
        self.stop_sequence = None
        if params.get("stop_sequence"):
            self.stop_sequence = tokenizer.encode(params.get("stop_sequence"))
        # note: sentecepiece tokenizer decode doesnt handle spaces directly
        # see: https://github.com/google/sentencepiece/blob/master/python/src/sentencepiece/__init__.py#L776
        # users must aggregate all generated tokens and decode full text each time
        # then send new chars
        self.generated_tokens = []
        self.generated_logits = torch.tensor([])
        self.num_generated_chars = 0
        self.max_generated_tokens = 120

        # TODO: check if this is correct
        if params.get("stop_sequence"):
            self.stop_sequence = tokenizer.encode(params.get("stop_sequence"), bos=False, eos=False)


class PrefillDecodeBackend:
    def __init__(
        self,
        model_version,
        batch_size,
        num_layers,
        max_seq_len,
        cache_root,
        verbose=False,
    ) -> None:
        """
        Initialize pybuda model and all infracstructures to continuously run decode
        Maintain a cur_prompts for decode.
        """
        self.max_users = 8
        self.num_users = None
        self.users = [None for _ in range(self.max_users)]
        self.use_cache = True
        # backend status
        self.time_last_status = time.time()
        self.update_period = 1  # status message period in seconds
        self.num_steps = 0
        self.verbose = verbose  # enable conditional debug logging
        self.model_version = model_version
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.default_top_p = 0.9
        self.default_top_k = 40
        self.default_temperature = 1.0
        self.timestamps_start = {}
        self.timestamps_stop = {}
        self.enable_profile_logging = True
        self.device = None
        self.cache_root = Path(cache_root)
        if not self.cache_root.exists():
            self.cache_root.mkdir(parents=True, exist_ok=True)
        # tt-metal init
        self.dtype = ttnn.bfloat8_b
        self.instruct_mode = True
        self.init_tt_metal()
        self.iteration = 0
        self.rot_emb_matrix_list = []
        self.batch_idx = 0  # keep track of what batch you are on to clear the kv cache
        # embed_on_device not currently supported, set to False to run embedding layer on CPU vs on TT device
        self.embed_on_device = False
        self.prefill_seq_len = 0  # 0 is default if there is no prefill
        self.max_generated_tokens = 120
        self.batch_counter = 0
        self.decode_counter = 1
        self.prev_decode_counter = 0

    def get_users(self):
        return [u for u in self.users if u]

    def get_user_param(self, param):
        return [user.generation_params[param] if user is not None else None for user in self.users]

    def timer_start(self, name):
        self.timestamps_start[name] = time.time()

    def timer_stop(self, name, log=False):
        if name in self.timestamps_start.keys():
            self.timestamps_stop[name] = time.time()
            timedelta = self.timestamps_stop[name] - self.timestamps_start[name]
            if log or self.enable_profile_logging:
                print(f"timedelta: {name}: {timedelta} seconds")
                logger.info(f"timedelta: {name}: {timedelta} seconds")

    def model_location_generator(self, model_version, model_subdir=""):
        model_cache_path = Path(self.cache_root) / "tt-metal-models" / model_version
        model_cache_path.mkdir(parents=True, exist_ok=True)
        return model_cache_path

    def get_tt_cache_path(self, model_version, model_subdir="", default_dir=""):
        tt_cache_path = Path(self.cache_root) / "tt-metal-cache" / model_version
        tt_cache_path.mkdir(parents=True, exist_ok=True)
        return tt_cache_path

    def tokenize_prompt(self, prompt: str, rag_context: str = None, add_special_tokens: bool = True, **kwargs):
        # if self.chat and add_special_tokens:
        #     if rag_context:
        #         messages = [
        #             Message(role="system", content=f"Please use the following context to answer the question:\n{rag_context}"),
        #             Message(role="user", content=prompt)
        #         ]
        #         return self.formatter.encode_dialog_prompt(messages)
        #     else:
        #         # encode as a single turn of dialog
        #         messages = [Message(role="user", content=prompt)]
        #         return self.formatter.encode_dialog_prompt(messages)
        # else:
        return self.tokenizer.encode(prompt)

    def add_users_from_context(self, context_enc_list, do_sample=True):
        """
        Add users from the given context_enc_list.

        Parameters:
        - context_enc_list (list): A list of encoded context tokens for each user.
        - do_sample (bool): True for top_k/top_p sampling, False for greedy decoding.

        Raises:
        - AssertionError: If the length of context_enc_list exceeds the maximum number of users.

        Returns:
        - None
        """
        assert len(context_enc_list) <= self.max_users
        # reset users
        for idx in range(len(self.get_users())):
            # reset memory
            self.decode_ids[idx, 0] = 0
            self.users[idx] = None

        params = {
            "top_p": self.default_top_p,
            "top_k": self.default_top_k if do_sample else 1,
            "temperature": self.default_temperature,
            "max_tokens": self.max_seq_len,
            "return_prompt": False,
        }

        for idx in range(len(context_enc_list)):
            self.users[idx] = UserInfo(
                user_id=idx,
                prompt=None,
                rag_context=None,
                context_tokens=context_enc_list[idx],
                params=params,
                tokenizer=self.tokenizer,
            )

    def prepare_batch_inputs(self):
        self.num_users = len(self.get_users())
        assert self.num_users <= self.max_users
        input_prompts = [user.prompt_tokens for user in self.get_users()]
        self.max_prompt_len = max([user.num_prefill_tokens for user in self.get_users()])
        self.min_prompt_len = min([user.num_prefill_tokens for user in self.get_users()])
        # limit total length to max_seq_len
        total_len = min(self.min_prompt_len, self.max_seq_len)
        if total_len < self.min_prompt_len:
            logger.warning(
                f"Truncating input prompt min_prompt_len:={self.min_prompt_len} to max_seq_len:={self.max_seq_len}"
            )
        # pad inputs, empty users get pad id
        prefill_tokens, input_text_mask, _ = initialize_inputs(
            tokenizer=self.tokenizer,
            prompt_tokens=input_prompts,
            bsz=len(input_prompts),
            total_len=total_len,
        )
        # where does intput_text_mask get used?
        self.input_text_mask = input_text_mask
        self.prefill_ids = prefill_tokens
        # decode_ids are padded to batch_size
        decode_ids = torch.full((self.batch_size, 1), self.tokenizer.pad_id, dtype=torch.long, device="cpu")
        decode_ids[: self.num_users, :1] = prefill_tokens[:, :1].clone()
        self.decode_ids = decode_ids

    def batch_preprocessing(self):
        # TODO: investigate changing when continous batching supported
        # note: the cur_pos index if shared between all users
        # this may change for the continuous batching implementation
        self.batch_start_time = time.time()
        self.prepare_batch_inputs()
        self.prev_pos = 0
        self.cur_pos = self.prev_pos + 1
        self.batch_counter += 1

    # def start_decode_loop(self):
    #     for user in self.get_users():
    #         if user.prefill_complete:
    #             user.start_decode_timer()
    #     self.timer_start("decode_batch")
    #     logger.info("Running inference decode and pushing results ...")

    def get_batch_stats(self, log=True):
        # self.timer_stop("decode_batch") # TODO turn back on later
        batch_duration = time.time() - self.batch_start_time

        # actual prefill tokens
        prefill_batch_tokens = self.prefill_batch_size * self.prefill_seq_len
        prefill_time = self.timestamps_stop["prefill"] - self.timestamps_start["prefill"]

        # prefill-via-decode + decode generation tokens
        decode_batches = self.decode_counter - self.prev_decode_counter
        decode_batch_tokens = decode_batches * self.batch_size
        decode_batch_e2e_time = self.timestamps_stop["decode_batch"] - self.timestamps_start["decode_batch"]
        decode_batch_time = self.timer_sums["decode"]
        self.timer_sums["decode"] = 0

        self.prev_decode_counter = self.decode_counter

        batch_stats = {
            "batch_counter": self.batch_counter,
            "decode_counter": self.decode_counter,
            "batch_duration": round(batch_duration, 3),
            "batch_users": self.num_users,
            "prefill": {
                "prefill_batch_size": self.prefill_batch_size,
                "prefill_batch_tokens": prefill_batch_tokens,
                "e2e_throughput_tps": round(prefill_batch_tokens / prefill_time, 3),
            },
            "decode": {
                "decode_batch_tokens": decode_batch_tokens,
                "e2e_throughput_tps": round(decode_batch_tokens / decode_batch_e2e_time, 3),
                "e2e_latency_ms": round((decode_batch_e2e_time / decode_batches) * 1000, 2),
                "decode_throughput_tps": round(decode_batch_tokens / decode_batch_time, 3),
                "decode_latency_ms": round((decode_batch_time / decode_batches) * 1000, 2),
            },
        }
        if log:
            logger.info(batch_stats)
        return batch_stats

    def teardown(self):
        logger.info("teardown ...")
        if not os.environ.get("MOCK_MODEL"):
            self.teardown_tt_metal_device()

    def teardown_tt_metal_device(self):
        logger.info("teardown_tt_metal_device ...")
        import tt_lib as ttl

        ttnn.DumpDeviceProfiler(self.device, True)
        ttnn.DeallocateBuffers(self.device)
        ttnn.Synchronize(self.device)
        ttnn.CloseDevice(self.device)

    def init_tt_metal_device(self):
        import tt_lib as ttl

        logger.info("init_tt_metal_device ...")
        device_ids = ttnn.get_device_ids()
        device_id = device_ids[0]
        num_devices = ttnn.GetNumPCIeDevices()
        assert device_id < num_devices, "CreateDevice not supported for non-mmio device"
        self.device = ttnn.CreateDevice(device_id)
        ttnn.SetDefaultDevice(self.device)
        self.device.enable_program_cache()

    def init_tt_metal(self):
        self.init_tt_metal_device()

        logger.info("init_tt_metal model ...")
        self.model_args = TtModelArgs(self.device, instruct=self.instruct_mode)

        self.tokenizer = Tokenizer(self.model_args.tokenizer_path)

        logger.info("Loading weights...")
        state_dict = torch.load(self.model_args.consolidated_weights_path)
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if (
                any([f"layers.{i}." in k for i in range(self.model_args.n_layers)])
                or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
            )
        }
        logger.info("Loading weights finished!")

        # TODO Should we keep initial embedding on host?
        self.embd = Emb()
        self.embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

        # needs full batchsize inputs always
        compile_prompts = ["COMPILE_PROMPT"] * self.batch_size

        (
            _,
            _,
            _,
            _,
            self.rot_emb_matrix_list,
            self.prefill_seq_len,
            _,
        ) = preprocess_inputs_prefill(
            compile_prompts, self.tokenizer, self.model_args, self.dtype, self.embd, self.instruct_mode, self.device
        )

        self.generation_start_pos = self.prefill_seq_len

        logger.info("Caching attention ops...")
        cache_attention(self.device, state_dict, self.model_args, self.rot_emb_matrix_list, self.dtype, 120)

        if self.instruct_mode:
            self.tokenizer._model.pad_id = self.tokenizer._model.eos_id

        # Load TTNN mistral model
        logger.info("Loading weights to device...")
        self.tt_model = TtTransformer(
            args=self.model_args,
            device=self.device,
            dtype=self.dtype,
            state_dict=state_dict,
            weight_cache_path=self.model_args.weight_cache_path(self.dtype),
            layers=list(range(self.model_args.n_layers)),
            rot_mat=self.rot_emb_matrix_list,
            start_pos=self.generation_start_pos,
        )
        self.tt_embd = TtMistralEmbedding(
            device=self.device,
            args=self.model_args,
            weight_cache_path=self.model_args.weight_cache_path(self.dtype),
            state_dict=state_dict,
            dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
        )

        logger.info("Finished loading weights to device. Starting inference...")

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

            user_info = UserInfo(user_id, prompt, 0, params, self.tokenizer)
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

    def prepare_inputs(self):
        # input_prompts = [user_info.prompt for user_info in self.users if user_info]
        # note: current implementation assumes full 32 prompts input always
        input_prompts = [user_info.prompt if user_info is not None else "" for user_info in self.users]
        logger.info("INPUT PROMPTS: ", input_prompts)
        self.timer_start("preprocess_inputs")
        (
            self.pt_encoded_input,
            self.tt_decode_input,
            self.pt_prefill_input,
            self.input_mask,
            self.rot_emb_matrix_list,
            self.prefill_seq_len,
            _,
        ) = preprocess_inputs_prefill(
            input_prompts,
            self.tokenizer,
            self.model_args,
            self.dtype,
            self.embd,
            self.instruct_mode,
            self.device,
        )
        # set kv cache to zeros if not first batch, to avoid context leaking
        if self.batch_idx != 0:
            for layer in self.tt_model.layers:
                k_cache, v_cache = layer.attention.layer_past_list[0]
                k_cache = k_cache * 0
                v_cache = v_cache * 0
                # Deallocation is necessary to avoid memory leaks and running out of L1 in later batches
                layer.attention.layer_past_list[0][0].deallocate(True)
                layer.attention.layer_past_list[0][1].deallocate(True)
                layer.attention.layer_past_list[0] = [k_cache, v_cache]

        self.timer_stop("preprocess_inputs")
        self.iteration = 0

    def prefill(self):
        if self.prefill_seq_len > 0:
            logger.info(f"Starting prefill [{self.prefill_seq_len} tokens]...")
            rot_mats_prefill = get_prefill_rot_mat(
                self.model_args.head_dim, self.model_args.max_seq_len, self.device, seq_len=self.prefill_seq_len
            )
            head_dim = self.model_args.dim // self.model_args.n_heads
            transformation_mat_torch = get_rot_transformation_mat(head_dim)
            transformation_mats = ttnn.as_tensor(
                transformation_mat_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for user_id in range(self.batch_size):
                prefill_input, attn_mask, _ = prepare_inputs_ttnn_prefill(
                    self.pt_prefill_input[user_id],
                    self.device,
                )
                tt_out = self.tt_model(
                    prefill_input,
                    0,  # Current position
                    attn_mask,
                    rot_mats_prefill,
                    transformation_mats,
                    user_id=user_id,
                    mode="prefill",
                )

            logger.info(f"Prefill finished [{self.prefill_seq_len} tokens]!")

    def decode(self, return_logits=False):
        self.decode_counter += 1
        curr_pos = self.generation_start_pos + self.iteration
        self.timer_stop("all_but_decode")
        self.timer_start("decode_preprocessing")

        decode_input, current_pos = prepare_inputs_ttnn(
            self.pt_encoded_input,
            curr_pos,
            self.model_args.dim,
            self.model_args.sliding_window,
            self.tt_model.device,
        )
        self.timer_stop("decode_preprocessing")
        self.timer_start("decode")
        # Run ttnn mistral model
        tt_out = self.tt_model(decode_input, current_pos)
        self.timer_stop("decode")
        self.timer_start("decode_get_logits")

        tt_output_torch = ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)[: self.model_args.max_batch_size, :, :]
        self.timer_stop("decode_get_logits")
        self.timer_start("token_selection")

        # TODO argmax on device
        # tt_out = ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)
        # tt_out = ttnn.permute(tt_out, (2, 1, 0, 3))
        # tt_out = ttnn.reshape(tt_out, (tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]))  # Squeeze(1)
        # tt_out_argmax = ttnn.experimental.tensor.argmax(tt_out, dim=-1)
        # Typecast from bf16 to uint32 for embedding
        # tt_out_tok = ttnn.clone(tt_out_argmax, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.uint32)
        # tt_out_tok = ttnn.experimental.tensor.typecast(tt_out_tok, dtype=ttnn.uint32)

        out_tok = self.select_tokens(
            logits=tt_output_torch,
            skip_token=self.tokenizer.eos_id,
        ).reshape([self.batch_size, 1])

        # tt_out_tok = sample(tt_output_torch, temperature=0, top_p=0.8)

        self.timer_stop("token_selection")
        self.timer_start("embeddings")

        if self.iteration < self.input_mask.shape[1]:  # If prefill
            # If token is pad token, start generating new token, otherwise, push the next prompt token to the model
            out_tok = torch.where(
                self.input_mask[:, self.iteration], self.tt_decode_input[:, self.iteration], out_tok[:, 0]
            ).unsqueeze(1)

        # embed_on_device not currently working
        if self.embed_on_device:
            tt_out_tok = ttnn.from_torch(out_tok, device=self.device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            self.pt_encoded_input = self.tt_embd(tt_out_tok)
        else:
            self.pt_encoded_input = self.embd(out_tok)
        self.timer_stop("embeddings")
        self.iteration += 1
        self.timer_start("all_but_decode")

    def select_tokens(
        self,
        logits,
        skip_token,
        return_probs=False,
    ):
        out_tokens = []
        for idx, user in enumerate(self.users):
            if not user:
                # skip None users, fill with skip token
                token = torch.tensor([skip_token])
            elif not user.prefill_complete:
                token = self.tt_decode_input[idx, self.iteration].unsqueeze(0)
                if user.return_prompt:
                    user.generated_tokens.append(token.item())
                    user.num_tokens_generated += 1
                # TODO: better way of counting prefill that handles input mask being non-contiguous
                if self.iteration == (torch.count_nonzero(self.input_mask[idx]).item() - 1):
                    user.prefill_complete = True
            elif user.decode_complete:
                logger.error(f"user.decode_complete={user.decode_complete}, and is still generating. Should be None")
            else:
                user.num_tokens_decoded += 1
                token = top_pk_logits_efficient(
                    logits[idx],
                    user.generation_params.get("top_p"),
                    user.generation_params.get("top_k"),
                    user.generation_params.get("temperature"),
                    return_probs=return_probs,
                    skip_token=skip_token,
                )
                user.generated_tokens.append(token.item())
                user.num_tokens_generated += 1
                if token == self.tokenizer.eos_id:
                    user.decode_complete = True
                elif user.num_tokens_generated > user.max_tokens:
                    user.decode_complete = True
                elif (user.stop_sequence is not None) and (token == user.stop_sequence):
                    user.decode_complete = True
            out_tokens.append(token)
            logger.info(f"Concatenated result shape at iteration {self.iteration}: {len(out_tokens)}")
        return torch.concat(out_tokens)

    def push_outputs(self, output_q):
        # Sentencepiece tokenizer doesn't handle spaces per token, must decode full text
        # then push new chars to output queue
        for idx, user in enumerate(self.users):
            if user is None or not user.generated_tokens:
                continue
            if user.generated_tokens[-1] == self.tokenizer.eos_id:
                # must pass end_of_sequence_str to frontend to close response
                out_text = "<|endoftext|>"
            else:
                full_text = self.tokenizer.decode(user.generated_tokens)
                out_text = full_text[user.num_generated_chars :]
                user.num_generated_chars = len(full_text)
            out = (user.user_id, out_text)
            output_q.put(out)
            if (user.decode_complete and out_text != "<|endoftext|>",):
                # send eos str to frontend in all cases
                output_q.put(
                    (
                        user.user_id,
                        "<|endoftext|>",
                    )
                )
            if self.verbose:
                logger.debug(f"user_id:{user.user_id}, {out_text}")

    def reset_user_memory(self, idx, user):
        # not needed for this implementation
        pass

    def log_user_stats(self, idx, user):
        # TODO: record user stats, e.g. prompt length, num generated tokens, time
        pass

    def update_users(self):
        for idx, user in enumerate(self.users):
            if user is None or not user.generated_tokens:
                continue
            token_id = user.generated_tokens[-1]
            if (token_id == self.tokenizer.eos_id) or user.decode_complete:
                if not user.decode_complete:
                    logger.error(f"user_id: {user.user_id} from index {idx} had EOS token but decode_complete=False.")
                if not (token_id == self.tokenizer.eos_id):
                    logger.error(
                        f"user_id: {user.user_id} from index {idx} did not have EOS token but decode_complete=True."
                    )
                if self.verbose:
                    logger.debug(f"Evicted user_id: {user.user_id} from index {idx} in user list")
                self.reset_user_memory(idx, user)
                self.log_user_stats(idx, user)
                self.users[idx] = None

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

    def run_generate(self, prompt_q, output_q, status_q, run_once=False):
        """
        Continuously pop prompt from prompt_q and push generated tokens to output_q
        while running decode. Automatically swap users from prompt_q
        prompt_q: {'user_id1': 'prompt1', 'user_id2': 'prompt2'...}
        output_q: {'user_id1': 'generated_1', 'user_id3': 'generated_1', 'user_id1': 'generated_2'...}
        """
        logger.info("starting run_generate ...")
        LOOP_FOREVER = True
        while LOOP_FOREVER:
            if self.verbose:
                logger.debug(f"run_generate step: {self.num_steps}")
            self.pick_prompts(prompt_q)  # we update to self.users
            self.prepare_inputs()
            self.prefill()
            logger.info("Running inference decode and pushing results ...")
            logger.info("prompts: ", prompt_q)
            while not all([user.decode_complete for user in self.get_users()]):
                self.decode()
                self.push_outputs(output_q)
                self.update_users()
                self.send_status(prompt_q, status_q)
            self.num_steps += 1
            self.batch_idx += 1
            if run_once:
                break

    def generate_n(self, n_tokens, return_logits=False):
        """
        use with add_users_from_context()
        """
        self.batch_preprocessing()
        self.prefill()
        # self.start_decode_loop()
        while not all([user.num_tokens_decoded >= n_tokens or user.decode_complete for user in self.get_users()]):
            self.decode(return_logits=return_logits)
        self.get_batch_stats(log=True)
        if return_logits:
            return torch.concat([user.generated_logits[:n_tokens, :].unsqueeze(0) for user in self.get_users()])
        else:
            return [user.generated_tokens[:n_tokens] for user in self.get_users()]


def top_pk_logits_efficient(
    logits,
    p,
    k,
    temperature,
    return_probs=False,
    skip_token=11,
):
    # do not keep the entire vocab size after top k. Instead, keep the k size tensor and record the associated indices
    top_k_values, top_k_indices = torch.topk(logits, k=k)
    # replace any nans with 0's
    top_k_values = torch.where(torch.isnan(top_k_values), torch.zeros_like(top_k_values), top_k_values)
    top_p_values = top_k_top_p_filtering(top_k_values, top_p=p)
    probs = F.softmax(top_p_values / temperature, dim=-1)
    top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
    return token


# def run_backend(prompt_q, output_q, status_q, verbose=True, run_once=False):
#     logger.info("starting run_backend ...")
#     with torch.no_grad():
#         backend = PrefillDecodeBackend(
#             model_version=inference_config.model_config.model_version,
#             batch_size=inference_config.model_config.batch_size,
#             num_layers=inference_config.model_config.num_layers,
#             max_seq_len=inference_config.model_config.max_seq_len,
#             cache_root=inference_config.cache_root,
#             verbose=verbose,
#         )
#         try:
#             # run generate
#             backend.run_generate(prompt_q, output_q, status_q, run_once)
#         except Exception as e:
#             logger.error(e)
#             # Capture the stack trace
#             stack_trace = traceback.format_exc()
#             logger.error(stack_trace)
#             # Re-raise the exception if you want the process to exit with an error
#             raise e
#         finally:
#             backend.teardown()
