# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import ttnn.experimental as tt_lib
import tt_lib as ttl
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor, ListMeshToTensor

from loguru import logger

import copy
from models.utility_functions import torch2tt_tensor, pad_by_zero, tt2torch_tensor, nearest_32
from models.experimental.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized as TtLlamaModel
from models.experimental.llama2_70b.tt.llama_common import BASE_URL
from models.experimental.llama2_70b.tt.model_config import (
    get_model_config,
)


class TtLlamaModelForGeneration:
    def __init__(self, configuration, state_dict, model_args, tt_args):
        ## Get state dict
        configuration = copy.deepcopy(configuration)
        self.params = configuration
        self.model_args = model_args
        self.tt_args = tt_args
        self.device_mesh = tt_args.device_mesh
        self.n_devices = tt_args.n_devices
        self.n_layers = model_args.n_layers
        self.max_batch_size = model_args.max_batch_size

        model_config = get_model_config(
            llama_version=model_args.llama_version,
            batch=model_args.max_batch_size,
            seq_len=1,  # initial model_config is in decode mode
            num_devices=tt_args.n_devices,
            max_batch_size=model_args.max_batch_size,
            max_context_len=model_args.max_kv_context_len,
        )

        # TT model -------------------------------------------------------------
        self.tt_model = TtLlamaModel(
            self.device_mesh,
            state_dict,
            BASE_URL,
            self.n_layers,
            model_config,
            configuration,
            cache_path=tt_args.cache_path,
            read_cache=False,
        )

        del state_dict

    def forward(self, tokens: torch.Tensor, start_pos: int, *args, **kwargs):
        # First, determine whether this is decode or prefill based on shape of the input
        assert len(tokens.shape) == 2
        batch, seq_len = tokens.shape
        seq_len = 1 if kwargs.get("decode_only", False) else seq_len
        if seq_len == 1:
            return self.decode_forward(tokens, start_pos, *args, **kwargs)
        else:
            return self.prefill_forward(tokens, start_pos, *args, **kwargs)

    def decode_forward(self, tokens: torch.Tensor, start_pos: int, *args, **kwargs):
        self._update_model_config("decode", tokens.shape)
        tt_inp_emb, start_pos, rot_mat, attn_mask = self.tt_model.prepare_inputs(tokens, start_pos)

        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            attn_mask,
        )

        del tt_inp_emb
        del rot_mat
        del attn_mask

        logits = self._process_logits(tt_logits)
        del tt_logits

        return logits

    def prefill_forward(self, tokens: torch.Tensor, start_pos: int, *args, **kwargs):
        self._update_model_config("prefill", tokens.shape)
        batch, seq_len = tokens.shape
        output_logits = torch.zeros(batch, seq_len, self.params.vocab_size)
        padded_seq_len = 128 if seq_len <= 128 else 2048
        # pad tokens to 128 or 2048
        prefill_ids = torch.cat([tokens, torch.zeros(batch, padded_seq_len - seq_len).long()], dim=-1)

        for user_id in range(batch):
            logger.info(f"Filling kv cache for user {user_id + 1}")

            tt_inp_emb, start_pos, rot_mat, attn_mask = self.tt_model.prepare_inputs(
                prefill_ids[user_id : user_id + 1], start_pos=0, valid_seq_len=seq_len
            )

            tt_logits = self.tt_model(
                tt_inp_emb,
                rot_mat,
                start_pos,
                attn_mask,
                user_id=user_id,
            )

            del tt_inp_emb
            del rot_mat
            del attn_mask

            logits = self._process_logits(tt_logits)

            del tt_logits

            output_logits[user_id] = logits[:, :seq_len, :]

        logger.info(f"Finished prefill for all users up to {seq_len} tokens, Starting decode...")

        return output_logits

    def _update_model_config(self, mode: str, shape: tuple):
        if self.tt_model.model_config["LLM_MODE"] != mode:
            logger.info(f"Changing mode to {mode}")
            batch, seq_len = shape
            if mode == "prefill":
                assert seq_len <= 2048  # TODO: make general for long context
                seq_len = 128 if seq_len <= 128 else 2048
            model_config = get_model_config(
                llama_version=self.model_args.llama_version,
                batch=batch,
                seq_len=seq_len,  # initial model_config is in decode mode
                num_devices=self.tt_args.n_devices,
                max_batch_size=self.model_args.max_batch_size,
                max_context_len=self.model_args.max_kv_context_len,
            )
            self.tt_model.set_model_config(model_config)

    def _process_logits(self, tt_logits):
        logits = ttnn.to_torch(
            tt_logits, device=self.device_mesh, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=3)
        )
        logits = logits[..., : self.params.vocab_size].float()
        return logits.permute(2, 1, 0, 3).squeeze().unsqueeze(1)
