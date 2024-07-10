# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
import numpy as np
from loguru import logger
from sklearn.metrics import top_k_accuracy_score

# Set Mixtral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["MIXTRAL_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor
import time
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import prepare_inputs_ttnn, get_single_rot_mat
from models.demos.t3000.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.t3000.mixtral8x7b.reference.model import Transformer
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.utility_functions import comp_pcc, comp_allclose


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@pytest.mark.parametrize(
    "n_layers",
    (32,),
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 6154240}], indirect=True)
def test_mixtral_model_inference(
    t3k_device_mesh,
    use_program_cache,
    reset_seeds,
    n_layers,
):
    pcc = 0.97
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(t3k_device_mesh.get_device(0))
    model_args.n_layers = n_layers

    model_args.max_seq_len = 32
    model_args.max_batch_size = 32

    state_dict = model_args.load_state_dict()

    tokenizer = Tokenizer(model_args.tokenizer_path)
    prompts = ["Once"] * 32
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    reference_model = Transformer(args=model_args)
    reference_model.load_state_dict(state_dict)
    reference_model.eval()

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    logger.info("Loading Model")
    # Load TTNN model
    tt_model = TtTransformer(
        device_mesh=t3k_device_mesh,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
    )

    generation_start_pos = 0
    generation_length = 10
    all_tests_pass = True

    seqlen = 1  # Generating one token per user at a time
    batch = 32

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = torch.zeros(batch, seqlen, 4096)
    tt_decode_input = pt_decode_input

    start_pos = 0

    logger.info("Compiling Model for the first time")
    decode_input, attn_mask = prepare_inputs_ttnn(
        tt_decode_input,
        model_args.dim,
        start_pos,
        model_args,
        tt_model.device_mesh,
    )
    decode_input = ttnn.to_device(decode_input, t3k_device_mesh, memory_config=ttnn.L1_MEMORY_CONFIG)
    attn_mask = ttnn.to_device(
        attn_mask,
        t3k_device_mesh,
        memory_config=ttnn.create_sharded_memory_config(
            shape=(32, model_args.max_seq_len),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    # print(ttnn.to_torch(tt_model.layers[0].attention.layer_past[1], mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0))[0])

    tt_model(decode_input, attn_mask)
    tt_model.start_pos = 0
    logger.info("Done Compiling Model")
    # print(ttnn.to_torch(tt_model.layers[0].attention.layer_past[1], mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0))[0])

    updated_k_cache = ttnn.as_tensor(
        torch.zeros(
            model_args.n_kv_heads,
            model_args.max_batch_size,
            model_args.max_seq_len,
            model_args.head_dim,
        ),
        mesh_mapper=ttnn.ShardTensorToMesh(t3k_device_mesh, dim=0),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
    )
    updated_v_cache = ttnn.as_tensor(
        torch.zeros(
            model_args.n_kv_heads,
            model_args.max_batch_size,
            model_args.max_seq_len,
            model_args.head_dim,
        ),
        mesh_mapper=ttnn.ShardTensorToMesh(t3k_device_mesh, dim=0),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Capture Decode Trace")
    decode_input, attn_mask = prepare_inputs_ttnn(
        tt_decode_input,
        model_args.dim,
        start_pos,
        model_args,
        tt_model.device_mesh,
    )
    decode_input = ttnn.to_device(decode_input, t3k_device_mesh, memory_config=ttnn.L1_MEMORY_CONFIG)
    attn_mask = ttnn.to_device(
        attn_mask,
        t3k_device_mesh,
        memory_config=ttnn.create_sharded_memory_config(
            shape=(32, model_args.max_seq_len),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    trace_id = ttnn.begin_trace_capture(t3k_device_mesh, cq_id=0)
    tt_out = tt_model(decode_input, attn_mask)
    ttnn.end_trace_capture(t3k_device_mesh, trace_id, cq_id=0)
    logger.info("Done Capturing Decode Trace")

    ttnn.copy_host_to_device_tensor(updated_k_cache, tt_model.layers[0].attention.layer_past[0])
    ttnn.copy_host_to_device_tensor(updated_v_cache, tt_model.layers[0].attention.layer_past[1])

    tt_model.start_pos = 0
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)
    tt_decode_input = pt_decode_input  # teacher forcing for PCC test
    for i in range(generation_length):
        start_pos = generation_start_pos + i
        updated_host_activation, updated_attn_mask = prepare_inputs_ttnn(
            tt_decode_input,
            model_args.dim,
            start_pos,
            model_args,
            tt_model.device_mesh,
        )
        start_time = time.time()
        ttnn.copy_host_to_device_tensor(updated_host_activation, decode_input)
        ttnn.copy_host_to_device_tensor(updated_attn_mask, attn_mask)

        ttnn.execute_trace(t3k_device_mesh, trace_id, cq_id=0, blocking=False)

        tt_out_host = ttnn.from_device(tt_out)
        # Convert ttnn tensor to torch tensor
        tt_output_torch = (
            ttnn.to_torch(tt_out_host, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0))[0]
            .squeeze(1)
            .view(batch, seqlen, -1)
            .detach()
            .float()
        )
        print(f"Time taken for iteration {i}: {time.time() - start_time}")

        # Measure PCC
        positions = torch.LongTensor([start_pos])
        ref_output = reference_model(pt_decode_input, positions).detach().float()

        passing, pcc_message = comp_pcc(
            ref_output.view(batch, seqlen, -1), tt_output_torch.view(batch, seqlen, -1), 0.99
        )
        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        reference_top1 = np.argmax(ref_output, axis=-1).squeeze()
        top1_acc = top_k_accuracy_score(
            reference_top1, tt_output_torch.squeeze(), k=1, labels=np.arange(tt_output_torch.shape[-1])
        )
        top5_acc = top_k_accuracy_score(
            reference_top1, tt_output_torch.squeeze(), k=5, labels=np.arange(tt_output_torch.shape[-1])
        )
        logger.info(f"Mean Top-1: {top1_acc}")
        logger.info(f"Mean Top-5: {top5_acc}")

        ref_token_batch = ref_output.squeeze().argmax(axis=-1)
        tt_token_batch = tt_output_torch.squeeze().argmax(axis=-1)
        logger.info(f"ref_output: {tokenizer.decode(ref_token_batch[0].item())}")
        logger.info(f"tt_output: {tokenizer.decode(tt_token_batch[0].item())}")
        pt_decode_input = embd(ref_token_batch).view(batch, seqlen, -1)
        tt_decode_input = pt_decode_input  # teacher forcing for PCC test

        if passing:
            logger.info("Mistral Model Passed!")
        else:
            logger.warning("Mistral Model Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All {generation_length} Mistral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral decode Failed!")
        assert all_tests_pass, f"PCC value is lower than the expected {0.99} for some of the outputs. Check Warnings!"
