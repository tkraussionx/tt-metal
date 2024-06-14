# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
import numpy as np
from loguru import logger
from sklearn.metrics import top_k_accuracy_score
import time

# Set Mixtral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["MIXTRAL_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor

from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    prepare_inputs_ttnn_prefill,
    prepare_rotation_mat_ttnn,
    get_rot_transformation_mat,
)
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
    "seq_len",
    (8192,),
)
@pytest.mark.parametrize(
    "n_layers",
    (1,),
)
def test_mixtral_model_inference(t3k_device_mesh, use_program_cache, reset_seeds, n_layers, seq_len):
    pcc = 0.96
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(t3k_device_mesh.get_device(0))
    model_args.n_layers = n_layers
    batch = 1
    if seq_len > 2048:
        model_args.max_seq_len = seq_len
        model_args.max_batch_size = 1
    state_dict = model_args.load_state_dict()

    tokenizer = Tokenizer(model_args.tokenizer_path)
    prompt = "Once upon a time, in a charming countryside, there lived three little pigs named Porky, Petunia, and Percy. They were siblings and loved to play together all day long. But one day, their mother knew it was time for them to build their own homes and be independent. Remember, my little ones, their mother said, the world can be tricky, so build your houses strong and sturdy to keep you safe from harm. With hugs and kisses, the three little pigs bid farewell to their mother and set off on their journey to find the perfect spot to build their homes. Porky, being the laziest of the bunch, quickly found a pile of straw nearby and decided it was the perfect place to build his house. With little effort, he constructed a cozy straw house and declared, I'm done! Now I can relax and play all day. Petunia was a bit more hardworking. She found a bunch of sticks and twigs and began building her house. It took a bit longer, but she managed to create a charming little house. Percy, the wisest of the three, knew that hard work pays off. He searched for the sturdiest materials he could find and finally decided on bricks. He carefully stacked and cemented the bricks together, creating a strong and reliable house. One evening, as the sun was setting, a big bad wolf happened upon the three little pigs. He was hungry and had his eyes set on the tasty pigs. The wolf first came across Porky's straw house. The wolf knocked on the flimsy door and called out in a deep, menacing voice, Little pig, little pig, let me in. Porky trembled inside his straw house but managed to reply, Not by the hair on my chinny-chin-chin. The wolf growled and said, Then I'll huff, and I'll puff, and I'll blow your house in. With a mighty breath, the wolf blew down the straw house, sending pieces of straw flying in all directions. Porky squealed in fear and dashed over to Petunia's stick house, just as the wolf began his chase. Petunia welcomed her frightened brother and quickly bolted the door behind him. But the wolf was relentless. He soon arrived at the stick house and repeated his sinister request, Little pigs, little pigs, let me in. Petunia and Porky shouted back in unison, Not by the hairs on our chinny-chin-chins. The wolf snarled, Then I'll huff, and I'll puff, and I'll blow your house in. With even greater force, the wolf blew down the stick house, scattering sticks and twigs all around. Petunia and Porky fled as fast as their legs could carry them to Percy’s brick house, hoping it would offer them the safety they desperately needed. When they reached Percy’s house, they hurried inside and locked the door. Percy, calm and collected, assured his siblings, Don't worry. This house is strong. The wolf won't get in. The wolf soon arrived at Percy’s brick house, now even more determined. He knocked on the sturdy door and repeated his demand, Little pigs, little pigs, let me in. The three little pigs chorused together, Not by the hairs on our chinny-chin-chins. The wolf's eyes narrowed as he replied, Then I'll huff, and I'll puff, and I'll blow your house in. He huffed, and he puffed, but no matter how hard he tried, the brick house stood firm. Exhausted and frustrated, the wolf decided to change his tactic. He began to climb up the house, intending to enter through the chimney. But Percy, ever the clever pig, had foreseen this possibility. He quickly set a large pot of water to boil in the fireplace. As the wolf began to descend the chimney, he felt the heat rising and knew he was in trouble. With a yelp, he reversed course and scampered back up, singed and thoroughly defeated. Percy, Petunia, and Porky celebrated their victory and the security of their brick house. The wolf, nursing his burns and his pride, slunk away into the forest, never to bother the three little pigs again. The siblings learned valuable lessons from their adventure. Porky understood the importance of hard work, Petunia realized that taking time to do a job right is crucial, and Percy reaffirmed his belief in preparation and forethought. They lived happily ever after in their strong, sturdy brick house, safe from harm and always together. With the wolf gone and peace restored, the three little pigs settled into a happy routine in their brick house. They each took on different responsibilities to ensure their home remained strong and their lives comfortable. Percy, always the planner, managed the construction and maintenance of the house, ensuring every brick was in place and every corner was secure. Petunia, with her eye for detail, took care of the garden, growing fresh vegetables and beautiful flowers that surrounded their home, creating a pleasant and bountiful environment. Porky, having learned the value of hard work, found his niche in the kitchen, experimenting with recipes and ensuring there were always delicious meals for them to enjoy. One bright morning, as the three pigs were having breakfast, they heard a knock on the door. They looked at each other, a bit wary after their previous encounter with the wolf, but Percy, ever cautious, peered through the peephole. To their surprise, it was a young, friendly-looking rabbit holding a basket of carrots. The rabbit introduced himself as Remy and explained that he was their new neighbor. He had heard about their encounter with the wolf and wanted to offer them a gift from his garden as a token of friendship. The pigs, relieved and delighted, welcomed Remy inside. They shared stories and laughter, forging a new friendship. Remy, who was quite knowledgeable about the forest, offered valuable advice on how to spot signs of danger and how to grow the best crops. In return, the pigs taught him about bricklaying and cooking, creating a mutually beneficial exchange of skills. As the days turned into weeks, the pigs and Remy’s bond grew stronger. They decided to build a network of tunnels connecting their homes, providing a safe passageway in case of any future threats. Together, they formed a close-knit community, looking out for one another and working together to improve their lives. Word of their cooperative and resilient community spread throughout the countryside. Soon, other animals began to visit, seeking advice and wishing to join their harmonious neighborhood. The pigs, with Remy’s help, guided their new friends in building strong homes and fostering a sense of unity and support. One evening, as the sun set and cast a golden glow over the flourishing garden, Percy, Petunia, and Porky sat together, reflecting on their journey. They had come a long way from the day they left their mother’s home. Their experiences had taught them the value of hard work, cooperation, and friendship. They knew that whatever challenges might come their way, they would face them together, stronger than ever. The little community continued to thrive, and the story of the three little pigs and their friends became a cherished tale, passed down through generations, reminding everyone that strength lies not just in bricks and mortar, but in the bonds of friendship and the spirit of working together. And so, in their strong, sturdy brick house, surrounded by friends and laughter, the three little pigs lived happily ever after."
    print("Prompt: ", len(prompt))
    prompt = prompt * 5
    print("Prompt length: ", len(tokenizer.encode(prompt)))
    encoded_prompts = tokenizer.encode(prompt)[:seq_len]
    reference_model = Transformer(args=model_args)
    reference_model.load_state_dict(state_dict)
    reference_model.eval()

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})
    rot_mats = prepare_rotation_mat_ttnn(
        model_args.head_dim, model_args.max_seq_len, t3k_device_mesh, mode="prefill", seq_len=seq_len
    )
    head_dim = model_args.dim // model_args.n_heads
    transformation_mat_torch = get_rot_transformation_mat(head_dim)
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_device_mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(t3k_device_mesh),
    )
    # Load TTNN model
    tt_model = TtTransformer(
        device_mesh=t3k_device_mesh,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
    )

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    # pt_decode_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1  #
    pt_decode_input = embd(encoded_prompts_tensor).view(batch, seq_len, -1)
    tt_decode_input = pt_decode_input

    start_pos = 0
    current_pos = start_pos % model_args.sliding_window

    for iter in range(1):
        start_time = time.time()
        decode_input, attn_mask, attn_mask_torch = prepare_inputs_ttnn_prefill(
            tt_decode_input,
            tt_model.device_mesh,
        )

        # Run TT model
        tt_out = tt_model(
            decode_input, start_pos, current_pos, attn_mask, rot_mats, transformation_mats, 0, mode="prefill"
        )
        # Work around program cache issue https://github.com/tenstorrent/tt-metal/issues/7159
        del decode_input, attn_mask
        # Convert ttnn tensor to torch tensor
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0))[0]
            .squeeze(1)
            .view(batch, seq_len, -1)
            .detach()
            .float()
        )

        # logger.info(f"seqlen: {seq_len}, iter: {iter}, TTNN Inference time: {time.time() - start_time:.2f} sec")

    # Measure PCC
    positions = torch.LongTensor(range(seq_len))
    ref_output = reference_model(pt_decode_input, positions, attn_mask_torch, mode="prefill").detach().float()

    passing, pcc_message = comp_pcc(ref_output.view(batch, seq_len, -1), tt_output_torch.view(batch, seq_len, -1), pcc)
    logger.info(comp_allclose(ref_output, tt_output_torch))
    logger.info(pcc_message)
    for layer in range(n_layers):
        ref = reference_model.layers[layer]
        tt = tt_model.layers[layer]
        for mod in range(len(ref.comps)):
            passing, pcc_message = comp_pcc(
                ref.comps[mod].view(batch, seq_len, -1), tt.comps[mod].view(batch, seq_len, -1), pcc
            )
            print("layer: ", layer, "mod: ", mod, pcc_message)
    if passing:
        logger.info(f"Mistral decode Passed!")
    else:
        logger.warning("Mistral decode Failed!")
        assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
