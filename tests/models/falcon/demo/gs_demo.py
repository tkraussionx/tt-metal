from functools import partial

import torch
from loguru import logger

from transformers import AutoTokenizer

from tests.models.falcon.falcon_causallm import TtFalconCausalLM

from tests.models.falcon.reference.hf_modeling_falcon import FalconForCausalLM
from tests.models.falcon.model_config import (
    get_model_config,
    get_tt_cache_path,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, dump_tensor

def post_process(logits, input_ids, last_index=-1):
    dump_tensor("logits", "tt", logits)
    dump_tensor("input_ids", "tt", input_ids)
    next_token_logits = logits[:, last_index, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    print(f"topk {torch.topk(next_token_logits, 20)[1]}")
    dump_tensor("topk_output", "tt", torch.topk(next_token_logits, 5)[1])
    ids = next_tokens[:, None]
    print("OUTPUT ID", ids)
    return ids


def test_gs_demo_kv(device):
    model_version = "tiiuae/falcon-7b-instruct"
    model_config = get_model_config("BFLOAT16-DRAM")
    tt_cache_path = get_tt_cache_path(model_version)

    batch_size = 32
    seq_len = 5
    max_seq_len = 32
    num_layers = 32

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_version)

    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    base_url = ""
    max_position_embeddings = max_seq_len
    head_dim = configuration.hidden_size // configuration.n_head
    use_cache = True

    post_processor = partial(post_process)

    logger.info("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    # input_prompts = ["Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"] * batch_size
    input_prompts = ["Write a poem about Valencia"]

    logger.info("Tokenizing inputs")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        input_prompts, padding="max_length", max_length=32, add_special_tokens=False, return_tensors="pt"
    )
    prefill_ids = tokenized_inputs["input_ids"]

    logger.info("Creating TT Model")
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
    logger.info("Created TT Model")

    # TODO: Generate embeddings and attention_mask on device
    logger.info("Setting up inputs and attention masks")
    # NOTE: tt_decode_attention_mask is only valid for one decode right after prefill here
    # TODO: If we loop, we need to decouple generation for embeddings and just generate new attention_mask

    tt_layer_past = ()
    k_cache = torch.zeros(batch_size, 1, max_position_embeddings, head_dim)
    v_cache = torch.zeros(batch_size, 1, max_position_embeddings, head_dim)
    for _ in range(num_layers):
        tt_k_cache = torch2tt_tensor(k_cache, device)
        tt_v_cache = torch2tt_tensor(v_cache, device)
        tt_layer_past += ((tt_k_cache, tt_v_cache),)

    (
        tt_prefill_embeddings,
        tt_prefill_attention_mask,
    ) = tt_FalconCausalLM.model_preprocessing(prefill_ids, 0, "prefill", seq_len=seq_len)
    assert tt_prefill_attention_mask is not None

    # PREFILL
    logger.info(f"Falcon prefill for seq_len {seq_len} and one user only")
    tt_logits, tt_layer_present = tt_FalconCausalLM(
        input_embeddings=tt_prefill_embeddings,
        llm_mode="prefill",
        attention_mask=tt_prefill_attention_mask,
        user_id=0,
        layer_past=tt_layer_past,
        layer_past_len=0,
        use_cache=use_cache,
    )
    logger.info("finished prefill stage")
    tt_prefill_embeddings.deallocate()
    if tt_prefill_attention_mask is not None:
        tt_prefill_attention_mask.deallocate()

    logits = tt2torch_tensor(tt_logits).squeeze(1)
    tt_logits.deallocate()
    output_ids = post_processor(logits=logits, input_ids=prefill_ids, last_index=seq_len-1)

    generated_ids = torch.concat((prefill_ids[..., :seq_len], output_ids), dim=1)

    for key, value in tt_layer_present:
        dump_tensor("cached_key", "tt", tt2torch_tensor(key)[:1, :, :32])
        dump_tensor("cached_value", "tt", tt2torch_tensor(value)[:1, :, :32])

    kv_cache_len = seq_len  # This will increment by one after each decode
    for output_token_index in range(0):
        assert output_ids.shape[0] == 1
        decode_ids = output_ids.expand(batch_size, -1) # Expand to 32 samples because decode stage only works with batch size of 32

        logger.info(f"Falcon decode token {output_token_index} for {batch_size} users")
        (
            tt_decode_embeddings,
            tt_decode_attention_mask,
        ) = tt_FalconCausalLM.model_preprocessing(
            decode_ids, kv_cache_len, "decode", seq_len=kv_cache_len + 1
        )
        assert tt_decode_attention_mask is not None

        tt_logits, tt_layer_present = tt_FalconCausalLM(
            input_embeddings=tt_decode_embeddings,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=tt_layer_present,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        tt_decode_embeddings.deallocate()
        if tt_decode_attention_mask is not None:
            tt_decode_attention_mask.deallocate()

        logits = tt2torch_tensor(tt_logits).squeeze(1)
        tt_logits.deallocate()

        decode_ids = decode_ids[:1] # Slice back to 1 sample
        output_ids = post_processor(logits=logits, input_ids=decode_ids)

        generated_ids = torch.concat((generated_ids, output_ids), dim=1)
        kv_cache_len += 1

    generated_ids = generated_ids.tolist()
    output_prompts = tokenizer.batch_decode(generated_ids)

    for input_prompt, output_prompt in zip(input_prompts, output_prompts):
        logger.info(f"input: {input_prompt}")
        logger.info(f"output: {output_prompt}")
