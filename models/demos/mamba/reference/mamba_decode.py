import torch
from transformers import AutoTokenizer
import torch.nn.functional as F


def generate_through_selective_scan(
    model, tokenizer, prompt: str, n_tokens_to_gen: int = 51, sample: bool = False, top_k: int = None
):
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    for token_n in range(n_tokens_to_gen):
        with torch.no_grad():
            indices_to_input = input_ids
            next_token_logits = model(indices_to_input)[:, -1]

        probs = F.softmax(next_token_logits, dim=-1)
        (batch, vocab_size) = probs.shape

        if top_k is not None:
            (values, indices) = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = 0
            probs = probs / probs.sum(axis=1, keepdims=True)

        if sample:
            next_indices = torch.multinomial(probs, num_samples=1)
        else:
            next_indices = torch.argmax(probs, dim=-1)[:, None]

        input_ids = torch.cat([input_ids, next_indices], dim=1)

    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]

    return output_completions


def generate_through_decode(
    model, tokenizer, prompt: str, n_tokens_to_gen: int = 51, sample: bool = False, top_k: int = None
):
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_token_counts = len(input_ids[0])
    promt_plus_generated_n_tokens = prompt_token_counts + n_tokens_to_gen - 1
    for token_n in range(promt_plus_generated_n_tokens):
        with torch.no_grad():
            indices_to_input = input_ids
            next_token_logits = model(indices_to_input[:, token_n])

        probs = F.softmax(next_token_logits, dim=-1)
        (batch, vocab_size) = probs.shape

        if top_k is not None:
            (values, indices) = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = 0
            probs = probs / probs.sum(axis=1, keepdims=True)

        if sample:
            next_indices = torch.multinomial(probs, num_samples=1)
        else:
            next_indices = torch.argmax(probs, dim=-1)[:, None]

        if token_n >= prompt_token_counts - 1:
            input_ids = torch.cat([input_ids, next_indices], dim=1)

    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]

    return output_completions


from model import Mamba, ModelArgs

# One of:
#     'state-spaces/mamba-2.8b-slimpj'
#     'state-spaces/mamba-2.8b'
#     'state-spaces/mamba-1.4b'
#     'state-spaces/mamba-790m'
#     'state-spaces/mamba-370m'
#     'state-spaces/mamba-130m'
pretrained_model_name = "state-spaces/mamba-370m"

model = Mamba.from_pretrained(pretrained_model_name)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
print("Output from selective scan:")
print(generate_through_selective_scan(model, tokenizer, "Mamba is the"))


from decode_model import MambaDecode

model_decode = MambaDecode.from_pretrained(pretrained_model_name)
print("Output from decode only mode:")
print(generate_through_decode(model_decode, tokenizer, "Mamba is the"))
