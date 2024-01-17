from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cpu"  # the device to load the model onto
print("model...")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
)
print("tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

prompt = "My favourite condiment is"
print("model inputs...")
model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
print("model to device...")
model.to(device)
print("generate ids...")
generated_ids = model.generate(**model_inputs, max_new_tokens=80, do_sample=True, pad_token_id=tokenizer.eos_token_id)
print("tokenizer...")
out = tokenizer.batch_decode(generated_ids)[0]
print(out)
