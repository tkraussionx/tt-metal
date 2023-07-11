from transformers import CodeGenTokenizer

from transformers import CodeGenConfig, CodeGenForCausalLM

checkpoint = "Salesforce/codegen-350M-mono"

configuration = CodeGenConfig(checkpoint)

#model = CodeGenModel(configuration)

model  = CodeGenForCausalLM.from_pretrained(checkpoint)


tokenizer = CodeGenTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
text = "def hello_world():"

completion = model.generate(**tokenizer(text, return_tensors="pt"))

print(tokenizer.decode(completion[0]))
