from models.demos.falcon40b.reference.hf_modeling_falcon import FalconForCausalLM, FalconConfig
import torch

# Specify the model configuration and model name
model_name = "tiiuae/falcon-40b-instruct"
config = FalconConfig.from_pretrained(model_name)

# Download the entire model
model = FalconForCausalLM.from_pretrained(model_name, num_hidden_layers=1)

# Specify the path to save the first layer weights
first_layer_weights_path = (
    "models/demos/falcon40b/datasets/tt_dnn-models/tt/Falcon/tiiuae/falcon-40b-instruct/transformer.h.0.pt"
)

# Extract and save the weights of the first layer
# first_layer_weights = model.transformer.h[0].state_dict()
# torch.save(first_layer_weights, first_layer_weights_path)

# torch.save(model.state_dict()["transformer.word_embeddings.weight"], "models/demos/falcon40b/datasets/tt_dnn-models/tt/Falcon/tiiuae/falcon-40b-instruct/embedding.pt")
torch.save(
    model.state_dict()["transformer.ln_f.weight"],
    "models/demos/falcon40b/datasets/tt_dnn-models/tt/Falcon/tiiuae/falcon-40b-instruct/ln_f.weight.pt",
)
torch.save(
    model.state_dict()["transformer.ln_f.bias"],
    "models/demos/falcon40b/datasets/tt_dnn-models/tt/Falcon/tiiuae/falcon-40b-instruct/ln_f.bias.pt",
)
print(f"First layer parameters saved to: {first_layer_weights_path}")
