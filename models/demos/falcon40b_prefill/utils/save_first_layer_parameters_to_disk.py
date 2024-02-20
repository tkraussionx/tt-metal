from models.demos.falcon40b_prefill.reference.hf_modeling_falcon import FalconForCausalLM, FalconConfig
import torch

# Specify the model configuration and model name
model_name = "tiiuae/falcon-40b-instruct"
config = FalconConfig.from_pretrained(model_name)

# Download the entire model
model = FalconForCausalLM.from_pretrained(model_name)

# Specify the path to save the first layer weights
first_layer_weights_path = (
    "models/demos/falcon40b_prefill/datasets/tt_dnn-models/tt/Falcon/tiiuae/falcon-40b-instruct/transformer.h.0.pt"
)

# Extract and save the weights of the first layer
first_layer_weights = model.transformer.h[0].state_dict()
torch.save(first_layer_weights, first_layer_weights_path)

print(f"First layer parameters saved to: {first_layer_weights_path}")
