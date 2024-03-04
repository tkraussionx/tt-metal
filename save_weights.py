from models.demos.falcon40b.reference.hf_modeling_falcon import FalconForCausalLM, FalconConfig
import torch

# Specify the model configuration and model name
model_name = "tiiuae/falcon-40b-instruct"
config = FalconConfig.from_pretrained(model_name)

print("Loaded config")

# Download the entire model
model = FalconForCausalLM.from_pretrained(model_name)

print("Loaded model")

# Specify the path to save the first layer weights
layer_weights_path = "models/demos/falcon40b/datasets/tt_dnn-models/tt/Falcon/tiiuae/falcon-40b-instruct/transformer.h."

print("Done")
# Extract and save the weights of the first layer
for i in range(31, 60):
    layer_weights = model.transformer.h[i].state_dict()
    path = layer_weights_path + str(i) + ".pt"
    torch.save(layer_weights, path)

    print(f"First layer parameters saved to: {path}")
