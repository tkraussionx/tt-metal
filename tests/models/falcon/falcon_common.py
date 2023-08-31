import torch

MODEL_VERSION = "tiiuae/falcon-7b-instruct"

from tests.models.falcon.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)

class PytorchFalconCausalLM(torch.nn.Module):
    def __init__(self, hf_reference_model, num_layers=None):
        super().__init__()
        self.model = hf_reference_model

        if num_layers is None:
            pass
        else:
            self.model.transformer.h = self.model.transformer.h[:num_layers]

        # Disable dropout
        self.model.eval()

    def forward(self, input_ids, past_key_values, use_cache):
        # this method is returning the logits
        result = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache, return_dict=False)
        return result


def load_from_weka_or_hf_cache(model_version, model_subdir):
    try:
        model_name = model_location_generator(model_version, model_subdir=model_subdir)
        model = FalconForCausalLM.from_pretrained(model_name)
    except OSError:
        logger.warning("Failed loading the weights from weka. Loading them from HF cache instead")
        model = FalconForCausalLM.from_pretrained(model_version)
    return model
