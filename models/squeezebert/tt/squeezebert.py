from models.squeezebert.tt.squeezebert_for_question_answering import (
    TtSqueezeBertForQuestionAnswering,
)
from transformers import (
    SqueezeBertForQuestionAnswering as HF_SqueezeBertForQuestionAnswering,
)


def _squeezebert(config, state_dict, base_address, device):
    return TtSqueezeBertForQuestionAnswering(
        config=config,
        state_dict=state_dict,
        base_address=base_address,
        device=device,
    )


def squeezebert_for_question_answering(device) -> TtSqueezeBertForQuestionAnswering:
    model_name = "squeezebert/squeezebert-uncased"
    model = HF_SqueezeBertForQuestionAnswering.from_pretrained(model_name)

    model.eval()
    state_dict = model.state_dict()
    config = model.config
    base_address = f""
    model = _squeezebert(config, state_dict, base_address, device)
    return model
