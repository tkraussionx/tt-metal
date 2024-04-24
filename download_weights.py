from models.demos.falcon40b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)


def test_download_weights(
    model_location_generator,
):
    print("DLing weights")
    model_name = model_location_generator("tiiuae/falcon-40b-instruct", model_subdir="Falcon")

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name)
    hugging_face_reference_model.eval()
