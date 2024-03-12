# Mistral7B Demo

Demo showcasing Mistral-7B-instruct running on Wormhole, using ttnn.

## How to Run

To run the model for a single user you can use the command line input:

`pytest --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/demos/mistral7b/demo/demo.py`

To run the demo using pre-written prompts for a batch of 32 users run (currently only supports same token-length inputs):

`pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/mistral7b/demo/input_data_questions.json' models/demos/mistral7b/demo/demo.py`


## Inputs

A sample of input prompts for 32 users is provided in `input_data_question.json` in the demo directory. If you wish you to run the model using a different set of input prompts you can provide a different path, e.g.:

`pytest --disable-warnings -q -s --input-method=json --input-path='path_to_input_prompts.json' models/demos/mistral7b/demo/demo.py`


## Details

This model uses the configs and weights from HuggingFace [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).
It expects the three weights checkpoint files `pytorch_model-0000X-of-00003.bin` as well as the tokenizer model.
You can provide the path to the folder containing these by adding the path argument to `TtModelArgs(model_base_path=<weights_folder>)`.

For more configuration settings, please check the file `tt/model_config.py`.

The first time you run the model, the weights will be processed into the target data type and stored on your machine, which will take a few minutes for the full model. In future runs, the weights will be loaded from your machine and it will be faster.
