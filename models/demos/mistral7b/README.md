# Mistral7B Demo

## How to Run

To run the model for a single user you can use the command line input:

`pytest --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/demos/mistral7b/demo/demo.py`

To run the demo using prewritten prompts for a batch of 32 users run (currently only supports same token-length inputs):

`pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/mistral7b/demo/input_data.json' models/demos/mistral7b/demo/demo.py`

## Inputs

A sample of input prompts for 32 users is provided in `input_data.json` in demo directory. If you wish you to run the model using a different set of input prompts you can provide a different path, e.g.:

`pytest --disable-warnings -q -s --input-method=json --input-path='path_to_input_prompts.json' models/demos/mistral7b/demo/demo.py`

## Details

This model uses the configs and weights from the original mistral code (mistral-7B-v0.1).
It expects the weights to be consolidated in a single file: `consolidated.00.pth`. You can provide the path to the folder containing the weights in `TtModelArgs(model_base_path=<weights_folder>)`.

The first time you run the model, the weights will be processed into the target data type and stored on your machine, which will take a few minutes for the full model. The second time you run the model on your machine, the weights are being read from your machine and it will be faster.
