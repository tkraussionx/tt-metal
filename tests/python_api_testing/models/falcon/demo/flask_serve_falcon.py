from loguru import logger
import os
import json
from tests.python_api_testing.models.falcon.demo.cpu_model import Falcon
from loguru import logger
model = None

from flask import Flask, request

app = Flask(__name__)


@app.route("/predictions/falcon7b/", methods=["POST"])
def infer_bart():
    data = request.get_json()
    text = data["text"]
    seq_len = data.get('seq_len', None)
    logger.info(text)
    logger.info(f"seq len:::: {seq_len}")
    global model
    if seq_len is None:
        response = model(text)
    else:
        response = model(text, seq_len)
    return {"output": response}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flask API Falcon model args")
    model = Falcon()
    # parser.add_argument(
    #     "-bs",
    #     "--batch_size",
    #     default=32,
    #     type=int,
    #     help="The batch size to compile with.",
    # )
    # parser.add_argument(
    #     "-md", "--model_dir", default="final_model", help="Path to model folder"
    # )


    # model = Falcon()

    app.run(host="0.0.0.0", port=5973, threaded=False)
