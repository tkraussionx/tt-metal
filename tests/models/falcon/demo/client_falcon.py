import requests
from loguru import logger
import ast

def run_client():
    url = 'http://127.0.0.1:5973/predictions/falcon7b'
    prompt = {'text': 'write me a poem about Valencia',}

    x = requests.post(url, json = prompt)
    content = ast.literal_eval(x.content.decode('utf-8'))
    logger.info(content['output'][0])


if __name__ == "__main__":
    run_client()
