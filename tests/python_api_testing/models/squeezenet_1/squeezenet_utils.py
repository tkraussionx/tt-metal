import os
import torch


def download_image(path):
    # os.system(f"wget -P {path} https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
    # dog *****
    # torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", os.path.join(path, "input_image.jpg"))
    # water snake *****
    torch.hub.download_url_to_file(
        "https://upload.wikimedia.org/wikipedia/commons/1/11/Viperine_water_snake_%28Natrix_maura%29.jpg",
        os.path.join(path, "input_image.jpg"),
    )
