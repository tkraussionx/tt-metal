import torch
from torchview import draw_graph

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Images
imgs = ["https://ultralytics.com/images/zidane.jpg"]  # batch of images
print("the type of imgs is: ", type(imgs))
print("the type of imgs[0] is: ", type(imgs[0]))
# Inference
results = model(imgs)
print("\n\n\n\n\nthe type of results is: ", type(results))
# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)


## Example usage

input_tensor = torch.randn(1, 3, 384, 640)  # Batch size of 1, 3 channels (RGB), 384x640 input
output_tensor = model(input_tensor)
print("the model is: ", model)
model_graph = draw_graph(
    model,
    input_size=(1, 3, 384, 640),
    dtypes=[torch.float32],
    expand_nested=True,
    graph_name="yolov5s",
    depth=1000,
    directory=".",
)
model_graph.visual_graph.render(format="pdf")


# import torch
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
#
# dummy_input = torch.randn(1, 3, 384, 640)
# writer = SummaryWriter("runs/yolov5")
# writer.add_graph(model, dummy_input)
# writer.close()
