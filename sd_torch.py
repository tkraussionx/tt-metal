import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
prompt = "A girl doing cartwheels in the park."

generator = torch.manual_seed(174)
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=30, generator=generator).images[0]

## you can save the image with
image.save(f"LATEST-torch-result.png")
