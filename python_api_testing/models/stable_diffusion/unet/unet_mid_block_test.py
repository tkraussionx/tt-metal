from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")


import torch.nn as nn
import torch.nn.functional as F
import torch

from pymetal import ttlib as ttl
from utility_functions import tilize_to_list, print_diff_argmax
from diffusers import StableDiffusionPipeline

from python_api_testing.models.stable_diffusion.unet.unet_2d_blocks import UNetMidBlock2D, DownEncoderBlock2D


def run_down_encoder_block_inference(device):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    vae = pipe.vae
    vae.eval()
    state_dict = vae.state_dict()
    down_block = pipe.vae.encoder.down_blocks[block_id]

    block_id = 1

    num_layers = 2
    resnet_act_fn = "silu"
    downsample_padding = 0

    if block_id == 2:
        in_channels = 512
        out_channels = 512
        add_downsample = False
        input_shape  = [1, 512, 64, 64]

    if block_id == 1:
        in_channels = 256
        out_channels = 512
        add_downsample = True
        input_shape  = [1, 256, 64, 64]

    input = torch.randn(input_shape)
    torch_out = down_block(input)

    tt_input = ttl.tensor.Tensor(tilize_to_list(input), input_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)
    tt_down_block = DownEncoderBlock2D(in_channels=in_channels,
                out_channels=out_channels,
                add_downsample=add_downsample,
                num_layers=num_layers,
                resnet_act_fn=resnet_act_fn,
                downsample_padding=downsample_padding,
                state_dict=state_dict,
                device=device,
                host=host,
                base_address=f"encoder.down_blocks.{block_id}")

    tt_out = tt_down_block(tt_input).to(host).data()
    tt_untilized_output = untilize(torch.Tensor(tt_out).reshape(torch_out.shape))

    print_diff_argmax(tt_untilized_output, torch_out)
    assert np.allclose(torch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)
    print("down block executed successfully")


def run_mid_block_inference(device):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    vae = pipe.vae
    vae.eval()
    state_dict = vae.state_dict()
    vae_encoder = pipe.vae.encoder
    mid_block = vae_encoder.mid_block


    in_channels = 512
    eps = 1e-06
    resnet_groups = 32
    input_shape  = [1, 512, 64, 64]

    input = torch.randn(input_shape)
    torch_out = mid_block(input, None)

    tt_input = ttl.tensor.Tensor(tilize_to_list(input), input_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)
    tt_mid_block = UNetMidBlock2D(in_channels=in_channels, temb_channels=None, resnet_act_fn="silu", attn_num_head_channels=1, state_dict=state_dict, device=device, host=host,)
    tt_out = tt_mid_block(tt_input, None).to(host).data()
    tt_untilized_output = untilize(torch.Tensor(tt_out).reshape(torch_out.shape))

    print_diff_argmax(tt_untilized_output, torch_out)
    assert np.allclose(torch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)
    print("mid block executed successfully")


if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_mid_block_inference(device)
    run_down_encoder_block_inference(device)
    ttl.device.CloseDevice(device)
