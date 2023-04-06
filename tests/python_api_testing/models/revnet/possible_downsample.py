import torch

#TODO: Work on this when Cat and max pooling are ready.

def possible_downsample(x, in_channels, out_channels, stride=1, padding=1,
                        dilation=1):
    _, _, H_in, W_in = x.size()

    _, _, H_out, W_out = size_after_residual(x.size(), out_channels, 3, stride, padding, dilation)


    ## Shrink size of skipped input to match F(x) or G(x) size.
    ## TODO: Does this happen on device?
    # Downsample image
    if H_in > H_out or W_in > W_out:
        out = F.avg_pool2d(x, 2*dilation+1, stride, padding)


    ## Pad skipped input with zero tensors to match F(x) or G(x).
    ## TODO: Does this happen on device?
    # Pad with empty channels
    if in_channels < out_channels:

        try: out
        except: out = x

        pad = Variable(torch.zeros(
            out.size(0),
            (out_channels - in_channels) // 2,
            out.size(2), out.size(3)
        ), requires_grad=True)

        temp = torch.cat([pad, out], dim=1)
        out = torch.cat([temp, pad], dim=1)

    # If we did nothing, add zero tensor, so the output of this function
    # depends on the input in the graph
    ## TODO: Does this happen on device?
    try: out
    except:
        injection = Variable(torch.zeros_like(x.data), requires_grad=True)

#        if CUDA:
#            injection.cuda()

        out = x + injection

    return out
