
    @staticmethod
    def residual(x, in_channels, out_channels, params, buffers, training,
                 stride=1, padding=1, dilation=1, no_activation=False):
        """Compute a pre-activation residual function.

        Args:
            x (Variable): The input variable
            in_channels (int): Number of channels of x
            out_channels (int): Number of channels of the output

        Returns:
            out (Variable): The result of the computation

        """
        out = x

        if not no_activation:
            out = F.batch_norm(out, buffers[0], buffers[1], params[0],
                               params[1], training)
            out = F.relu(out)

        out = F.conv2d(out, params[-6], params[-5], stride, padding=padding,
                       dilation=dilation)

        out = F.batch_norm(out, buffers[-2], buffers[-1], params[-4],
                           params[-3], training)
        out = F.relu(out)
        out = F.conv2d(out, params[-2], params[-1], stride=1, padding=1,
                       dilation=1)

        return out

### TT Implementation

    @staticmethod
    def TT_residual(x, in_channels, out_channels, params, buffers, training,
                 stride=1, padding=1, dilation=1, no_activation=False):
        """Compute a pre-activation residual function.

        Args:
            x (Variable): The input variable
            in_channels (int): Number of channels of x
            out_channels (int): Number of channels of the output

        Returns:
            out (Variable): The result of the computation

        """
        #print this in example, see if I can have an array or arrays?
        
        params_ = []
        buffers_ = []
        for ii in len(params):
            params_[ii] = pad_weight(params[ii])

        for jj in len(buffers):
            buffers_[jj] = pad_weight(buffers[ii])

        x = pad_activation(x)
        x_ = tilize_to_list(x)
        out = x_

        if not no_activation:
            out = F.batch_norm(out, buffers[0], buffers[1], params[0],
                               params[1], training)
            out = F.relu(out)

        out = F.conv2d(out, params[-6], params[-5], stride, padding=padding,
                       dilation=dilation)

        out = F.batch_norm(out, buffers[-2], buffers[-1], params[-4],
                           params[-3], training)
        out = F.relu(out)
        out = F.conv2d(out, params[-2], params[-1], stride=1, padding=1,
                       dilation=1)

        return out

