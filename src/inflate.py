import torch
from torch.nn.parameter import Parameter


def inflate_convolution(conv2d, time_dim=3, time_padding=0, time_stride=1, time_dilation=1):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
    stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
    dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
    conv3d = torch.nn.Conv3d(conv2d.in_channels, conv2d.out_channels,
                             kernel_dim, padding=padding, dilation=dilation)
    # Repeat filter time_dim times along time dimension
    weight_3d = conv2d.weight.unsqueeze(2).repeat(1, 1, time_dim, 1, 1).data
    weight_3d = weight_3d / time_dim

    # Assign new params
    conv3d.weight = Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d


def inflate_batch_norm(batch2d):
    # In pytorch 0.2.0 the 2d and 3d versions work identically
    # except for the check that verifies the input dimensions

    batch3d = torch.nn.BatchNorm3d(batch2d.num_features)
    # retrieve 3d _check_input_dim function
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d
