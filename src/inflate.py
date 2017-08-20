import torch
from torch.nn.parameter import Parameter

def inflate_convolution(conv2d, time_dim=3):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    padding = (0, 1, 1)
    conv3d = torch.nn.Conv3d(conv2d.in_channels, conv2d.out_channels, time_dim, padding=padding)
    # Repeat filter time_dim times along time dimension
    weight_3d = conv2d.weight.unsqueeze(2).repeat(1, 1, time_dim, 1, 1).data
    weight_3d = weight_3d/time_dim

    # Assign new params
    conv3d.weight = Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d
