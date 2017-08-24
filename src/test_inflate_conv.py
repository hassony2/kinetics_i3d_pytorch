import torch
from torch.autograd import Variable

from src.inflate import inflate_conv

def test_inflate_conv_no_padding():
    torch.manual_seed(0)

    input_space_dim = 10  # Dimensions of input image
    in_channels = 3  # input feature map dim
    out_channels = 2  # output feature map dim

    filter_space_dim = 5  # conv filter spatial dim
    filter_time_dim = 3  # conv filter temporal dim

    frame_nb = 5

    # Initialize inputs with batch dimension
    input_img = torch.rand(in_channels, input_space_dim, input_space_dim)
    input_2d_var = Variable(input_img.unsqueeze(0))
    input_3d = input_img.unsqueeze(1).repeat(1, frame_nb, 1, 1)
    input_3d_var = Variable(input_3d.unsqueeze(0))

    # Initialize convolutions
    conv2d = torch.nn.Conv2d(in_channels, out_channels, filter_space_dim, padding=1)
    conv3d = inflate_conv(conv2d, filter_time_dim)

    # Compute outputs
    out_2d = conv2d(input_2d_var)
    out_3d = conv3d(input_3d_var)
    expected_out_3d = out_2d.data.unsqueeze(2).repeat(1, 1, frame_nb - 2*int(filter_time_dim/2), 1, 1)

    output_diff = out_3d.data - expected_out_3d
    assert(output_diff.max() < 0.00001)


def test_inflate_conv_padding():
    torch.manual_seed(0)

    input_space_dim = 10  # Dimensions of input image
    in_channels = 3  # input feature map dim
    out_channels = 2  # output feature map dim

    filter_space_dim = 5  # conv filter spatial dim
    filter_time_dim = 3  # conv filter temporal dim

    frame_nb = 5
    batch_size = 4
    
    # Padding params
    pad_size = int(filter_time_dim/2)
    time_pad = torch.nn.ReplicationPad3d((0, 0, 0, 0, pad_size, pad_size))

    # Initialize inputs with batch dimension
    input_img = torch.rand(batch_size, in_channels, input_space_dim, input_space_dim)
    input_2d_var = Variable(input_img)
    input_3d = input_img.unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)
    input_3d_var = time_pad(input_3d)

    # Initialize convolutions
    conv2d = torch.nn.Conv2d(in_channels, out_channels, filter_space_dim, padding=1)
    conv3d = inflate_conv(conv2d, filter_time_dim)

    # Compute outputs
    out_2d = conv2d(input_2d_var)
    out_3d = conv3d(input_3d_var)
    expected_out_3d = out_2d.data.unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)

    output_diff = out_3d.data - expected_out_3d
    assert(output_diff.max() < 0.00001)
