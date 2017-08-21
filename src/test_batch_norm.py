import torch
from torch.autograd import Variable

from src.inflate import inflate_batch_norm

def test_inflate_batch_norm():
    torch.manual_seed(0)

    input_space_dim = 10  # Dimensions of input image
    in_channels = 3  # input feature map dim
    out_channels = 2  # output feature map dim

    filter_space_dim = 5  # conv filter spatial dim
    filter_time_dim = 3  # conv filter temporal dim

    frame_nb = 5
    batch_dim = 10

    # Initialize inputs
    input_img = torch.rand(batch_dim, in_channels, input_space_dim, input_space_dim)
    input_2d_var = Variable(input_img)
    input_3d = input_img.unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)
    input_3d_var = Variable(input_3d)

    # Initialize batch_norm
    batch2d = torch.nn.BatchNorm2d(in_channels)
    batch3d = inflate_batch_norm(batch2d)

    # Compute outputs
    out_2d = batch2d(input_2d_var)
    out_3d = batch3d(input_3d_var)
    expected_out_3d = out_2d.data.unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)

    output_diff = out_3d.data - expected_out_3d
    assert(output_diff.max() == 0)
    batch2d.eval()
    batch3d.eval()
    out_2d = batch2d(input_2d_var)
    out_3d = batch3d(input_3d_var)
    expected_out_3d = out_2d.data.unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)

    output_diff = out_3d.data - expected_out_3d
    assert(output_diff.max() == 0)
