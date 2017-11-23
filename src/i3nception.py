import math
import os

import numpy as np
import torch
from torch.nn import ReplicationPad3d
import torchvision

from src import inflate


class Unit3Dpy(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding=0,
                 use_bias=False,
                 use_bn=True):
        super(Unit3Dpy, self).__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=use_bias)
        self.batch3d = torch.nn.BatchNorm3d(out_channels)
        if activation == 'relu':
            self.activation = torch.nn.functional.relu
        else:
            raise ValueError(
                'activation "{}" not recognized'.format(activation))

    def forward(self, inp):
        conv_out = self.conv3d(inp)
        batch_out = self.batch3d(conv_out)
        out = torch.nn.functional.relu(batch_out)
        return out


class I3nception(torch.nn.Module):
    def __init__(self, num_classes, spatial_squeeze=True, name='inception'):
        super(I3nception, self).__init__()
        self.name = name
        self.num_classes = num_classes
        self.spatial_squeeze = spatial_squeeze
        conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64,
            in_channels=3,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2))
        self.conv3d_1a_7x7 = conv3d_1a_7x7

    def forward(self, inp):
        out_1a = self.conv3d_1a_7x7(inp)
        return out_1a


def get_conv_params(sess, name):
    # Get conv weights
    conv_weights_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'w:0'))
    conv_weights = sess.run(conv_weights_tensor)
    conv_shape = conv_weights.shape

    kernel_shape = conv_shape[0:3]
    in_channels = conv_shape[3]
    out_channels = conv_shape[4]

    conv_op = sess.graph.get_operation_by_name(
        os.path.join(name, 'convolution'))
    padding_name = conv_op.get_attr('padding')
    padding = _get_padding(padding_name, kernel_shape)
    all_strides = conv_op.get_attr('strides')
    strides = all_strides[1:4]
    return conv_weights, kernel_shape, in_channels, out_channels, strides, padding


def get_bn_params(sess, name):
    moving_mean_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_mean:0'))
    moving_var_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_variance:0'))
    beta_tensor = sess.graph.get_tensor_by_name(os.path.join(name, 'beta:0'))
    moving_mean = sess.run(moving_mean_tensor)
    moving_var = sess.run(moving_var_tensor)
    beta = sess.run(beta_tensor)
    return moving_mean, moving_var, beta


def _get_padding(padding_name, conv_shape):
    padding_name = padding_name.decode("utf-8")
    if padding_name == "VALID":
        return [0, 0]
    elif padding_name == "SAME":
        #return [math.ceil(int(conv_shape[0])/2), math.ceil(int(conv_shape[1])/2)]
        return [
            math.floor(int(conv_shape[0]) / 2),
            math.floor(int(conv_shape[1]) / 2),
            math.floor(int(conv_shape[2]) / 2)
        ]
    else:
        raise ValueError('Invalid padding name ' + padding_name)


def load_conv3d(state_dict, name_pt, sess, name_tf):
    # Transfer convolution params
    conv_name_tf = os.path.join(name_tf, 'conv_3d')
    conv_weights, kernel_shape, in_channels, out_channels, strides, padding = get_conv_params(
        sess, conv_name_tf)
    # state_dict[name_pt + '.conv3d.weight'] = torch.from_numpy(conv_weights)
    conv_weights_rs = np.transpose(
        conv_weights, (4, 3, 0, 1,
                       2))  # to pt format (out_c, in_c, depth, height, width)
    state_dict['conv3d.weight'] = torch.from_numpy(conv_weights).permute(
        4, 3, 0, 1, 2)

    conv_tf_name = os.path.join(name_tf, 'batch_norm')
    moving_mean, moving_var, beta = get_bn_params(sess, conv_tf_name)

    # Transfer batch norm params
    out_planes = conv_weights_rs.shape[0]
    # state_dict[name_pt + '.batch3d.weight'] = torch.ones(out_planes)
    # state_dict[name_pt + '.batch3d.bias'] = torch.from_numpy(beta)
    # state_dict[name_pt
    #            + '.batch3d.running_mean'] = torch.from_numpy(moving_mean)
    # state_dict[name_pt + '.batch3d.running_var'] = torch.from_numpy(moving_var)
    state_dict['batch3d.weight'] = torch.ones(out_planes).squeeze()
    state_dict['batch3d.bias'] = torch.from_numpy(beta).squeeze()
    state_dict['batch3d.running_mean'] = torch.from_numpy(
        moving_mean).squeeze()
    state_dict['batch3d.running_var'] = torch.from_numpy(moving_var).squeeze()

    # conv_out = sess.graph.get_operation_by_name(
    #    'RGB/inception_i3d/Conv3d_1a_7x7/conv_3d/convolution').outputs[
    #        0].eval()
