import math

import torch
import torch.nn.functional as F
from torch.nn import ReplicationPad3d
import torchvision

from src import inflate


class I3DenseNet(torch.nn.Module):
    def __init__(self, densenet2d, frame_nb, inflate_block_convs=False):
        super(I3DenseNet, self).__init__()
        self.frame_nb = frame_nb
        self.features, transition_nb = inflate_features(
            densenet2d.features, inflate_block_convs=inflate_block_convs)
        self.final_time_dim = frame_nb // int(math.pow(
            2,
            transition_nb))  # time_dim is divided by two for each transition
        self.classifier = inflate.inflate_linear(densenet2d.classifier,
                                                 self.final_time_dim)

    def forward(self, inp):
        features = self.features(inp)
        out = torch.nn.functional.relu(features)
        out = torch.nn.functional.avg_pool3d(out, kernel_size=(1, 7, 7))
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(
            -1, self.final_time_dim * 1024)
        out = self.classifier(out)
        return out


class _DenseLayer3d(torch.nn.Sequential):
    def __init__(self, denselayer2d, inflate_convs=False):
        super(_DenseLayer3d, self).__init__()

        self.inflate_convs = inflate_convs
        for name, child in denselayer2d.named_children():
            if isinstance(child, torch.nn.BatchNorm2d):
                self.add_module(name, inflate.inflate_batch_norm(child))
            elif isinstance(child, torch.nn.ReLU):
                self.add_module(name, child)
            elif isinstance(child, torch.nn.Conv2d):
                kernel_size = child.kernel_size[0]
                if inflate_convs and kernel_size > 1:
                    # Pad input in the time dimension
                    assert kernel_size % 2 == 1, 'kernel size should be\
                            odd be got {}'.format(kernel_size)
                    pad_size = int(kernel_size / 2)
                    pad_time = ReplicationPad3d((0, 0, 0, 0, pad_size,
                                                 pad_size))
                    self.add_module('padding.1', pad_time)
                    # Add time dimension of same dim as the space one
                    self.add_module(name,
                                    inflate.inflate_conv(child, kernel_size))
                else:
                    self.add_module(name, inflate.inflate_conv(child, 1))
            else:
                raise ValueError(
                    '{} is not among handled layer types'.format(type(child)))
        self.drop_rate = denselayer2d.drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer3d, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition3d(torch.nn.Sequential):
    def __init__(self, transition2d, inflate_conv=False):
        """
        Inflates transition layer from transition2d
        """
        super(_Transition3d, self).__init__()
        for name, layer in transition2d.named_children():
            if isinstance(layer, torch.nn.BatchNorm2d):
                self.add_module(name, inflate.inflate_batch_norm(layer))
            elif isinstance(layer, torch.nn.ReLU):
                self.add_module(name, layer)
            elif isinstance(layer, torch.nn.Conv2d):
                if inflate_conv:
                    pad_time = ReplicationPad3d((0, 0, 0, 0, 1, 1))
                    self.add_module('padding.1', pad_time)
                    self.add_module(name, inflate.inflate_conv(layer, 3))
                else:
                    self.add_module(name, inflate.inflate_conv(layer, 1))
            elif isinstance(layer, torch.nn.AvgPool2d):
                self.add_module(name, inflate.inflate_pool(layer, 2))
            else:
                raise ValueError(
                    '{} is not among handled layer types'.format(type(layer)))


def inflate_features(features, inflate_block_convs=False):
    """
    Inflates the feature extractor part of DenseNet by adding the corresponding
    inflated modules and transfering the inflated weights
    """
    features3d = torch.nn.Sequential()
    transition_nb = 0  # Count number of transition layers
    for name, child in features.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            features3d.add_module(name, inflate.inflate_batch_norm(child))
        elif isinstance(child, torch.nn.ReLU):
            features3d.add_module(name, child)
        elif isinstance(child, torch.nn.Conv2d):
            features3d.add_module(name, inflate.inflate_conv(child, 1))
        elif isinstance(child, torch.nn.MaxPool2d) or isinstance(
                child, torch.nn.AvgPool2d):
            features3d.add_module(name, inflate.inflate_pool(child))
        elif isinstance(child, torchvision.models.densenet._DenseBlock):
            # Add dense block
            block = torch.nn.Sequential()
            for nested_name, nested_child in child.named_children():
                assert isinstance(nested_child,
                                  torchvision.models.densenet._DenseLayer)
                block.add_module(nested_name,
                                 _DenseLayer3d(
                                     nested_child,
                                     inflate_convs=inflate_block_convs))
            features3d.add_module(name, block)
        elif isinstance(child, torchvision.models.densenet._Transition):
            features3d.add_module(name, _Transition3d(child))
            transition_nb = transition_nb + 1
        else:
            raise ValueError(
                '{} is not among handled layer types'.format(type(child)))
    return features3d, transition_nb
