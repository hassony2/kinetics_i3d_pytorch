import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src import inflate

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
dataset = datasets.ImageFolder('/sequoia/data1/yhasson/datasets/test-dataset',
        transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

densenet = torchvision.models.densenet121(pretrained=True)
features = densenet.features
seq2d = torch.nn.Sequential(
    features.conv0, features.norm0, features.relu0, features.pool0)
i3features = torch.nn.Sequential(
    inflate.inflate_conv(features.conv0, 3),
    inflate.inflate_batch_norm(features.norm0),
    features.relu0,
    inflate.inflate_pool(features.pool0, 1))

features_2d = densenet.features

class _Transition3d(torch.nn.Sequential):
    def __init__(self, transition2d):
        super(_Transition3d, self).__init__()
        for name, layer in transition2d.named_children():
            if isinstance(layer, torch.nn.BatchNorm2d):
                self.add_module(name, inflate.inflate_batch_norm(layer))
            elif isinstance(layer, torch.nn.ReLU):
                self.add_module(name, layer)
            elif isinstance(layer, torch.nn.Conv2d):
                self.add_module(name, inflate.inflate_conv(layer, 1))
            elif isinstance(layer, torch.nn.AvgPool2d):
                self.add_module(name, inflate.inflate_pool(layer, 1))
            else:
                raise ValueError('{} is not among handled layer types'.format(type(layer)))

class _DenseLayer3d(torch.nn.Sequential):
    def __init__(self, denselayer2d):
        super(_DenseLayer3d, self).__init__()
        for name, child in denselayer2d.named_children():
            if isinstance(child, torch.nn.BatchNorm2d):
                self.add_module(name, inflate.inflate_batch_norm(child))
            elif isinstance(child, torch.nn.ReLU):
                self.add_module(name, child)
            elif isinstance(child, torch.nn.Conv2d):
                self.add_module(name, inflate.inflate_conv(child, 1))
            else:
                raise ValueError('{} is not among handled layer types'.format(type(child)))
        self.drop_rate = denselayer2d.drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer3d, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

def inflate_features(features):
    features3d = torch.nn.Sequential()
    for name, child in features.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            features3d.add_module(name, inflate.inflate_batch_norm(child))
        elif isinstance(child, torch.nn.ReLU):
            features3d.add_module(name, child)
        elif isinstance(child, torch.nn.Conv2d):
            features3d.add_module(name, inflate.inflate_conv(child, 1))
        elif isinstance(child, torch.nn.MaxPool2d) or isinstance(child, torch.nn.AvgPool2d):
            features3d.add_module(name, inflate.inflate_pool(child))
        elif isinstance(child, torchvision.models.densenet._DenseBlock):
            block = torch.nn.Sequential()
            for nested_name, nested_child in child.named_children():
                assert isinstance(nested_child, torchvision.models.densenet._DenseLayer)
                block.add_module(nested_name, _DenseLayer3d(nested_child))
            features3d.add_module(name, block)
        elif isinstance(child, torchvision.models.densenet._Transition):
            features3d.add_module(name, _Transition3d(child))
        else:
            raise ValueError('{} is not among handled layer types'.format(type(child)))
    return features3d
                


classifier_2d = densenet.classifier

frame_nb = 4
class i3DenseNet(torch.nn.Module):
    def __init__(self, densenet2d, frame_nb):
        super(i3DenseNet, self).__init__()
        self.features = inflate_features(densenet2d.features)
        self.classifier = inflate.inflate_linear(densenet2d.classifier, frame_nb)
        
    def forward(self, inp):
        features = self.features(inp)
        out = torch.nn.functional.relu(features)
        out = torch.nn.functional.avg_pool3d(out, kernel_size=(1, 7, 7))
        out = out.view(-1, frame_nb*1024)
        out = self.classifier(out)
        return out


loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
i3densenet = i3DenseNet(densenet, frame_nb)
for i, (input_2d, target) in enumerate(loader):
    target = target.cuda()
    target_var = torch.autograd.Variable(target)
    input_2d_var = torch.autograd.Variable(input_2d)
    out2d = densenet(input_2d_var)
    
    input_3d = input_2d.unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)
    # # Add padding
    # time_pad = torch.nn.ReplicationPad3d((0, 0, 0, 0, 1, 1))
    # input_3d_var = time_pad(input_3d) 
    # out3d = i3features(input_3d_var)
    input_3d_var = torch.autograd.Variable(input_3d) 
    out3d = i3densenet(input_3d_var)

    expected_out_3d = out2d.data.unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)
    import pdb; pdb.set_trace()
    out_diff = expected_out_3d - out3d.data
    print(out_diff.max())
    assert(out_diff.max() < 0.0001)
