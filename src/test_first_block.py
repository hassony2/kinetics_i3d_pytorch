import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src import inflate

def test_input_block():
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
    seq3d = torch.nn.Sequential(
        inflate.inflate_conv(features.conv0, 3),
        inflate.inflate_batch_norm(features.norm0),
        features.relu0,
        inflate.inflate_pool(features.pool0, 1))

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    frame_nb = 4
    for i, (input_2d, target) in enumerate(loader):
        target = target.cuda()
        target_var = torch.autograd.Variable(target)
        input_2d_var = torch.autograd.Variable(input_2d)
        out2d = seq2d(input_2d_var)
        time_pad = torch.nn.ReplicationPad3d((0, 0, 0, 0, 1, 1))
        input_3d = input_2d.unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)
        input_3d_var = time_pad(input_3d) 
        out3d = seq3d(input_3d_var)
        expected_out_3d = out2d.data.unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)
        out_diff = expected_out_3d - out3d.data
        print(out_diff.max())
        assert(out_diff.max() < 0.0001)
