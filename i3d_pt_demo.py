import os

import numpy as np
from matplotlib import pyplot as plt
from src import i3dtf, i3nception
from src.i3nception import I3nception

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

rgb_pt_checkpoint = 'model/model_rgb.pth'
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
im_size = 224
dataset = datasets.ImageFolder(
    '/sequoia/data1/yhasson/datasets/test-dataset',
    transforms.Compose([
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
        #                                   normalize,
    ]))

# Initialize input params
in_channels = 3
batch_size = 2
frame_nb = 16  # Number of items in depth (temporal) dimension
class_nb = 400

# Initialize dataset
loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=False)

# Initialize pytorch I3D
i3nception_pt = I3nception(num_classes=400)
i3nception_pt.eval()
i3nception_pt.load_state_dict(torch.load(rgb_pt_checkpoint))
i3nception_pt.cuda()

criterion = torch.nn.L1Loss()


#@profile
def run(model, dataloader, criterion):
    # Load data
    for i, (input_2d, target) in enumerate(dataloader):
        # Prepare data for pytorch forward pass
        target_var = torch.autograd.Variable(target)
        input_3d = input_2d.clone().unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)
        input_3d_var = torch.autograd.Variable(input_3d.cuda())

        # Pytorch forward pass
        out_pt = model(input_3d_var)
        out = out_pt.data.cpu().numpy()
        loss = criterion(out_pt, torch.ones_like(out_pt))
        loss.backward()


run(i3nception_pt, loader, criterion)
