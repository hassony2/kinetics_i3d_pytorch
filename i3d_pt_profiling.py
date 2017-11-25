from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.i3nception import I3nception

# Use this code to profile with kernprof
# Install using `pip install line_profiler`
# Launch `kernprof -lv i3d_pt_profiling.py`

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
batch_size = 2
frame_nb = 16  # Number of items in depth (temporal) dimension

# Initialize dataset
loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=False)

# Initialize pytorch I3D
i3nception_pt = I3nception(num_classes=400)
i3nception_pt.eval()
i3nception_pt.load_state_dict(torch.load(rgb_pt_checkpoint))
i3nception_pt.train()
i3nception_pt.cuda()

l1_loss = torch.nn.L1Loss()
sgd = torch.optim.SGD(i3nception_pt.parameters(), lr=0.001, momentum=0.9)


@profile
def run(model, dataloader, criterion, optimizer):
    # Load data
    for i, (input_2d, target) in enumerate(dataloader):
        optimizer.zero_grad
        # Prepare data for pytorch forward pass
        input_3d = input_2d.clone().unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)
        input_3d_var = torch.autograd.Variable(input_3d.cuda())

        # Pytorch forward pass
        out_pt = model(input_3d_var)
        loss = criterion(out_pt, torch.ones_like(out_pt))
        loss.backward()
        optimizer.step()


run(i3nception_pt, loader, l1_loss, sgd)
