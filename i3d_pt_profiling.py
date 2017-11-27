import argparse

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.i3dpt import I3D

# Use this code to profile with kernprof
# Install using `pip install line_profiler`
# Launch `kernprof -lv i3d_pt_profiling.py`


@profile
def run(model, dataloader, criterion, optimizer, frame_nb):
    # Load data
    for i, (input_2d, target) in enumerate(dataloader):
        optimizer.zero_grad
        # Prepare data for pytorch forward pass
        input_3d = input_2d.clone().unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)
        input_3d_var = torch.autograd.Variable(input_3d.cuda())

        # Pytorch forward pass
        out_pt, _ = model(input_3d_var)
        loss = criterion(out_pt, torch.ones_like(out_pt))
        loss.backward()
        optimizer.step()


def run_profile(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Use pytorch image dataset, each image is duplicated in the
    # temporal dimension in order to produce a proxy for a
    # spatio-temporal video input
    dataset_path = 'data/dummy-dataset'
    dataset = datasets.ImageFolder(dataset_path,
                                   transforms.Compose([
                                       transforms.CenterCrop(args.im_size),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    # Initialize input params
    batch_size = 2

    # Initialize dataset
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)

    # Initialize pytorch I3D
    i3nception_pt = I3D(num_classes=400)
    i3nception_pt.eval()
    i3nception_pt.load_state_dict(torch.load(args.rgb_weights_path))
    i3nception_pt.train()
    i3nception_pt.cuda()

    l1_loss = torch.nn.L1Loss()
    sgd = torch.optim.SGD(i3nception_pt.parameters(), lr=0.001, momentum=0.9)

    run(i3nception_pt, loader, l1_loss, sgd, frame_nb=args.frame_nb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Runs inflated inception v1 network on\
    cricket sample from tensorflow demo (generate the network weights with\
    i3d_tf_to_pt.py first)')
    parser.add_argument(
        '--rgb_weights_path',
        type=str,
        default='model/model_rgb.pth',
        help='Path to model state_dict')
    parser.add_argument(
        '--frame_nb',
        type=int,
        default='16',
        help='Number of video_frames to use (should be a multiple of 8)')
    parser.add_argument(
        '--im_size', type=int, default='224', help='Size of center crop')
    args = parser.parse_args()
    run_profile(args)
