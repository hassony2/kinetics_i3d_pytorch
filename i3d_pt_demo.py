import argparse

import numpy as np
import torch

from src.i3nception import I3nception

rgb_pt_checkpoint = 'model/model_rgb.pth'

# Initialize input params
in_channels = 3


def run_demo(args):
    # Initialize pytorch I3D
    i3nception_pt = I3nception(num_classes=400)
    i3nception_pt.eval()
    i3nception_pt.load_state_dict(torch.load(args.rgb_weights_path))
    i3nception_pt.cuda()

    sample = np.load(args.sample_path).transpose(0, 4, 1, 2, 3)

    sample_var = torch.autograd.Variable(torch.from_numpy(sample).cuda())
    out_var = i3nception_pt(sample_var)
    out_tensor = out_var.data.cpu()

    kinetics_classes = [x.strip() for x in open(args.classes_path)]
    top_val, top_idx = torch.sort(out_tensor, 1, descending=True)

    print('Top {} classes and associated probabilities: '.format(args.top_k))
    for i in range(args.top_k):
        print(
            '[{}]: {}'.format(kinetics_classes[top_idx[0, i]], top_val[0, i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Runs inflated inception v1 network on\
    cricket sample from tensorflow demo (generate the network weights with\
    i3d_tf_to_pt.py first)')
    parser.add_argument(
        '--rgb_weights_path',
        type=str,
        default='model/model_rgb.pth',
        help='Path to model state_dict')
    parser.add_argument(
        '--sample_path',
        type=str,
        default='data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy',
        help='Path to kinetics rgb numpy sample')
    parser.add_argument(
        '--classes_path',
        type=str,
        default='data/kinetic-samples/label_map.txt',
        help='Number of video_frames to use (should be a multiple of 8)')
    parser.add_argument(
        '--top_k',
        type=int,
        default='5',
        help='When display_samples, number of top classes to display')
    args = parser.parse_args()
    run_demo(args)
