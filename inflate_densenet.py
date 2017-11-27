import argparse
import copy
import json

from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.i3dense import I3DenseNet


# To profile uncomment @profile and run `kernprof -lv inflate_densenet.py`
# @profile
def run_inflater(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder('data/dummy-dataset',
                                   transforms.Compose([
                                       transforms.CenterCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    class_idx = json.load(open('data/imagenet_class_index.json'))
    imagenet_classes = [class_idx[str(k)][1] for k in range(len(class_idx))]

    densenet = torchvision.models.densenet121(pretrained=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    i3densenet = I3DenseNet(
        copy.deepcopy(densenet), args.frame_nb, inflate_block_convs=True)
    i3densenet.train()
    i3densenet.cuda()

    for i, (input_2d, target) in enumerate(loader):
        target = target.cuda()
        target_var = torch.autograd.Variable(target)
        input_2d_var = torch.autograd.Variable(input_2d)
        out2d = densenet(input_2d_var)

        input_3d = input_2d.unsqueeze(2).repeat(1, 1, args.frame_nb, 1, 1)
        input_3d_var = torch.autograd.Variable(input_3d.cuda())

        out3d = i3densenet(input_3d_var)
        out_diff = out2d.data - out3d.cpu().data

        # Computing errors between final predictions of inflated and uninflated
        # dense networks
        print(
            'Batch {i} maximum error between 2d and inflated predictions: {err}'.
            format(i=i, err=out_diff.max()))
        assert (out_diff.max() < 0.0001)

        if args.display_samples:
            max_vals, max_indexes = out3d.max(1)
            for sample_idx in range(out3d.shape[0]):
                sample_out = out3d.data[sample_idx]

                top_val, top_idx = torch.sort(sample_out, 0, descending=True)

                print('Top {} classes and associated scores: '.format(
                    args.top_k))
                for i in range(args.top_k):
                    print('[{}]: {}'.format(imagenet_classes[top_idx[i]],
                                            top_val[i]))

                sample_img = input_2d[sample_idx].numpy().transpose(1, 2, 0)
                sample_img = (sample_img - sample_img.min()) * (1 / (
                    sample_img.max() - sample_img.min()))
                plt.imshow(sample_img)
                plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Inflates the 121 version of densenet and runs\
    it on dummy dataset to compare outputs from original and inflated networks\
    (should be the same)')
    parser.add_argument(
        '--display_samples',
        action='store_true',
        help='Whether to display samples and associated\
        scores for 3d inflated densenet')
    parser.add_argument(
        '--top_k',
        type=int,
        default='5',
        help='When display_samples, number of top classes to display')
    parser.add_argument(
        '--frame_nb',
        type=int,
        default='8',
        help='Number of video_frames to use (should be a multiple of 8)')
    args = parser.parse_args()
    run_inflater(args)
