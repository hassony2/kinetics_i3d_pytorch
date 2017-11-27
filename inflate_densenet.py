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
def run_inflater():
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

    frame_nb = 8
    densenet = torchvision.models.densenet121(pretrained=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    i3densenet = I3DenseNet(
        copy.deepcopy(densenet), frame_nb, inflate_block_convs=True)
    i3densenet.train()
    i3densenet.cuda()
    for i, (input_2d, target) in enumerate(loader):
        target = target.cuda()
        target_var = torch.autograd.Variable(target)
        input_2d_var = torch.autograd.Variable(input_2d)
        out2d = densenet(input_2d_var)

        input_3d = input_2d.unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)
        input_3d_var = torch.autograd.Variable(input_3d.cuda())

        out3d = i3densenet(input_3d_var)
        max_vals, max_indexes = out3d.max(1)
        for sample_idx in range(out3d.shape[0]):
            sample_out = out3d.data[sample_idx]

            top_val, top_idx = torch.sort(sample_out, 0, descending=True)

            show_best = 10
            print('Top {} classes and associated scores: '.format(show_best))
            for i in range(show_best):
                print('[{}]: {}'.format(imagenet_classes[top_idx[i]], top_val[
                    i]))
                out_diff = out2d.data - out3d.cpu().data

            sample_img = input_2d[sample_idx].numpy().transpose(1, 2, 0)
            sample_img = (sample_img - sample_img.min()) * (1 / (
                sample_img.max() - sample_img.min()))
            plt.imshow(sample_img)
            plt.show()

        # Computing errors between final predictions of inflated and uninflated
        # dense networks
        print('For batch {i} , maximum error between final predictions: {err}'.
              format(i=i, err=out_diff.max()))
        assert (out_diff.max() < 0.0001)


if __name__ == "__main__":
    run_inflater()
