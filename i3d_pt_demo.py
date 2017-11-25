from matplotlib import pyplot as plt
import numpy as np
import torch

from src.i3nception import I3nception

rgb_pt_checkpoint = 'model/model_rgb.pth'

# Initialize input params
in_channels = 3

# Initialize pytorch I3D
i3nception_pt = I3nception(num_classes=400)
i3nception_pt.eval()
i3nception_pt.load_state_dict(torch.load(rgb_pt_checkpoint))
i3nception_pt.cuda()
sample_path = '../kinetics-i3d/data/v_CricketShot_g04_c01_rgb.npy'
sample = np.load(sample_path).transpose(0, 4, 1, 2, 3)

sample_var = torch.autograd.Variable(torch.from_numpy(sample).cuda())
out_var = i3nception_pt(sample_var)
out_tensor = out_var.data.cpu()

classes_path = '../kinetics-i3d/data/label_map.txt'
kinetics_classes = [x.strip() for x in open(classes_path)]
top_val, top_idx = torch.sort(out_tensor, 1, descending=True)

show_best = 10
print('Top {} classes and associated probabilities: '.format(show_best))
for i in range(show_best):
    print('[{}]: {}'.format(kinetics_classes[top_idx[0, i]], top_val[0, i]))
