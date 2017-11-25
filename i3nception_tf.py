import os

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from src import i3dtf, i3nception
from src.i3dtf import Unit3Dtf, InceptionI3d
from src.i3nception import Unit3Dpy, I3nception

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def compare_outputs(tf_out, py_out):
    out_diff = np.abs(py_out - tf_out)
    mean_diff = out_diff.mean()
    max_diff = out_diff.max()
    print('===============')
    print('max diff : {}, mean diff : {}'.format(max_diff, mean_diff))
    print('mean val: tf {tf_mean} pt {pt_mean}'.format(
        tf_mean=tf_out.mean(), pt_mean=py_out.mean()))
    print('max vals: tf {tf_max} pt {pt_max}'.format(
        tf_max=tf_out.max(), pt_max=py_out.max()))
    print('===============')


intermediate_feature = False
rgb_tf_checkpoint = '../kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt'
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

# Initialzie tensorflow I3D
with tf.variable_scope('RGB'):
    rgb_model = InceptionI3d(
        class_nb, spatial_squeeze=True, final_endpoint='Predictions')
    # Tensorflow forward pass
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(batch_size, frame_nb, im_size, im_size, in_channels))
    rgb_logits, _ = rgb_model(
        rgb_input, is_training=False, dropout_keep_prob=1.0)

# Get params for tensorflow weight retreival
rgb_variable_map = {}
for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'RGB':
        rgb_variable_map[variable.name.replace(':0', '')] = variable

criterion = torch.nn.L1Loss()
rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
with tf.Session() as sess:
    # Load saved tensorflow weights
    rgb_saver.restore(sess, rgb_tf_checkpoint)

    # Transfer weights from tensorflow to pytorch
    i3nception_pt.eval()
    i3nception_pt.load_tf_weights(sess)

    # Load data
    for i, (input_2d, target) in enumerate(loader):
        input_2d = torch.from_numpy(input_2d.numpy())

        # Prepare data for pytorch forward pass
        target_var = torch.autograd.Variable(target)
        input_3d = input_2d.clone().unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)
        input_3d_var = torch.autograd.Variable(input_3d)

        # Prepare data for tensorflow pass
        feed_dict = {}
        input_3d_tf = input_3d.numpy().transpose(0, 2, 3, 4, 1)
        feed_dict[rgb_input] = input_3d_tf

        # Tensorflow forward pass
        tf_out3dsample = sess.run(rgb_logits, feed_dict=feed_dict)
        out_tf_np = tf_out3dsample

        if intermediate_feature:
            # Reshape intermediary input to insure they are comparable
            out_tf_np = tf_out3dsample.transpose((0, 4, 1, 2, 3))
        else:
            out_tf = torch.from_numpy(out_tf_np)

        # Pytorch forward pass
        out_pt = i3nception_pt(input_3d_var)
        out_pt_np = out_pt.data.numpy()

        # Make sure the tensorflow and pytorch outputs have the same shape
        assert out_tf_np.shape == out_pt_np.shape, 'tf output: {} != pt output : {}'.format(
            out_tf_np.shape, out_pt_np.shape)
        compare_outputs(out_tf_np, out_pt_np)

        # Display slices of filter map for intermediate features
        # for visual comparison
        if intermediate_feature:
            filter_idx = 219
            img_tf = out_tf_np[0][filter_idx][0]
            img_pt = out_pt_np[0][filter_idx][0]

            max_v = max(img_tf.max(), img_pt.max())
            min_v = min(img_tf.min(), img_pt.min())
            plt.subplot(2, 2, 1)
            plt.imshow(img_pt, vmax=max_v, vmin=min_v)
            plt.subplot(2, 2, 2)
            plt.imshow(img_tf, vmax=max_v, vmin=min_v)
            plt.subplot(2, 2, 3)
            plt.imshow(img_tf - img_pt)
            plt.show()
            print('min val : {}, max_val : {}, mean val : {}'.format(
                min_v, max_v, out_pt_np.mean()))
        loss = criterion(out_pt, torch.ones_like(out_pt))
        loss.backward()

# Save pytorch weights for future loading
i3nception_state_dict = i3nception_pt.cpu().state_dict()
torch.save(i3nception_state_dict, rgb_pt_checkpoint)
