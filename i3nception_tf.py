import os

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from src import i3dtf, i3nception
from src.i3dtf import Unit3Dtf, InceptionI3d
from src.i3nception import Unit3Dpy, I3nception

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def compare_outputs(tf_out, py_out):
    out_diff = np.abs(py_out - tf_out)
    mean_diff = out_diff.mean()
    max_diff = out_diff.max()
    print('===============')
    print('max diff : {}, mean diff : {}'.format(max_diff, mean_diff))
    print('===============')
    # assert (max_diff < 0.0001)


rgb_checkpoint = '../kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt'
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

# Network params
in_channels = 3
batch_size = 2
loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=False)
class_nb = 400
out_channels = 64
kernel_shape = [7, 7, 7]
stride = [2, 2, 2]
unittf = Unit3Dtf(
    output_channels=out_channels,
    kernel_shape=kernel_shape,
    stride=stride,
    name='conv1a')
unitpy = Unit3Dpy(
    in_channels,
    out_channels,
    kernel_size=tuple(kernel_shape),
    stride=tuple(stride),
    activation='relu',
    padding='SAME',
    use_bias=False,
    use_bn=True)
i3nception_pt = I3nception(num_classes=400)

frame_nb = 20
with tf.variable_scope('RGB'):
    rgb_model = InceptionI3d(
        class_nb, spatial_squeeze=True, final_endpoint='Conv3d_2b_1x1')
    # Tensorflow forward pass
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(batch_size, frame_nb, im_size, im_size, in_channels))
    rgb_logits, _ = rgb_model(
        rgb_input, is_training=False, dropout_keep_prob=1.0)
    tf_out3d = unittf(rgb_input, is_training=False)

rgb_variable_map = {}
for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'RGB':
        rgb_variable_map[variable.name.replace(':0', '')] = variable

rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
with tf.Session() as sess:
    rgb_saver.restore(sess, rgb_checkpoint)
    for i, variable in enumerate(tf.global_variables()):
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable
            print(variable.name)

    # init = tf.global_variables_initializer()
    # sess.run(init)
    for i, (input_2d, target) in enumerate(loader):
        input_2d = torch.from_numpy(input_2d.numpy())
        target = target.cuda()
        target_var = torch.autograd.Variable(target)

        # Pytorch forward pass
        input_3d = input_2d.clone().unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)
        input_3d_var = torch.autograd.Variable(input_3d)

        feed_dict = {}
        input_3d_tf = input_3d.numpy().transpose(0, 2, 3, 4, 1)  #
        feed_dict[rgb_input] = input_3d_tf

        # Get output
        tf_out3dsample = sess.run(rgb_logits, feed_dict=feed_dict)
        out_tf_np = tf_out3dsample.transpose((0, 4, 1, 2, 3))
        out_tf = torch.from_numpy(out_tf_np)

        i3nception_pt.eval()
        i3nception_pt.load_tf_weights(sess)
        i3nception_pt
        out_pt = i3nception_pt(input_3d_var).data
        out_pt_np = out_pt.numpy()
        filter_idx = 0

        assert out_tf_np.shape == out_pt_np.shape, 'tf output: {} != pt output : {}'.format(
            out_tf_np.shape, out_pt_np.shape)
        # Plot slices
        filter_idx = 0
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

        # conv_name = os.path.join(unit_name_tf, 'conv_3d')
        # batchnorm_name = os.path.join(unit_name_tf, 'batch_norm')
        # conv_params = i3nception.get_conv_params(sess, conv_name)
        # batch_params = i3nception.get_bn_params(sess, batchnorm_name)

        # Compare outputs
        compare_outputs(out_tf_np, out_pt_np)
        import pdb
        pdb.set_trace()
