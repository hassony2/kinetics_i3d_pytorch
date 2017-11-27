import argparse

from matplotlib import pyplot as plt
import tensorflow as tf
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.i3dtf import InceptionI3d
from src.i3dpt import I3D
from src.monitorutils import compare_outputs


def transfer_weights(tf_checkpoint, pt_checkpoint, batch_size, modality='rgb'):
    intermediate_feature = False
    im_size = 224
    dataset = datasets.ImageFolder(
        'data/dummy-dataset',
        transforms.Compose([
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            #                                   normalize,
        ]))

    # Initialize input params
    if modality == 'rgb':
        in_channels = 3
    elif modality == 'flow':
        in_channels = 2
    else:
        raise ValueError(
            '{} not among known modalities [rgb|flow]'.format(modality))

    frame_nb = 16  # Number of items in depth (temporal) dimension
    class_nb = 400

    # Initialize dataset
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)

    # Initialize pytorch I3D
    i3nception_pt = I3D(num_classes=400, modality=modality)

    # Initialzie tensorflow I3D
    if modality == 'rgb':
        scope = 'RGB'
    elif modality == 'flow':
        scope = 'Flow'

    with tf.variable_scope(scope):
        rgb_model = InceptionI3d(class_nb, final_endpoint='Predictions')
        # Tensorflow forward pass
        rgb_input = tf.placeholder(
            tf.float32,
            shape=(batch_size, frame_nb, im_size, im_size, in_channels))
        rgb_logits, _ = rgb_model(
            rgb_input, is_training=False, dropout_keep_prob=1.0)

    # Get params for tensorflow weight retreival
    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == scope:
            rgb_variable_map[variable.name.replace(':0', '')] = variable

    criterion = torch.nn.L1Loss()
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
    with tf.Session() as sess:
        # Load saved tensorflow weights
        rgb_saver.restore(sess, tf_checkpoint)

        # Transfer weights from tensorflow to pytorch
        i3nception_pt.eval()
        i3nception_pt.load_tf_weights(sess)

        # Save pytorch weights for future loading
        i3nception_state_dict = i3nception_pt.cpu().state_dict()
        torch.save(i3nception_state_dict, pt_checkpoint)

        # Load data
        for i, (input_2d, target) in enumerate(loader):
            input_2d = torch.from_numpy(input_2d.numpy())
            if modality == 'flow':
                input_2d = input_2d[:, 0:2]  # Remove one dimension

            # Prepare data for pytorch forward pass
            target_var = torch.autograd.Variable(target)
            input_3d = input_2d.clone().unsqueeze(2).repeat(
                1, 1, frame_nb, 1, 1)
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

            # Pytorch forward pass
            out_pt, _ = i3nception_pt(input_3d_var)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Transfers the kinetics rgb pretrained i3d\
    inception v1 weights from tensorflow to pytorch and saves the weights as\
    as state_dict')
    parser.add_argument(
        '--rgb', action='store_true', help='Convert RGB pretrained network')
    parser.add_argument(
        '--rgb_tf_checkpoint',
        type=str,
        default='model/tf_rgb_imagenet/model.ckpt',
        help='Path to tensorflow weight checkpoint trained on rgb')
    parser.add_argument(
        '--rgb_pt_checkpoint',
        type=str,
        default='model/model_rgb.pth',
        help='Path for pytorch state_dict saving')
    parser.add_argument(
        '--flow', action='store_true', help='Convert Flow pretrained network')
    parser.add_argument(
        '--flow_tf_checkpoint',
        type=str,
        default='model/tf_flow_imagenet/model.ckpt',
        help='Path to tensorflow weight checkpoint trained on flow')
    parser.add_argument(
        '--flow_pt_checkpoint',
        type=str,
        default='model/model_flow.pth',
        help='Path for pytorch state_dict saving')
    parser.add_argument(
        '--batch_size',
        type=int,
        default='2',
        help='Batch size for comparison between tensorflow and pytorch outputs'
    )
    args = parser.parse_args()

    if args.rgb:
        transfer_weights(
            args.rgb_tf_checkpoint,
            args.rgb_pt_checkpoint,
            batch_size=args.batch_size,
            modality='rgb')
    if args.flow:
        transfer_weights(
            args.flow_tf_checkpoint,
            args.flow_pt_checkpoint,
            batch_size=args.batch_size,
            modality='flow')
