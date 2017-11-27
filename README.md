I3D models transfered from Tensorflow to PyTorch
================================================

This repo contains several scripts that allow to transfer the weights from the tensorflow implementation of I3D
from the paper [*Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset*](https://arxiv.org/abs/1705.07750) by Joao Carreira and Andrew Zisserman to PyTorch.

The original (and official!) tensorflow code can be found [here](https://github.com/deepmind/kinetics-i3d/).

The heart of the transfer is the `i3d_tf_to_pt.py` script

Launch it with `python i3d_tf_to_pt.py --rgb` to generate the rgb checkpoint weight pretrained from ImageNet inflated initialization.

To generate the flow weights, use `python i3d_tf_to_pt.py --flow`.

You can also generate both in one run by using both flags simultaneously `python i3d_tf_to_pt.py --rgb --flow`.

This simple version does not take advantage of the uneven padding used in tensorflow but is therefore compatible with pytorch 0.2


## Demo

There is a slight drift in the weights that impacts the predictions, however, it seems to only marginally affect the final predictions, and therefore, the converted weights should serve as a valid initialization for further finetuning.

This can be observed by evaluating the same sample as the [original implementation](https://github.com/deepmind/kinetics-i3d/).

For a demo, launch `python i3d_pt_demo.py --rgb --flow`.
This script will print the scores produced by the pytorch model.

Pytorch Flow + RGB predictions:
```
1.0 43.0998 playing cricket
2.2176e-08 25.4755 hurling sport
1.4613e-08 25.0585 catching or throwing baseball
2.7955e-09 23.4045 catching or throwing softball
1.1410e-09 22.50846 hitting baseball
7.1152e-11 19.7336 playing tennis
```

Tensorflow Flow + RGB predictions:
```
1.0 41.8137 playing cricket
1.49717e-09 21.494 hurling sport
3.84311e-10 20.1341 catching or throwing baseball
1.54923e-10 19.2256 catching or throwing softball
1.13601e-10 18.9153 hitting baseball
8.80112e-11 18.6601 playing tennis
```

## Time profiling

To time the forward and backward passes, you can install [kernprof](https://github.com/rkern/line_profiler), an efficient line profiler, and then launch

`kernprof -lv i3d_pt_profiling.py --frame_nb 16`

This launches a basic pytorch training script on a dummy dataset that consists of replicated images as spatio-temporal inputs.

On my GeForce GTX TITAN Black (6Giga) a forward+backward pass takes roughly 0.2-0.25 seconds.


