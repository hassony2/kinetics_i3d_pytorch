I3D models transfered from Tensorflow to PyTorch
================================================

This repo contains several scripts that allow to transfer the weights from the tensorflow implementation of I3D
from the paper [*Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset*](https://arxiv.org/abs/1705.07750) by Joao Carreira and Andrew Zisserman to PyTorch.

The original (and official!) tensorflow code can be found [here](https://github.com/deepmind/kinetics-i3d/).

The heart of the transfer is the `i3d_tf_to_pt.py` script

Launch it with `python i3d_tf_to_pt.py` to generate the rgb checkpoint weight pretrained from ImageNet inflated initialization


## Demo

There is a slight drift in the weights that impacts the predictions, however, it seems to only marginally affect the final predictions, and therefore, the converted weights should serve as a valid initialization for further finetuning.

This can be observed by evaluating the same sample as the [original implementation](https://github.com/deepmind/kinetics-i3d/).

For a demo, launch `python i3d_pt_demo.py`.

PyTorch RGB predictions:
```
[playing cricket]: 0.999998
[playing kickball]: 3.890206e-07
[catching or throwing baseball]: 2.721246e-07
[catching or throwing softball]: 1.210907e-07
[shooting goal (soccer)]: 1.108749e-07
```

Tensorflow RGB predictions:
```
[playing cricket]: 0.999997
[playing kickball]: 1.33535e-06
[catching or throwing baseball]: 4.55313e-07
[shooting goal (soccer)]: 3.14343e-07
[catching or throwing softball]: 1.92433e-07
```


## Time profiling

To time the forward and backward passes, you can install [kernprof](https://github.com/rkern/line_profiler), an efficient line profiler, and then launch

`kernprof -lv i3d_pt_profiling.py --frame_nb 16`

This launches a basic pytorch training script on a dummy dataset that consists of replicated images as spatio-temporal inputs.

On my GeForce GTX TITAN Black (6Giga) a forward+backward pass takes roughly 0.25-0.3 seconds.


