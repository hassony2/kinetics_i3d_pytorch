I3D models transfered from Tensorflow to PyTorch
================================================

This repo contains several scripts that allow to transfer the weights from the tensorflow implementation of I3D
from the paper [*Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset*](https://arxiv.org/abs/1705.07750) by Joao Carreira and Andrew Zisserman to PyTorch.

The original (and official!) tensorflow code can be found [here](https://github.com/deepmind/kinetics-i3d/).

The heart of the transfer is the `i3d_tf_to_pt.py` script

Launch it with `python i3d_tf_to_pt.py --rgb` to generate the rgb checkpoint weight pretrained from ImageNet inflated initialization.

To generate the flow weights, use `python i3d_tf_to_pt.py --flow`.

You can also generate both in one run by using both flags simultaneously `python i3d_tf_to_pt.py --rgb --flow`.

Note that the master version requires PyTorch 0.3 as it relies on the recent addition of ConstantPad3d that has been included in this latest release.

If you want to use pytorch 0.2 checkout the branch pytorch-02 which contains a simplified model with even padding on all sides (and the corresponding pytorch weight checkpoints).
The difference is that the 'SAME' option for padding in tensorflow allows it to pad unevenly both sides of a dimension, an effect reproduced on the master branch.

This simpler model produces scores a bit closer to the original tensorflow model on the demo sample and is also a bit faster.

## Demo

There is a slight drift in the weights that impacts the predictions, however, it seems to only marginally affect the final predictions, and therefore, the converted weights should serve as a valid initialization for further finetuning.

This can be observed by evaluating the same sample as the [original implementation](https://github.com/deepmind/kinetics-i3d/).

For a demo, launch `python i3d_pt_demo.py --rgb --flow`.
This script will print the scores produced by the pytorch model.

Pytorch Flow + RGB predictions:
```
1.0          46.11447 playing cricket
1.364149e-09 25.70173 hurling sport
4.665193e-10 24.62874 catching or throwing baseball
2.186982e-10 23.87114 catching or throwing softball
1.686102e-10 23.61104 hitting baseball
1.738439e-11 1.339023 playing tennis
```

Tensorflow Flow + RGB predictions:
```
1.0         41.8137 playing cricket
1.49717e-09 21.4943 hurling sport
3.84311e-10 20.1341 catching or throwing baseball
1.54923e-10 19.2256 catching or throwing softball
1.13601e-10 18.9153 hitting baseball
8.80112e-11 18.6601 playing tennis
```



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

PyTorch Flow predictions:
```
[playing cricket]: 0.999998
[playing kickball]: 3.890206e-07
[catching or throwing baseball]: 2.721246e-07
[catching or throwing softball]: 1.210907e-07
[shooting goal (soccer)]: 1.108749e-07
```

Tensorflow Flow predictions:
```
[playing cricket]: 0.928604
[hurling (sport)]: 0.0406825
[playing tennis]: 0.00415417
[playing squash or racquetbal]: 0.00247407
[hitting baseball]: 0.00138002
```

## Time profiling

To time the forward and backward passes, you can install [kernprof](https://github.com/rkern/line_profiler), an efficient line profiler, and then launch

`kernprof -lv i3d_pt_profiling.py --frame_nb 16`

This launches a basic pytorch training script on a dummy dataset that consists of replicated images as spatio-temporal inputs.

On my GeForce GTX TITAN Black (6Giga) a forward+backward pass takes roughly 0.25-0.3 seconds.


