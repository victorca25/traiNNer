
## Tensorboard Logger (tb_logger)

In addition to logging outputs to files, [tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) is also an optional visualization tool for visualizing/comparing training loss, validation metrics and even validation images.

You can turn it on/off in option file with the key: `use_tb_logger`.

### Install

Either the official tensorboard (`pip install tensorboard`) or tensorboardX (`pip install tensorboardX`) should work automatically if the option is set.

### Run
1. In the terminal open tensorboard directing it to the directory with the outputs of your experiment in this directory ([tb_logger](https://github.com/victorca25/BasicSR/tree/master/tb_logger)) like: `tensorboard --logdir xxx/xxx`.
2. Open the tensorboard UI at http://localhost:6006 in your browser
