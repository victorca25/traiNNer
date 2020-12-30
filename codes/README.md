# Code Framework

The overall code framework is shown in the following figure. It mainly consists of four parts - [Config], [Data],
[Model], and [Network].

<p align="center">
   <img src="https://github.com/xinntao/public_figures/blob/master/BasicSR/code_framework.png" height="450">
</p>

For example, once [train.py] is called (i.e. `python train.py -i options/train/train_template.yml`):

1.  [Config] - Reads the configuration from [/options/train/train_template.yml] which is a [YAML] file. Then passes
    the configuration values from it down through the following steps.
2.  [Data] - Creates the train and validation dataloaders.
3.  [Model] - Creates the chosen model.
4.  [Network] - Creates the chosen network.
5.  Finally - Starts to train the model. Other actions like logging, saving intermediate models, validation, updating
    learning rate, e.t.c. are also done during training.

Moreover, there are also [Utilities](#utils) and [Useful script](#scripts) available to use for various operations,
like configuring your dataset.

[train.py]: https://github.com/victorca25/BasicSR/blob/master/codes/train.py

[/options/train/train_template.yml]: https://github.com/victorca25/BasicSR/blob/master/codes/options/train/train_template.yml

[/options]: https://github.com/victorca25/BasicSR/tree/master/codes/options

[Config]: #config

[Data]: #data

[Model]: #model

[Network]: #network

[YAML]: https://yaml.org

## Config

### [/options] Configure the options for data loader, network structure, model, training strategies

-   [YAML] files are used to configure options and [/options/\_\_init__.py] will convert the YAML file to a python
    dictionary (where missing keys return `None` instead of `Exception`).
-   [YAML] files use `~` for `None`; and officially supports comments with `#`.
-   Supports `debug` mode, i.e, if the model name contains `debug`, it will trigger debug mode.
-   The configuration file and descriptions can be found in [/options].

Note: `JSON` files are also supported but may be phased out, I don't recommend using the available JSON option files.
If you need to use the values they suggest, simply make a new YML file using the wanted data from it instead. JSON
support could be removed at any minute.

[/options/\_\_init__.py]: https://github.com/victorca25/BasicSR/blob/master/codes/options/__init__.py

## Data

### [/data] A data loader to provide data for training, validation, and testing

-   A separate data loader module. You can modify/create data loader to meet your own needs.
-   Uses [cv2] package to do image processing, which provides rich operations.
-   Supports reading files from image folder or [lmdb] file. For faster IO during training (even if on an SSD) it is
    recommended to create and use an [lmdb] dataset. More details can be found in [xinntao]'s [lmdb wiki].
-   [/data/util.py] provides useful tools. For example, the `MATLAB bicubic` operation; rgb&lt;-->ycbcr as MATLAB. We
    also provide [MATLAB bicubic imresize wiki] and [Color conversion in SR wiki].
-   Now, we convert the images to format NCHW, \[0,1], RGB, torch float tensor.

[/data]: https://github.com/victorca25/BasicSR/tree/master/codes/data

[/data/util.py]: https://github.com/victorca25/BasicSR/blob/master/codes/data/util.py

[cv2]: https://github.com/skvark/opencv-python

[lmdb]: https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database

[xinntao]: https://github.com/xinntao

[lmdb wiki]: https://github.com/xinntao/BasicSR/wiki/Faster-IO-speed

[MATLAB bicubic imresize wiki]: https://github.com/victorca25/BasicSR/wiki/MATLAB-bicubic-imresize

[Color conversion in SR wiki]: https://github.com/victorca25/BasicSR/wiki/Color-conversion-in-SR

## Model

### [/models] Construct different models for training and testing

-   A model mainly consists of two parts - [Network] structure, and [Model] definition (e.g. loss definition,
    optimization, e.t.c.).
-   Based on the [/models/base_model.py], we define different models, e.g., [SR], [SRGAN], [SRRaGAN], and [SFTGAN_ACD].

[/models]: https://github.com/victorca25/BasicSR/tree/master/codes/models

[/models/base_model.py]: https://github.com/victorca25/BasicSR/blob/master/codes/models/base_model.py

[SR]: https://github.com/victorca25/BasicSR/blob/master/codes/models/SR_model.py

[SRGAN]: https://github.com/victorca25/BasicSR/blob/master/codes/models/SRGAN_model.py

[SRRaGAN]: https://github.com/victorca25/BasicSR/blob/master/codes/models/SRRaGAN_model.py

[SFTGAN_ACD]: https://github.com/victorca25/BasicSR/blob/master/codes/models/SFTGAN_ACD_model.py

## Network

### [/models/modules] Construct different network architectures

-   The network is constructed in [/models/network.py] and the detailed structures are in [/models/modules].
-   We provide some useful blocks in [/models/modules/block.py] and it is flexible to construct your network
    structures with these pre-defined blocks.
-   You can also easily write your own network architecture in a separate file like [/models/modules/sft_arch.py].

[/models/modules]: https://github.com/victorca25/BasicSR/tree/master/codes/models/modules

[/models/modules/block.py]: https://github.com/victorca25/BasicSR/blob/master/codes/models/modules/block.py

[/models/modules/sft_arch.py]: https://github.com/victorca25/BasicSR/blob/master/codes/models/modules/sft_arch.py

[/models/network.py]: https://github.com/victorca25/BasicSR/blob/master/codes/models/networks.py

## Utils

### [/utils] Provides useful utilities

-   [/utils/logger.py] provides logging service during training and testing.
-   [/utils/progress_bar.py] provides a progress bar which can print the progress.
-   Support to use [tensorboard] to visualize and compare training loss, validation PSNR, SSIM, LPIPS, e.t.c.

[/utils]: https://github.com/victorca25/BasicSR/tree/master/codes/utils

[/utils/logger.py]: https://github.com/victorca25/BasicSR/blob/master/codes/utils/logger.py

[/utils/progress_bar.py]: https://github.com/victorca25/BasicSR/blob/master/codes/utils/progress_bar.py

[tensorboard]: https://tensorflow.org/programmers_guide/summaries_and_tensorboard

## Scripts

### [/scripts](https://github.com/victorca25/BasicSR/tree/master/codes/scripts) Provides useful scripts
