# Code Framework

The overall code framework is shown in the following figure. It mainly consists of four parts - [Config], [Data],
[Model], and [Network].

<p align="center">
   <img src="https://user-images.githubusercontent.com/41912303/121567451-dff2e800-ca1e-11eb-9e46-a6b45a72a9ff.png" height="450">
</p>

For example, once [train.py] is called (i.e. `python train.py -opt options/train/train_template.yml`), a sequence of actions will follow this command:

1.  [Config] - Reads the configuration from [/options/train/train_template.yml] which is a [YAML] file. Then passes
    the configuration values from it down through the following steps.
2.  [Data] - Creates the train and validation dataloaders.
3.  [Model] - Creates the chosen model.
4.  [Network] - Creates the chosen network.
5.  Finally [train.py] - Starts to train the model. Other actions like logging, saving intermediate models, validation, updating
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

### [/options] Configure the options for data loader, network structure, losses, training strategies and other components

-   [YAML] (or JSON) files are used to configure options and [/options/options.py] will convert the file to a python
    dictionary (where missing keys return `None` instead of `Exception`).
-   [YAML] files use `~` for `None`; and officially supports comments with `#`. (JSON files use `null` for `None`; and supports `//` comments instead).
-   Supports `debug` mode, i.e, if the model name contains `debug`, it will trigger debug mode.
-   The configuration file and descriptions can be found in [/options].

[/options/options.py]: https://github.com/victorca25/BasicSR/blob/master/codes/options/options.py

## Data

### [/data] A data loader to provide data for training, validation, and testing

-   A separate data loader module. You can modify/create data loader to meet your own needs. The data loaders are 
    constructed in [/data/\_\_init__.py].
-   [/dataops] hosts a large variety of data operations, like: filters, colors, images augmentation, batch augmentations, 
    differential augmentations, and others that can be used at different points of the architecture when needed.
-   By default, the package uses [cv2] package to do image processing, which provides rich operations. However, it is 
    possible to integrate other options if required.
-   Supports reading files from image folder or [lmdb] file. For faster IO during training (even if on an SSD) it is
    recommended to create and use an [lmdb] dataset if possible.
-   [/dataops/common.py] provides useful tools. For example, the `MATLAB bicubic` operation; rgb&lt;-->ycbcr as MATLAB. We
    also provide [MATLAB bicubic imresize wiki] and [Color conversion in SR wiki].
-   The standard tensor format used is NCHW, \[0,1], RGB, torch float tensor, but they can be normalized using the `znorm` 
    option if needed.

[/data]: https://github.com/victorca25/BasicSR/tree/master/codes/data

[/data/\_\_init__.py]: https://github.com/victorca25/BasicSR/blob/master/codes/data/__init__.py

[/dataops]: https://github.com/victorca25/BasicSR/tree/master/codes/dataops

[/dataops/common.py]: https://github.com/victorca25/BasicSR/blob/master/codes/dataops/common.py

[cv2]: https://github.com/skvark/opencv-python

[lmdb]: https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database

[MATLAB bicubic imresize wiki]: https://github.com/xinntao/BasicSR/wiki/MATLAB-bicubic-imresize

[Color conversion in SR wiki]: https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR

## Model

### [/models] Construct different models for training and testing

-   The chosen model is constructed in [/models/\_\_init__.py] according to selected model type. 
-   A model mainly consists of two parts - a [Network] structure, and a [Model] definition (e.g. loss configuration,
    optimization step, e.t.c.).
-   Based on the [/models/base_model.py], we define different models, e.g., [SRRaGAN], [SFTGAN_ACD], [VSR].
-   [SRRaGAN] is used to train super resolution models (ESRGAN, PAN, SRGAN, etc), but based on it, additional 
    models that require specific parameters optimization strategy or options can be defined, such as [PPON].

[/models]: https://github.com/victorca25/BasicSR/tree/master/codes/models

[/models/\_\_init__.py]: https://github.com/victorca25/BasicSR/blob/master/codes/models/__init__.py

[/models/base_model.py]: https://github.com/victorca25/BasicSR/blob/master/codes/models/base_model.py

[SRRaGAN]: https://github.com/victorca25/BasicSR/blob/master/codes/models/SRRaGAN_model.py

[PPON]: https://github.com/victorca25/BasicSR/blob/master/codes/models/ppon_model.py

[VSR]: https://github.com/victorca25/BasicSR/blob/master/codes/models/VSR_model.py

[SFTGAN_ACD]: https://github.com/victorca25/BasicSR/blob/master/codes/models/SFTGAN_ACD_model.py

## Network

### [/models/modules] Construct different network architectures

-   The network is constructed in [/models/network.py] and the detailed structures are in [/models/modules/architectures].
-   Some useful blocks can be found in [/models/modules/architectures/block.py], which in general are common to multiple 
    networks and it is flexible to construct your network structures with these pre-defined blocks.
-   You can also easily write your own network architecture in a separate file like [/models/modules/architectures/sft_arch.py]
    or [/models/modules/architectures/PAN_arch.py].

[/models/modules]: https://github.com/victorca25/BasicSR/tree/master/codes/models/modules

[/models/modules/architectures]: https://github.com/victorca25/BasicSR/tree/master/codes/models/modules/architectures

[/models/modules/architectures/block.py]: https://github.com/victorca25/BasicSR/blob/master/codes/models/modules/architectures/block.py

[/models/modules/architectures/sft_arch.py]: https://github.com/victorca25/BasicSR/blob/master/codes/models/modules/architectures/sft_arch.py

[/models/modules/architectures/PAN_arch.py]: https://github.com/victorca25/BasicSR/blob/master/codes/models/modules/architectures/PAN_arch.py

[/models/network.py]: https://github.com/victorca25/BasicSR/blob/master/codes/models/networks.py

## Utils

### [/utils] Provides useful utilities

-   [/utils/util.py] provides logging service during training and testing.
-   [/utils/progress_bar.py] provides a progress bar which can print the progress.
-   Support to use [tensorboard] to visualize and compare training loss, validation PSNR, SSIM, LPIPS, e.t.c.

[/utils]: https://github.com/victorca25/BasicSR/tree/master/codes/utils

[/utils/util.py]: https://github.com/victorca25/BasicSR/blob/master/codes/utils/util.py

[/utils/progress_bar.py]: https://github.com/victorca25/BasicSR/blob/master/codes/utils/progress_bar.py

[tensorboard]: https://tensorflow.org/programmers_guide/summaries_and_tensorboard

## Scripts

### [/scripts](https://github.com/victorca25/BasicSR/tree/master/codes/scripts) Provides useful scripts