# BasicSR



[![Python Version](https://img.shields.io/badge/python-3-informational?style=flat)](https://python.org)
[![License](https://img.shields.io/github/license/victorca25/BasicSR?style=flat)](https://github.com/victorca25/BasicSR/blob/master/LICENSE)
[![DeepSource](https://deepsource.io/gh/victorca25/BasicSR.svg/?label=active+issues&show_trend=true)](https://deepsource.io/gh/victorca25/BasicSR/?ref=repository-badge)
[![Issues](https://img.shields.io/github/issues/victorca25/BasicSR?style=flat)](https://github.com/victorca25/BasicSR/issues)
[![PR's Accepted](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://makeapullrequest.com)

BasicSR (Basic Super Restoration) is an open source image and video restoration (super-resolution, denoising, deblurring and others) and image to image translation toolbox based on PyTorch.

Here you will find: boilerplate code for training and testing computer vision (CV) models, different methods and strategies integrated in a single pipeline and modularity to add and remove components as needed, including new network architectures and templates for different training strategies. The code is under a constant state of change, so if you find an issue or bug please open a [issue](https://github.com/victorca25/BasicSR/issues), a [discussion](https://github.com/victorca25/BasicSR/discussions) or write in one of the [Discord channels](##additional-help) for help.

Different from other repositories, here the focus is not only on repeating previous papers' results, but to enable more people to train their own models more easily, using their own custom datasets, as well as integrating new ideas to increase the performance of the models. For these reasons, a lot of the code is made in order to automatically take care of fixing potential issues, whenever possible.

Details of the currently supported architectures can be found [here](https://github.com/victorca25/BasicSR/blob/master/docs/architectures.md).

For a changelog and general list of features of this repository, check [here](https://github.com/victorca25/BasicSR/blob/master/docs/changes.md).
    
## Table of Contents
1.  [Dependencies](#dependencies)
2.  [Codes](#codes)
3.  [Usage](#usage)
4.  [Pretrained models](#pretrained-models)
5.  [Datasets](#datasets)
6.  [How to help](#how-to-help)


### Dependencies

-   Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
-   [PyTorch >= 0.4.0](https://pytorch.org/). PyTorch >= 1.7.0 required to enable certain features (SWA, AMP, others), as well as [torchvision](https://pytorch.org/vision/stable/index.html).
-   NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
-   Python packages: `pip install numpy opencv-python`
-   `JSON` files can be used for the configuration option files, but in order to use `YAML`, the `PyYAML` python package is also a dependency: [`pip install PyYAML`](https://pyyaml.org/)

#### Optional Dependencies

-   Python package: [`pip install tensorboardX`](https://github.com/lanpa/tensorboardX), for visualizing curves.
-   Python package: [`pip install lmdb`](https://github.com/jnwatson/py-lmdb), for lmdb database support.
-   Python package: [`pip install scipy`](https://www.scipy.org/) to use [CEM](https://github.com/victorca25/BasicSR/blob/master/codes/models/modules/architectures/CEM/README.md).
-   Python package: [`pip install Pillow`](https://python-pillow.org/) to use as an alternative image backend (default is OpenCV).

## Codes

This repository is a full framework for training different kinds of networks, with multiple enhancements and options. In [`./codes`](https://github.com/victorca25/BasicSR/tree/master/codes) you will find a more detailed explaination of the **code framework** ).

You will also find:
1.  Some useful scripts. More details in [`./codes/scripts`](https://github.com/victorca25/BasicSR/tree/master/codes/scripts). 
2.  [Evaluation codes](https://github.com/victorca25/BasicSR/tree/master/metrics), e.g., PSNR/SSIM metric.

Additionally, it is complemented by other repositories like [DLIP](https://github.com/victorca25/DLIP), that can be used in order to extract estimated kernels and noise patches from real images, using a modified KernelGAN and patches extraction code. Detailed instructions about how to use the estimated kernels are available [here](https://github.com/victorca25/BasicSR/blob/master/docs/kernels.md)

## Usage

### Training

#### Data and model preparation

In order to train your own models, you will need to create a [dataset](#datasets) consisting of images, and prepare these images, both considering [IO](https://github.com/victorca25/BasicSR/wiki/IO-speed) constrains, as well as the task the model should target. Detailed data preparation can be seen in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data).

[**Pretrained models**](#pretrained-models) that can be used for fine-tuning are available.

Detailed instructions on [how to train](https://github.com/victorca25/BasicSR/blob/master/docs/howtotrain.md) are also available.

### How to Test

#### For simple testing
The recommended way to get started with some of the models produced by the training codes available in this repository is by getting the [pretrained models](#pretrained-models) to be tested and run them in the companion repository [iNNfer](https://github.com/victorca25/iNNfer), with the purpose of model inference.

Additionally, you can also use a GUI (for [ESRGAN models](https://github.com/n00mkrad/cupscale), for [video](https://github.com/n00mkrad/flowframes)) or a smaller repo for inference (for [ESRGAN](https://github.com/JoeyBallentine/ESRGAN), for [video](https://github.com/JoeyBallentine/Video-Inference)). 

If you are interested in obtaining results that can automatically return evaluation metrics, it is also possible to do inference of batches of images and some additional options with the instructions in [how to test](https://github.com/victorca25/BasicSR/blob/master/docs/howtotest.md).


## Pretrained models
The most recent community pretrained models can be found in the [Wiki](https://upscale.wiki/wiki/Model_Database), Discord channels ([game upscale](https://discord.gg/nbB4A5F) and [animation upscale](https://discord.gg/vMaeuTEPh9)) and [nmkd's models](https://nmkd.de/?esrgan).

For more details about the original and experimental pretrained models, please see [`pretrained models`](https://github.com/victorca25/BasicSR/tree/master/docs/pretrained.md).

You can put the downloaded models in the default `experiments/pretrained_models` directory and use them in the options files with the corresponding network architectures.


### Model interpolation
Models that were trained using the same pretrained model or are derivates of the same pretrained model are able to be interpolated to combine the properties of both. The original author demostrated this by interpolating the PSNR pretrained model (which is not perceptually good, but results in smooth images) with the ESRGAN resulting models that have more details but sometimes is excessive to control a balance in the resulting images, instead of interpolating the resulting images from both models, giving much better results.

The capabilities of linearly interpolating models are also explored in "DNI": [Deep Network Interpolation for Continuous Imagery Effect Transition](https://xinntao.github.io/projects/DNI) (CVPR19) with very interesting results and examples. The script for interpolation can be found in the [net_interp.py](https://github.com/victorca25/BasicSR/blob/master/codes/scripts/net_interp.py) file. This is an alternative to create new models without additional training and also to create pretrained models for easier fine tuning. Below is an example of interpolating between a PSNR-oriented and a perceptual `ESRGAN` model (first row), and examples of interpolating `CycleGAN` style transfer models.

<p align="center">
   <img src="https://camo.githubusercontent.com/913baa366ba395595a9638ab6282a9cbb088ab98/68747470733a2f2f78696e6e74616f2e6769746875622e696f2f70726f6a656374732f444e495f7372632f7465617365722e6a7067" height="300">
</p>

More details and explanations of interpolation can be found [here](https://github.com/victorca25/BasicSR/wiki/Interpolation) in the Wiki.

## Datasets

Many [datasets](https://github.com/victorca25/BasicSR/blob/master/docs/datasets.md) are publicly available and used to train models in a way that can be benchmarked and compared with other models. You are also able to create your own datasets with your own images.

Any dataset can be augmented to expose the model to information that might not be available in the images, such a noise and blur. For this reason, a [data augmentation](https://github.com/victorca25/BasicSR/wiki/Dataset-Augmentation) pipeline has been added to the options in this repository. It is also possible to add other types of augmentations, such as `Batch Augmentations` to apply them to minibatches instead of single images. Lastly, if your dataset is small, you can make use of `Differential Augmentations` to allow the discriminator to extract more information from the available images and train better models. More information can be found in the [augmentations](https://github.com/victorca25/BasicSR/blob/master/docs/augmentations.md) document.

# How to help

There are multiple ways to help this project. The first one is by using it and trying to train your own models. You can open an [issue](https://github.com/victorca25/BasicSR/issues) if you find any bugs or start a [discussion](https://github.com/victorca25/BasicSR/discussions) if you have ideas, questions or would like to showcase your results.

If you would like to contribute in the form of adding or fixing code, you can do so be cloning this repo and creating a [PR](https://github.com/victorca25/BasicSR/pulls). Ideally, it's better for PR to be precise and not changing many parts of the code at the same time, so it can be reviewed and tested. If possible, open an issue or discussion prior to creating the PR and we can talk about any ideas.

You can also join the [discord servers](#additional-Help) and share results and questions with other users.

Lastly, after it has been suggested many times before, now there are options to donate to show your support to the project and help stir it in directions that will make it even more useful. Below you will find those options that were suggested.

<p align="left">
   <a href="https://patreon.com/victorca25">
      <img src="https://github.githubassets.com/images/modules/site/icons/funding_platforms/patreon.svg" height="30">
      Patreon
   </a>
</p>

<p align="left">
   <a href="https://user-images.githubusercontent.com/41912303/121814560-fba1fc80-cc71-11eb-9b98-17c3ce0f06d6.png">
      <img src="https://user-images.githubusercontent.com/41912303/121814516-ca293100-cc71-11eb-9ddf-ffda840cd36d.png" height="30">
      <img src="https://user-images.githubusercontent.com/41912303/121814560-fba1fc80-cc71-11eb-9b98-17c3ce0f06d6.png" height="30">
   </a>
   Bitcoin Address: 1JyWsAu7aVz5ZeQHsWCBmRuScjNhCEJuVL
</p>

<p align="left">
   <a href="https://user-images.githubusercontent.com/41912303/121814692-aa463d00-cc72-11eb-99b2-c1bae3f63fdc.png">
      <img src="https://user-images.githubusercontent.com/41912303/121814599-36a43000-cc72-11eb-974a-146661e5e665.png" height="30">
      <img src="https://user-images.githubusercontent.com/41912303/121814692-aa463d00-cc72-11eb-99b2-c1bae3f63fdc.png" height="30">
   </a>
   Ethereum Address: 0xa26AAb3367D34457401Af3A5A0304d6CbE6529A2
</p>

* * *
## Additional Help

If you have any questions, we have a couple of discord servers ([game upscale](https://discord.gg/nbB4A5F) and [animation upscale](https://discord.gg/vMaeuTEPh9)) where you can ask them and a [Wiki](https://upscale.wiki/) with more information.

* * *

## Acknowledgement

Code architecture is originally inspired by [pytorch-cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and the first version of [BasicSR](https://github.com/xinntao/BasicSR).