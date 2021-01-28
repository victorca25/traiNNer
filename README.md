# BasicSR

BasicSR (Basic Super Restoration) is an open source image and video restoration toolbox (super-resolution, denoising, deblurring and others) based on PyTorch.

[![Python Version](https://img.shields.io/badge/python-3-informational?style=flat)](https://python.org)
[![License](https://img.shields.io/github/license/victorca25/BasicSR?style=flat)](https://github.com/victorca25/BasicSR/blob/master/LICENSE)
[![DeepSource](https://deepsource.io/gh/victorca25/BasicSR.svg/?label=active+issues&show_trend=true)](https://deepsource.io/gh/victorca25/BasicSR/?ref=repository-badge)
[![Issues](https://img.shields.io/github/issues/victorca25/BasicSR?style=flat)](https://github.com/victorca25/BasicSR/issues)
[![PR's Accepted](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://makeapullrequest.com)


This is a heavily modified fork of the original BasicSR. What you will find here: boilerplate code for training and testing computer vision (CV) models, different CV methods and strategies integrated in a single pipeline and modularity to add and remove components as needed, including new network architectures. A large rewrite of code was made to reduce code redundancy and duplicates, reorganize the code and make it more modular.

Details of the supported architectures can be found [here](https://github.com/victorca25/BasicSR/blob/master/docs/architectures.md).

(README currently WIP)

Some of the new things in the latest version of this code:
-   The filters and image manipulations used by the different functions (HFEN, SSIM/MS-SSIM, SPL, TV/DTV, etc) are now consolidated in filters.py and colors.py
-   Reusable loss builder to reduce the changes needed when using a new model and adding new losses only once for all models
-   Metrics builder to include only the selected ones during validation
-   Automatic Mixed Precision (AMP: <https://pytorch.org/docs/master/amp.html>) is now properly integrated. (Code updated to work with Pytorch 1.6.0 and 1.3.0). Option "use_amp".
-   Contextual Loss (<https://arxiv.org/abs/1803.02077>, <https://arxiv.org/abs/1803.04626>). Option: 'cx_type'.
-   Differential Augmentations for efficient gan training (<https://arxiv.org/pdf/2006.10738>). Option: 'diffaug'.
-   batch augmentations (based on <https://arxiv.org/abs/2004.00448>). Option: 'mixup'.
-   ESRGAN+ improvements to the ESRGAN network (<https://arxiv.org/pdf/2001.08073>). Options: 'gaussian' and 'plus'.
-   adapted frequency filtering per loss function (<https://arxiv.org/pdf/1911.07850>). Option: 'fs'.
-   enabled option to use the feature maps from the VGG-like discriminator in training for feature similarity (<https://arxiv.org/abs/1712.05927>). Option: 'discriminator_vgg_128_fea'.
-   PatchGAN option for the discriminator (<https://arxiv.org/pdf/1611.07004v3.pdf>). Option: 'patchgan'.
-   Multiscale PatchGAN option for the discriminator (<https://arxiv.org/pdf/1711.11585.pdf>). Option: 'multiscale'.
-   Added a modified Pixel Attention Network for Efficient Image Super-Resolution (<https://arxiv.org/pdf/2010.01073.pdf>), which includes a self-attention layer in the residual path, among other changes. A basic pretrained model for 4x scale can be found [here](https://mega.nz/file/mpRgVThY#tRi1q_PrY5OX4MVOTtjWlXzBXcLZs2tP1duo-mEkWSs)
-   Stochastic Weight Averaging (SWA: <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/>, <https://arxiv.org/pdf/1803.05407.pdf>) added as an option. Currently the change only applies to the generator network, changing the original learning rate scheduler to the SWA scheduler after a defined number of iterations have passed (the original paper refers to the later 25% part of training). The resulting SWA model can be converted to a regular model after training using the scripts/swa2normal.py script. Option "use_swa" and configure the swa scheduler.
-   Added the basic idea behind "Freeze Discriminator: A Simple Baseline for Fine-tuning GANs" (<https://arxiv.org/pdf/2002.10964.pdf>) to accelerate training with transfer learning. It is possible to use a pretrained discriminator model and freeze the initial (bottom) X number of layers. Option: "freeze_loc", enabled for any of the VGG-like discriminators or patchgan (muliscale patchgan not yet added).
-   Other changes: added graceful interruption of training to continue from where it was interrupted, virtual batch option, "strict" model loading flag, support for using YAML or JSON options files, color transfer script (color_transfer.py) with multiple algorithms to transfer image statistics (colors) from a reference image to another, integrated the "forward_chop" function into the SR model to crop images into patches before upscaling for inference in VRAM constrained systems (use option test_mode: chop), general fixes and code refactoring.

WIP:
-   Added on the fly use of realistic image kernels extracted with KernelGAN (<https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Kernel_Modeling_Super-Resolution_on_Real_Low-Resolution_Images_ICCV_2019_paper.pdf>) and injection of noise extracted from real images patches (<https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Image_Blind_Denoising_CVPR_2018_paper.pdf>)
-   Change to use openCV-based composable transformation for augmentations (<https://github.com/victorca25/opencv_transforms>) with a new dataloader 
-   Use of configuration presets for reuse instead of editing full configuration files 
-   Video network for optical flow and video super-resolution (<http://arxiv.org/abs/2001.02129>. Pretrained model using 3 frames, trained on a subset of REDS dataset [here](https://mega.nz/file/28JmyLrK#xhRP-EZKR7Vg7UjIRZQqotiFLix21JaGGLSvZq7cjt4)) 
-   Added option to use different image upscaling networks with the HR optical flow estimation for video (Pretrained using 3 frames and default ESRGAN as SR network [here](https://mega.nz/file/TwwEWD7Q#wCfUvVudI17weYc1JLeM3nTeK2xiMlVdc_JN1Nov3ac))
-   Initial integration of RIFE (<https://arxiv.org/abs/2011.06294>) architecture for Video Frame Interpolation (Converted trained model from three pickle files into a single pth model [here](https://mega.nz/file/DhBWgRYQ#hLkR4Eiks6s3ZvwLCl4eA57J3baR0eDXjyaV9yzmTeM))
-   Video ESRGAN (EVSRGAN) and SR3D networks using 3D convolution for video super-resolution, inspired on "3DSRnet: Video Super-resolution using 3D Convolutional Neural Networks" (<https://arxiv.org/pdf/1812.09079.pdf>). (EVSRGAN Pretrained using 3 frames and default arch options [here](https://u.pcloud.link/publink/show?code=XZ2Wg8XZebryABNV8Q0GsSE2ifkLdh9NzzaX))
-   Real-time Deep Video Deinterlacing (https://arxiv.org/pdf/1708.00187.pdf) training and testing codes implemented. (Pretraineds DVD models can be found [here](https://u.pcloud.link/publink/show?code=kZIIfQXZYLGBJF4sQVJ2aONxgwiPr8iQPxo7))

(Previous changes can be found [here](https://github.com/victorca25/BasicSR/blob/master/docs/changes.md))
    
## Table of Contents
1.  [Dependencies](#dependencies)
2.  [Codes](#codes)
3.  [Usage](#usage)
4.  [Datasets](#datasets)
5.  [Pretrained models](#pretrained-models)

### Dependencies

-   Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
-   [PyTorch >= 0.4.0](https://pytorch.org/). PyTorch >= 1.7.0 required to enable certain features (SWA, AMP, others).
-   NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
-   Python packages: `pip install numpy opencv-python`

#### Optional Dependencies

-   Python package: [`pip install tensorboardX`](https://github.com/lanpa/tensorboardX), for visualizing curves.
-   Python package: [`pip install lmdb`](https://github.com/jnwatson/py-lmdb), for lmdb database support.

## Codes

[`./codes`](https://github.com/victorca25/BasicSR/tree/master/codes). We provide a detailed explaination of the **code framework** in [`./codes`](https://github.com/victorca25/BasicSR/tree/master/codes).

<p align="center">
   <img src="https://github.com/xinntao/public_figures/blob/master/BasicSR/code_framework.png" height="300">
</p>

We also provide:

1.  Some useful scripts. More details in [`./codes/scripts`](https://github.com/victorca25/BasicSR/tree/master/codes/scripts). 
2.  [Evaluation codes](https://github.com/victorca25/BasicSR/tree/master/metrics), e.g., PSNR/SSIM metric.

To extract the estimated kernels and noise patches from images, use the modified KernelGAN and patches extraction code in: [DLIP](https://github.com/victorca25/DLIP). Detailed instructions to use the estimated kernels are available [here](https://github.com/victorca25/BasicSR/blob/master/docs/kernels.md)

## Usage

### Data and model preparation

The common **SR datasets** can be found in [Datasets](#datasets). Detailed data preparation can be seen in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data).

We provide **pretrained models** in [Pretrained models](#pretrained-models).

### How to Test

#### For simple testing
The recommended way to get started with some of the models produced by the training codes available in this repository is by getting the pretrained models to be tested and either a GUI (for [ESRGAN models](https://github.com/n00mkrad/cupscale), for [video](https://github.com/n00mkrad/flowframes)) or a smaller repo for inference (for [ESRGAN](https://github.com/JoeyBallentine/ESRGAN), for [video](https://github.com/JoeyBallentine/Video-Inference)). 

Otherwise, it is also possible to do inference of batches of images with the code in this repository as follow.

#### Test Super Resolution models (ESRGAN, PPON, PAN, others)

1.  Modify the configuration file `options/test/test_ESRGAN.yml` (or `options/test/test_ESRGAN.json`)
2.  Run command: `python test.py -opt options/test/test_ESRGAN.yml` (or `python test.py -opt options/test/test_ESRGAN.json`)

#### Test SFTGAN models

1.  Obtain the segmentation probability maps: `python test_seg.py`
2.  Run command: `python test_sftgan.py`

#### Test VSR models

1.  Modify the configuration file `options/test/test_video.yml`
2.  Run command: `python test_vsr.py -opt options/test/test_video.yml`

### How to Train
[How to train](https://github.com/victorca25/BasicSR/blob/master/docs/howtotrain.md)

## Datasets

Several common SR datasets are list below. 

<table>
  <tr>
    <th>Name</th>
    <th>Datasets</th>
    <th>Short Description</th>
    <th>Google Drive</th>
    <th>Other</th>
  </tr>
  <tr>
    <td rowspan="3">Classical SR Training</td>
    <td>T91</td>
    <td><sub>91 images for training</sub></td>
    <td rowspan="9"><a href="https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU?usp=sharing">Google Drive</a></td>
    <td rowspan="9"><a href="https://pan.baidu.com/s/1q_1ERCMqALH0xFwjLM0pTg">Other</a></td>
  </tr>
 <tr>
    <td><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">BSDS200</a></td>
    <td><sub>A subset (train) of BSD500 for training</sub></td>
  </tr>
  <tr>
    <td><a href="http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html">General100</a></td>
    <td><sub>100 images for training</sub></td>
  </tr>
  <tr>
    <td rowspan="6">Classical SR Testing</td>
    <td>Set5</td>
    <td><sub>Set5 test dataset</sub></td>
  </tr>
  <tr>
    <td>Set14</td>
    <td><sub>Set14 test dataset</sub></td>
  </tr>
  <tr>
    <td><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">BSDS100</a></td>
    <td><sub>A subset (test) of BSD500 for testing</sub></td>
  </tr>
  <tr>
    <td><a href="https://sites.google.com/site/jbhuang0604/publications/struct_sr">urban100</a></td>
    <td><sub>100 building images for testing (regular structures)</sub></td>
  </tr>
  <tr>
    <td><a href="http://www.manga109.org/en/">manga109</a></td>
    <td><sub>109 images of Japanese manga for testing</sub></td>
  </tr>
  <tr>
    <td>historical</td>
    <td><sub>10 gray LR images without the ground-truth</sub></td>
  </tr>
   
  <tr>
    <td rowspan="3">2K Resolution</td>
    <td><a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">DIV2K</a></td>
    <td><sub>proposed in <a href="http://www.vision.ee.ethz.ch/ntire17/">NTIRE17</a>(800 train and 100 validation)</sub></td>
    <td rowspan="3"><a href="https://drive.google.com/drive/folders/1B-uaxvV9qeuQ-t7MFiN1oEdA6dKnj2vW?usp=sharing">Google Drive</a></td>
    <td rowspan="3"><a href="https://pan.baidu.com/s/1CFIML6KfQVYGZSNFrhMXmA">Other</a></td>
  </tr>
 <tr>
    <td><a href="https://github.com/LimBee/NTIRE2017">Flickr2K</a></td>
    <td><sub>2650 2K images from Flickr for training</sub></td>
  </tr>
 <tr>
    <td>DF2K</td>
    <td><sub>A merged training dataset of DIV2K and Flickr2K</sub></td>
  </tr>
  
  <tr>
    <td rowspan="2">OST (Outdoor Scenes)</td>
    <td>OST Training</td>
    <td><sub>7 categories images with rich textures</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/u/1/folders/1iZfzAxAwOpeutz27HC56_y5RNqnsPPKr">Google Drive</a></td>
    <td rowspan="2"><a href="https://pan.baidu.com/s/1neUq5tZ4yTnOEAntZpK_rQ#list/path=%2Fpublic%2FSFTGAN&parentPath=%2Fpublic">Other</a></td>
  </tr>
 <tr>
    <td>OST300</td>
    <td><sub>300 test images of outdoor scences</sub></td>
  </tr>
  
  <tr>
    <td >PIRM</td>
    <td>PIRM</td>
    <td><sub>PIRM self-val, val, test datasets</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/folders/17FmdXu5t8wlKwt8extb_nQAdjxUOrb1O?usp=sharing">Google Drive</a></td>
    <td rowspan="2"><a href="https://pan.baidu.com/s/1gYv4tSJk_RVCbCq4B6UxNQ">Other</a></td>
  </tr>
</table>

Any dataset can be augmented to expose the model to information that might not be available in the images, such a noise and blur. For this reason, [Data Augmentation](https://github.com/victorca25/BasicSR/wiki/Dataset-Augmentation) has been added to the options in this repository and it can be extended to include other types of augmentations.


## Pretrained models
The most recent community pretrained models can be found in the [Wiki](https://upscale.wiki/wiki/Model_Database), [Discord](https://discord.gg/nbB4A5F) and [nmkd's models](https://nmkd.de/?esrgan).

You can put the downloaded models in the default `experiments/pretrained_models` folder.

Models that were trained using the same pretrained model or are derivates of the same pretrained model are able to be interpolated to combine the properties of both. The original author demostrated this by interpolating the PSNR pretrained model (which is not perceptually good, but results in smooth images) with the ESRGAN resulting models that have more details but sometimes is excessive to control a balance in the resulting images, instead of interpolating the resulting images from both models, giving much better results.

The authors continued exploring the capabilities of linearly interpolating models in their new work "DNI" (CVPR19): [Deep Network Interpolation for Continuous Imagery Effect Transition](https://xinntao.github.io/projects/DNI) with very interesting results and examples. The script for interpolation can be found in the [net_interp.py](https://github.com/victorca25/BasicSR/blob/master/codes/scripts/net_interp.py) file, but a new version with more options will be commited at a later time. This is an alternative to create new models without additional training and also to create pretrained models for easier fine tuning. 

<p align="center">
   <img src="https://camo.githubusercontent.com/913baa366ba395595a9638ab6282a9cbb088ab98/68747470733a2f2f78696e6e74616f2e6769746875622e696f2f70726f6a656374732f444e495f7372632f7465617365722e6a7067" height="300">
</p>

More details and explanations of interpolation can be found [here](https://github.com/victorca25/BasicSR/wiki/Interpolation) in the Wiki.

Following are the original pretrained models that the authors made available for ESRGAN, SFTGAN and PPON:

<table>
  <tr>
    <th>Name</th>
    <th>Models</th>
    <th>Short Description</th>
    <th>Google Drive</th>
    <th>Other</th>
  </tr>
  <tr>
    <th rowspan="2">ESRGAN</th>
    <td>RRDB_ESRGAN_x4.pth</td>
    <td><sub>final ESRGAN model we used in our paper</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY">Google Drive</a></td>
    <td rowspan="2"><a href="https://pan.baidu.com/s/1-Lh6ma-wXzfH8NqeBtPaFQ)">Other</a></td>
  </tr>
  <tr>
    <td>RRDB_PSNR_x4.pth</td>
    <td><sub>model with high PSNR performance</sub></td>
  </tr>
   
  <tr>
    <th rowspan="4">SFTGAN</th>
    <td>segmentation_OST_bic.pth</td>
     <td><sub> segmentation model</sub></td>
    <td rowspan="4"><a href="https://drive.google.com/drive/folders/1WR2X4_gwiQ9REb5fHfNnBfXOdeuDS8BA?usp=sharing">Google Drive</a></td>
    <td rowspan="4"><a href="">Other</a></td>
  </tr>
  <tr>
    <td>sft_net_ini.pth</td>
    <td><sub>sft_net for initilization</sub></td>
  </tr>
  <tr>
    <td>sft_net_torch.pth</td>
    <td><sub>SFTGAN Torch version (paper)</sub></td>
  </tr>
  <tr>
    <td>SFTGAN_bicx4_noBN_OST_bg.pth</td>
    <td><sub>SFTGAN PyTorch version</sub></td>
  </tr>
  
  <tr>
    <td >SRGAN<sup>*1</sup></td>
    <td>SRGAN_bicx4_303_505.pth</td>
     <td><sub> SRGAN(with modification)</sub></td>
    <td><a href="https://drive.google.com/drive/folders/1WR2X4_gwiQ9REb5fHfNnBfXOdeuDS8BA?usp=sharing">Google Drive</a></td>
    <td><a href=""></a></td>
  </tr>
  
  <tr>
    <td >SRResNet<sup>*2</sup></td>
    <td>SRResNet_bicx4_in3nf64nb16.pth</td>
     <td><sub> SRResNet(with modification)</sub></td>
    <td><a href="https://drive.google.com/drive/folders/1WR2X4_gwiQ9REb5fHfNnBfXOdeuDS8BA?usp=sharing">Google Drive</a></td>
    <td><a href=""></a></td>
  </tr>
  
  <tr>
    <td>PPON<sup>*2</sup></td>
    <td>PPON.pth</td>
     <td><sub>PPON model presented in the paper</sub></td>
    <td><a href="https://github.com/Zheng222/PPON">Original Repo</a></td>
    <td><a href=""></a></td>
  </tr>

  <tr>
    <td>PAN</td>
    <td>PAN.pth</td>
     <td><sub>4x pretrained modified PAN model with self-attention</sub></td>
    <td><a href=""></a></td>
    <td><a href="https://mega.nz/file/mpRgVThY#tRi1q_PrY5OX4MVOTtjWlXzBXcLZs2tP1duo-mEkWSs">Other</a></td>
  </tr>

  <tr>
    <td>SOFVSR</td>
    <td>SOFVSR.pth</td>
     <td><sub>4x pretrained SOFVSR model, using 3 frames</sub></td>
    <td><a href=""></a></td>
    <td><a href="https://mega.nz/file/28JmyLrK#xhRP-EZKR7Vg7UjIRZQqotiFLix21JaGGLSvZq7cjt4">Other</a></td>
  </tr>

  <tr>
    <td>SOFVESRGAN</td>
    <td>SOFVESRGAN.pth</td>
     <td><sub>4x pretrained modified SOFVSR model using ESRGAN network for super-resolution, using 3 frames</sub></td>
    <td><a href=""></a></td>
    <td><a href="https://mega.nz/file/TwwEWD7Q#wCfUvVudI17weYc1JLeM3nTeK2xiMlVdc_JN1Nov3ac">Other</a></td>
  </tr>

  <tr>
    <td>RIFE</td>
    <td>RIFE.pth</td>
     <td><sub>Converted pretrained RIFE model from the three original pickle files into a single pth model</sub></td>
    <td><a href=""></a></td>
    <td><a href="https://mega.nz/file/DhBWgRYQ#hLkR4Eiks6s3ZvwLCl4eA57J3baR0eDXjyaV9yzmTeM">Other</a></td>
  </tr>
</table>

For more details about the original pretrained models, please see [`experiments/pretrained_models`](https://github.com/victorca25/BasicSR/tree/master/experiments/pretrained_models).

* * *
## Additional Help

If you have any questions, we have a [discord server](https://discord.gg/nbB4A5F) where you can ask them and a [Wiki](https://upscale.wiki/) with more information.

* * *

## Acknowledgement

-   Code architecture is inspired by [pytorch-cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and based on the original version of [BasicSR](https://github.com/xinntao/BasicSR).