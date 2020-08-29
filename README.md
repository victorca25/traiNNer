# BasicSR [[ESRGAN]](https://github.com/xinntao/ESRGAN) [[SFTGAN]](https://github.com/xinntao/SFTGAN) [[PPON]](https://github.com/Zheng222/PPON)

WARNING: This is a development branch to merge many changes and rewrite the codebase, some things may change, some things may break, create an issue if you find anything.

This is a heavily modified fork of BasicSR. What you will find here: boilerplate code for training and testing CV models, different CV methods and strategies integrated in a single pipeline and modularity to add and remove components as needed, including new network architectures. A large rewrite of code is being made to: reduce code redundancy and duplicates, reorganize the code and make it more modular.

Some of the new things in the branch:
- The filters and image manipulations used by the different functions (HFEN, SSIM/MS-SSIM, SPL, TV/DTV, etc) are now consolidated in filters.py and colors.py
- Reusable loss builder to reduce the changes needed when using a new model and adding new losses only once for all models
- Metrics builder to include only the selected ones during validation
- Automatic Mixed Precision (AMP: https://pytorch.org/docs/master/amp.html) is now properly integrated. (Code updated to work with Pytorch 1.6.0 and 1.3.0).
- Added the Contextual Loss (https://arxiv.org/abs/1803.02077, https://arxiv.org/abs/1803.04626) option: 'cx_type', Differential Augmentations for efficient gan training (https://arxiv.org/pdf/2006.10738) option: 'diffaug', batch augmentations (based on https://arxiv.org/abs/2004.00448) option: 'mixup', ESRGAN+ improvements to the ESRGAN network (https://arxiv.org/pdf/2001.08073) options: 'gaussian' and 'plus', adapted frequency filtering per loss function (https://arxiv.org/pdf/1911.07850), enabled using the feature maps from the discriminator in training for feature similarity (https://arxiv.org/abs/1712.05927), additional fixes and general code refactoring.
- Other changes: added graceful interruption of training to continue from where it was interrupted, virtual batch option, "strict" model loading flag





:black_square_button: TODO

- [ ] Test TV loss/regularization (needs to balance loss weight with other losses). 
- [ ] Test HFEN loss (needs to balance loss weight with other losses). 
- [ ] Test [Partial Convolution based Padding](https://github.com/NVIDIA/partialconv) (PartialConv2D).
- [ ] Test PartialConv2D with random masks.
- [ ] Add automatic model scale change (preserve conv layers, estimate upscale layers).
- [ ] Add automatic loading of old models and new ESRGAN models.
- [ ] Downscale images before and/or after inference. Helps in cleaning up some noise or bring images back to the original scale.
- [ ] Adopt SRPGAN's extraction of features from the discriminator to test if it reduces compute usage
- [ ] Import GMFN's recurrent network and add the feature loss to their MSE model, should have better MSE results with SRGAN's features/textures (Needs testing)
- [ ] Test PPON training code. Inference is the same as the PPON repo.

Done
- [:white_check_mark:] Add on the fly augmentations (gaussian noise, blur, JPEG compression).
- [:white_check_mark:] Add TV loss/regularization options. Useful for denoising tasks, reduces Total Variation.
- [:white_check_mark:] Add HFEN loss. Useful to keep high frequency information. Used Gaussian filter to reduce the effect of noise.
- [:white_check_mark:] Add [Partial Convolution based Padding](https://github.com/NVIDIA/partialconv) (PartialConv2D). It should help prevent edge padding issues. Zero padding is the default and typically has best performance, PartialConv2D has better performance and converges faster for segmentation and classification (https://arxiv.org/pdf/1811.11718.pdf). Code has been added, but the switch makes pretained models using Conv2D incompatible. Training new models for testing. (May be able to test inpainting and denoising)
- [:white_check_mark:] Added SSIM and MS-SSIM loss functions. Originally needed to replicate the PPON training code, it can also be used on ESRGAN models
- [:white_check_mark:] Import PPON's inference network to train using BasicSR's framework. They use dilated convolutions to increase receptive field and compare against ESRGAN with perceptually good results
- [:white_check_mark:] Almost complete implementation of the PPON training, based on the original published paper. It's missing the Multiscale L1 loss in phase 2 (currently it only does the L1 calculation at full scale, together with the MS-SSIM loss). Added TV Loss to phase 1 (Content Reconstruction), HFEN to phase 2 (Structure Reconstruction) and left phase 3 (Perceptual Reconstruction) with the same GAN and VGG_Feature loss as the original and can use an alternative learning rate scheme (MultiStepLR_Restart) or the original StepLR_Restart from the paper (all these options are configurable in the JSON file). Training doesn't necessarily have to stop after finishing phase 3, but it should to be the same as in the paper.


An image super-resolution toolkit flexible for development. It now provides:

1. **PSNR-oriented SR** models (e.g., SRCNN, SRResNet and etc). You can try different architectures, e.g, ResNet Block, ResNeXt Block, Dense Block, Residual Dense Block, Poly Block, Dual Path Block, Squeeze-and-Excitation Block, Residual-in-Residual Dense Block and etc.
<!--   1. want to compare more structures for SR. e.g. ResNet Block, ResNeXt Block, Dense Block, Residual Dense Block, Poly Block, Dual Path Block, Squeeze-and-Excitation Block and etc.
   1. want to provide some useful tricks for training SR networks.
   1. We are also curious to know what is the upper bound of PSNR for bicubic downsampling kernel by using an extremely large model.-->
2. [**Enhanced SRGAN**](https://github.com/xinntao/ESRGAN) model (It can also train the **SRGAN** model). Enhanced SRGAN achieves consistently better visual quality with more realistic and natural textures than [SRGAN](https://arxiv.org/abs/1609.04802) and won the first place in the [PIRM2018-SR Challenge](https://www.pirm2018.org/PIRM-SR.html). For more details, please refer to [Paper](https://arxiv.org/abs/1809.00219), [ESRGAN repo](https://github.com/xinntao/ESRGAN). (If you just want to test the model, [ESRGAN repo](https://github.com/xinntao/ESRGAN) provides simpler testing codes.)
<p align="center">
  <img height="350" src="https://github.com/xinntao/ESRGAN/blob/master/figures/baboon.jpg">
</p>

3. [**SFTGAN**](https://github.com/xinntao/CVPR18-SFTGAN) model. It adopts Spatial Feature Transform (SFT) to effectively incorporate other conditions/priors, like semantic prior for image SR, representing by segmentation probability maps. For more details, please refer to [Paper](https://arxiv.org/abs/1804.02815), [SFTGAN repo](https://github.com/xinntao/CVPR18-SFTGAN).
<p align="center">
  <img height="220" src="https://github.com/xinntao/SFTGAN/blob/master/figures/network_structure.png">
</p>

4. [**PPON**](https://github.com/Zheng222/PPON) model. The model for "Progressive Perception-Oriented Network for Single Image Super-Resolution", which the authors compare favorably against ESRGAN. Training is done progressively, by freezing and unfreezing layers in phases, which are: Content Reconstruction, Structure Reconstruction and Perceptual Reconstruction. For more details, please refer to [Paper](https://arxiv.org/abs/1907.10399), [PPON repo](https://github.com/Zheng222/PPON). The pretrained model for download can also be found in the original repo.
<p align="center">
   <img height="220" src="https://github.com/Zheng222/PPON/raw/master/figures/Structure.png">
</p>

    
## Table of Contents
1. [Dependencies](#dependencies)
1. [Codes](#codes)
1. [Usage](#usage)
1. [Datasets](#datasets)
1. [Pretrained models](#pretrained-models)

### Dependencies

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 0.4.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python`

#### Optional Dependencies

- Python package: [`pip install tensorboardX`](https://github.com/lanpa/tensorboardX), for visualizing curves.
- Python package: [`pip install lmdb`](https://github.com/jnwatson/py-lmdb), for lmdb database support.

# Codes
[`./codes`](https://github.com/victorca25/BasicSR/tree/master/codes). We provide a detailed explaination of the **code framework** in [`./codes`](https://github.com/victorca25/BasicSR/tree/master/codes).
<p align="center">
   <img src="https://github.com/xinntao/public_figures/blob/master/BasicSR/code_framework.png" height="300">
</p>

We also provide:

1. Some useful scripts. More details in [`./codes/scripts`](https://github.com/victorca25/BasicSR/tree/master/codes/scripts). 
1. [Evaluation codes](https://github.com/victorca25/BasicSR/tree/master/metrics), e.g., PSNR/SSIM metric.
1. [Wiki](https://github.com/xinntao/BasicSR/wiki), e.g., How to make high quality gif with full (true) color, Matlab bicubic imresize and etc.

# Usage
### Data and model preparation
The common **SR datasets** can be found in [Datasets](#datasets). Detailed data preparation can be seen in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data).

We provide **pretrained models** in [Pretrained models](#pretrained-models).

## How to Test
### Test ESRGAN (SRGAN) models
1. Modify the configuration file `options/test/test_esrgan.json` 
1. Run command: `python test.py -opt options/test/test_esrgan.json`

### Test SR models
1. Modify the configuration file `options/test/test_sr.json` 
1. Run command: `python test.py -opt options/test/test_sr.json`

### Test SFTGAN models
1. Obtain the segmentation probability maps: `python test_seg.py`
1. Run command: `python test_sftgan.py`

### Test PPON models
1. Modify the configuration file `options/test/test_ppon.json` 
1. Run command: `python test_ppon.py -opt options/test/test_ppon.json`

## How to Train
### Train ESRGAN (SRGAN) models
We use a PSNR-oriented pretrained SR model to initialize the parameters for better quality. According to the author's paper and some testing, this will also stabilize the GAN training and allows for faster convergence. 

1. Prepare datasets, usually the DIV2K dataset. More details are in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data) and [
(Faster IO speed)](https://github.com/xinntao/BasicSR/wiki/Faster-IO-speed). 
1. Optional: If the intention is to replicate the original paper here you would prerapre the PSNR-oriented pretrained model. You can also use the original `RRDB_PSNR_x4.pth` as the pretrained model for that purpose, otherwise *any* existing model will work as pretrained.
1. Modify the configuration file  `options/train/train_esrgan.json`
1. Run command: `python train.py -opt options/train/train_esrgan.json`

### Train SR models
1. Prepare datasets, usually the DIV2K dataset. More details are in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data). 
1. Modify the configuration file `options/train/train_sr.json`
1. Run command: `python train.py -opt options/train/train_sr.json`

### Train SFTGAN models 
*Pretraining is also important*. We use a PSNR-oriented pretrained SR model (trained on DIV2K) to initialize the SFTGAN model.

1. First prepare the segmentation probability maps for training data: run [`test_seg.py`](https://github.com/victorca25/BasicSR/blob/master/codes/test_seg.py). We provide a pretrained segmentation model for 7 outdoor categories in [Pretrained models](#pretrained-models). We use [Xiaoxiao Li's codes](https://github.com/lxx1991/caffe_mpi) to train our segmentation model and transfer it to a PyTorch model.
1. Put the images and segmentation probability maps in a folder as described in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data).
1. Transfer the pretrained model parameters to the SFTGAN model. 
    1. First train with `debug` mode and obtain a saved model.
    1. Run [`transfer_params_sft.py`](https://github.com/victorca25/BasicSR/blob/master/codes/scripts/transfer_params_sft.py) to initialize the model.
    1. We provide an initialized model named `sft_net_ini.pth` in [Pretrained models](#pretrained-models)
1. Modify the configuration file in `options/train/train_sftgan.json`
1. Run command: `python train.py -opt options/train/train_sftgan.json`

### Train PPON models 
1. Prepare datasets, usually the DIV2K dataset. More details are in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data). 
1. Modify the configuration file `options/train/train_PPON.json`
1. Run command: `python train_ppon.py -opt options/train/train_PPON.json`

### Resuming Training 
When resuming training, just pass a option with the name `resume_state`, like , <small>`"resume_state": "../experiments/debug_001_RRDB_PSNR_x4_DIV2K/training_state/200.state"`. </small>


# Datasets
Several common SR datasets are list below. 

<table>
  <tr>
    <th>Name</th>
    <th>Datasets</th>
    <th>Short Description</th>
    <th>Google Drive</th>
    <th>Baidu Drive</th>
  </tr>
  <tr>
    <td rowspan="3">Classical SR Training</td>
    <td>T91</td>
    <td><sub>91 images for training</sub></td>
    <td rowspan="9"><a href="https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU?usp=sharing">Google Drive</a></td>
    <td rowspan="9"><a href="https://pan.baidu.com/s/1q_1ERCMqALH0xFwjLM0pTg">Baidu Drive</a></td>
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
    <td rowspan="3"><a href="https://pan.baidu.com/s/1CFIML6KfQVYGZSNFrhMXmA">Baidu Drive</a></td>
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
    <td rowspan="2"><a href="https://pan.baidu.com/s/1neUq5tZ4yTnOEAntZpK_rQ#list/path=%2Fpublic%2FSFTGAN&parentPath=%2Fpublic">Baidu Drive</a></td>
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
    <td rowspan="2"><a href="https://pan.baidu.com/s/1gYv4tSJk_RVCbCq4B6UxNQ">Baidu Drive</a></td>
  </tr>
</table>

Any dataset can be augmented to expose the model to information that might not be available in the images, such a noise and blur. For this reason, [Data Augmentation](https://github.com/victorca25/BasicSR/wiki/Dataset-Augmentation) has been added to the options in this repository and it can be extended to include other types of augmentations.


# Pretrained models
The most recent community pretrained models can be found in the [Wiki](https://github.com/alsa64/AI-wiki/wiki/Model-Database).

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
    <th>Baidu Drive</th>
  </tr>
  <tr>
    <th rowspan="2">ESRGAN</th>
    <td>RRDB_ESRGAN_x4.pth</td>
    <td><sub>final ESRGAN model we used in our paper</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY">Google Drive</a></td>
    <td rowspan="2"><a href="https://pan.baidu.com/s/1-Lh6ma-wXzfH8NqeBtPaFQ)">Baidu Drive</a></td>
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
    <td rowspan="4"><a href="">Baidu Drive</a></td>
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
    <td><a href="">Baidu Drive</a></td>
  </tr>
  
  <tr>
    <td >SRResNet<sup>*2</sup></td>
    <td>SRResNet_bicx4_in3nf64nb16.pth</td>
     <td><sub> SRResNet(with modification)</sub></td>
    <td><a href="https://drive.google.com/drive/folders/1WR2X4_gwiQ9REb5fHfNnBfXOdeuDS8BA?usp=sharing">Google Drive</a></td>
    <td><a href="">Baidu Drive</a></td>
  </tr>
  
  <tr>
    <td >PPON<sup>*2</sup></td>
    <td>PPON.pth</td>
     <td><sub>PPON model presented in the paper</sub></td>
    <td><a href="https://github.com/Zheng222/PPON">Original Repo</a></td>
    <td><a href=""></a></td>
  </tr>
</table>

For more details about the original pretrained models, please see [`experiments/pretrained_models`](https://github.com/victorca25/BasicSR/tree/master/experiments/pretrained_models).

---

## Additional Help 

If you have any questions, we have a [discord server](https://discord.gg/SxvYsgE) where you can ask them and a [Wiki](https://github.com/alsa64/AI-wiki/wiki) with more information.

---

## Acknowledgement

- Code architecture is inspired by [pytorch-cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- Thanks to *Wai Ho Kwok*, who contributes to the initial version.

### BibTex

    @InProceedings{wang2018esrgan,
        author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
        title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
        booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
        month = {September},
        year = {2018}
    }
    @InProceedings{wang2018sftgan,
        author = {Wang, Xintao and Yu, Ke and Dong, Chao and Loy, Chen Change},
        title = {Recovering realistic texture in image super-resolution by deep spatial feature transform},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2018}
    }
    @article{Hui-PPON-2019,
        title={Progressive Perception-Oriented Network for Single Image Super-Resolution},
        author={Hui, Zheng and Li, Jie and Gao, Xinbo and Wang, Xiumei},
        booktitle={arXiv:1907.10399v1},
        year={2019}
    }
    @InProceedings{Liu2019abpn,
        author = {Liu, Zhi-Song and Wang, Li-Wen and Li, Chu-Tak and Siu, Wan-Chi},
        title = {Image Super-Resolution via Attention based Back Projection Networks},
        booktitle = {IEEE International Conference on Computer Vision Workshop(ICCVW)},
        month = {October},
        year = {2019}
    }
