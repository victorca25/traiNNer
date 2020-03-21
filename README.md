# BasicSR (Enhanced)

This is a fork of victorca25's [BasicSR](https://github.com/victorca25/BasicSR/) branch. Most of the documentation is there if you need any information regarding BasicSR. This readme will focus specifically on the differences of this fork.

## Table of Contents
1. [Dependencies](#dependencies)
3. [BasicSR Enhanced features](#features)
4. [Datasets](#datasets)
5. [Pretrained models](#pretrained-models)

### Dependencies

- All [BasicSR dependencies](https://github.com/victorca25/BasicSR/) as documented at victorca25's branch.
- [ImageMagick](https://imagemagick.org/script/download.php) for the image manipulation library. 
- Python package: [`pip install wand`](https://pypi.org/project/Wand/), to access IM from Python.

# BasicSR Enhanced features
This features are configured in the training `.json` file.

### Load state via CPU
Lower end graphics card with low VRAM may have difficulty resuming from a state. If you get a out of memory error when continuing a training session, then set `"load2CPU":true` so that it is loaded to the system RAM instead.


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
