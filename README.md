# BasicSR with Automatic Mixed Precision

This branch of BasicSR utilises the highly experimental PyTorch CUDA AMP library, which allows Volta/Turing GPUs to take advantage of their Tensor cores that results in reducing training time by +/- 50%.

*Most of the documentation is at the Master branch. This branch only deals with AMP specific features.

## Table of Contents
1. [Dependencies](#dependencies)
2. [Features](#features)
3. [To Do](#todo)

### Dependencies
- PyTorch 1.6, which is currently the latest nightly builds.
- All [BasicSR dependencies](https://github.com/victorca25/BasicSR/) as documented at victorca25's branch.
- [ImageMagick](https://imagemagick.org/script/download.php) for the image manipulation library. 
- Python package: [`pip install wand`](https://pypi.org/project/Wand/), to access IM from Python.

## Features
To run this branch requires you to own a Volta/Turing GPU. Minimum entry level card is perhaps a GeForce RTX 2060. Running it on normal GTX may bring only slight improvements, or may even be counter-productive. You may read at [nVidia's dev website](https://developer.nvidia.com/automatic-mixed-precision) on how it works.

Currently, the SSIM loss function is **broken** when using mixed precision, so do not use it in your training options.

---

## Acknowledgement
- Big thanks to *victorca25* for encouraging the creation of this fork.
- Thanks to *Twittman* for sharing how Fatality's training tiles and screentones were generated.
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
