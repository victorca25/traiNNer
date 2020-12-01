

1. **PSNR-oriented SR** models (e.g., SRCNN, SRResNet and etc). You can try different architectures, e.g, ResNet Block, ResNeXt Block, Dense Block, Residual Dense Block, Poly Block, Dual Path Block, Squeeze-and-Excitation Block, Residual-in-Residual Dense Block and etc.
<!--   1. want to compare more structures for SR. e.g. ResNet Block, ResNeXt Block, Dense Block, Residual Dense Block, Poly Block, Dual Path Block, Squeeze-and-Excitation Block and etc.
   1. want to provide some useful tricks for training SR networks.
   1. We are also curious to know what is the upper bound of PSNR for bicubic downsampling kernel by using an extremely large model.-->

2. [**Enhanced SRGAN**](https://github.com/xinntao/ESRGAN) model (It can also train the **SRGAN** model). Enhanced SRGAN achieves consistently better visual quality with more realistic and natural textures than [SRGAN](https://arxiv.org/abs/1609.04802) and won the first place in the [PIRM2018-SR Challenge](https://www.pirm2018.org/PIRM-SR.html). For more details, please refer to [Paper](https://arxiv.org/abs/1809.00219), [ESRGAN repo](https://github.com/xinntao/ESRGAN). (If you just want to test the model, [ESRGAN repo](https://github.com/xinntao/ESRGAN) provides simpler testing codes.)
<p align="center">
  <img height="350" src="https://github.com/xinntao/ESRGAN/blob/master/figures/baboon.jpg">
</p>
[**ESRGAN+**](https://github.com/ncarraz/ESRGANplus) [Paper](https://arxiv.org/pdf/2001.08073). Details TBD.

3. [**SFTGAN**](https://github.com/xinntao/CVPR18-SFTGAN) model. It adopts Spatial Feature Transform (SFT) to effectively incorporate other conditions/priors, like semantic prior for image SR, representing by segmentation probability maps. For more details, please refer to [Paper](https://arxiv.org/abs/1804.02815), [SFTGAN repo](https://github.com/xinntao/CVPR18-SFTGAN).
<p align="center">
  <img height="220" src="https://github.com/xinntao/SFTGAN/blob/master/figures/network_structure.png">
</p>

4. [**PPON**](https://github.com/Zheng222/PPON) model. The model for "Progressive Perception-Oriented Network for Single Image Super-Resolution", which the authors compare favorably against ESRGAN. Training is done progressively, by freezing and unfreezing layers in phases, which are: Content Reconstruction, Structure Reconstruction and Perceptual Reconstruction. For more details, please refer to [Paper](https://arxiv.org/abs/1907.10399). The pretrained model for download can also be found in the original repo.
<p align="center">
   <img height="220" src="https://github.com/Zheng222/PPON/raw/master/figures/Structure.png">
</p>

5. [**PAN**](https://github.com/zhaohengyuan1/PAN) Pixel Attention Network for Efficient Image Super-Resolution. [Paper](https://arxiv.org/pdf/2010.01073.pdf). Details TBD.

6. [**SOFVSR**](https://github.com/LongguangWang/SOF-VSR/tree/master/TIP) Deep Video Super-Resolution using HR Optical Flow Estimation. [Paper](http://arxiv.org/abs/2001.02129). Details TBD.

7. [**RIFE**](https://github.com/hzwer/arXiv2020-RIFE). [Paper](https://arxiv.org/abs/2011.06294). Details TBD.

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
