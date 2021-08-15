# Supported architectures

Below are the network architectures and, in general, training strategies supported in this repository. Note that it is possible to combine ideas from all those below, from training strategy to even modifying the networks with components from different networks. They can serve as a baseline for your experiments as well.

## Super-Resolution

1. **[SRGAN](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)** (2017), that originally uses the `SRResNet` network, and introduced the idea of using a generative adversarial network (`GAN`) for image superresolution (`SR`). It was the first framework capable of inferring photo-realistic natural images for 4× with a loss function which consists of an adversarial loss (GAN), a feature loss (using a pretrained VGG classification network) and a content (pixel) loss.

2. [**Enhanced SRGAN**](https://arxiv.org/abs/1809.00219) (2018). Enhanced SRGAN achieves consistently better visual quality with more realistic and natural textures than `SRGAN` and won the first place in the [PIRM2018-SR Challenge](https://www.pirm2018.org/PIRM-SR.html). Originally uses the `RRDB` network. ESRGAN remains until today (2021) as the base for many projects and research papers that continuing building upon it. For more details, please refer to [ESRGAN repo](https://github.com/xinntao/ESRGAN).
<p align="center">
  <img height="250" src="https://user-images.githubusercontent.com/41912303/121768586-2e0f0500-cb5f-11eb-8ca7-9c3fc24b4bfe.png">
</p>

3. [**ESRGAN+**](https://arxiv.org/pdf/2001.08073) [Repo](https://github.com/ncarraz/ESRGANplus) (2020). A follow up paper that introduced two main changes to ESRGAN's `RRDB` network and can be enabled with the network options `plus` and `gaussian`.

4. [**SFTGAN**](https://arxiv.org/abs/1804.02815) (2018). Adopts Spatial Feature Transform (SFT) to incorporate other conditions/priors, like semantic prior for image `SR`, representing by segmentation probability maps. For more details, please refer to [SFTGAN repo](https://github.com/xinntao/CVPR18-SFTGAN).
<p align="center">
  <img height="170" src="https://user-images.githubusercontent.com/41912303/121768709-d2914700-cb5f-11eb-90f5-c37b86476bb8.png">
</p>

5. [**PPON**](https://arxiv.org/abs/1907.10399) (2019). The model and training strategy for "Progressive Perception-Oriented Network for Single Image Super-Resolution", which the authors compare favorably against ESRGAN. Training is done progressively, by freezing and unfreezing layers in phases, which are: Content Reconstruction, Structure Reconstruction and Perceptual Reconstruction. For more details, please refer to [PPON repo](https://github.com/Zheng222/PPON).
<p align="center">
   <img height="220" src="https://user-images.githubusercontent.com/41912303/121768753-184e0f80-cb60-11eb-9c44-15328416bada.png">
</p>

6. [**PAN**](https://arxiv.org/pdf/2010.01073.pdf) (2020). Pixel Attention Network for Efficient Image Super-Resolution. Aims at designing a lightweight network for image super resolution (`SR`) that can potentially be used in real-time. More details in [PAN repo](https://github.com/zhaohengyuan1/PAN).
<p align="center">
   <img height="220" src="https://user-images.githubusercontent.com/41912303/107143307-af962280-6934-11eb-90e6-0489158d7168.png">
</p>

7. The Consistency Enforcing Module (CEM) module from [**Explorable-Super-Resolution**](http://openaccess.thecvf.com/content_CVPR_2020/papers/Bahat_Explorable_Super_Resolution_CVPR_2020_paper.pdf) (2020). Can be used to wrap **any** network (during training or testin) around a module that has no trainable parameters, but enforces results to be consistent with the `LR` images, instead of just the `HR` images as is the common case. More information on CEM [here](https://github.com/victorca25/traiNNer/tree/master/codes/models/modules/architectures/CEM). Note that the rest of the explorable `SR` framework is TBD, but is available in the [**ESR repo**](https://github.com/YuvalBahat/Explorable-Super-Resolution/).

8. [**SRFlow**](https://arxiv.org/pdf/2006.14200.pdf) (2020). [Repo](https://github.com/andreas128/SRFlow). Aims at fixing one common pitfall of other frameworks, in that the results of the models are deterministic. SRFlow proposes using a normalizing flow (based on [GLOW](https://arxiv.org/pdf/1807.03039.pdf)) which allows the network to learn the conditional distribution of the output given the low-resolution input. It doesn't require the `GAN` formulation and can be trained using only the Negative Log Likelihood (`NLL`). In this repo, it has also been modified to use any of the regular losses on the deterministic version of the super-resolved image. Check  [how to train](https://github.com/victorca25/traiNNer/blob/master/docs/howtotrain.md#srflow-models) for more details.
<p align="center">
   <img height="220" src="https://user-images.githubusercontent.com/41912303/107157089-77b5cc00-6982-11eb-83f3-05773ff46610.png">
</p>

In addition, since they are based on `ESRGAN` and don't modify the general training strategy or the network architecture, but only the data used for training, [Real-SR](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf) (2020), [BSRGAN](https://arxiv.org/pdf/2103.14006v1.pdf) (2021) and [Real-ESRGAN](https://arxiv.org/pdf/2107.10833) (2021) are supported. `Real-SR` by means of the [realistic kernels](https://github.com/victorca25/traiNNer/blob/master/docs/kernels.md) and noise injection from image patches and `BSRGAN` and `Real-ESRGAN` through the on the fly augmentations pipeline. More information in the [augmentations](https://github.com/victorca25/traiNNer/blob/master/docs/augmentations.md) document. These strategies can be combined with **any** of the networks above.


## Image to image translation

1. [**pix2pix**](https://arxiv.org/pdf/1611.07004.pdf) (2017) Image-to-Image Translation with Conditional Adversarial Networks. Uses the conditional GANs formulation as a general-purpose solution to image-to-image translation problems when paired images are available, in a way that doesn't require hand-engineered mapping functions or losses. More information in [how to train](https://github.com/victorca25/traiNNer/blob/master/docs/howtotrain.md#image-to-image-translation), the [Pix2pix Pytorch repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and the [project page](https://phillipi.github.io/pix2pix/).

<p align="center">
   <img height="220" src="https://camo.githubusercontent.com/c10e6bc28b817a8741c2611e685eec2f6e2634587227699290dece8dd7e13d0c/68747470733a2f2f7068696c6c6970692e6769746875622e696f2f706978327069782f696d616765732f7465617365725f76332e706e67">
</p>

2. [***CycleGAN***](https://arxiv.org/pdf/1703.10593.pdf) (2017) Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. Different to previous approaches, `CycleGAN` was one of the first works to use an approach for learning to translate an image from a source domain A to a target domain B in the absence of paired examples. More information in [how to train](https://github.com/victorca25/traiNNer/blob/master/docs/howtotrain.md#image-to-image-translation), the [CycleGAN Pytorch repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and the [project page](https://junyanz.github.io/CycleGAN/).

<p align="center">
   <img height="220" src="https://camo.githubusercontent.com/16fa02525bf502bec1aac77a3eb5b96928b0f25d73f7d9dedcc041ba28c38751/68747470733a2f2f6a756e79616e7a2e6769746875622e696f2f4379636c6547414e2f696d616765732f7465617365725f686967685f7265732e6a7067">
</p>

3. [***WBC***](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Learning_to_Cartoonize_Using_White-Box_Cartoon_Representations_CVPR_2020_paper.pdf) (2020) Learning to Cartoonize Using White-box Cartoon Representations. Unlike the black-box strategies like `Pix2pix` and `CycleGAN` use, white-box cartoonization (`WBC`) is designed to use domain knowledge about how cartoons (anime) are made and decomposes the training task in image representations that correspond to the cartoon images workflow, each with different objectives. In general, the representations are: smooth surfaces (`surface`), sparse color blocks (`structure`) and contours and fine textures (`texture`). Like `CycleGAN`, it uses unpaired images and by tuning the scale of each representation, as well as the scale of the guided filter, different results can be obtained. More information in [how to train](https://github.com/victorca25/traiNNer/blob/master/docs/howtotrain.md#image-to-image-translation). You can build your own datasets, but for reference the ones used by `WBC` are:
    * landscape photos: the photos for the style transfer `CycleGAN` [dataset](https://github.com/victorca25/traiNNer/blob/master/docs/datasets.md#image-to-image-translation) (`6227`).
    * landscape cartoon: frames extracted and cropped from Miyazaki Hayao (`3617`), Hosoda Mamoru (`5107`) and Shinkai Makoto (`5891`) films.
    * face photos: [FFHQ](https://github.com/NVlabs/ffhq-dataset) photos (`#00000-10000`).
    * face cartoon: faces extracted from works by PA Works (`5000`) and Kyoto Animation (`5000`).

<p align="center">
   <img height="220" src="https://user-images.githubusercontent.com/41912303/126795194-17627f01-84dd-467c-9604-c02a3cc57585.png">
</p>


## Video

**Important**: Video network training can be considered fully functional, but experimental, with an overhaul to the pipeline pending for now (Help welcomed).

### Video Super-Resolution (VSR)

1. [**SOFVSR**](http://arxiv.org/pdf/2001.02129.pdf) (2020) Deep Video Super-Resolution using HR Optical Flow Estimation. Instead of the usual strategy of estimating optical flow for temporal consistency in the low-resolution domain, SOFVSR does so at the high-resolution level to prevent inconsistencies between low-resolution flows and high-resolution frames. This network has been modifified in this repo to also work with an `ESRGAN` network in the super-resolution step, as well as using 3 channel images as input, but requires more testing. More information in the [SOFVSR repo](https://github.com/LongguangWang/SOF-VSR/tree/master/TIP).

<p align="center">
   <img height="200" src="https://user-images.githubusercontent.com/41912303/121771007-1cccf500-cb6d-11eb-8cf5-b707648d1034.png">
</p>

2. ***EVSRGAN*** Video ESRGAN and ***SR3D*** networks, inspired by the paper [3DSRnet](https://arxiv.org/pdf/1812.09079.pdf): "Video Super-resolution using 3D Convolutional Neural Networks". `EVSRGAN` uses the regular `ESRGAN` network as backbone, but modifies it with 3D Convolutions to account for the time dimension, while `SR3D` more closely resembles the network proposed in `3DSRnet`. Require more testing.

<p align="center">
   <img height="200" src="https://user-images.githubusercontent.com/41912303/121771130-c8764500-cb6d-11eb-94b1-2e69965fa591.png">
</p>

3. [***EDVR***](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Wang_EDVR_Video_Restoration_With_Enhanced_Deformable_Convolutional_Networks_CVPRW_2019_paper.pdf) (2019): Video Restoration with Enhanced Deformable Convolutional Networks. Uses deformable convolutions to align frames at a feature level, instead of explicitly estimating optical flow. More information in [project page](https://xinntao.github.io/projects/EDVR).

<p align="center">
   <img height="200" src="https://user-images.githubusercontent.com/41912303/121771199-39b5f800-cb6e-11eb-97a1-081ed48a2224.png">
</p>


### Frame Interpolation (FI)

1. [***DVD***](https://arxiv.org/pdf/1708.00187.pdf) (2017) Real-time Deep Video Deinterlacing, implemented for the specific case of efficient video de-interlacing.

<p align="center">
   <img height="200" src="https://user-images.githubusercontent.com/41912303/121771179-168b4880-cb6e-11eb-8748-8ba6f4b4463b.png">
</p>

2. Initial integration of [**RIFE**](https://arxiv.org/pdf/2011.06294.pdf) (2020). Combining all 3 separate model files in a single structure. [RIFE repo](https://github.com/hzwer/arXiv2020-RIFE). (Training not yet available, pending for video pipeline overhaul).

<p align="center">
   <img height="200" src="https://user-images.githubusercontent.com/41912303/122081727-a3990080-cdff-11eb-987d-34ba541c7624.png">
</p>


### BibTex
    @misc{traiNNer,
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/victorca25/traiNNer}}
    }
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
    @inproceedings{bahat2020explorable,
        title={Explorable Super Resolution},
        author={Bahat, Yuval and Michaeli, Tomer},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={2716--2725},
        year={2020}
    }
    @inproceedings{lugmayr2020srflow,
        title={SRFlow: Learning the Super-Resolution Space with Normalizing Flow},
        author={Lugmayr, Andreas and Danelljan, Martin and Van Gool, Luc and Timofte, Radu},
        booktitle={ECCV},
        year={2020}
    }
    @inproceedings{zhang2021designing,
        title={Designing a Practical Degradation Model for Deep Blind Image Super-Resolution},
        author={Zhang, Kai and Liang, Jingyun and Van Gool, Luc and Timofte, Radu},
        booktitle={arxiv},
        year={2021}
    }
    @Article{wang2021realesrgan,
        title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
        author={Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
        journal={arXiv:2107.10833},
        year={2021}
    }
    @inproceedings{CycleGAN2017,
        title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
        author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
        booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
        year={2017}
    }
    @inproceedings{isola2017image,
        title={Image-to-Image Translation with Conditional Adversarial Networks},
        author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
        booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
        year={2017}
    }
    @InProceedings{Wang_2020_CVPR,
        author = {Wang, Xinrui and Yu, Jinze},
        title = {Learning to Cartoonize Using White-Box Cartoon Representations},
        booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2020}
    }
    @Article{Wang2020tip,
        author    = {Longguang Wang and Yulan Guo and Li Liu and Zaiping Lin and Xinpu Deng and Wei An},
        title     = {Deep Video Super-Resolution using {HR} Optical Flow Estimation},
        journal   = {{IEEE} Transactions on Image Processing},
        year      = {2020},
    }
    @InProceedings{wang2019edvr,
        author = {Wang, Xintao and Chan, Kelvin C.K. and Yu, Ke and Dong, Chao and Loy, Chen Change},
        title = {EDVR: Video Restoration with Enhanced Deformable Convolutional Networks},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        month = {June},
        year = {2019}
    }
    @article{zhu2017real,
        title={Real-time Deep Video Deinterlacing},
        author={Zhu, Haichao and Liu, Xueting and Mao, Xiangyu and Wong, Tien-Tsin},
        journal={arXiv preprint arXiv:1708.00187},
        year={2017}
    }
    @article{huang2020rife,
        title={RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
        author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
        journal={arXiv preprint arXiv:2011.06294},
        year={2020}
    }

