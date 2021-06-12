# Pretrained models

Following are the original trained models that authors made available for the different architectures, compiled here only for convenience. Also some experimental models that have been trained using this repository for use as pretrained models have been added.

Additional custom models can be found in the [model database](https://upscale.wiki/wiki/Model_Database).

In order to use these models, they can be downloaded and saved in any directory and the path should be added to the configuration file (usually `pretrain_model_G`). The default path to use is `experiments/pretrained_models`.

## Super Resolution

<table>
  <tr>
    <th>Name</th>
    <th>Models</th>
    <th>Short Description</th>
    <th>Source</th>
  </tr>

  <tr>
    <th rowspan="2">ESRGAN</th>
    <td>4x_RRDB_ESRGAN.pth, 4x_RRDB_ESRGAN_modarch.pth</td>
    <td><sub>the final ESRGAN model used in the paper and the modified architecture version from current repo</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/folders/1O9qLq2CjywbS4FJXvJMggX8DjX4e-GAV?usp=sharing">Google Drive</a></td>
  </tr>

  <tr>
    <td>4x_RRDB_PSNR.pth, 4x_RRDB_PSNR_modarch.pth</td>
    <td><sub>model with high PSNR performance and the modified architecture version from current repo</sub></td>
  </tr>

  <tr>
    <td >SRGAN<sup>*1</sup></td>
    <td>SRGAN_bicx4_303_505.pth</td>
     <td><sub> 4x SRGAN model (with modification), trained on DIV2K, w/o BN, bicubic downsampling.</sub></td>
    <td><a href="https://drive.google.com/drive/folders/1WR2X4_gwiQ9REb5fHfNnBfXOdeuDS8BA?usp=sharing">Google Drive</a></td>
  </tr>

  <tr>
    <td >SRResNet<sup>*2</sup></td>
    <td>**multiple, see notes below**, SRResNet_bicx4_in3nf64nb16.pth</td>
     <td><sub> 4x SRResNet model (with modification), trained on DIV2K, w/o BN, bicubic downsampling.</sub></td>
    <td><a href="https://drive.google.com/drive/folders/1Yrql9XI9goMAmsoQHHP8jG1k3RIAVhbK?usp=sharing">Google Drive</a></td>
  </tr>

  <tr>
    <td>PPON</td>
    <td>PPON.pth</td>
     <td><sub>PPON model presented in the paper</sub></td>
    <td><a href="https://drive.google.com/drive/folders/1uWz7Dc6-2zSpMmK9ks2mKAALnnCsjTCJ?usp=sharing">Google Drive</a></td>
  </tr>

  <tr>
    <td>PAN</td>
    <td>PAN.pth</td>
     <td><sub>4x pretrained modified PAN model with self-attention</sub></td>
    <td><a href="https://drive.google.com/drive/folders/1q7NRm_qTv3Kd4PzVOHkQHXw8tcwVTzHA?usp=sharing">Google Drive</a></td>
  </tr>

  <tr>
    <th rowspan="2">SRFLOW</th>
    <td>4x_srflow.pth</td>
     <td><sub>4x SRFlow model trained using a ESRGAN model from the original architecture as base (not fully trained)</sub></td>
    <td><a href="https://drive.google.com/drive/folders/1q7NRm_qTv3Kd4PzVOHkQHXw8tcwVTzHA?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>srflow_original.zip</td>
    <td><sub>the original SRFlow models at 4x and 8x scales (used the modified ESRGAN arch as base model)</sub></td>
  </tr>

  <tr>
    <th rowspan="4">SFTGAN</th>
    <td>segmentation_OST_bic.pth</td>
     <td><sub> segmentation model for bicubic downsampled images, outdoor scenes</sub></td>
    <td rowspan="4"><a href="https://drive.google.com/drive/folders/1pWP96xnGxnHOJ2A8tP9IJNhiNFZP6yZQ?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>sft_net_ini.pth</td>
    <td><sub>initialized SFTGAN model, initializing the SR generator with SRGAN_bicx4_303_505 parameters</sub></td>
  </tr>
  <tr>
    <td>sft_net_torch.pth</td>
    <td><sub>torch version of SFTGAN model (paper)</sub></td>
  </tr>
  <tr>
    <td>SFTGAN_bicx4_noBN_OST_bg.pth</td>
    <td><sub>PyTorch version of SFTGAN model: trained on OST dataset and use DIV2K as background images, w/o BN, bicubic downsampling</sub></td>
  </tr>

</table>



## Image to image translation

These initial models are the same as the ones in the original [`pix2pix and CycleGAN`](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repo and use the default UNet and ResNet networks configuration respectively (ie. using `ConvTranspose2d` for upsample layers).

<table>
  <tr>
    <th>Name</th>
    <th>Models</th>
    <th>Short Description</th>
    <th>Source</th>
  </tr>

  <tr>
    <th rowspan="6">pix2pix</th>
    <td>facades_label2photo.pth</td>
     <td><sub>models originally available in the pix2pix repo. These correspond to the Pix2pix <a href="https://github.com/victorca25/BasicSR/blob/master/docs/datasets.md#image-to-image-translation">datasets</a>.</sub></td>
    <td rowspan="6"><a href="https://drive.google.com/drive/folders/1gYeEOM1QIO-o3CG3aywtvKcCNmREf0bB?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>sat2map.pth</td>
  </tr>
  <tr>
    <td>map2sat.pth</td>
  </tr>
  <tr>
    <td>edges2shoes.pth</td>
  </tr>
  <tr>
    <td>edges2handbags.pth</td>
  </tr>
  <tr>
    <td>day2night.pth</td>
  </tr>

  <tr>
    <th rowspan="11">CycleGAN</th>
    <td>facades_label2photo.pth, facades_photo2label.pth</td>
     <td><sub>models originally available in the CycleGAN repo. These correspond to the CycleGAN <a href="https://github.com/victorca25/BasicSR/blob/master/docs/datasets.md#image-to-image-translation">datasets</a>. In some cases, only one generator of the cycle was provided, but the missing generator can be trained with the original dataset.</sub></td>
    <td rowspan="11"><a href="https://drive.google.com/drive/folders/1MTE_uQmTcHI5ieXo3-p2h9HftCLB03-t?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>sat2map.pth, map2sat.pth</td>
  </tr>
  <tr>
    <td>horse2zebra.pth, zebra2horse.pth</td>
  </tr>
  <tr>
    <td>summer2winter_yosemite.pth, winter2summer_yosemite.pth</td>
  </tr>
  <tr>
    <td>cityscapes_photo2label.pth, cityscapes_label2photo.pth</td>
  </tr>
  <tr>
    <td>apple2orange.pth, orange2apple.pth</td>
  </tr>
  <tr>
    <td>monet2photo.pth, photo2monet.pth</td>
  </tr>
  <tr>
    <td>photo2ukiyoe.pth</td>
  </tr>
  <tr>
    <td>photo2cezanne.pth</td>
  </tr>
  <tr>
    <td>photo2vangogh.pth</td>
  </tr>
  <tr>
    <td>iphone2dslr_flower.pth</td>
  </tr>

</table>

## Video (Experimental)

Following are some video models that have been trained using the networks available in this repository, many of them are experimental, but are useful to use as pretrained models for testing.

<table>
  <tr>
    <th>Name</th>
    <th>Models</th>
    <th>Short Description</th>
    <th>Source</th>
  </tr>

  <tr>
    <td>SOFVSR</td>
    <td>SOFVSR.pth</td>
     <td><sub>4x pretrained SOFVSR model, using 3 frames</sub></td>
    <td><a href="https://drive.google.com/drive/folders/1xDynI8v6s5oio4gKjks-bZlAYofCrGyz?usp=sharing">Google Drive</a></td>
  </tr>

  <tr>
    <td>SOFVESRGAN</td>
    <td>SOFVESRGAN.pth</td>
     <td><sub>4x pretrained modified SOFVSR model using ESRGAN network for super-resolution, using 3 frames</sub></td>
    <td><a href="https://drive.google.com/drive/folders/1BtL63OKyMlyam9BwE_nUiz0hr71MbpzE?usp=sharing">Google Drive</a></td>
  </tr>

  <tr>
    <td>EVSRGAN</td>
    <td>4x_EVSRGAN_REDS_pretrained.pth</td>
     <td><sub>4x EVSRGAN Pretrained using 3 frames and default arch options</sub></td>
    <td><a href="https://u.pcloud.link/publink/show?code=XZ2Wg8XZebryABNV8Q0GsSE2ifkLdh9NzzaX">pcloud</a></td>
  </tr>

  <tr>
    <td>DVD</td>
    <td>DVD_REDS-Deinterlace-*_G.pth</td>
     <td><sub>Real-time Deep Video Deinterlacing</sub></td>
    <td><a href="https://u.pcloud.link/publink/show?code=kZIIfQXZYLGBJF4sQVJ2aONxgwiPr8iQPxo7">pcloud</a></td>
  </tr>

  <tr>
    <td>RIFE</td>
    <td>RIFE.pth</td>
     <td><sub>Converted pretrained RIFE model from the three original pickle files into a single pth model</sub></td>
    <td><a href="https://drive.google.com/drive/folders/1-Wzwz8SmWkJm04Y32FuSKxH09n1hJDRz?usp=sharing">Google Drive</a></td>
  </tr>

</table>


* * *

# Below are some additional notes on some models:


## SRResNet (EDSR)

Through experiments it was found that using:

-   no batch normalization
-   residual block style: Conv-ReLU-Conv

are the best network settings for this network.

### Qualitative results [PSNR/dB] 

Besides the mentioned `SRResNet_bicx4_in3nf64nb16.pth`, other pretrained SRResNet models can be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1flWWpziKp5_i-8wbxyjUkP-e8EBtY37W?usp=sharing) folder, with where trained using different network configurations. Below is a table that compares the results of each configuration (note the name convention, also explained after the table, to identify each configuration):

| Model | Scale | Channel | DIV2K<sup>2</sup> | Set5| Set14 | BSD100 | Urban100 |
|--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SRResNet_bicx2_in3nf64nb16<sup>1</sup> | 2 | RGB | 34.720<sup>3</sup> | 35.835 | 31.643 | | |
|  |   |   | 36.143<sup>3</sup> | 37.947 | 33.682 | | |
| SRResNet_bicx3_in3nf64nb16 | 3 | RGB | 31.019 | 32.442  |  28.499 | | |
|  |   |   | 32.449 | 34.428  | 30.371  | | |
| SRResNet_bicx4_in3nf64nb16 | 4 | RGB | 29.051 | 30.278 | 26.853 | | |
|  |   |   | 30.486 | 32.180 | 28.645 | | |
| SRResNet_bicx8_in3nf64nb16 | 8 | RGB | 25.429 | 25.357 | 23.348 | | |
|  |   |   | 26.885 | 27.070 | 24.996 | | |
| SRResNet_bicx2_in1nf64nb16 | 2 | Y | 35.870 | 37.864 | 33.581 | | |
| SRResNet_bicx3_in1nf64nb16 | 3 | Y | 32.182 | 34.263 | 30.186 | | |
| SRResNet_bicx4_in1nf64nb16 | 4 | Y | 30.224 | 32.038<sup>4</sup> | 28.494 | | |
| SRResNet_bicx8_in1nf64nb16 | 8 | Y | 26.660 | 26.621 | 24.804 | | |

<sup>1</sup> **bic**: MATLAB bicubic downsampling; **in3**: input has 3 channels; **nf64**: 64 feature maps; **nb16**: 16 residual blocks.

<sup>2</sup> DIV2K 0801 ~ 0900 validation images.

<sup>3</sup> The first row is evaluated on RGB channels, while the secone row is evaluated on Y channel (of YCbCr).

<sup>4</sup> (31.901, 29.711)
