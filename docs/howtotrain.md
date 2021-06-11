# How to train

Here you will find detailed information on how to train different kinds of models:

1.  [Super-Resolution](#super-resolution)
2.  [Restoration](#restoration)
3.  [Image to image translation](#image-to-image-translation)
4.  [Video](#video)
5.  [Resuming training](#resuming-training)


# Super-Resolution

## Normal Single-Image Super-Resolution (ESRGAN, SRGAN, PAN, etc) models

In some cases, like SRGAN and ESRGAN, the recommendation is to use a PSNR-oriented pretrained SR model to initialize the parameters for better quality. According to the SRGAN author's paper and some testing, this will also stabilize the GAN training and allows for faster convergence, but it may not be necessary in all cases. As an example with ESRGAN, these could be the steps to follow:

1. Prepare datasets, usually the DIV2K dataset. More details are in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data).
2. Optional: If the intention is to replicate the original paper here you would prepare the PSNR-oriented pretrained model. You can also use the original `RRDB_PSNR_x4.pth` as the pretrained model for that purpose, otherwise *any* existing model will work as pretrained.
3. Modify one of the configuration template file, for example `options/train/train_template.yml` or `options/train/train_template.json`
4. Run command: `python train.py -opt options/train/train_template.yml` or `python train.py -opt options/train/train_template.json`

## Single-Image Super-Resolution PPON models

Note that while you can train PPON using the regular train.py file and the same steps as other SR models, these additional options have to be set in the training options file (using example values):

Select ppon model type:
```
    model: ppon
```


Set the ppon Generator network:
```
    which_model_G: ppon
    mode: CNA
    nf: 64
    nb: 24
    in_nc: 3
    out_nc: 3
    group: 1
```

You need to configure the losses (type, weights, etc) as you would normally first:
```    
    pixel_criterion: l1
    pixel_weight: 1
    feature_criterion: l1
    feature_weigh": 1
    ssim_type: ms-ssim
    ssim_weight: 1e-2
    ms_criterion: multiscale-l1
    ms_weight: 1e-2
    gan_type: vanilla
    gan_weight: 8e-3
```

And then pick which of the configured losses will be used for each stage (the names used are matched out of the names as they are logged during training, so `pixel_criterion` corresponds to `pix`, `feature_criterion` to `fea` and `cx_type: contextual` to `contextual`, for example):
```
    p1_losses: [pix] # from the paper: l1 pixel_weigh: 1
    p2_losses: [pix-multiscale, ms-ssim] # from the paper: multiscale_weight: 1, ms-ssim_weight: 1
    p3_losses: [fea] # from the paper: VGG feature_weight: 1 gan_weight: 0.005
    ppon_stages: [1000, 2000] # The first value here is where phase 2 (structure) will start and the second is where phase 3 (features) starts
```

The same losses can be used in multiple stages (it can be repeated) and take into consideration that the first stage is the one with the most capacity of the network and the other two stages depend on it.

The Discriminator is enabled only on the last phase at the moment, following the paper. You can configure any of the losses in any of the phases, I recommend testing "contextual" (cx) if possible, specially on phases 2 and 3.

You may also want to adjust your scheduler and coordinate the ppon_stages to match the steps, the original paper used "StepLR_Restart".

Lastly, you can control what phase you want to train with the ppon_stages option. For example, if you set it to [0, 0] it will start on phase 3 from the beginning of the training session, while [0, 1000000] will start in phase 2, and phase 3 will begin after 1000000 iterations. Similarly, with [1000000, 1000000], only phase 1 will be trained for 1000000 iterations.

## SRFlow models

SRFlow allows for the use of any differentiable architecture for the LR encoding network, since ir itself does not need to be invertible. SRFlow uses by default an RRDB network (ESRGAN) network for this purpose. In the original work, a pretrained ESRGAN model is loaded and according to the paper, the remaining flow network is trained for half the training time and the RRDB module is only unfrozen after that period. The option "train_RRDB_delay: 0.5" does that automatically, but you can lower it to start earlier if required. Besides these main differences, the training process is similar to other SR networks.

1. Prepare datasets, usually the DIV2K dataset. More details are in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data).
2. Optional: If the intention is to replicate the original paper here you would use an ESRGAN pretrained model. The original paper used the ESRGAN modified architecture model for this purpose. You can also use the original `RRDB_PSNR_x4.pth` as the pretrained model for that purpose, otherwise *any* existing model will work as pretrained. In `options/train/train_srflow.yml` set path.pretrain_model_G: `RRDB_ESRGAN_x4_mod_arch.pth` (or any ESRGAN model) and path.load_submodule: `true` for this purpose. If using an SRFlow model as pretrained, only setting pretrain_model_G is required.
3. Modify the configuration file, `options/train/train_srflow.yml` as needed.
4. Run command: `python train_srflow.py -opt options/train/train_srflow.yml`

Notes:
- While SRFlow only needs the nll to train, it is possible to add any of the losses (except GAN) from the regular training template for training and they will work. They will operate on the deterministic version of the super resolved image with temperature τ= 0. 
- SRFlow is more memory intensive than ESRGAN, specially if using the regular losses that need to calculate reconstructed SR from the latent space `z` (with `reverse=True`)
- To remain stable, SRFlow needs a large batch size. batch=1 produces NaN results. If real batch sizes>1 are not possible on the hardware, using virtual batch can solve this stability issue.
- During validation and inference it's known that reconstructed images will output NaN values, which are reduced with more training. More details are discussed [here](https://github.com/andreas128/SRFlow/issues/2)
- During validation, as many images as set in the `heats: [ 0.0, 0.5, 0.75, 1.0 ]` times `n_sample: 3` will be generated. This example means 3 random samples from each of the heat values configured there, 12 images in total for each validation image.

## SFTGAN models
Note: these are the instructions from the original repository and the whole process is in need on behing updated, but it should work if you want to experiment.

*Pretraining is also important*. Use a PSNR-oriented pretrained SR model (trained on DIV2K) to initialize the SFTGAN model.

1. First prepare the segmentation probability maps for training data: run [`test_seg.py`](https://github.com/victorca25/BasicSR/blob/master/codes/test_seg.py). We provide a pretrained segmentation model for 7 outdoor categories in [Pretrained models](#pretrained-models). Use [Xiaoxiao Li's codes](https://github.com/lxx1991/caffe_mpi) to train the segmentation model and transfer it to a PyTorch model.
2. Put the images and segmentation probability maps in a folder as described in [`codes/data`](https://github.com/victorca25/BasicSR/tree/master/codes/data).
3. Transfer the pretrained model parameters to the SFTGAN model. 
    1. First train with `debug` mode and obtain a saved model.
    2. Run [`transfer_params_sft.py`](https://github.com/victorca25/BasicSR/blob/master/codes/scripts/transfer_params_sft.py) to initialize the model.
    3. We provide an initialized model named `sft_net_ini.pth` in [Pretrained models](#pretrained-models)
4. Modify the configuration file in `options/train/train_sftgan.json`
5. Run command: `python train.py -opt options/train/train_sftgan.json`


# Restoration

Restoration models (Deblurring, denoising, etc) are fundamentally the same as [Super Resolution](super-resolution) models, with the difference that they usually operate without scaling the images (1x scale). The steps to train are the same as in super-resolution models, only make sure that the network supports operating in 1x scale.

Super-Resolution and restoration are tasks that can be done simultaneously, in which case the low quality input data will be both a factor of the size of the high quality target, but also contains one or more degradations.

# Image to image translation

## Preprocessing
Images can be resized and cropped in different ways using `preprocess` option. 
- The default option `resize_and_crop` resizes the image to be of size (`load_size`, `load_size`) and does a random crop of size (`crop_size`, `crop_size`). 
- `crop` skips the resizing step and only performs random cropping, in the same way as SR cases.
- `center_crop` will always do the same center crop of size (`center_crop_size`, `center_crop_size`) to all images.
- `scale_width` resizes the image to have width `crop_size` while keeping the aspect ratio.
- `scale_width_and_crop` first resizes the image to have width `load_size` and then does random cropping of size (`crop_size`, `crop_size`).
- `none` tries to skip all these preprocessing steps. However, if the image size is not a multiple of some number depending on the number of downsamplings of the generator, you will get an error because the size of the output image may be different from the size of the input image. Therefore, `none` option still tries to adjust the image size to be a multiple of 4. You might need a bigger adjustment if you change the generator architecture. Please see `dataops/augmentations.py` to see how all these were implemented.
- **Note**: Options can be concatenated usin `_and_` like `center_crop_and_resize` or `resize_and_crop`.

## About image size

Since the generator architecture in CycleGAN involves a series of downsampling / upsampling operations, the size of the input and output image may not match if the input image size is not a multiple of 4. As a result, you may get a runtime error because the L1 identity loss cannot be enforced with images of different size. Therefore, we slightly resize the image to become multiples of 4 even with `preprocess: none` option, as explained above. For the same reason, `crop_size` needs to be a multiple of 4.

## Training/Testing with high res images

CycleGAN is quite memory-intensive as four networks (two generators and two discriminators) need to be loaded on one GPU, so a large image cannot be entirely loaded. In this case, we recommend training with cropped images. For example, to generate 1024px results, you can train with `preprocess: scale_width_and_crop`, `load_size: 1024` `crop_size: 360`, and test with `preprocess: scale_width`, `load_size: 1024`. This way makes sure the training and test will be at the same scale. At test time, you can afford higher resolution because you don’t need to load all the networks.

## Training/Testing with rectangular images

Both pix2pix and CycleGAN can work for rectangular images. To make them work, you need to use different preprocessing flags. Let's say that you are working with 360x256 images. During training, you can specify `preprocess: crop` and `crop_size: 256`. This will allow your model to be trained on randomly cropped `256x256` images during training time. During test time, you can apply the model on `360x256` images with the flag `preprocess: none`.

There are practical restrictions regarding image sizes for each generator architecture. For `unet256`, it only supports images whose width and height are divisible by **`256`**. For `unet128`, the width and height need to be divisible by **`128`**. For `resnet_6blocks` and `resnet_9blocks`, the width and height need to be divisible by **`4`**.

## About batch size

For all experiments in the original pix2pix and CycleGAN papers, the batch size was set to be 1. If there is room for memory, you can use higher batch size with batch norm or instance norm. (Note that the default `batchnorm` does not work well with multi-GPU training. You may consider using [synchronized batchnorm](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) instead). But please be aware that it can impact the training. In particular, even with Instance Normalization, different batch sizes can lead to different results. Moreover, increasing `crop_size` may be a good alternative to increasing the `batch_size`.

## Identity loss (CycleGAN)

The identity loss can regularize the generator to be close to an identity mapping when fed with real samples from the target domain. If something already looks like from the target domain, you should preserve the image without making additional changes. The generator trained with this loss will often be more conservative for unknown content, meaning that if you want to allow more liberty to the model to be able to change the images, the identity loss should be disabled.

## Using resize-conv to reduce checkerboard artifacts

This Distill [blog](https://distill.pub/2016/deconv-checkerboard/) discussed one of the potential causes of the checkerboard artifacts. You can fix that issue by switching from `deconv` ("deconvolution") to an `upconv` (regular upsampling followed by regular convolution). Currently the network parameters for both pix2pix and CycleGAN allow to make this change (using the `upsample_mode` option), but here's an alternative reference implementation using `ReflectionPad2d`:

```nn.Upsample(scale_factor = 2, mode='bilinear'),
nn.ReflectionPad2d(1),
nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0),
```

Sometimes the checkboard artifacts will go away if you train long enough. Maybe you can try training your model a bit longer.



# Video


## Video Super-Resolution (VSR) models
TBD


### Video Frame Interpolation (RIFE) models
TBD


# Resuming Training 

When resuming training, just set the `resume_state` option in the configuration file under `path`, like: <small>`resume_state: "../experiments/debug_001_RRDB_PSNR_x4_DIV2K/training_state/200.state"`. </small>

# Fine-tuning

To fine-tune a pre-trained model, just set the path in `pretrain_model_G` to your pretrained generator model (`pretrain_model_G_A` and `pretrain_model_G_B` in the case of CycleGAN) and it will train with your current configuration. The program will initialize the training from iteration 0.

You can also use pretrained Discriminator networks with the corresponding `pretrain_model_D`, which is particularly useful in cases where you would like to evaluate transfer learning, either by using the Discriminator as is or by freezing layers using the `FreezeD` option. Also, can be useful when combined with `feature matching` to use the discriminator feature maps to calculate feature loss.

