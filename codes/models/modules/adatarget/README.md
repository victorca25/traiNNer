# Adaptive Target Generation (AdaTarget)

In Super-Resolution (SR) tasks (blind or non-blind), many target HR patches can result in the same LR patch when downscaled, making it an ill-posed (under-determined) problem.

Limiting the solution of an SR network to just one from the given ground truth can penalize an acceptable output, even when they are mathematically valid candidates, according to the training framework. To tackle this issue [AdaTarget](https://openaccess.thecvf.com/content/CVPR2021/papers/Jo_Tackling_the_Ill-Posedness_of_Super-Resolution_Through_Adaptive_Target_Generation_CVPR_2021_paper.pdf), proposes an adaptive target strategy that relaxes the limitation on the possible solutions.

<p align="center">
   <img src="https://user-images.githubusercontent.com/41912303/130649558-ac2d0d64-3288-44f6-a4b1-c33d96593808.png" height="250">
</p>

This is achieved by transforming images in a way that they become spatially consistent, using affine transformations that deform the images in a patch-wise manner to be aligned with each other.

<p align="center">
   <img src="https://user-images.githubusercontent.com/41912303/130650108-5c745fd7-7607-42f2-9ab4-4cf8f69d8c5d.png" height="200">
</p>

While the proposal in AdaTarget is for the target HR images to be the ones modified, they obtained better results by instead using the pretrained network and deforming the outputs of the network with the inverse affine transformation, fully exploiting the original high resolution details of the target images without harming the results.

Note that this strategy is synergistic with the SR framework (using adversarial losses with GAN) and the [augmentations](https://github.com/victorca25/traiNNer/blob/master/docs/augmentations.md) options for real-world models in this repository and it can be enabled for any of the supported architectures.

In order to use AdaTarget, in your [options file](https://github.com/victorca25/traiNNer/blob/master/codes/options/sr/train_sr.yml), similar to how other options like AMP are enabled, add the `use_atg` variable to enable this feature:

```yaml
use_atg: true
```

And in addition to that, you need to define the path to the pretrained localization network, similar to how the pretrained generator (`pretrain_model_G`) and discriminator (`pretrain_model_D`) paths are defined:

```yaml
pretrain_model_Loc: '../experiments/pretrained_models/locnet.pth
```

This pretrained network can be downloaded from [here](https://drive.google.com/file/d/1ZIDGYO1sDQDBPY1dhbLT-PGPNzOtpr-2/view?usp=sharing).

Lastly, and similar to Stochastic Weight Averaging (SWA), AdaTarget will start training the localization network together with the generator network after training has progressed substantially with it being frozen.

In the original work, the generator network is trained with the frozen localization network for `100000` iterations, with a learning rate of `1e-4`, while the localization network only trains for the remaining `20000` iterations (for a total of `120000` iterations), with a learning rate of `1e-5`.

In other words, the localization network remains frozen for about `83%` of the training.

Assuming your total iterations (`niter`) match the original paper, you can either define, right before the losses to use are specified in the training options, a specific iteration when the localization network will start training like:

```yaml
    # For AdaTarget
    atg_start_iter: 100000
```

Or alternatively define the relative iteration out of the total iterations at which it will be enabled, with `83%` being:

```yaml
    # For AdaTarget
    atg_start_iter_rel: 0.83
```

A learning rate scheduler like `MultiStepLR` can be used to automatically match the decrease in the learning rate used when the localization network starts training.

Another consideration to take into account is that the ATG divides output images into patches of size 7x7, so in order to prevent border issues, the `crop_size` should be a multiple of `7`. The original paper used `crop_size: 140`, but values like `112` and `224` that are both 7 times a power of 2 also work well.

## Citation

```
@InProceedings{jo2021adatarget,
    author = {Jo, Younghyun and Oh, Seoung Wug and Vajda, Peter and Kim, Seon Joo},
    title = {Tackling the Ill-Posedness of Super-Resolution through Adaptive Target Generation},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2021}
}
```
