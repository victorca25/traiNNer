# Presets files

These files are used to define the augmentations pipeline and configuration for each augmentation to be used during training.

Below are more details regarding the presets and configuring the augmentation pipeline.

1. [General](#general)
2. [Resizing stages details](#resizing)
3. [Augmentations types](#augmentations-types)
4. [Overriding](#overriding)

## General

Each set of presets has three elements, corresponding to the mathematical model of low resolution images **`'y'`** in relation to the high resolution images **`'x'`**:

```y = (x ⊗ k) ↓s + n```

Where **`'k'`** is a blurring kernel that is convolved with the ground truth image **`'x'`**, before being downsampled with a scaling algorithm **`'s'`** and finally noise **`'n'`** is applied to complete the degradation.

The presets are applied in overlays, where the `base` presets are the default configurations, which mainly contain the default augmentations configurations and typically don't need to be changed. Additional presets are then used to define the pipeline and custom configurations and finally, the main training options [file](https://github.com/victorca25/traiNNer/blob/master/codes/options/) can be used to both select the proper presets as well as easily overriding any value if needed.

Four sample presets configurations are included:
- [Real-SR](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf) (`realsr`): uses realistic kernels for downscaling (pre-pipeline) and real images patches to inject noise. Note that these have to be extracted offline beforehand by following the instructions in [DLIP](https://github.com/victorca25/DLIP/) and the paths to the kernels and image patches must be provided in `dataroot_kernels` and `noise_data`.
- [BSRGAN](https://arxiv.org/pdf/2103.14006v1.pdf) (`bsrgan`): which notably applies two blur operations (`iso` and `aniso`), two noise operations (`gaussian` and `camera` noise) and random in-pipeline scaling. These augmentations are shuffled and followed by `jpeg` compression.
- [Real-ESRGAN](https://arxiv.org/pdf/2107.10833.pdf) (`resrgan`): very similar to BSRGAN, but adds `sinc` filter to the two blur operations, replaces the realistic `camera` noise for a simpler `poisson` noise augmentation and adds a second in-pipeline scaling operation. Instead of randomly shuffling the degradations, repeats the pipeline twice in the original form (blur -> scaling -> noise), with a `jpeg` compression between each and finishing with a random order of an additional `sinc` filter or `scaling`+`jpeg`. Note that additionally, the paper presents an optional use of an `unsharp` filter applied to `HR` images to increase sharpness of the result, which is a strategy that was already demostrated to work in this repository a couple of years back, and can be enabled by uncommenting the two lines in the `resrgan_noise.yaml` file.
- Combination (`combo`): is an example preset that combines the previous three. Note that, unless disabled, it also requires the `dataroot_kernels` and `noise_data` to be provided.

There are many more augmentations available than those shown in the sample presets and can be used to better match the desired outcome of the model in training. For more details, refer to the three base preset files that contain the default configuration for all augmentations.

In order to use the presets, it is only required to use the `augs_strategy` variable in the in the `train` dataset in the options file with the name of the presets, for example:

```yaml
augs_strategy: bsrgan
```

Alternativelly, each of the blur, resize and noise presets can be defined individually, like:

```yaml
add_blur_preset: bsrgan_blur
add_resize_preset: bsrgan_resize
add_noise_preset: bsrgan_noise
```

Which allows to do custom mix and matching of the presets. For example, if using the `bsrgan` strategy, but want to instead use the `resrgan` resizing options, this can be achieved with:

```yaml
augs_strategy: bsrgan
add_resize_preset: resrgan_resize
```

Or using three different presets, including custom ones, can be done with:

```yaml
add_blur_preset: combo_blur
add_resize_preset: custom1_resize
add_noise_preset: custom2_noise
```

These names correspond directly to the presets `yaml` files names. For example, `add_blur_preset: combo_blur` will look for `./presets/combo_blur.yaml`. Similarly, `augs_strategy: bsrgan` will look for the three presets that start with `bsrgan_*` in the presets folder.

`add_blur_preset`, `add_resize_preset` and `add_noise_preset` can also work with full paths.

If you need to use custom `base` configurations, you can similarly override the default base with:

```yaml
base_blur_preset: base2_blur
base_resize_preset: base2_resize
base_noise_preset: base2_noise
```

[Back to index](#presets-files)

## Resizing

There are multiple things to consider regarding the pipeline image resizing.

One of this is controlled with the variable `resize_strat`, which typically is either `pre` (for pre-pipeline image scaling) or `in` (for in-pipeline image scaling). Depending on your dataset and each case, one or the other can be used. The presets come with the configuration used in each case, but this can be overriden in the main options file.

The code will automatically deal with images in cases such as:
- only the `HR`/`GT` image is provided: will generate the `LR`/`LQ` pair on the fly.
- the `LR`/`LQ` image has the same scale as `HR`/`GT`: will scale the image either pre-pipeline or in-pipeline, according to the configuration.
- the `LR`/`LQ` image already has the final desired scale: if using pre-pipeline image scaling, images will be used unchanged, while if using in-pipeline scaling, an adaptive scale will be selected such that the final sizes remain inside the configured minimum and maximum and doesn't produce excessively small images.
- if `LR`/`LQ` images are provided in any other scale that the same as `HR`/`GT` or the correct target scale, they will be resized to the target scale before the pipeline and automatically use the adaptive scale.

It is possible to provide only some `LR`/`LQ` images, in which case they will be used, and if not provided, they will be created from the `HR`/`GT` and augmented on-the-fly by the pipeline, following the considerations above for both cases.

On the other hand, if a portion or all `LR`/`LQ` images are provided, `aug_downscale` can be used in a scale from `0.0` to `1.0` to define the probability that the provided `LR`/`LQ` images are ignored and instead created on-the-fly from `HR`/`GT`. This allows to benefit from both the provided images (that can be generated using any degradation process) and the degradations pipeline.

If set to `true`, the `pre_crop` option can be convenient to crop the image pairs to the `crop_size` before entering the paired and unpaired augmentations, which may help accelerate processing times.

[Back to index](#presets-files)

## Augmentations types

Multiple augmentation options need a secondary variable where `types` of the augmentation are defined. For example, `lr_blur_types` can be `gaussian`, `iso` (isotropic), `aniso` (anisotropic) and others.

These types options can be defined either as lists like: [`gaussian`, `poisson`, `camera`, `patches`] (noise types) and represent a uniform distribution (same probability for each type), meaning that the probability of any of them being applied is the same (25% in this example).

Alternatively, dictionaries can be used like: {`sinc`: `0.1`, `iso`: `0.58`, `aniso`: `0.32`} (blur types) in which case the numbers represent the probabilities for each case to be applied. In this example the probabilities sum up to 1, but it doesn't have to be the case. Another option is to use something like: {`gaussian`: `1`, `jpeg`: `1`, `clean`: `4`} where the probability also increases with the values assigned to each type, and in this case, `clean` (no augmentation) will happen 4 out of every 6 (1+1+4) times.

[Back to index](#presets-files)

## Overriding

The main options file can be used to override any configuration in the base presets or the additional presets. This maintains the original behavior of the options file from before the presets strategy was introduced.

The means to override the configurations is to simply add the parameters in the `train` dataset.

For example, any of the below options can be uncommented (if needed) and added on the options file.

```yaml
datasets:  # configure the datasets
  train:  # the stage the dataset will be used for (training)
    # ...
    # ...
    # ...
    # ...
    # ...

    # Color space conversion: 'color' for both LR and HR, 'color_LR' for LR independently, 'color_HR' for HR independently. Default: no conversion (RGB), Options: 'y' for Y in YCbCr | 'gray' to convert RGB to grayscale | 'RGB' to convert gray to RGB
    # color: y
    # color_LR: y
    # color_HR: y
    
    # LR and HR modifiers. Random flip LR and HR or ignore provided LRs and generate new ones on the fly with defined probability:
    # rand_flip_LR_HR: false  # true # flip LR and HR during training.
    # flip_chance: 0.05  # Example: 0.05 = 5% chance of LR and HR flipping during training.
    aug_downscale: 0.2  # Example: 0.6 = 60% chance of generating LR on the fly, even if LR dataset is provided.

    # Configure random downscaling of HR target image (will match LR input to correct size)
    hr_downscale: true
    hr_downscale_amt: [2, 1.75, 1.5, 1]  # the random scales to downscale to
    pre_crop: true  # enable to crop the images before scaling for speed improvement (relevant when using hr_downscale or generating LRs on the fly)

    # Fix LR size if it doesn't match the scale of HR. Options: `reshape_lr` to modify only LR to HR/scale or reshape_hr to modify both LR and HR in respect to each other.
    shape_change: reshape_lr
    
    # Configure on the fly generation of LR: (else, it will automatically default to Matlab-like antialiased downscale algorithm when/if required.
    # The scaling options are: cv2_nearest, cv2_linear, cv2_area, cv2_cubic, cv2_lanczos4, cv2_linear_exact, linear, box , lanczos2, lanczos3, bicubic, mitchell, hermite, lanczos4, lanczos5, bell, catrom, hanning, hamming, gaussian, sinc2, sinc3, sinc4, sinc5, blackman2, blackman3, blackman4, blackman5, nearest_aligned, down_up, realistic
    resize_strat: pre  # in | pre
    lr_downscale: true  # true | false
    lr_downscale_types: [linear, bicubic]  # scaling interpolation options.
    lr_downscale2: false
    lr_downscale_types2: [linear, bicubic]
    dataroot_kernels: '../training/kernels/results/'  # location of the image kernels extracted with KernelGAN, for use with the `realistic` downscale type below
    # realk_scale: 4  # if using a specific KernelGAN extracted kernel scale that differs from the model scale
    down_up_types: [linear, bicubic, mitchell]
    final_scale: true
    final_scale_types: [area, linear, bicubic]

    # Noise and blur augmentations:
    # In both cases, the options will consist of a dictionary or a list, for example: {gaussian: 1, clean: 3} where the probability of an option being applied depends on the number set. In this example, there 1/4 (25%) chance of `gaussian` being applied, while `clean` will happen 3/4 (75%) of the time. [gaussian, clean] will be interpreted as 50% chance of any of the two options being used.
    
    # The blur options are: "iso", "aniso", "sinc", "average", "box", "gaussian", "bilateral", "median", "motion", "complexmotion" or "clean"
    lr_blur: false # true | false
    lr_blur_types: {gaussian: 1, clean: 3}
    blur_prob: 1.0
    lr_blur2: true
    lr_blur_types2: {sinc: 0.1, iso: 0.58, aniso: 0.32}
    blur_prob2: 0.8
    final_blur: [sinc]
    final_blur_prob: 0.8
    
    # The noise options are: "gaussian", "poisson", "dither", "s&p", "speckle", "jpeg", "webp", "quantize", "km_quantize", "simplequantize", "clahe", "patches", "camera" or "clean"
    noise_data: ../noise_patches/normal/ # location of the noise patches extracted from real images to use for noise injection with noise option "patches"
    lr_noise: false # true | false
    lr_noise_types: {gaussian: 1, jpeg: 1, clean: 4}
    lr_noise2: false # true | false
    lr_noise_types2: {dither: 2, clean: 2}
    hr_noise: false # true | false
    hr_noise_types:  {gaussian: 1, clean: 4}

    # Compression augmentations
    compression: [jpeg]
    final_compression: [jpeg]
    
    # Color augmentations
    lr_fringes: true # true | false
    lr_fringes_chance: 0.4
    lr_auto_levels: true # add auto levels to LR images to expand dynamic range.
    lr_rand_auto_levels: 0.7 # Example: 0.4 = 40% chance of adding auto levels to images on the fly
    hr_auto_levels: true # add auto levels to HR images to expand dynamic range.
    hr_rand_auto_levels: 0.7 # Example: 0.4 = 40% chance of adding auto levels to images on the fly
    lr_unsharp_mask: true # add a unsharpening mask to LR images.
    lr_rand_unsharp: 1 # Example: 0.5 = 50% chance of adding unsharpening mask to LR images on the fly
    hr_unsharp_mask: true # add a unsharpening mask to HR images. Can work well together with the HFEN loss function.
    hr_rand_unsharp: 1 # Example: 0.5 = 50% chance of adding unsharpening mask to HR images on the fly
    
    # Augmentations for classification or (maybe) inpainting networks:
    lr_cutout: false # true | false
    lr_erasing: false # true | false

    # shuffle degradations option
    shuffle_degradations: false

```

Any option in the `pipeline` section of any of the presets can also be added the `train` dataset and override the preset.

[Back to index](#presets-files)

