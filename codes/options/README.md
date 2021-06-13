# Options configuration
-   These files are used to pass all the configuration needed to train or test models.
-   You can Use **yaml** or **json** files to configure options.
-   The code converts the the option file to a python dictionary in `options.py`, which is used later in all modules.
-   In `yaml` files `#` are used for comments, while in `json` files it's with  `//`. Use `null` for `None` in both cases.
-   The recommendation is to find the template file for the task (network,strategy, etc) you're interested in and create a copy you can edit to customize.
-   The main change to get up and running, for testing is to add the path to the generator model in `pretrain_model_G`. For training, it is also usually recommended to use a pretrain model in `pretrain_model_G`, but you need to provide the dataset to train on. `dataroot_B` (or `dataroot_HR`, either name can be used) for the target images and if not generating the input pair on the fly, also the `dataroot_A` (or `dataroot_LR`, either name can be used) path.

# Common

All configuration files have parts that will be the same and are described here (using `yaml` as an example). Note that every commented line represents optional configuration options that can be added as needed.

Click in each section below for detailed explanations for each one.

1. [General](#general)
2. [Dataset options](#dataset-options) (also for training srgan)
3. [Network options](#network-options)
4. [Training strategy](#training-strategy)
5. [Logger and checkpoint configuration](#logger-and-checkpoint-configuration)

## General

First, there's a general section of options used to configure the experiment training session. The following examples will be using the default `ESRGAN` configuration, but each template will contain the default configuration recommended as a starting point for each case.

```yaml
name: 001_template  # the name that defines the experiment and the directory that will be created in the experiments directory.
# name: debug_001_template  # use the "debug" or "debug_nochkp" prefix in the name to run a test session and check everything is working. Does validation and state saving every 8 iterations. Remove "debug" to run the real training session.
use_tb_logger: true  # wheter to enable Tensorboard logging or not. Output will be saved in: https://github.com/victorca25/BasicSR/tree/master/tb_logger
model: srragan  # the model training strategy to be used. Depends on the type of model, from: https://github.com/victorca25/BasicSR/tree/master/codes/models
scale: 4  # the scale factor that will be used for training for super-resolution cases. Default is "1".
gpu_ids: [0]  # the list of `CUDA_VISIBLE_DEVICES` that will be used during training, ie. for two GPUs, use [0, 1]. The batch size should be a multiple of the number of 'gpu_ids', since images will be distributed from the batch to each GPU.
use_amp: true  # select to use PyTorch's Automatic Mixed Precision package to train in low-precision FP16 mode (lowers VRAM requirements).
# use_swa: false  # select to use Stochastic Weight Averaging
# use_cem: false  # select to use CEM during training. https://github.com/victorca25/BasicSR/tree/master/codes/models/modules/architectures/CEM
```

[Back to index](#common)

## Dataset options:

Here the options to configure the datasets to use are selected. Options will vary between super-resolution, video, image to image translation and other cases, following here with the example of SR and on the fly augmentations.

For training a `train` dataset is required:

```yaml
datasets:  # configure the datasets
  train:  # the stage the dataset will be used for (training)
    name: DIV2K  # the name of your dataset (only informative)
    mode: aligned  # dataset mode: https://github.com/victorca25/BasicSR/tree/master/codes/data
    # here you provide the paths to the input (A) and target (B) datasets.
    # it can be a list of directories as follows:
    dataroot_B: [
      '../datasets/train/hr1',
      '../datasets/train/hr2',
      '../datasets/train/hr3'
      ] # high resolution / ground truth images
    dataroot_A: [
      '../datasets/train/lr1',
      '../datasets/train/lr2' #,
      # '../datasets/train/lr3'
      ] # low resolution images. If there are missing LR images, they will be generated on the fly from HR
    # you can also use repeated HR/B images for multiple LR/As:
    # dataroot_B: [
    #   '../datasets/train/hr1',
    #   '../datasets/train/hr1',
    #   '../datasets/train/hr2',
    #   '../datasets/train/hr2'
    #   ]
    # dataroot_A: [
    #   '../datasets/train/lr1a',
    #   '../datasets/train/lr1b',
    #   '../datasets/train/lr2a',
    #   '../datasets/train/lr2b'
    #   ]
    # they can be a single directory for each like:
    # dataroot_B: '../datasets/train/hr'  # target
    # dataroot_A: '../datasets/train/lr'  # input
    # or they can also be `lmdb` databases like:
    # dataroot_B: '../datasets/train/hr.lmdb'  # target
    # dataroot_A: '../datasets/train/lr.lmdb'  # input
    
    subset_file: null  # to use a subset of an image folder
    use_shuffle: true  # shuffle the dataset
    # znorm: false  # true | false: Normalizes images in [-1, 1] range if true. Default = None (range [0,1]).
    n_workers: 4  # number of PyTorch data load workers. Use 0 to disable CPU multithreading, or an integrer representing CPU threads to use for dataloading.
    batch_size: 8  # the training minibatch size.
    virtual_batch_size: 8  # if needed, a virtual minibatch can also be used, in case VRAM is not enough for larger `batch_size`.
    preprocess: crop  # how to process the images after loading
    crop_size: 128  # target image patch size. Default: 128. (Needs to be coordinated with the patch size of the networks)
    image_channels: 3 # number of channels to load images in

    # Color space conversion: 'color' for both LR and HR, 'color_LR' for LR independently, 'color_HR' for HR independently. Default: no conversion (RGB), Options: 'y' for Y in YCbCr | 'gray' to convert RGB to grayscale | 'RGB' to convert gray to RGB
    # color: 'y'
    # color_LR: 'y'
    # color_HR: 'y'
    
    # LR and HR modifiers. Random flip LR and HR or ignore provided LRs and generate new ones on the fly with defined probability:
    # rand_flip_LR_HR: false  # true # flip LR and HR during training.
    # flip_chance: 0.05  # Example: 0.05 = 5% chance of LR and HR flipping during training.
    # aug_downscale: 0.2  # Example: 0.6 = 60% chance of generating LR on the fly, even if LR dataset is provided.

    # Configure random downscaling of HR target image (will match LR input to correct size)
    # hr_downscale: true
    # hr_downscale_amt: [2, 1.75, 1.5, 1]  # the random scales to downscale to
    # #pre_crop: true  # enable to crop the images before scaling for speed improvement (relevant when using hr_downscale or generating LRs on the fly)

    # Fix LR size if it doesn't match the scale of HR. Options: `reshape_lr` to modify only LR to HR/scale or reshape_hr to modify both LR and HR in respect to each other.
    # shape_change: reshape_lr
    
    # Configure on the fly generation of LR: (else, it will automatically default to Matlab-like antialiased downscale algorithm when/if required
    lr_downscale: true  # true | false
    # dataroot_kernels: '../training/kernels/results/'  # location of the image kernels extracted with KernelGAN, for use with the `realistic` downscale type below
    lr_downscale_types: ["linear", "bicubic"]  # scaling interpolation options. select from ['cv2_nearest', 'cv2_linear', 'cv2_area', 'cv2_cubic', 'cv2_lanczos4', 'cv2_linear_exact', 'linear', 'box' ,'lanczos2', 'lanczos3', 'bicubic', 'mitchell', 'hermite', 'lanczos4', 'lanczos5', 'bell', 'catrom', 'hanning', 'hamming', 'gaussian', 'sinc2', 'sinc3', 'sinc4', 'sinc5', 'blackman2', 'blackman3', 'blackman4', 'blackman5', 'realistic']

    # Rotations augmentations:
    use_flip: true  # whether use horizontal and vertical flips
    use_rot: true  # whether use rotations: 90, 190, 270 degrees
    use_hrrot: false # rotate images in free-range random degress between -45 and 45
    
    # Noise and blur augmentations:
    # In both cases, the options will consist of a dictionary, for example: {gaussian: 1, clean: 3} where the probability of an option being applied depends on the number set. In this example, there 1/4 (25%) chance of `gaussian` being applied, while `clean` will happen 3/4 (75%) of the time.
    
    # The blur options are: "average", "box", "gaussian", "bilateral", "median", "motion", "complexmotion" or "clean"
    # lr_blur: false # true | false
    # lr_blur_types: {gaussian: 1, clean: 3}
    
    # The noise options are: "gaussian", "poisson", "dither", "s&p", "speckle", "jpeg", "webp", "quantize", "km_quantize", "simplequantize", "clahe", "patches" or "clean"
    # noise_data: ../noise_patches/normal/ # location of the noise patches extracted from real images to use for noise injection with noise option "patches"
    # lr_noise: false # true | false
    # lr_noise_types: {gaussian: 1, jpeg: 1, clean: 4}
    # lr_noise2: false # true | false
    # lr_noise_types2: {dither: 2, clean: 2}
    # hr_noise: false # true | false
    # hr_noise_types:  {gaussian: 1, clean: 4}
    
    # Color augmentations
    # lr_fringes: true # true | false
    # lr_fringes_chance: 0.4
    # auto_levels: HR # "HR" | "LR" | "Both" #add auto levels to the images to expand dynamic range. Can use with SPL loss or (MS)-SSIM.
    # rand_auto_levels: 0.7 # Example: 0.4 = 40% chance of adding auto levels to images on the fly
    # lr_unsharp_mask: true # add a unsharpening mask to LR images.
    # lr_rand_unsharp: 1 # Example: 0.5 = 50% chance of adding unsharpening mask to LR images on the fly
    # hr_unsharp_mask: true # add a unsharpening mask to HR images. Can work well together with the HFEN loss function.
    # hr_rand_unsharp: 1 # Example: 0.5 = 50% chance of adding unsharpening mask to HR images on the fly
    
    # Augmentations for classification or (maybe) inpainting networks:
    # lr_cutout: false # true | false
    # lr_erasing: false # true | false
```

If needed, a `validation` dataset can also be included to evaluate progress during training. This is needed in order to calculate training metrics (`psnr`, `ssim` or `lpips`) and those metrics are required in the case that the ReduceLROnPlateau optimizer is used (Note: SRFlow can use `nll` as the metric instead). The options are a subset of training dataset options.

```yaml
  val:  # validation dataset configurations
    name: val_set14_part
    mode: aligned
    dataroot_B: '../datasets/val/hr'
    dataroot_A: '../datasets/val/lr'
    
    znorm: false
    
    # Color space conversion:
    # color: 'y'
    # color_LR: 'y'
    # color_HR: 'y'
    
    # hr_crop: false #disabled
    # In case that no LR image is available for validation, HR can be provided and lr_downscale enabled to generate an LR on the fly.
    lr_downscale: false
    lr_downscale_types: ["linear", "bicubic"] # scaling interpolation options, same as in train dataset
```

[Back to index](#common)

## Network options

The next part of the options files is the networks configuration.

First, the path where either pretrained models are found (in case of fine-tuning or inference, otherwise they can be commented out or `null`) or where the resume state is found, to continue a previous training session.

```yaml
path:
    root: '../'  # the root where the training files will be stored, inside a 'experiments' directory
    pretrain_model_G: '../experiments/pretrained_models/RRDB_PSNR_x4.pth'  # load a pretrained generator G for fine-tuning or inference
    # pretrain_model_D: '../experiments/pretrained_models/patchgan.pth'  # load a pretrained discriminator D for direct use or for transfer learning (if using FreezeD)
    #resume_state: '../experiments/debug_001_RRDB_ESRGAN_x4_DIV2K/training_state/latest.state'  # resume a previous training session
```

Then, you configure the network options, for both generator and discriminator. If you are using the default network configurations, you can use the network name and the defaults will be used. Check [`defaults.py`](https://github.com/victorca25/BasicSR/blob/master/codes/options/defaults.py) for the details. For example:

```yaml
# Generator options:
network_G: esrgan
```

is equivalent to:

```yaml
network_G:  # configurations for the Generator network
  which_model_G: RRDB_net  # check:  https://github.com/victorca25/BasicSR/tree/master/codes/models/modules/architectures
    norm_type: null  # norm type, null | "batch"
    mode: CNA  # Convolution mode: CNA for Conv-Norm_Activation or NAC
    nf: 64  # number of features (filters) for each layer
    nb: 23  # number of RRDB blocks
    nr: 3  # number of residual layers in each RRDB block
    in_nc: 3  # of input image channels: 3 for RGB and 1 for grayscale
    out_nc: 3  # of output image channels: 3 for RGB and 1 for grayscale
    gc: 32  # channel growth (dense block)
    convtype: Conv2D  # convolution type in: Conv2D | PartialConv2D | DeformConv2D | Conv3D
    net_act: leakyrelu  # network activation in: relu | leakyrelu | swish
    gaussian: true  # add gaussian noise in the generator
    plus: false  # enable ESRGAN+ changes
    upsample_mode: upconv  # upsampling model in: upconv | pixelshuffle
```

If you want to override only some of the defaults, you can do so like this example to enable the ESRGAN+ modifications to ESRGAN:

```yaml
network_G:
  which_model_G: esrgan
    plus: true  # enable ESRGAN+ changes
```

Similarly, to use a patchGAN discriminator, you can use

```yaml
# Discriminator options:
network_D: patchgan
```

And it will be loaded with the default options. Changing to a VGG-like discriminator is as simple as replacing `patchgan` with `discriminator_vgg`.

An additional option that both generators and discriminators have is the `strict` key, which if set to `false` will allow the pretrained model to be loaded into the configured network, even if not all the parameters match. It can be useful when trying to do transfer learning or reusing a trained model for another purpose.

Another additional option for both cases is the configuration of the network initialization, which is relevant when training from scratch (without using a pretrain model). The default is `kaiming` with a scale of `0.1`, but those can be changed using the `init_type` and `init_scale` accordingly. Other options are: `normal`, `xavier`, and `orthogonal`, which can be recommended for certain networks.

[Back to index](#common)

## Training strategy

The next part of the options relates to the training strategy. This includes the `optimizers` used to update the model weights in search of the minima, the `schedulers` to modify the learning rate during training and the `loss` functions used to evaluate the results and calculate the errors that will be backpropagated.

Multiple Pytorch standard and additional [`optimizers`](https://github.com/victorca25/BasicSR/blob/master/codes/models/optimizers.py), [`schedulers`](https://github.com/victorca25/BasicSR/blob/master/codes/models/schedulers.py) and [`losses`](https://github.com/victorca25/BasicSR/blob/master/codes/models/modules/loss.py) are available to select. The templates will use the default configuration for each case, based on the original papers, but you can experiment with other options that can help produce better results for your particular case.

If using discriminators, then options for `optimizers` and `schedulers` must be provided.

Also in this block the configurations for frequency separation, batch augmentations and differential augmentations are found. More information in the [augmentations](https://github.com/victorca25/BasicSR/blob/master/docs/augmentations.md) document.

First, the optimizers options. Similar to the networs, if you want to use the default parameters (which are used in many other cases and are considered to be "safe values"), you can opt for selecting only the optimizer algorithm name (check the details in [`optimizers`](https://github.com/victorca25/BasicSR/blob/master/codes/models/optimizers.py)). For example, to use the default `adam` optimizer for both generator and discriminator, use:

```yaml
train:
    # Optimizer options:
    optim_G: adam  # generator optimizer
    optim_D: adam  # discriminator optimizer
```

And this will be equivalent to using the full configuration:

```yaml
train:
    # Optimizer options:
    optim_G: adam  # generator optimizer
    lr_G: 0.0001  # generator starting learning rate
    weight_decay_G: 0  # generator weight decay
    beta1_G: 0.9  # generator momentum
    beta2_G: 0.999  # generator beta 2
    optim_D: adam  # discriminator optimizer
    lr_D: 0.0001  # discriminator starting learning rate
    weight_decay_D: 0  # discriminator weight decay
    beta1_D: 0.9  # discriminator momentum
    beta2_D: 0.999  # discriminator beta 2
```

You can provide partial overrides if needed.

Similarly, the schedulers can be configured next. Here, using the MultiStepLR schedule from ESRGAN. You can use `lr_steps` to specifically set in which iterations the learning rate will be changed (multiplied by `lr_gamma`) or use the relative option `lr_steps_rel`, where you can set fractions of the total number of iterations (`niter`) at which the changes will take place. The two options in the example below are equivalente, given that `niter: 5e5`.

```yaml
    # Schedulers options:
    lr_scheme: MultiStepLR  # learning rate decay scheduler
    # lr_steps: [50000, 100000, 200000, 300000] # iterations at which the learning rate will be decayed (use if training from scratch)
    # lr_steps: [50000, 75000, 85000, 100000]  # finetuning
    lr_steps_rel: [0.1, 0.2, 0.4, 0.6] # to use lr steps relative to % of training niter instead of fixed lr_steps
    lr_gamma: 0.5 # lr change at every step (multiplied by)
```

In the case you want to use Stochastic Weight Averaging (`use_swa: true`), here you will also configure the SWA scheduler.

```yaml
    # For SWA scheduler
    # swa_start_iter: 375000 #Just reference: 75% of 500000. Can be any value, including 0 to start right away with a pretrained model.
    swa_start_iter_rel: 0.75 # to use swa_start_iter relative to % of training niter instead of fixed swa_start_iter
    swa_lr: 1e-4  # has to be ~order of magnitude of a stable lr for the regular scheduler
    swa_anneal_epochs: 10
    swa_anneal_strategy: "cos"
```

The next options of the training strategy section is for the loss function. These losses should be selected for each case, according to the task. In this example with `ESRGAN`, the loss function is the same defined as in `SRGAN`, using pixel loss (to stabilize the colors of the outputs), feature loss (to evaluate the loss in the feature space, instead of the pixel space) and the adversarial loss. More details about the weight (contribution) of each component can be found in the original papers.

```yaml
    # Losses:
    pixel_criterion: l1  # pixel (content) loss
    pixel_weight: 1e-2
    feature_criterion: l1 # feature loss (VGG feature network)
    feature_weight: 1
    # Adversarial loss:
    gan_type: vanilla  # GAN type
    gan_weight: 5e-3
```

Next you would also configure the frequency separation, batch augmentations and differential augmentations, which in this case will not be used, since it was not in the original `ESRGAN` paper.

```yaml
    # Differentiable Augmentation for Data-Efficient GAN Training
    # diffaug: true
    # dapolicy: 'color,transl_zoom,flip,rotate,cutout' # smart "all" (translation, zoom_in and zoom_out are exclusive)
    
    # Batch (Mixup) augmentations
    # mixup: true
    # mixopts: [blend, rgb, mixup, cutmix, cutmixup] # , "cutout", "cutblur"]
    # mixprob: [1.0, 1.0, 1.0, 1.0, 1.0] #, 1.0, 1.0]
    # mixalpha: [0.6, 1.0, 1.2, 0.7, 0.7] #, 0.001, 0.7]
    # aux_mixprob: 1.0
    # aux_mixalpha: 1.2
    ## mix_p: 1.2
    
    # Frequency Separator
    # fs: true
    # lpf_type: average # "average" | "gaussian"
    # hpf_type: average # "average" | "gaussian"
```

The last part of the training strategy is

```yaml
    # Other training options:
    manual_seed: 0  # a set seed for repeatability
    niter: 5e5  # the total number of training iterations
    # warmup_iter: -1  # number of warm up iterations, -1 for no warm up
    val_freq: 5e3  # the frequency at which validation will be executed
    # overwrite_val_imgs: true  # select to overwrite previous validation images
    # val_comparison: true  # select to save validation images comparing LR and HR
    metrics: 'psnr,ssim,lpips' # select from: "psnr,ssim,lpips" or a combination separated by comma ","
```

[Back to index](#common)

## Logger and checkpoint configuration

```yaml
logger:  # logger configurations
    print_freq: 200  # the frequency at which statistics are logged in the console and log files
    save_checkpoint_freq: 5e3  # the frequency at which the training models and states are checkpointed to disk
    overwrite_chkp: false  # whether if the models and states will be overwriten each time they are saved (ideal for storage contrained cases)
```

[Back to index](#common)