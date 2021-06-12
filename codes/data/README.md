
# General notes for dataloaders

- use OpenCV (`cv2`) to read and process images by default, but the main ones can also use Pillow (`PIL`) as an alternative. Some benchmarking comparisons between `cv2` and `PIL` can be found [here](https://github.com/victorca25/opencv_transforms/blob/master/opencv_transforms/)

- read from **image** files OR from **.lmdb** for faster speed. Refer to [`IO-speed`](https://github.com/victorca25/BasicSR/wiki/IO-speed) for some tips regarding data IO speed.
    - Note that when preparing the **.lmdb** database on Windows it is currently required to set `n_workers: 0` in the dataloader options, else there can be a `PermissionError` due to multiple processes accesing the image database.

- images can be downsampled on-the-fly using `matlab`-like `imresize` function. It can add a lot more variety to the training, but the speed is slower than when using other optimized downscaling algorithms like the `cv2` one. Implemented in [`imresize.py`](https://github.com/victorca25/BasicSR/blob/master/codes/dataops/imresize.py). For more information about why this is an important consideration, check [here](https://github.com/victorca25/BasicSR/blob/master/docs/augmentations.md#downscaling-methods-and-augmentation-pipeline)

- it is also possible to add different kinds of augmentations to images on the fly during training. More information about the augmentations can be found [here](https://github.com/victorca25/BasicSR/blob/master/docs/augmentations.md#augmentations)


## Contents

- `base_dataset.py` implements an base class for datasets. It also includes common functions which are used by the other dataset files.

- `single_dataset`: includes a dataset class that can load a set of single images specified by the path `dataroot_*: /path/to/data`. It only reads single images (`LR`, `LQ`, `A`, etc) in test (inference) phase where there is no `GT/B` image. It can be used for generating CycleGAN results only for one side of the cycle generators.

- `aligned_dataset`: a dataset class that can load image pairs from image folder or lmdb files and on-the-fly augmentation options. If only `HR/B` images are provided or the specific configuration is provided, it will generate the paired images on-the-fly. Used for training on paired images cases (Super-Resolution, Super-Restoration, Pix2pix, etc) training and validation phase. It can work with either one path for each side of the pair (ie, `dataroot_A: /path/to/dataA` and `dataroot_B: /path/to/dataB`) or a single image directory `dataroot_AB: /path/to/data`, which contains image pairs in the form of {A,B}, like the pix2pix original [datasets](https://github.com/victorca25/BasicSR/blob/master/docs/datasets.md#image-to-image-translation).

- `unaligned_dataset.py`: a dataset class that can load unaligned/unpaired datasets. It assumes that two directories to host training images from domain A `dataroot_A: /path/to/dataA` and from domain B `dataroot_B: /path/to/dataB` respectively.

- `LRHR_seg_bg_dataset.py`: reads HR images, segmentations and generates LR images, category. Used in SFTGAN training and validation phase.

- `LRHRPBR_dataset.py`: experimental dataset for working with the PBR training model.

- `Vid_dataset.py`: experimental dataset for loading video datasets in the form of frames in a directory containing one directory for each scene. Based on the structure of the [REDS datasets](https://github.com/victorca25/BasicSR/blob/master/docs/datasets.md#video).

- `DVD_dataset.py`: experimental dataset for loading video datasets, specifically for the interlaced video case. Interlaced frame is expected to be "combed" from the progressive pair. It will read interlaced and progressive frame triplets (pairs of three).


## How To Prepare Data
### Super-Resolution and restoration
- Prepare the images. You can find the links to download **classical SR** datasets (including BSD200, T91, General100; Set5, Set14, urban100, BSD100, manga109; historical) or **DIV2K** dataset from [datasets](https://github.com/victorca25/BasicSR/blob/master/docs/datasets.md#super-resolution) or prepare your own dataset.


### SFTGAN
SFTGAN is used for a part of outdoor scenes. 

1. Download OutdoorScene training dataset and OutdoorScene testing dataset from [datasets](https://github.com/victorca25/BasicSR/blob/master/docs/datasets.md). The training dataset is a little different from that in project page, e.g., image size and format).
2. Generate the segmenation probability maps for training and testing dataset using [`codes/test_seg.py`](https://github.com/victorca25/BasicSR/blob/master/codes/test_seg.py).
3. Put the images in a folder named `img` and put the segmentation .pth files in a folder named `bicseg` as the following figure shows.

<p align="center">
  <img src="https://c1.staticflickr.com/2/1726/42730268851_9179e94f48.jpg" width="100">
</p>

4. The same for validation (you can choose some from the test folder) and test folder.

### Image to image translation
- Similar to SR cases, you will find sample datasets for both paired and unpaired cases in [datasets](https://github.com/victorca25/BasicSR/blob/master/docs/datasets.md#image-to-image-translation) or you can use your own datasets.
- In the case of Pix2pix trainin, the corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., `/path/to/data/A/train/1.jpg` is considered to correspond to `/path/to/data/B/train/1.jpg` and the size at which the network will use the images to train must be coordinated in the network configuration and the `load_size` option.
- For CycleGAN, you similarly need two directories that contain images from domain `A` and `B`. You should not expect the method to work on just any random combination of input and output datasets (e.g. `cats<->keyboards`). From experiments, it works better if two datasets share similar visual content. For example, `landscape painting<->landscape photographs` works much better than `portrait painting <-> landscape photographs`. `zebras<->horses` achieves compelling results while `cats<->dogs` completely fails.

More details about the data configuration for image to image translation [here](https://github.com/victorca25/BasicSR/blob/master/docs/howtotrain.md#image-to-image-translation)

## General Data Process

### Data augmentation

By default random crop and random flip/rotation are used for data augmentation. However, multiple additional on-the-fly options are available. More information about dataset augmnetation can be found [here](https://github.com/victorca25/BasicSR/wiki/Dataset-Augmentation) and [here](https://github.com/victorca25/BasicSR/blob/master/docs/augmentations.md).