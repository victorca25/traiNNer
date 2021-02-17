
LR/HR Dataloaders

- use opencv (`cv2`) to read and process images.

- read from **image** files OR from **.lmdb** for fast IO speed.
    - How to create .lmdb file? Please see [`codes/scripts/create_lmdb.py`](https://github.com/victorca25/BasicSR/blob/master/codes/scripts/create_lmdb.py). Note that it is currently required to set `n_workers: 0` in the dataloader options to use lmdb (at least on Windows), else there can be a PermissionError due to multiple processes accesing the image database.
    
- images can be downsampled on-the-fly using `matlab bicubic` function. However, the speed is a bit slow compared to using other optimized downscaling algorithms like those in `cv2`. Implemented in [`common.py`](https://github.com/victorca25/BasicSR/blob/master/codes/dataops/common.py). More about [`matlab bicubic` function](https://github.com/victorca25/BasicSR/wiki/MATLAB-like-imresize).


## Contents

- `LR_dataset`: only reads LR images in test phase where there is no GT images.
- `LRHR_dataset`: reads LR and HR pairs from image folder or lmdb files. If only HR images are provided, downsample the images on-the-fly. Used in SR and SRGAN training and validation phase.
- `LRHROTF_dataset` and `LRHRC_dataset`: are customized dataloaders that add on-the-fly augmentation options
- `LRHR_seg_bg_dataset`: reads HR images, segmentations and generates LR images, category. Used in SFTGAN training and validation phase.


## How To Prepare Data
### SR, SRGAN
1. Prepare the images. You can download **classical SR** datasets (including BSD200, T91, General100; Set5, Set14, urban100, BSD100, manga109; historical) from [Google Drive](https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU?usp=sharing). DIV2K dataset can be downloaded from [DIV2K offical page](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

2. Refer to see [`IO-speed`](https://github.com/victorca25/BasicSR/wiki/IO-speed) for some tips regarding data IO speed.


### SFTGAN
SFTGAN is now used for a part of outdoor scenes. 

1. Download OutdoorScene training dataset from [Google Drive](https://drive.google.com/drive/folders/16PIViLkv4WsXk4fV1gDHvEtQxdMq6nfY?usp=sharing) (the training dataset is a little different from that in project page, e.g., image size and format) and OutdoorScene testing dataseet from [Google Drive](https://drive.google.com/drive/u/1/folders/1_uB4EJ2HBLfz1R_F5_zlvIf-SfB-gMzw).
1. Generate the segmenation probability maps for training and testing dataset using [`codes/test_seg.py`](https://github.com/victorca25/BasicSR/blob/master/codes/test_seg.py).
1. Put the images in a folder named `img` and put the segmentation .pth files in a folder named `bicseg` as the following figure shows.

<p align="center">
  <img src="https://c1.staticflickr.com/2/1726/42730268851_9179e94f48.jpg" width="100">
</p>

4. The same for validation (you can choose some from the test folder) and test folder.

## General Data Process

### data augmentation

By default the random crop, random flip/rotation, (random scale) for data augmentation. However, multiple additional on-the-fly options are available when using `LRHRC_dataset`. More information about dataset augmnetation can be found [here](https://github.com/victorca25/BasicSR/wiki/Dataset-Augmentation)