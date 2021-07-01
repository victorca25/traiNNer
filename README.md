# augmeNNt

This repository is intended first as a faster drop-in replacement of Pytorch's Torchvision default augmentations in the "transforms" [package](https://github.com/pytorch/vision/tree/master/torchvision/transforms), based on NumPy and OpenCV (PIL-free) for computer vision pipelines. Additionally, many useful functions and augmentations for image to image translation, super-resolution and restoration (deblur, denoise, etc) are also available.

## Supported Augmentations

Most functions from the original Torchvision transforms are reimplemented, with some considerations:
1.  ToPILImage is not implemented or needed, we use OpenCV instead (ToCVImage). However, the original ToPILImage in ~transforms can be used to save the tensor as a PIL image if required. Once transformed into tensor format, images have RGB channel order in both cases.
2.  OpenCV images are Numpy arrays. OpenCV supports uint8, int8, uint16, int16, int32, float32, float64. Certain operations (like `cv.CvtColor()`) do require to convert the arrays to OpenCV type (with `cv.fromarray()`).
3.  The affine transform in the original one only has 5 degrees of freedom, YU-Zhiyang implemented an Affine transform with 6 degress of freedom called `RandomAffine6` (can be found in [transforms.py](augmennt/transforms.py)). The original method `RandomAffine` is also available and reimplemented with OpenCV.
4.  The rotate function is clockwise, however the original one is anticlockwise.
5.  Some new augmentations have been added, in comparison to Torchvision's, refer to the list [below](#support).
6.  **The outputs of the OpenCV versions are almost the same as the original one's (it's possible to test by running [test.py](/test.py)) directly with test images**.

These are the basic augmentations, equivalent to torchvision's:

-   `Compose`, `ToTensor`, `ToCVImage`, `Normalize`,
-   `Resize`, `CenterCrop`, `Pad`,
-   `Lambda` (see [note](#attention)),
-   `RandomApply`, `RandomOrder`, `RandomChoice`, `RandomCrop`,
-   `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomResizedCrop`,
-   `FiveCrop`, `TenCrop`, `LinearTransformation`, `ColorJitter`,
-   `RandomRotation`, `RandomAffine`,
-   `Grayscale`, `RandomGrayscale`, `RandomErasing`,

The additional transforms can be used to train models such as [Noise2Noise](https://arxiv.org/pdf/1803.04189.pdf), [BSRGAN](https://arxiv.org/pdf/2103.14006v1.pdf) [White-box Cartoonization](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Learning_to_Cartoonize_Using_White-Box_Cartoon_Representations_CVPR_2020_paper.pdf) and [EdgeConnect](https://openaccess.thecvf.com/content_ICCVW_2019/papers/AIM/Nazeri_EdgeConnect_Structure_Guided_Image_Inpainting_using_Edge_Prediction_ICCVW_2019_paper.pdf), among others. There are some general augmentations:
-   `RandomAffine6`, `Cutout`, `RandomPerspective`,

Noise augmentations, with options for artificial noises and realistic noise generation:
-   `RandomGaussianNoise`, `RandomPoissonNoise`, `RandomSPNoise`,
-   `RandomSpeckleNoise`, `RandomCompression`,
-   `BayerDitherNoise`, `FSDitherNoise`, `AverageBWDitherNoise`,`BayerBWDitherNoise`,
-   `BinBWDitherNoise`, `FSBWDitherNoise`, `RandomBWDitherNoise`,
-   `RandomCameraNoise`, `RandomChromaticAberration`

Blurs and different kind of kernels generation and use, with standard blurs, isotropic and anisotropic Gaussian filters and simple and complex motion blur kernels:
-   `RandomAverageBlur`, `RandomBilateralBlur`, `RandomBoxBlur`,
-   `RandomGaussianBlur`, `RandomMedianBlur`,
-   `RandomMotionBlur`, `RandomComplexMotionBlur`,
-   `RandomAnIsoBlur`, `AlignedDownsample`, `ApplyKernel`,

Filters to modify the images, including color quantization, superpixel segmentation and CLAHE:
-   `FilterMaxRGB`, `FilterColorBalance`, `FilterUnsharp`,
-   `SimpleQuantize`, `RandomQuantize`, `RandomQuantizeSOM`,
-   `CLAHE`, `RandomGamma`, `Superpixels`

Edge filters:
-   `FilterCanny`,

## Requirements

-   python >= 3.5.2
-   numpy >= 1.10 ('@' operator may not be overloaded before this version)
-   pytorch >= 0.4.1
-   A working installation of OpenCV. **Tested with OpenCV version 3.4.2, 4.1.0**
-   Tested on Windows 10 and Ubuntu 18.04.

## Optional requirements

-   torchvision >= 0.2.1

In order to use the additional Superpixel options (skimage SLIC and Felzenszwalb algorithms), as well as segment reduction algorithms (selective search and RAG merging) and the Menon demosaicing algorithm, there are additional requirements:
-   scikit-image >= 0.17.2
-   scipy >= 1.6.2

## Usage

1.  git clone <https://github.com/victorca25/augmeNNt.git> .
2.  Add `augmennt` to your python path.
3.  Add `from augmennt import augmennt as transforms` in your python file.
4.  From here, almost everything should work exactly as the original `transforms`.

### Example: Image resizing

```python
import numpy as np
from augmennt import augmennt as transforms
image = np.random.randint(low=0, high=255, size=(1024, 2048, 3))
resize = transforms.Resize(size=(256,256))
image = resize(image)
```

Should be 1.5 to 10 times faster than PIL. See benchmarks

### Example: Composing transformations

```py
transform = transforms.Compose([
   transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 0)),
   transforms.Resize(size=(350, 350), interpolation="BILINEAR"),
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

More examples can be found in the  official Pytorch [tutorials](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

## Attention

The multiprocessing used in Pytorch's dataloader may have issues with lambda functions (using `Lambda` in [transforms.py](torchvision/transforms/transforms.py)) in Windows, as lambda functions can't be pickled (<https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled>). This issue also happens with Torchvision's `Lambda` function.

These issues happen when using, `num_workers > 0` in a Pytorch `DataLoader` class when the transformations are initialized in the class init. The issue can be prevented either by using proper functions (not lambda) when composing the transformations or by initializing it in the `DataLoader` call instead.

## Performance

The following are the performance tests as executed by jbohnslav. 

-   Most transformations are between 1.5X and ~4X faster in OpenCV. Large image resizes are up to 10 times faster in OpenCV.
-   To reproduce the following benchmarks, download the [Cityscapes dataset](https://www.cityscapes-dataset.com/). 
-   An example benchmarking file that jbohnslav used can be found in the notebook **benchmarking_v2.ipynb** where the Cityscapes default directories are wrapped with a HDF5 file for even faster reading (Note: this file has not been updated or tested for a very long time, but can serve as a reference).

![resize](benchmarks/benchmarking_Resize.png)
![random crop](benchmarks/benchmarking_Random_crop_quarter_size.png)
![change brightness](benchmarks/benchmarking_Color_brightness_only.png)
![change brightness and contrast](benchmarks/benchmarking_Color_constrast_and_brightness.png)
![change contrast only](benchmarks/benchmarking_Color_contrast_only.png)
![random horizontal flips](benchmarks/benchmarking_Random_horizontal_flip.png)

The changes start to add up when you compose multiple transformations together.
![composed transformations](benchmarks/benchmarking_Resize_flip_brightness_contrast_rotate.png)

Compared to regular Pillow, cv2 is around three times faster than PIL, as shown in this [article](https://www.kaggle.com/vfdev5/pil-vs-opencv).

Additionally, the [Albumentations project](https://github.com/albumentations-team/albumentations), mostly based on Numpy and OpenCV also has shown better performance than other options, including torchvision with a fast Pillow-SIMD backend.

But it can also be the case that Pillow-SIMD can be faster in some cases, as tested in this [article](https://python-pillow.org/pillow-perf/)

## Alternatives

There are multiple image augmentation and manipulation frameworks available, each with its own strengths and limitations. Some of these alternatives are:

-   [Torchvision](https://github.com/pytorch/vision): Based on [Pillow (default)](https://python-pillow.org/), [Pillow-SIMD](https://github.com/uploadcare/pillow-simd), [accimage](https://github.com/pytorch/accimage), [libpng](http://www.libpng.org/pub/png/libpng.html), [libjpeg](http://ijg.org/) or [libjpeg-turbo](https://libjpeg-turbo.org/)
-   [Kornia](https://github.com/kornia/kornia): Inspired by OpenCV, for differentiable tensor image functions
-   [Albumentations](https://github.com/albumentations-team/albumentations): Based on pure NumPy, [OpenCV](https://github.com/opencv/opencv) and [imgaug](https://github.com/aleju/imgaug), with a large variety of transformations
-   [Rising](https://github.com/PhoenixDL/rising): For differentiable 2D and 3D image functions
-   [TorchIO](https://github.com/fepegar/torchio): For 3D medical imaging

## Postscript
-   This repository originally merged [jbohnslav](https://github.com/jbohnslav/opencv_transforms) and [YU-Zhiyang](https://github.com/YU-Zhiyang/opencv_transforms_torchvision) repositories (which had the same purpose), and my own OpenCV-based augmentations from [BasicSR](https://github.com/victorca25/BasicSR), in order to allow to refactor the project's data flow and streamline to use the Torchvision's API as a standard. This enables changing or combining different base frameworks (OpenCV, Pillow/Pillow-SIMD, etc) to add more augmentations only by modifying the imported library and also to easily switch to other replacements like [Kornia](https://github.com/kornia/kornia), [Albumentations](https://github.com/albumentations-team/albumentations), or [Rising](https://github.com/PhoenixDL/rising), based on the user's needs. An example with a backend change is DinJerr's [fork](https://github.com/DinJerr/BasicSR), using [wand](https://github.com/emcconville/wand)+[ImageMagick](https://imagemagick.org/) for augmentations.
-   Each backend has it's pros and cons, but important points to consider when choosing are: available augmentation types, performance, external dependencies, features (for example, Kornia's differentiable augmentations) and user preference (all previous points being equal).
