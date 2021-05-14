import random
import argparse

import os
import os.path
import sys
import glob

import numpy as np
import cv2
import dataops.common as util
from dataops.common import fix_img_channels, get_image_paths, read_img, np2tensor
from dataops.minisom import MiniSom
from dataops.debug import *
from dataops.imresize import resize as imresize  # resize # imresize_np

import dataops.opencv_transforms.opencv_transforms as transforms
from dataops.opencv_transforms.opencv_transforms.common import wrap_cv2_function, wrap_pil_function
from torch.utils.data.dataset import Dataset #TODO TMP, move NoisePatches to a separate dataloader

# from dataops.augmentations import Scale, get_blur, get_noise, get_pad, translate_chan, NoisePatches, KernelDownscale, RandomQuantize, RandomNoisePatches #, MLResize, RandomQuantize, KernelDownscale, NoisePatches, RandomNoisePatches, get_resize, get_blur, get_noise, get_pad



try:
    from PIL import Image
    pil_available = True
except ImportError:
    pil_available = False
    # pass

try:
    import cv2
    cv2_available =  True
except ImportError:
    cv2_available = False

def set_transforms(loader_type=None):
    if not hasattr(set_transforms, 'loader_type') or set_transforms.loader_type != loader_type:
        global transforms
        if loader_type == 'pil' and pil_available:
            import torchvision.transforms as transforms
        elif cv2_available:
            import dataops.opencv_transforms.opencv_transforms as transforms
        else:
            Exception("No suitable image loader available. Need either PIL or OpenCV.")

        set_transforms.loader_type = loader_type

set_transforms()


custom_ktypes = [794, 793, 792, 791, 790, 789, 788, 787, 786, 785, 784,
                783, 782, 781, 780, 779, 778, 777, 776, 775, 774, 773,]

_n_interpolation_to_str = {
    794:'blackman5', 793:'blackman4', 792:'blackman3',
    791:'blackman2', 790:'sinc5', 789:'sinc4', 788:'sinc3',
    787:'sinc2', 786:'gaussian', 785:'hamming', 784:'hanning',
    783:'catrom', 782:'bell', 781:'lanczos5', 780:'lanczos4',
    779:'hermite', 778:'mitchell', 777:'cubic', 776:'lanczos3',
    775:'lanczos2', 774:'box', 773:'linear',}

def Scale(img=None, scale: int = None, algo=None, ds_kernel=None, resize_type=None):

    ow, oh = img.shape[0], img.shape[1]
    ## original rounds down:
    # w = int(oh/scale)
    # h = int(ow/scale)
    ## rounded_up = -(-numerator // denominator)     # math.ceil(scale_factor * in_sz)
    w = int(-(-oh//scale))
    h = int(-(-ow//scale))

    if (h == oh) and (w == ow):
        return img, None

    resize, resize_type = get_resize(size=(h, w), scale=scale, ds_algo=algo, ds_kernel=ds_kernel, resize_type=resize_type)
    return resize(np.copy(img)), resize_type


class MLResize(object):
    """Resize the input numpy ndarray to the given size using the Matlab-like
    algorithm (warning an order of magnitude slower than OpenCV).

    Args:
        scale (sequence or int): Desired amount to scale the image. 
        interpolation (int, optional): Desired interpolation. Default is
            ``cubic`` interpolation, other options are: "lanczos2", 
            "lanczos3", "box", "linear"
    """

    def __init__(self, scale, antialiasing=True, interpolation='cubic'):

        self.scale = 1/scale
        self.interpolation = interpolation
        self.antialiasing = antialiasing

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be scaled.
        Returns:
            numpy ndarray: Rescaled image.
            
        """
        # return util.imresize_np(img=img, scale=self.scale, antialiasing=self.antialiasing, interpolation=self.interpolation)
        return imresize(img, self.scale, antialiasing=self.antialiasing, interpolation=self.interpolation)


def get_resize(size, scale=None, ds_algo=None, ds_kernel=None, resize_type=None):

    if ds_algo is None:
        #scaling interpolation options
        ds_algo = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT]
    elif isinstance(ds_algo, int):
        ds_algo = [ds_algo]
    #elif(!isinstance(resize_types, list)):
        #Error, unexpected type

    resize = None
    if resize_type == 0:
        pass
    elif not resize_type:
        resize_type = random.choice(ds_algo)

    # print(resize_type)
    if resize_type in custom_ktypes:
        # print('matlab')
        resize = MLResize(scale=scale, interpolation=_n_interpolation_to_str[resize_type]) #(np.copy(img_LR))
    elif resize_type == 999: # use realistic downscale kernels
        # print('kernelgan')
        if ds_kernel:
            resize = ds_kernel
    else: # use the provided OpenCV2 algorithms
        #TODO: tmp fix in case ds_kernel was not provided, default to something
        if resize_type == 999:
            #TODO: add a log message to explain ds_kernel is missing
            resize_type = cv2.INTER_CUBIC
        _cv2_interpolation_to_str = {cv2.INTER_NEAREST:'NEAREST',
                                cv2.INTER_LINEAR:'BILINEAR',
                                cv2.INTER_AREA:'AREA',
                                cv2.INTER_CUBIC:'BICUBIC',
                                cv2.INTER_LANCZOS4:'LANCZOS'}
        resize = transforms.Resize(size, interpolation=_cv2_interpolation_to_str[resize_type]) #(np.copy(img_LR))
        # print('cv2')

    return resize, resize_type


#TODO: use options to set the blur types parameters if configured, else random_params=True
def get_blur(blur_types: list = []):

    blur = None
    if(isinstance(blur_types, list)):
        blur_type = random.choice(blur_types)
        if blur_type == 'average':
            # print('average')
            blur = transforms.RandomAverageBlur(
                    max_kernel_size=11, random_params=True, p=1)
        elif blur_type == 'box':
            # print('box')
            blur = transforms.RandomBoxBlur(
                    max_kernel_size=11, random_params=True, p=1)
        elif blur_type == 'gaussian':
            # print('gaussian')
            blur = transforms.RandomGaussianBlur(
                    max_kernel_size=11, random_params=True, p=1)
        elif blur_type == 'median':
            # print('median')
            blur = transforms.RandomMedianBlur(
                    max_kernel_size=11, random_params=True, p=1)
        elif blur_type == 'bilateral':
            # print('bilateral')
            blur = transforms.RandomBilateralBlur(
                    sigmaSpace=200, sigmaColor=200,
                    max_kernel_size=11, random_params=True, p=1)
        elif blur_type == 'motion':
            # print('motion')
            blur = transforms.RandomMotionBlur(
                     max_kernel_size=7, random_params=True, p=1)
        elif blur_type == 'complexmotion':
            # print('complexmotion')
            blur = transforms.RandomComplexMotionBlur(
                     max_kernel_size=7, random_params=True, p=1,
                     size=100, complexity=1.0)        
        #elif blur_type == 'clean':
    return blur


#TODO: use options to set the noise types parameters if configured, else random_params=True
def get_noise(noise_types: list = [], noise_patches=None, noise_amp: int=1):

    noise = None
    if(isinstance(noise_types, list)):
        noise_type = random.choice(noise_types)

        # if noise_type == 'dither':
        if 'dither' in noise_type:
            # print(noise_type)
            #TODO: need a dither type selector, there are multiple options including b&w dithers
            # dither = 'fs'
            # if dither == 'fs':
            if ('fs' in noise_type and 'bw' not in noise_type) or noise_type == 'dither':
                noise = transforms.FSDitherNoise(p=1)
            # elif dither == 'bayer':
            elif 'bayer' in noise_type and 'bw' not in noise_type:
                noise = transforms.BayerDitherNoise(p=1)
            # elif dither == 'fs_bw':
            elif 'fs_bw' in noise_type:
                noise = transforms.FSBWDitherNoise(p=1)
            # elif dither == 'avg_bw':
            elif 'avg_bw' in noise_type:
                noise = transforms.AverageBWDitherNoise(p=1)
            # elif dither == 'bayer_bw':
            elif 'bayer_bw' in noise_type:
                noise = transforms.BayerBWDitherNoise(p=1)
            # elif dither == 'bin_bw':
            elif 'bin_bw' in noise_type:
                noise = transforms.BinBWDitherNoise(p=1)
            # elif dither == 'rnd_bw':
            elif 'rnd_bw' in noise_type:
                noise = transforms.RandomBWDitherNoise(p=1)
            # print("dither")
        elif noise_type == 'simplequantize':
            #TODO: find a useful rgb_range for SimpleQuantize in [0,255]
            # the smaller the value, the closer to original colors
            noise = transforms.SimpleQuantize(p=1, rgb_range = 50) #30
            # print("simplequantize")
        elif noise_type == 'quantize':
            noise = RandomQuantize(p=1, num_colors=32)
            # print("quantize")
        elif noise_type == 'gaussian':
            noise = transforms.RandomGaussianNoise(p=1, random_params=True, gtype='bw')
            # print("gaussian")
        elif noise_type.lower() == 'jpeg':
            noise = transforms.RandomCompression(p=1, random_params=True, image_type='.jpeg')
            # print("JPEG")
        elif noise_type.lower() == 'webp':
            noise = transforms.RandomCompression(p=1, random_params=True, image_type='.webp')
        elif noise_type == 'poisson':
            noise = transforms.RandomPoissonNoise(p=1)
            # print("poisson")
        elif noise_type == 's&p':
            noise = transforms.RandomSPNoise(p=1)
            # print("s&p")
        elif noise_type == 'speckle':
            noise = transforms.RandomSpeckleNoise(gtype='bw', p=1)
            # print("speckle")
        elif noise_type == 'maxrgb':
            noise = transforms.FilterMaxRGB(p=1)
            # print("maxrgb")
        # elif noise_type == 'canny':
        #     noise = transforms.FilterCanny(p=1)
        elif noise_type == 'patches' and noise_patches:
            noise = RandomNoisePatches(noise_patches, noise_amp)
            # print("patches")
        #elif noise_type == 'clean':
        elif noise_type == 'clahe':
            noise = transforms.CLAHE(p=1)

    return noise

def get_pad(img, size: int, fill = 0, padding_mode: str ='constant'):
    #TODO: update to work with PIL as well with image_size(img)
    h, w = img.shape[0], img.shape[1]

    if fill == 'random':
        fill_list = []
        for _ in range(len(img.shape)):
            fill_list.append(random.randint(0, 255))
        fill = tuple(fill_list)

    top = (size - h) // 2 if h < size else 0
    bottom = top + h % 2 if h < size else 0
    left = (size - w) // 2 if w < size else 0
    right = left + w % 2 if w < size else 0

    pad = transforms.Pad(padding=(top, bottom, left, right), padding_mode=padding_mode, fill=fill) #reflect

    return pad, fill



class RandomQuantize(object):
    r"""Color quantization using MiniSOM
    Args:
        img (numpy ndarray): Image to be quantized.
        num_colors (int): the target number of colors to quantize to
        sigma (float): the radius of the different neighbors in the SOM
        learning_rate (float): determines how much weights are adjusted 
            during each SOM iteration
        neighborhood_function (str): the neighborhood function to use 
            with SOM
    Returns:
        numpy ndarray: quantized version of the image.
    """

    def __init__(self, p = 0.5, num_colors=None, sigma=1.0, learning_rate=0.2, neighborhood_function='bubble'):
        # assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        self.p = p

        if not num_colors:
            N = int(np.random.uniform(2, 8))
        else:
            N = int(num_colors/2)
        # assert isinstance(N, numbers.Number) and N >= 0, 'N should be a positive value'
        # input_len corresponds to the shape of the pixels array (H, W, C)
        # x and y are the "palette" matrix shape. x=2, y=N means 2xN final colors, but 
        # could reshape to something like x=3, y=3 too
        # try sigma = 0.1 , 0.2, 1.0, etc
        self.som = MiniSom(x=2, y=N, input_len=3, sigma=sigma,
                    learning_rate=0.2, neighborhood_function=neighborhood_function)

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be quantized.

        Returns:
            np.ndarray: Quantized image.
        """
        if random.random() < self.p:

            img_type = img.dtype
            if np.issubdtype(img_type, np.integer):
                img_max = np.iinfo(img_type).max
            elif np.issubdtype(img_type, np.floating):
                img_max = np.finfo(img_type).max

            # reshape image as a 2D array
            pixels = np.reshape(img, (img.shape[0]*img.shape[1], 3))
            # print(pixels.shape)
            # print(pixels.min())
            # print(pixels.max())

            # initialize som
            self.som.random_weights_init(pixels)
            # save the starting weights (the image’s initial colors)
            starting_weights = self.som.get_weights().copy() 
            self.som.train_random(pixels, 500, verbose=False)
            #som.train_random(pixels, 100)

            # Vector quantization: quantize each pixel of the image to reduce the number of colors
            qnt = self.som.quantization(pixels) 
            clustered = np.zeros(img.shape)
            for i, q in enumerate(qnt): 
                clustered[np.unravel_index(i, dims=(img.shape[0], img.shape[1]))] = q
            img = np.clip(clustered, 0, img_max).astype(img_type)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class KernelDownscale(object):
    '''
    Use the previously extracted realistic kernels to downscale images with

    Ref:
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Kernel_Modeling_Super-Resolution_on_Real_Low-Resolution_Images_ICCV_2019_paper.pdf
    https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf
    '''
    #TODO: can make this class into a pytorch dataloader, as an iterator can load all the kernels in init

    def __init__(self, scale, kernel_paths, size=13, permute=True):
        self.scale = 1.0/scale

        # self.kernel_paths = glob.glob(os.path.join(opt['dataroot_kernels'], '*/*_kernel_x{}.npy'.format(scale)))
        # using the modified kernelGAN file structure.
        self.kernel_paths = glob.glob(os.path.join(kernel_paths, '*/kernel_x{}.npy'.format(scale)))
        if not self.kernel_paths:
            # try using the original kernelGAN file structure.
            self.kernel_paths = glob.glob(os.path.join(kernel_paths, '*/*_kernel_x{}.npy'.format(scale)))
        assert self.kernel_paths, "No kernels found for scale {} in path {}.".format(scale, kernel_paths)

        self.num_kernel = len(self.kernel_paths)
        # print('num_kernel: ', self.num_kernel)

        if permute:
            np.random.shuffle(self.kernel_paths)

        # making sure the kernel size (receptive field) is 13x13 (https://arxiv.org/pdf/1909.06581.pdf)
        self.pre_process = transforms.Compose([transforms.CenterCrop(size)])

        # print(kernel_path)
        # print(self.kernel.shape)

    def __call__(self, img):

        kernel_path = self.kernel_paths[np.random.randint(0, self.num_kernel)]
        with open(kernel_path, 'rb') as f:
            kernel = np.load(f)

        kernel = self.pre_process(kernel)
        # normalize to make cropped kernel sum 1 again
        # kernel = kernel / np.sum(kernel)
        kernel = kernel.astype(np.float32) / np.sum(kernel)
        # print(kernel.shape)

        input_shape = img.shape
        # By default, if scale-factor is a scalar assume 2d resizing and duplicate it.
        if isinstance(self.scale, (int, float)):
            scale_factor = [self.scale, self.scale]
        else:
            scale_factor = self.scale

        # if needed extend the size of scale-factor list to the size of the input 
        # by assigning 1 to all the unspecified scales
        if len(scale_factor) != len(input_shape):
            scale_factor = list(scale_factor)
            scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

        # Dealing with missing output-shape. calculating according to scale-factor
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

        # First run a correlation (convolution with flipped kernel)
        out_im = cv2.filter2D(img, -1, kernel)

        # Then subsample and return
        return out_im[np.round(np.linspace(0, out_im.shape[0] - 1 / scale_factor[0], output_shape[0])).astype(int)[:, None],
                np.round(np.linspace(0, out_im.shape[1] - 1 / scale_factor[1], output_shape[1])).astype(int), :]



# class NoisePatches(Object):
class NoisePatches(Dataset):
    '''
    Load the previously noise patches from real images to apply to the LR images
    Ref:
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Image_Blind_Denoising_CVPR_2018_paper.pdf
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Kernel_Modeling_Super-Resolution_on_Real_Low-Resolution_Images_ICCV_2019_paper.pdf

    '''
    def __init__(self, dataset=None, size=32, permute=True, grayscale=False):
        # size = opt['GT_size']/opt['scale']
        super(NoisePatches, self).__init__()
        assert os.path.exists(dataset)

        self.grayscale = grayscale
        self.noise_imgs = sorted(glob.glob(dataset + '*.png'))
        if permute:
            np.random.shuffle(self.noise_imgs)

        self.pre_process = transforms.Compose([transforms.RandomCrop(size), 
                                               transforms.RandomHorizontalFlip(p=0.5), 
                                               transforms.RandomVerticalFlip(p=0.5), 
                                               ])

    def __getitem__(self, index, out_nc=3):

        noise = self.pre_process(util.read_img(None, self.noise_imgs[index], out_nc))
        # describe_numpy(noise, all=True)
        # tmp_vis(noise, False)
        norm_noise = (noise - np.mean(noise, axis=(0, 1), keepdims=True, dtype=np.float32))
        # tmp_vis(np.mean(noise, axis=(0, 1), keepdims=True), False)
        # describe_numpy(np.mean(noise, axis=(0, 1), keepdims=True), all=True)
        # describe_numpy(norm_noise, all=True)

        #TODO: test adding noise to single channel images 
        if self.grayscale:
            norm_noise = util.bgr2ycbcr(norm_noise, only_y=True)
        return norm_noise #.astype('uint8')

    def __len__(self):
        return len(self.noise_imgs)


class RandomNoisePatches():
    def __init__(self, noise_patches, noise_amp=1):
        self.noise_patches = noise_patches
        self.noise_amp = noise_amp

    def __call__(self, img):
        # add noise from patches 
        noise = self.noise_patches[np.random.randint(0, len(self.noise_patches))]
        # tmp_vis(noise, False)
        # img = torch.clamp(img + noise, 0, 1)
        # describe_numpy(img, all=True)
        h, w = img.shape[0:2]
        n_h, n_w = noise.shape[0:2]
        if n_h < h or n_w < w:
            # pad noise patch to image size if smaller
            i = random.randint(0, h - n_h)
            j = random.randint(0, w - n_w)
            #top, bottom, left, right borders
            noise = transforms.Pad(padding=(i, h-(i+n_h), j, w-(j+n_w)))(noise)
        elif n_h > h or n_w > w:
            # crop noise patch to image size if larger
            noise = transforms.RandomCrop(size=(w,h))(noise)

        img = np.clip((img.astype('float32') + self.noise_amp*noise), 0, 255).astype('uint8')
        # describe_numpy(img, all=True)
        ## tmp_vis(img, False)
        return img




############# transformations
#TODO: adapt for video dataloaders, apply same transform to multiple frames

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    load_size = opt.get('load_size')
    if isinstance(load_size, list):
        load_size = random.choice(load_size)
    crop_size = opt.get('crop_size')
    center_crop_size = opt.get('center_crop_size')
    preprocess_mode = opt.get('preprocess', 'none')

    # if preprocess_mode == 'resize_and_crop':
    if 'resize_and_crop' in preprocess_mode:
        # assert load_size, "load_size not defined"
        new_h = new_w = load_size
    # elif preprocess_mode == 'scale_width_and_crop':
    elif 'scale_width_and_crop' in preprocess_mode:
        # assert load_size, "load_size not defined"
        new_w = load_size
        new_h = load_size * h // w
    elif 'scale_height_and_crop' in preprocess_mode:
        # assert load_size, "load_size not defined"
        new_w = load_size * w // h
        new_h = load_size
    # elif preprocess_mode == 'scale_shortside_and_crop':
    elif 'scale_shortside_and_crop' in preprocess_mode:
        # assert load_size, "load_size not defined"
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)
    elif 'center_crop' in preprocess_mode:
        # assert center_crop_size, "center_crop_size not defined"
        new_w = center_crop_size
        new_h = center_crop_size
    # elif 'fixed' in preprocess_mode:
    #     aspect_ratio = opt.get('aspect_ratio')
    #     assert aspect_ratio, "aspect_ratio not defined"

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5
    rot = random.random() > 0.5
    vflip = random.random() > 0.5

    return {'load_size': load_size,
            'crop_pos': (x, y),
            'flip': flip,
            'rot': rot,
            'vflip': vflip}


# TODO: could use the hasattr to set a value for all future calls
# TODO: need something similar to use the other PIL interpolation methods
def get_default_imethod(img_type='cv2'):
    if img_type == 'pil':
        return Image.BICUBIC
    else:
        return "BICUBIC"


# TODO: change HR_size to crop_size during parsing in options.py
def get_transform(opt, params=None, grayscale=False, method=None,
                        preprocess_mode=None):
    """
    Base paired transformations: crop, scale, flip, rotate.
    There are different modes to load images by specifying 'preprocess_mode' along with
        'load_size', 'crop_size' and 'center_crop_size'. Can use options such as:
        - 'resize': resizes the images into square images of side length 'load_size'.
        - 'crop': randomly crops images to 'crop_size'.
        - 'resize_and_crop': resizes the images into square images of side length 'load_size'
            and randomly crops to 'crop_size'.
        - scale_shortside_and_crop: scales the image to have a short side of length 'load_size'
        and crops to 'crop_size' x 'crop_size' square.
        - center_crop: can be used to do an initial center crop of the images of size
            'center_crop_size' x 'center_crop_size' before other pre-processing steps.
    .... more TBD

    Rotations:
        Horizontal flips and rotations (0, 90, 180, 270 degrees).
        Note: Vertical flip and transpose are used for rotation implementation.
    """
    transform_list = []
    load_size = params['load_size'] if params else opt.get('load_size', None)
    crop_size = opt.get('crop_size')
    center_crop_size = opt.get('center_crop_size', None)
    default_none = opt.get('default_none', 'power2')
    img_type = opt.get('img_loader', 'cv2')

    if not method:
        #TODO: important: if the method does not matches the image type, the error is not
        # helpful to debug, get the image type from opt dict and assert it's not None
        method = get_default_imethod(img_type)

    preprocess_mode = opt.get('preprocess') if preprocess_mode is None else preprocess_mode
    preprocess_mode = 'none' if not preprocess_mode else preprocess_mode

    # preprocess
    if 'center_crop' in preprocess_mode:
        transform_list.append(transforms.CenterCrop(center_crop_size))

    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    # TODO:
    # elif params and params('color'):
    #     # other colorspace changes, deal with CV2 and PIL

    if 'resize' in preprocess_mode:
        if isinstance(load_size, list):
            transform_list.append(
                transforms.RandomChoice([
                    transforms.Resize([osize, osize], method) for osize in load_size
                ]))
        elif isinstance(load_size, int):
            osize = [load_size, load_size]
            transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in preprocess_mode:
        transform_list.append(transforms.Lambda(
            lambda img: scale_width(img, load_size, crop_size, method)))
    elif 'scale_height' in preprocess_mode:
        transform_list.append(transforms.Lambda(
            lambda img: scale_height(img, load_size, crop_size, method)))
    elif 'scale_shortside' in preprocess_mode:
        transform_list.append(transforms.Lambda(
            lambda img: scale_shortside(img, load_size, method)))

    # if 'crop' in preprocess_mode and preprocess_mode != 'center_crop':
    if preprocess_mode == 'crop' or 'and_crop' in preprocess_mode:
        if params is None:
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.Lambda(
                lambda img: crop(img, params['crop_pos'],
                                size=crop_size, img_type=img_type)))

    if preprocess_mode == 'fixed':
        w = crop_size
        h = round(crop_size / opt.get('aspect_ratio'))
        transform_list.append(transforms.Lambda(
            lambda img: resize(img, w, h, method)))

    if preprocess_mode == 'none':
        # no preprocessing, fix dimensions if needed
        if default_none == 'power2':
            # only make sure image has dims of power 2
            base = 4  # 32
            transform_list.append(transforms.Lambda(
                lambda img: make_power_2(img, base=base, method=method)))
        elif default_none == 'modcrop':
            # only modcrop size according to scale
            transform_list.append(transforms.Lambda(
                lambda img: modcrop(
                    img, scale=opt.get('scale'), img_type=img_type)))
        elif default_none == 'padbase':
            # only pad dims to base
            base = 4  # 32
            transform_list.append(transforms.Lambda(
                lambda img: padbase(img, base=base, img_type=img_type)))

    # paired augmentations
    if opt.get('use_flip', None):
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(
                lambda img: flip(img, params['flip'], img_type=img_type)))

    if opt.get('use_rot', None):
        if params is None:
            if random.random() < 0.5:
                #TODO: degrees=(min, max)
                transform_list.append(
                    transforms.RandomRotation(degrees=(90,90)))
        elif params['rot']:
            transform_list.append(transforms.Lambda(
                lambda img: rotate90(
                    img, params['rot'], params['vflip'], img_type=img_type)))

    #TODO: missing hrrot function or replacement

    return transforms.Compose(transform_list)


def resize(img, w, h, method=None):
    img_type = image_type(img)
    if not method:
        method = get_default_imethod(img_type)

    return transforms.Resize((h,w), interpolation=method)(img)


def scale(img, scale, method=None):
    """
    Returns a rescaled image by a specific factor given in parameter.
    A scalefactor greater than 1 expands the image, between 0 and 1 
    contracts the image.
    :param scale: The expansion factor, as a float.
    :param resample: An optional resampling filter. Same values 
        possible as in the PIL.Image.resize function.
    :returns: An :py:class:`~PIL.Image.Image` object.
    """
    if scale <= 0:
        raise ValueError("the scale factor must be greater than 0")

    if not method:
        method = get_default_imethod(image_type(img))

    ow, oh = image_size(img)
    w = int(round(scale * ow))
    h = int(round(scale * oh))
    if h == oh and w == ow:
        return img

    return resize(img, w, h, method)


def make_power_2(img, base, method=None):
    ow, oh = image_size(img)
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    if not method:
        method = get_default_imethod(image_type(img))

    print_size_warning(ow, oh, w, h, base)
    return resize(img, w, h, method)


def modcrop(img, scale, img_type=None):
    """Modulo crop images, removing the remainder of 
        dividing each dimension by a scale factor.
    Args:
        img (ndarray or Image.Image): Input image.
        scale (int): Scale factor.
        img_type (str): 'pil' or 'cv2'
    Returns:
        Mod cropped image.
    """
    if not img_type:
        img_type = image_type(img)
    ow, oh = image_size(img)
    # get the remainder in each dim
    h, w = oh % scale, ow % scale
    if h == oh and w == ow:
        return img

    print_size_warning(ow, oh, ow-w, oh-h, scale)
    if img_type == 'pil':
        return img.crop((0, 0, ow-w, oh-h))
    else:
        return img[0:oh-h, 0:ow-w, ...]


def padbase(img, base, img_type=None):
    if not img_type:
        img_type = image_type(img)
    ow, oh = image_size(img)
    ph = ((oh - 1) // base + 1) * base
    pw = ((ow - 1) // base + 1) * base
    # padding = (0, pw - ow, 0, ph - oh)
    if ph == oh and pw == ow:
        return img

    print_size_warning(ow, oh, pw, ph, base)
    if img_type == 'pil':
        # Note: with PIL if crop sizes > sizes, it adds black padding
        return img.crop((0, 0, pw, ph))
    else:
        return transforms.Pad(padding=(0, ph-oh, 0, pw-ow))(img)


def scale_width(img, target_size, crop_size, method=None):
    ow, oh = image_size(img)
    if ow == target_size and oh >= crop_size:
        return img

    if not method:
        method = get_default_imethod(image_type(img))

    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return resize(img, w, h, method)


def scale_height(img, target_size, crop_size, method=None):
    ow, oh = image_size(img)
    if oh == target_size and ow >= crop_size:
        return img

    if not method:
        method = get_default_imethod(image_type(img))

    h = target_size
    w = int(max(target_size * ow / oh, crop_size))
    return resize(img, w, h, method)


def scale_shortside(img, target_width, method=None):
    ow, oh = image_size(img)
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img

    if not method:
        method = get_default_imethod(image_type(img))

    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return resize(img, nw, nh, method)


def crop(img, pos, size, img_type=None):
    if not img_type:
        img_type = image_type(img)
    ow, oh = image_size(img)
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        if img_type == 'pil':
            return img.crop((x1, y1, x1 + tw, y1 + th))
        else:
            return img[y1:y1 + th, x1:x1 + tw, ...]
    return img


def flip(img, flip, img_type=None):
    if not img_type:
        img_type = image_type(img)

    if flip:
        if img_type == 'pil':
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return np.flip(img, axis=1) #img[:, ::-1, :]
    return img


def rotate90(img, rotate, vflip=None, img_type=None):
    if not img_type:
        img_type = image_type(img)

    if rotate:
        if vflip: #-90 degrees, else 90 degrees
            if img_type == 'pil':
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                img = np.flip(img, axis=0) #img[::-1, :, :]
        if img_type == 'pil':
            return img.transpose(Image.ROTATE_90)
        else:
            return np.rot90(img, 1) #img.transpose(1, 0, 2)

    return img


def print_size_warning(ow, oh, w, h, base=4):
    """Print warning information about image size(only print once)"""
    if not hasattr(print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of {}. "
              "The loaded image size was ({}, {}), so it was adjusted to "
              "({}, {}). This adjustment will be done to all images "
              "whose sizes are not multiples of {}.".format(base,
              ow, oh, w, h, base))
        print_size_warning.has_printed = True



#TODO: using hasattr here, but there can be cases where I
# would like to check the image type anyways
def image_type(img):
    if not hasattr(image_type, 'img_type'):
        if pil_available and isinstance(img, Image.Image):
            img_type = 'pil'
            # return 'pil'
        elif isinstance(img, np.ndarray):
            img_type = 'cv2'
            # return 'cv2'
        else:
            raise Exception("Unrecognized image type")
        image_type.img_type = img_type
        return img_type
    else:
        return image_type.img_type

def image_size(img, img_type=None):
    if not img_type:
        img_type = image_type(img)
    if img_type == 'pil':
        return img.size
    elif img_type == 'cv2':
        return (img.shape[1], img.shape[0])
    else:
        raise Exception("Unrecognized image type")

def image_channels(img, img_type=None):
    if not img_type:
        img_type = image_type(img)
    if img_type == 'pil':
        return len(img.getbands())
    elif img_type == 'cv2':
        if len(img.shape) == 2:
            return 1
        else:
            return img.shape[2]
    else:
        raise Exception("Unrecognized image type")

def pil2cv(pil_image):
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    if len(open_cv_image.shape) == 2:
        open_cv_image = fix_img_channels(open_cv_image, 1)
    return open_cv_image[:, :, ::-1].copy()

def cv2pil(open_cv_image):
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(open_cv_image)


#TODO: move to debug
def tmp_vis_pil(image):
    from dataops.debug import tmp_vis
    if isinstance(image, Image.Image):
        tmp_vis(pil2cv(image), to_np=False, rgb2bgr=False)
    else:
        tmp_vis(image, to_np=False, rgb2bgr=False)

def scale_params(params, scale):
    if scale == 1 or not params:
        return params
    scaled_params = params.copy()
    x, y = scaled_params['crop_pos']
    x_scaled, y_scaled = int(x * scale), int(y * scale)
    scaled_params['crop_pos'] = (x_scaled, y_scaled)
    if scaled_params.get('load_size', None):
        scaled_params['load_size'] = scale * scaled_params['load_size']
    return scaled_params




# TODO: only has to be done once in init and reused
def scale_opt(opt, scale):
    if scale == 1:
        return opt
    scale = 1 / scale
    scaled_opt = opt.copy()
    scaled_opt['center_crop_size'] = scaled_opt.get('center_crop_size', None)
    scaled_opt['load_size'] = scaled_opt.get('load_size', None)
    
    scaled_opt['center_crop_size'] = int(scale * scaled_opt['center_crop_size']) if scaled_opt['center_crop_size'] else None
    scaled_opt['load_size'] = int(scale * scaled_opt['load_size']) if scaled_opt['load_size'] else None
    scaled_opt['crop_size'] = int(scale * scaled_opt['crop_size']) if scaled_opt['crop_size'] else None
    return scaled_opt


def random_downscale_B(img_A, img_B, opt):
    crop_size = opt.get('crop_size')
    scale = opt.get('scale')
    default_int_method = get_default_imethod(image_type(img_B))

    # HR downscale
    if opt.get('hr_downscale', None): # and random.random() > 0.5:
        ds_algo  = opt.get('hr_downscale_types', 777)
        hr_downscale_amt  = opt.get('hr_downscale_amt', 2)
        if isinstance(hr_downscale_amt, list):
            hr_downscale_amt = random.choice(hr_downscale_amt)
        if hr_downscale_amt <= 1:
            return img_A, img_B

        w, h = image_size(img_B)  # shape 1, 0
        w_A, h_A = image_size(img_A)

        if opt.get('pre_crop', False):
            # speed up downscaling by cropping first
            pc = PreCrop(img_A, img_B, scale, int(hr_downscale_amt*crop_size))
            img_Al, img_Bl = pc(img_A, img_B)
            img_A, img_B = img_Al[0], img_Bl[0]
            w, h = image_size(img_B)
            w_A, h_A = image_size(img_A)

        # will ignore if 1 or if result is smaller than crop size
        if hr_downscale_amt > 1 and h//hr_downscale_amt >= crop_size and w//hr_downscale_amt >= crop_size:
            if opt.get('img_loader', None) == 'pil':
                # TODO: simple solution for PIL, but limited
                img_B = transforms.Resize(
                    (int(h//hr_downscale_amt), int(w//hr_downscale_amt)), 
                    interpolation=default_int_method)(img_B)
            else:
                img_B, _ = Scale(img=img_B, scale=hr_downscale_amt, algo=ds_algo)
            img_B = make_power_2(img_B, base=4*scale)
            # Downscales LR to match new size of HR if scale does not match after
            w, h = image_size(img_B)
            if img_A is not None:  # and (h // scale != h_A or w // scale != w_A):
                # TODO: simple solution for PIL, but limited
                if opt.get('img_loader', None) == 'pil':
                    img_A = transforms.Resize(
                        # (int(h//scale), int(w//scale)),
                        (int(h_A//hr_downscale_amt), int(w_A//hr_downscale_amt)),
                        interpolation=default_int_method)(img_A)
                else:
                    img_A, _ = Scale(img=img_A, scale=hr_downscale_amt, algo=ds_algo)
                img_A = make_power_2(img_A, base=4*scale)
            
                # fix cases with rounding errors with some scales
                w_A, h_A = image_size(img_A)
                if abs(h_A*scale - h) <= 8*scale or abs(w_A*scale - w) <= 8*scale:
                    img_A = transforms.Resize((h//scale, w//scale), 
                            interpolation=default_int_method)(img_A)

    return img_A, img_B

def shape_change_fn(img_A, img_B, opt, scale, default_int_method):
    """Fix the images shapes in respect to each other
    """
    #TODO: test to be removed
    # w, h = image_size(img_B)
    # img_A = transforms.Resize((int(h/(2*scale)), int(w/(scale))), interpolation=default_int_method)(img_A)

    # Check that HR and LR have the same dimensions ratio, else use an option to process
    #TODO: add 'shape_change' options variable
    w, h = image_size(img_B)
    w_A, h_A = image_size(img_A)
    if h//h_A != w//w_A:
        # make B power2 or padbase before calculating, else dimensions won't fit
        img_B = make_power_2(img_B, base=scale)
        w, h = image_size(img_B)

        shape_change = opt.get('shape_change', 'reshape_lr')
        if shape_change == "reshape_lr":
            img_A = transforms.Resize((int(h/scale), int(w/scale)), interpolation=default_int_method)(img_A)
        elif shape_change == "reshape_hr":
            # for unaligned A-B pairs, forcing both to have the correct scale
            nh = h*(2*w_A)//(h_A+w_A)
            nw = w*(2*h_A)//(h_A+w_A)
            nh = ((nh - 1) // scale + 1) * scale
            nw = ((nw - 1) // scale + 1) * scale
            img_B = transforms.Resize((nh,nw), interpolation=default_int_method)(img_B)
            w, h = image_size(img_B)
            img_A = transforms.Resize((int(h/scale), int(w/scale)), interpolation=default_int_method)(img_A)
        else:
            # generate new A from B
            img_A = img_B

    # fix LR if at wrong scale (should be 1 or scale at this point)
    w, h = image_size(img_B)
    w_A, h_A = image_size(img_A)
    if not (h//h_A == scale or w//w_A == scale) and not (h//h_A == 1 or w//w_A == 1):
        img_A = transforms.Resize((int(h/scale), int(w/scale)),
                        interpolation=default_int_method)(img_A)

    return img_A, img_B

def dim_change_fn(img_A, img_B, opt, scale, default_int_method,
        crop_size, A_crop_size, ds_kernels):
    """Fix the images dimensions if smaller than crop sizes
    """
    #TODO: test to be removed
    # w, h = image_size(img_B)
    # w_A, h_A = image_size(img_A)
    # img_B = transforms.Resize((crop_size-10,w), interpolation=default_int_method)(img_B)
    # img_A = transforms.Resize(((A_crop_size)-(10//scale),w_A), interpolation=default_int_method)(img_A)
    # # img_A = img_B

    w, h = image_size(img_B)
    w_A, h_A = image_size(img_A)
    # Or if the images are too small, pad images or Resize B to the crop_size size and fit A pair to A_crop_size
    #TODO: add 'dim_change' options variable
    dim_change = opt.get('dim_change', 'pad')
    if h < crop_size or w < crop_size:
        if dim_change == "resize":
            # rescale B image to the crop_size
            img_B = transforms.Resize((crop_size, crop_size), interpolation=default_int_method)(img_B)
            # rescale B image to the B_size (The original code discarded the img_A and generated a new one on the fly from img_B)
            img_A = transforms.Resize((A_crop_size, A_crop_size), interpolation=default_int_method)(img_A)
        elif dim_change == "pad":
            # if img_A is img_B, padding will be wrong, downscaling LR before padding
            if scale != 1 and h_A == h and w_A == w:
                ds_algo = 777 # default to matlab-like bicubic downscale
                if opt.get('lr_downscale', None): # if manually set and scale algorithms are provided, then:
                    ds_algo  = opt.get('lr_downscale_types', 777)
                if opt.get('lr_downscale', None) and opt.get('dataroot_kernels', None) and 999 in opt["lr_downscale_types"]:
                    ds_kernel = ds_kernels
                else:
                    ds_kernel = None
                img_A, _ = Scale(img=img_A, scale=scale, algo=ds_algo, ds_kernel=ds_kernel)
            elif h_A != (-(-h // scale)) or w_A != (-(-w // scale)):
                img_A = transforms.Resize((h//scale, w//scale), interpolation=default_int_method)(img_A)

            HR_pad, fill = get_pad(img_B, crop_size, fill='random', padding_mode=opt.get('pad_mode', 'constant'))
            img_B = HR_pad(img_B)

            LR_pad, _ = get_pad(img_A, A_crop_size, fill=fill, padding_mode=opt.get('pad_mode', 'constant'))
            img_A = LR_pad(img_A)

    return img_A, img_B

# TODO: could use in paired_imgs_check() instead of inside the functions?
class PreCrop:
    def __init__(self, img_A, img_B, scale, B_crop_size):
        w_B, h_B = image_size(img_B)
        w_A, h_A = image_size(img_A)
        if w_B == w_A and h_B == h_A:
            scale = 1
            A_crop_size = B_crop_size
        else:
            ims = w_B // w_A
            scale = ims if ims != scale else scale
            A_crop_size = B_crop_size // scale
        
        x_A = random.randint(0, max(0, w_A - A_crop_size))
        y_A = random.randint(0, max(0, h_A - A_crop_size))
        
        x_B, y_B = int(x_A * scale), int(y_A * scale)

        self.crop_params = {'pos_A': (x_A, y_A),
                            'A_crop_size': A_crop_size,
                            'pos_B': (x_B, y_B),
                            'B_crop_size': B_crop_size}

    def __call__(self, img_A=None, img_B=None):
        B_rlt = []
        A_rlt = []
        
        if img_B is not None:
            if not isinstance(img_B, list):
                img_B = [img_B]
            for B in img_B:
                B = crop(B, self.crop_params['pos_B'], 
                        self.crop_params['B_crop_size'], img_type=None)
                B_rlt.append(B)

        if img_A is not None:
            if not isinstance(img_A, list):
                img_A = [img_A]
            for A in img_A:
                A = crop(A, self.crop_params['pos_A'], 
                        self.crop_params['A_crop_size'], img_type=None)
                A_rlt.append(A)
        return A_rlt, B_rlt


def generate_A_fn(img_A, img_B, opt, scale, default_int_method,
        crop_size, A_crop_size, ds_kernels):
    """ Generate A (from B if needed) during training if:
        - dataset A is not provided (img_A = img_B)
        - dataset A is not in the correct scale
        - Also to check if A is not at the correct scale already (ie. if img_A was changed to img_B)
    """
    #TODO: test to be removed
    # img_A = img_B

    w, h = image_size(img_B)
    w_A, h_A = image_size(img_A)
    
    # TODO: validate, to fix potential cases with rounding errors (hr_downscale, etc), should not be needed
    # if abs(h_A*scale - h) <= 40 or abs(w_A*scale - w) <= 40:
    #     img_A = transforms.Resize((h//scale, w//scale), 
    #             interpolation=default_int_method)(img_A)
    #     w_A, h_A = image_size(img_A)

    # if h_A != A_crop_size or w_A != A_crop_size:
    if h_A != h//scale or w_A != w//scale:
        #TODO: make B and A power2 or padbase before calculating, else dimensions won't fit
        img_B = make_power_2(img_B, base=scale)
        img_A = make_power_2(img_A, base=scale)

        if opt.get('pre_crop', False):
            # speed up A downscaling by cropping first
            pc = PreCrop(img_A, img_B, scale, crop_size)
            img_Al, img_Bl = pc(img_A, img_B)
            img_A, img_B = img_Al[0], img_Bl[0]

        w, h = image_size(img_B)
        ds_algo = 777  # default to matlab-like bicubic downscale
        if opt.get('lr_downscale', None): # if manually set and scale algorithms are provided, then:
            ds_algo  = opt.get('lr_downscale_types', 777)
        else: # else, if for some reason img_A is too large, default to matlab-like bicubic downscale
            #if not opt['aug_downscale']: #only print the warning if not being forced to use HR images instead of LR dataset (which is a known case)
            # print("LR image is too large, auto generating new LR for: ", LR_path)
            print("LR image is too large, auto generating new LR")
        if opt.get('lr_downscale', None) and opt.get('dataroot_kernels', None) and 999 in opt["lr_downscale_types"]:
            ds_kernel = ds_kernels #KernelDownscale(scale, kernel_paths, num_kernel)
        else:
            ds_kernel = None
        if opt.get('img_loader', None) == 'pil':
            # TODO: simple solution for PIL, but limited
            img_A = transforms.Resize(
                (h//scale, w//scale), interpolation=default_int_method)(img_A)
        else:
            img_A, _ = Scale(img=img_A, scale=scale, algo=ds_algo, ds_kernel=ds_kernel)

    return img_A, img_B

def get_ds_kernels(opt):
    """ kernelGAN estimated kernels """
    if opt.get('dataroot_kernels', None) and 999 in opt["lr_downscale_types"]:
        ds_kernels = KernelDownscale(scale=opt.get('scale', 4), kernel_paths=opt['dataroot_kernels'])
    else:
        ds_kernels = None

    return ds_kernels

def get_noise_patches(opt):
    if opt['phase'] == 'train' and opt.get('lr_noise_types', 3) and "patches" in opt.get('lr_noise_types', {}):
        assert opt['noise_data']
        # noise_patches = NoisePatches(opt['noise_data'], opt.get('HR_size', 128)/opt.get('scale', 4))
        noise_patches = NoisePatches(opt['noise_data'], opt.get('noise_data_size', 256))
    else:
        noise_patches = None

    return noise_patches

def paired_imgs_check(img_A, img_B, opt, ds_kernels=None):
    crop_size = opt.get('crop_size')
    scale = opt.get('scale')
    A_crop_size = crop_size//scale
    default_int_method = get_default_imethod(image_type(img_A))

    # Validate there's an img_A, if not, use img_B
    if img_A is None:
        img_A = img_B
        # print("Image LR: ", LR_path, ("was not loaded correctly, using HR pair to downscale on the fly."))
        print("Image was not loaded correctly, using pair to generate on the fly.")

    img_A, img_B = shape_change_fn(
        img_A, img_B, opt, scale, default_int_method)

    img_A, img_B = dim_change_fn(
        img_A, img_B, opt, scale, default_int_method,
        crop_size, A_crop_size, ds_kernels)

    img_A = generate_A_fn(
        img_A, img_B, opt, scale, default_int_method, ds_kernels)

    return img_A, img_B

def get_unpaired_params(opt):  #get_augparams
    # below are the On The Fly augmentations
    lr_augs = {}
    hr_augs = {}

    # Apply "auto levels" to images
    auto_levels = opt.get('auto_levels', None)  # LR, HR or both
    if auto_levels:
        rand_levels = opt.get('rand_auto_levels', None)  # probability
        if rand_levels:
            if auto_levels.lower() == 'lr':
                lr_augs['auto_levels'] = rand_levels
            elif auto_levels.lower() == 'hr':
                hr_augs['auto_levels'] = rand_levels
            elif auto_levels.lower() == 'both':
                lr_augs['auto_levels'] = rand_levels
                hr_augs['auto_levels'] = rand_levels

    # Apply unsharpening mask to images
    hr_unsharp_mask = opt.get('hr_unsharp_mask', None)  # true | false
    if hr_unsharp_mask:
        hr_rand_unsharp = opt.get('hr_rand_unsharp', None)  # probability
        if hr_rand_unsharp:
            hr_augs['unsharp_mask'] = hr_rand_unsharp

    lr_unsharp_mask = opt.get('lr_unsharp_mask', None)  # true | false
    if lr_unsharp_mask:
        lr_rand_unsharp = opt.get('lr_rand_unsharp', None)  # probability
        if lr_rand_unsharp:
            lr_augs['unsharp_mask'] = lr_rand_unsharp

    # Create color fringes
    # Caution: Can easily destabilize a model
    # Only applied to a small % of the images. Around 20% and 50% appears to be stable.
    # Note: this one does not have a transforms class
    lr_fringes = opt.get('lr_fringes', None)  # true | false
    if lr_fringes:
        lr_augs['fringes'] = opt.get('lr_fringes_chance', 0.4)  # probability

    # Add blur if blur AND blur types are provided
    lr_blur = opt.get('lr_blur', None)  # true | false
    if lr_blur:
        blur_types = opt.get('lr_blur_types', None)  # blur types
        if isinstance(blur_types, list):
            lr_augs['blur'] = [random.choice(blur_types)]

    # Add HR noise if enabled AND noise types are provided (for noise2noise and similar models)
    hr_noise = opt.get('hr_noise', None)  # true | false
    if hr_noise:
        noise_types = opt.get('hr_noise_types', None)  # noise types
        if isinstance(noise_types, list):
            hr_augs['noise'] = [random.choice(noise_types)]

    # LR primary noise: Add noise to LR if enabled AND noise types are provided
    lr_noise = opt.get('lr_noise', None)  # true | false
    if lr_noise:
        noise_types = opt.get('lr_noise_types', None)  # noise types
        if isinstance(noise_types, list):
            lr_augs['noise'] = [random.choice(noise_types)]

    # LR secondary noise: Add additional noise to LR if enabled AND noise types are provided, else will skip
    lr_noise2 = opt.get('lr_noise2', None)  # true | false
    if lr_noise2:
        noise_types = opt.get('lr_noise_types2', None)  # noise types
        if isinstance(noise_types, list):
            lr_augs['noise2'] = [random.choice(noise_types)]

    #v LR cutout / LR random erasing (for inpainting/classification tests)
    lr_cutout = opt.get('lr_cutout', None)
    lr_erasing = opt.get('lr_erasing', None)

    if lr_cutout and not lr_erasing:
        lr_augs['cutout'] = opt.get('lr_cutout_p', 1)
    elif lr_erasing and not lr_cutout:
        lr_augs['erasing'] = opt.get('lr_erasing_p', 1)
    elif lr_cutout and lr_erasing:
        # only do cutout or erasing, not both at the same time
        if random.random() > 0.5:
            lr_augs['cutout'] = opt.get('lr_cutout_p', 1)
        else:
            lr_augs['erasing'] = opt.get('lr_erasing_p', 1)

    # label the augmentation lists
    if lr_augs:
        lr_augs['kind'] = 'lr'

    if hr_augs:
        hr_augs['kind'] = 'hr'

    return lr_augs, hr_augs

def get_augmentations(opt, params=None, noise_patches=None):
    """ unpaired augmentations
    Note: This a good point to convert PIL images to CV2
        for the augmentations, can make an equivalent for
        CV2 to PIL if needed.
    """

    loader = set_transforms.loader_type
    set_transforms(loader_type='cv2')

    transform_list = []
    crop_size = opt.get('crop_size')
    if params and params['kind'] == 'lr':
        crop_size = crop_size // opt.get('scale', 1)

    # "auto levels"
    if 'auto_levels' in params:
        transform_list.append(transforms.FilterColorBalance(
            p=params['auto_levels'], percent=10, random_params=True))

    # unsharpening mask
    if 'unsharp_mask' in params:
        transform_list.append(transforms.FilterUnsharp(
            p=params['unsharp_mask']))

    # color fringes
    if 'fringes' in params:
        if random.random() > (1.- params['fringes']):
            transform_list.append(transforms.Lambda(
                lambda img: translate_chan(img)))

    # blur
    if 'blur' in params:
        blur_func = get_blur(params['blur'])
        if blur_func:
            transform_list.append(blur_func)

    # primary noise
    if 'noise' in params:
        noise_func = get_noise(params['noise'], noise_patches)
        if noise_func:
            transform_list.append(noise_func)

    # secondary noise
    if 'noise2' in params:
        noise_func = get_noise(params['noise2'], noise_patches)
        if noise_func:
            transform_list.append(noise_func)

    # cutout
    if 'cutout' in params:
        transform_list.append(transforms.Cutout(
            p=params['cutout'], mask_size=crop_size//2))

    # random erasing
    if 'erasing' in params:
        # transform_list.append(transforms.RandomErasing(p=params['erasing']))  # mode=[3]. With lambda?
        transform_list.append(transforms.Lambda(
                lambda img: transforms.RandomErasing(
                    p=params['erasing'])(img, mode=[3])))

    set_transforms(loader_type=loader)
    return transforms.Compose(transform_list)


# Note: these don't change, can be fixed from the dataloader init
def get_totensor_params(opt):
    params = {}

    params['znorm'] = opt.get('znorm', False)

    loader = opt.get('img_loader', 'cv2')
    if loader == 'pil':
        # for PIL the default is torchvision.transforms
        method = opt.get('toTensor_method', 'transforms')
    else:
        # for CV2 the default is np2tensor
        method = opt.get('toTensor_method', None)
    params['method'] = method

    # only required for 'transforms' normalization
    mean = opt.get('normalization_mean', None)
    if mean:
        params['mean'] = mean

    std = opt.get('normalization_std', None)
    if std:
        params['std'] = std

    params['normalize_first'] = opt.get('normalize_first', None)

    return params

# TODO: use better name for the function, normalize can be used with np images too
# Note: Alt normalization: transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
def get_totensor(opt, params=None, toTensor=True, grayscale=False, normalize=False):
    """ convert to tensor and/or normalize if needed"""
    transform_list = []

    if params['method'] == 'transforms':
        # to tensor
        if toTensor:
            transform_list += [transforms.ToTensor()]

        if params['znorm'] or normalize:
            # Default znorm will normalize the image in the range [-1,1].
            if grayscale:
                mean = params['mean'] if 'mean' in params else (0.5,)
                std = params['std'] if 'std' in params else (0.5,)
            else:
                mean = params['mean'] if 'mean' in params else (0.5, 0.5, 0.5)
                std = params['std'] if 'std' in params else (0.5, 0.5, 0.5)
            if params['normalize_first']:
                transform_list[0:0] = [transforms.Normalize(mean=mean, std=std)]
            else:
                transform_list += [transforms.Normalize(mean=mean, std=std)]
    else:
        transform_list.append(transforms.Lambda(
                lambda img: np2tensor(img, bgr2rgb=True,
                            normalize=params['znorm'], add_batch=False)))

    return transforms.Compose(transform_list)


def custom_pipeline(custom_transforms, transforms_cfg, noise_patches=None):
    #TODO: here I can add the get_noise() and get_blur() functions as
    # options to get random noise and blur added. It's a serial pipeline,
    # but can add those as parallel randomization too. And those could
    # also be used multiple times to have multiple parallel randomizations
    # (ie noise types or simple/complex/real motion blur kernels)

    transform_list = []
    for transform in custom_transforms:
        # auto levels
        if transform == 'auto_levels':
            transform_list.append(transforms.FilterColorBalance(
                p=transforms_cfg.get('auto_levels_p', 0.5),
                percent=10, random_params=True))

        # unsharpening mask
        elif transform == 'unsharp_mask':
            transform_list.append(transforms.FilterUnsharp(
                p=transforms_cfg.get('unsharp_mask_p', 0.5)))

        # color fringes
        elif transform == 'fringes':
            if random.random() > (1.- transforms_cfg.get('fringes_p', 0.5)):
                transform_list.append(transforms.Lambda(
                    lambda img: translate_chan(img)))

        # blur
        elif 'blur' in transform:
            if transform == 'averageblur':
                # print('average')
                transform_list.append(transforms.RandomAverageBlur(
                    p=transforms_cfg.get('averageblur_p', 0.5),
                    max_kernel_size=11, random_params=True))
            elif transform == 'boxblur':
                # print('box')
                transform_list.append(transforms.RandomBoxBlur(
                    p=transforms_cfg.get('boxblur_p', 0.5),
                    max_kernel_size=11, random_params=True))
            elif transform == 'gaussianblur':
                # print('gaussian')
                transform_list.append(transforms.RandomGaussianBlur(
                    p=transforms_cfg.get('gaussianblur_p', 0.5),
                    max_kernel_size=11, random_params=True))
            elif transform == 'medianblur':
                # print('median')
                transform_list.append(transforms.RandomMedianBlur(
                    p=transforms_cfg.get('medianblur_p', 0.5),
                    max_kernel_size=11, random_params=True))
            elif transform == 'bilateralblur':
                # print('bilateral')
                transform_list.append(transforms.RandomBilateralBlur(
                    p=transforms_cfg.get('bilateralblur_p', 0.5),
                    sigmaSpace=200, sigmaColor=200,
                    max_kernel_size=11, random_params=True))
            elif transform == 'motionblur':
                # print('motion')
                transform_list.append(transforms.RandomMotionBlur(
                    p=transforms_cfg.get('motionblur_p', 0.5),
                    max_kernel_size=7, random_params=True))
            elif transform == 'complexmotionblur':
                # print('complexmotion')
                transform_list.append(transforms.RandomComplexMotionBlur(
                    p=transforms_cfg.get('motionblur_p', 0.5),
                    max_kernel_size=7, random_params=True,
                    size=100, complexity=1.0))

        # noise
        elif 'dither' in transform:
            #TODO: need a dither type selector, there are multiple options including b&w dithers
            # if dither == 'fs':
            if ('fs' in transform and 'bw' not in transform) or transform == 'dither':
                transform_list.append(transforms.FSDitherNoise(
                    p=transforms_cfg.get('dither_p', 0.5)))
            # elif dither == 'bayer':
            elif 'bayer' in transform and 'bw' not in transform:
                transform_list.append(transforms.BayerDitherNoise(
                    p=transforms_cfg.get('dither_p', 0.5)))
            # elif dither == 'fs_bw':
            elif 'fs_bw' in transform:
                transform_list.append(transforms.FSBWDitherNoise(
                    p=transforms_cfg.get('dither_p', 0.5)))
            # elif dither == 'avg_bw':
            elif 'avg_bw' in transform:
                transform_list.append(transforms.AverageBWDitherNoise(
                    p=transforms_cfg.get('dither_p', 0.5)))
            # elif dither == 'bayer_bw':
            elif 'bayer_bw' in transform:
                transform_list.append(transforms.BayerBWDitherNoise(
                    p=transforms_cfg.get('dither_p', 0.5)))
            # elif dither == 'bin_bw':
            elif 'bin_bw' in transform:
                transform_list.append(transforms.BinBWDitherNoise(
                    p=transforms_cfg.get('dither_p', 0.5)))
            # elif dither == 'rnd_bw':
            elif 'rnd_bw' in transform:
                transform_list.append(transforms.RandomBWDitherNoise(
                    p=transforms_cfg.get('dither_p', 0.5)))
            # print("dither")
        elif transform == 'simplequantize':
            #TODO: find a useful rgb_range for SimpleQuantize in [0,255]
            # the smaller the value, the closer to original colors
            transform_list.append(transforms.SimpleQuantize(
                p=transforms_cfg.get('squantize_p', 0.5),
                rgb_range = 50)) #30
            # print("simplequantize")
        elif transform == 'quantize':
            transform_list.append(RandomQuantize(
                p=transforms_cfg.get('quantize_p', 0.5), num_colors=32))
            # print("quantize")
        elif transform == 'km_quantize':
            transform_list.append(transforms.RandomQuantize(
                p=transforms_cfg.get('km_quantize_p', 0.5), num_colors=32))
        elif transform == 'gaussian':
            transform_list.append(transforms.RandomGaussianNoise(
                p=transforms_cfg.get('gaussian_p', 0.5),
                random_params=True, gtype='bw'))
            # print("gaussian")
        elif transform.lower() == 'jpeg':
            transform_list.append(transforms.RandomCompression(
                p=transforms_cfg.get('compression_p', 0.5),
                random_params=True, image_type='.jpeg'))
            # print("JPEG")
        elif transform.lower() == 'webp':
            transform_list.append(transforms.RandomCompression(
                p=transforms_cfg.get('compression_p', 0.5),
                random_params=True, image_type='.webp'))
        elif transform == 'poisson':
            transform_list.append(transforms.RandomPoissonNoise(
                p=transforms_cfg.get('poisson_p', 0.5)))
            # print("poisson")
        elif transform == 's&p':
            transform_list.append(transforms.RandomSPNoise(
                p=transforms_cfg.get('s&p_p', 0.5)))
            # print("s&p")
        elif transform == 'speckle':
            transform_list.append(transforms.RandomSpeckleNoise(
                p=transforms_cfg.get('speckle_p', 0.5), gtype='bw'))
            # print("speckle")
        elif transform == 'maxrgb':
            transform_list.append(transforms.FilterMaxRGB(
                p=transforms_cfg.get('maxrgb_p', 0.5)))
            # print("maxrgb")
        elif transform == 'canny':
            transform_list.append(transforms.FilterCanny(
                p=transforms_cfg.get('canny_p', 0.5),
                # bin_thresh=transforms_cfg.get('canny_bin_thresh', True),
                # threshold=transforms_cfg.get('canny_threshold', 127)
                ))
        elif transform == 'clahe':
            transform_list.append(transforms.CLAHE(
                p=transforms_cfg.get('clahe_p', 0.5)))
        #TODO: needs transforms function
        elif transform == 'patches' and noise_patches:
            if random.random() > (1.- transforms_cfg.get('patches_p', 0.5)):
                transform_list.append(RandomNoisePatches(
                    noise_patches,
                    noise_amp=transforms_cfg.get('noise_amp', 1)))
        elif transform == 'superpixels':
            transform_list.append(transforms.Superpixels(
                p=transforms_cfg.get('superpixels_p', 0.5),
                n_segments=32,
                p_replace=1.0,
                max_size=None))

        # cutout
        elif transform == 'cutout':
            transform_list.append(transforms.Cutout(
                p=transforms_cfg.get('cutout_p', 0.5),
                mask_size=crop_size//2))

        # random erasing
        elif transform == 'erasing':
            # transform_list.append(transforms.RandomErasing(p=params['erasing']))  # mode=[3]. With lambda?
            transform_list.append(transforms.Lambda(
                    lambda img: transforms.RandomErasing(
                        p=transforms_cfg.get('erasing_p', 0.5))(img, mode=[3])))

    return transforms.Compose(transform_list)




@wrap_cv2_function
def apply_wrapped_cv2transform(img, transform):
    loader = set_transforms.loader_type
    set_transforms(loader_type='cv2')
    img = transform(img)
    set_transforms(loader_type=loader)
    return img

@wrap_pil_function
def apply_wrapped_piltransform(img, transform):
    loader = set_transforms.loader_type
    set_transforms(loader_type='pil')
    img = transform(img)
    set_transforms(loader_type=loader)
    return img

def apply_transform_list(img_list: list, transform, wrapper=None, loader=None):
    img_list_out = []
    for img in img_list:
        if wrapper == 'cv2' and wrapper != loader:
            tr_img = apply_wrapped_cv2transform(img, transform)
        elif wrapper == 'pil' and wrapper != loader:
            tr_img = apply_wrapped_piltransform(img, transform)
        else:
            tr_img = transform(img)
        img_list_out.append(tr_img)
    return img_list_out








########### below is the original augmentations file




IMAGE_EXTENSIONS = ['.png', '.jpg']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMAGE_EXTENSIONS)

def _get_paths_from_dir(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    image_list = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                image_path = os.path.join(dirpath, fname)
                image_list.append(image_path)
    assert image_list, '{:s} has no valid image file'.format(path)
    return image_list

def horizontal_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, ::-1, :]
    return image

def vertical_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[::-1, :, :]
    return image

def random_crop(image, crop_size=(224, 224)): 
    h, w, _ = image.shape
    
    if h > crop_size[0] and w > crop_size[1]:
        top = np.random.randint(0, h - crop_size[0])
        left = np.random.randint(0, w - crop_size[1])

        bottom = top + crop_size[0]
        right = left + crop_size[1]

        image = image[top:bottom, left:right, :]
    else:
        image, _ = resize_img(image, crop_size, algo=[4])
    
    return image
    
def random_crop_pairs(img_HR, img_LR, HR_size, scale):
    H, W, C = img_LR.shape
    
    LR_size = int(HR_size/scale)
    
    # randomly crop
    rnd_h = random.randint(0, max(0, H - LR_size))
    rnd_w = random.randint(0, max(0, W - LR_size))
    #print ("LR rnd: ",rnd_h, rnd_w)
    img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, ...]
    rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
    #print ("HR rnd: ",rnd_h_HR, rnd_w_HR)
    img_HR = img_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, ...]
    
    return img_HR, img_LR

def cutout(image_origin, mask_size, p=0.5):
    if np.random.rand() > p:
        return image_origin
    image = np.copy(image_origin)
    mask_value = image.mean()

    h, w, _ = image.shape
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    if top < 0:
        top = 0
    if left < 0:
        left = 0

    image[top:bottom, left:right, :].fill(mask_value)
    return image

def random_erasing(image_origin, p=0.5, s=(0.02, 0.4), r=(0.3, 3), modes=[0,1,2]): # erasing probability p, the area ratio range of erasing region sl and sh, and the aspect ratio range of erasing region r1 and r2.
    if np.random.rand() > p:
        return image_origin
    image = np.copy(image_origin)

    h, w, _ = image.shape
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])

    mask_aspect_ratio = np.random.rand() * r[1] + r[0]

    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1

    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width
    
    mode = random.choice(modes) #mode 0 fills with a random number, mode 1 fills with ImageNet mean values, mode 2 fills with random pixel values (noise)
    
    if mode == 0: # original code 
        mask_value = np.random.uniform(0., 1.)
        image[top:bottom, left:right, :].fill(mask_value) 
    elif mode == 1: # use ImageNet mean pixel values for each channel 
        if image.shape[2] == 3:
            mean=[0.4465, 0.4822, 0.4914] #OpenCV follows BGR convention and PIL follows RGB color convention
            image[top:bottom, left:right, 0] = mean[0]
            image[top:bottom, left:right, 1] = mean[1]
            image[top:bottom, left:right, 2] = mean[2]
        else: 
            mean=[0.4914]
            image[top:bottom, left:right, :].fill(mean[0])
    else: # replace with random pixel values (noise) (With the selected erasing region Ie, each pixel in Ie is assigned to a random value in [0, 1], respectively.)
        image[top:bottom, left:right, :] = np.random.rand(mask_height, mask_width, image.shape[2])
    return image

# scale image
def scale_img(image, scale, algo=None):
    h, w = image.shape[0], image.shape[1]
    newdim = (int(w/scale), int(h/scale))
    
    # randomly use OpenCV2 algorithms if none are provided
    if algo is None:
        scale_algos = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT] #scaling interpolation options
        interpol = random.choice(scale_algos)
        resized = cv2.resize(image, newdim, interpolation = interpol)
    # using matlab imresize
    else:
        if isinstance(algo, list):
            interpol = random.choice(algo)
        elif isinstance(algo, int):
            interpol = algo
        
        if interpol == 777: #'matlab_bicubic'
            resized = imresize(image, 1 / scale, antialiasing=True)
            # force to 3 channels
            # if resized.ndim == 2:
                # resized = np.expand_dims(resized, axis=2)
        else:
            # use the provided OpenCV2 algorithms
            resized = cv2.resize(image, newdim, interpolation = interpol)
    
    #resized = np.clip(resized, 0, 1)
    return resized, interpol

# resize image to a defined size 
def resize_img(image, newdim=(128, 128), algo=None):
    if algo is None:
        scale_algos = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT] #scaling interpolation options
        interpol = random.choice(scale_algos)
    else:
        if isinstance(algo, list):
            interpol = random.choice(algo)
        elif isinstance(algo, int):
            interpol = algo
            
    if interpol == 777: #'matlab_bicubic'
        resized = imresize(image, 1 / scale, antialiasing=True)
        # force to 3 channels
        # if resized.ndim == 2:
            # resized = np.expand_dims(resized, axis=2)
    else:
        # use the provided OpenCV2 algorithms
        resized = cv2.resize(image, newdim, interpolation = interpol)
    
    return resized, interpol

def random_resize_img(img_LR, crop_size=(128, 128), new_scale=0, algo=None):
    mu, sigma = 0, 0.3
    if new_scale == 0:
        new_scale = np.random.uniform(-0.5, 0.) #resize randomly from 0 to 2x
        scale = 1+new_scale
    else:
        scale = new_scale
    
    if algo is None:
        scale_algos = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT] #scaling interpolation options
        interpol = random.choice(scale_algos)
    else:
        interpol = random.choice(algo)
    
    img_LR, interpol = scale_img(img_LR, scale, interpol) 
    img_LR = random_crop(img_LR, crop_size)
    
    return img_LR, interpol

def rotate(image, angle, center=None, scale=1.0, border_value=0, auto_bound=False):
    """Rotate an image.
    Args:
        image (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple): Center of the rotation in the source image, by default
            it is the center of the image.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        auto_bound (bool): Whether to increase the image size to contain the 
            whole rotated image.
    Returns:
        ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    #(h, w) = image.shape[:2]
    h, w = image.shape[:2]
    if center is None:
        #center = (w // 2, h // 2)
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)
     
    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))

    #rotated = cv2.warpAffine(image, matrix, (w, h))
    rotated = cv2.warpAffine(image, matrix, (w, h), borderValue=border_value)

    return rotated

def random_rotate(image, angle=0, center=None, scale=1.0):    
    if angle == 0:
        angle = int(np.random.uniform(-45, 45))
    
    h,w,c = image.shape
    
    if np.random.rand() > 0.5: #randomly upscaling 2x like HD Mode7 to reduce jaggy lines after rotating, more accurate underlying "sub-pixel" data. cv2.INTER_LANCZOS4?
        image, _ = scale_img(image, 1/2, cv2.INTER_CUBIC)
    
    rotated = rotate(image, angle) #will change image shape unless auto_bound is used
    rotated = random_crop(image, crop_size=(w, h)) #crop back to the original size
    return rotated, angle 

def random_rotate_pairs(img_HR, img_LR, HR_size, scale, angle=0, center=0):
    if angle == 0:
        angle = int(np.random.uniform(-45, 45))
    
    if np.random.rand() > 0.5: #randomly upscaling 2x like HD Mode7 to reduce jaggy lines after rotating, more accurate underlying "sub-pixel" data. cv2.INTER_LANCZOS4?
        img_HR, _ = scale_img(img_HR, 1/2, cv2.INTER_CUBIC) 
        img_LR, _ = scale_img(img_LR, 1/2, cv2.INTER_CUBIC) 
    
    img_HR = rotate(img_HR, angle) #will change image shape if auto_bound is used    
    img_LR = rotate(img_LR, angle) #will change image shape if auto_bound is used
    
    img_HR, img_LR = random_crop_pairs(img_HR, img_LR, HR_size, scale) #crop back to the original size if needed 
    return img_HR, img_LR

def blur_img(img_LR, blur_algos=['clean'], kernel_size = 0):
    h,w,c = img_LR.shape
    blur_type = random.choice(blur_algos)
    
    if blur_type == 'average':
        #Averaging Filter Blur (Homogeneous filter)
        if kernel_size == 0:
            kernel_size = int(np.random.uniform(3, 11))
        if kernel_size > h:
            kernel_size = int(np.random.uniform(3, h/2))
        
        kernel_size = int(np.ceil(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size+=1 
        
        blurred = cv2.blur(img_LR,(kernel_size,kernel_size))
        
    elif blur_type == 'box':
        #Box Filter Blur 
        if kernel_size == 0:
            kernel_size = int(np.random.uniform(3, 11))
        if kernel_size > h:
            kernel_size = int(np.random.uniform(3, h/2))
            
        kernel_size = int(np.ceil(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size+=1 
            
        blurred = cv2.boxFilter(img_LR,-1,(kernel_size,kernel_size))
    
    elif blur_type == 'gaussian':
        #Gaussian Filter Blur
        if kernel_size == 0:
            kernel_size = int(np.random.uniform(3, 11))
        if kernel_size > h:
            kernel_size = int(np.random.uniform(3, h/2))
        
        kernel_size = int(np.ceil(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size+=1 
            
        blurred = cv2.GaussianBlur(img_LR,(kernel_size,kernel_size),0)
    
    elif blur_type == 'bilateral':
        #Bilateral Filter
        sigma_bil = int(np.random.uniform(20, 120))
        if kernel_size == 0:
            kernel_size = int(np.random.uniform(3, 9))
        if kernel_size > h:
            kernel_size = int(np.random.uniform(3, h/2))
        
        kernel_size = int(np.ceil(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size+=1 
        
        blurred = cv2.bilateralFilter(img_LR,kernel_size,sigma_bil,sigma_bil)

    elif blur_type == 'clean': # Pass clean image, without blur
        blurred = img_LR

    #img_LR = np.clip(blurred, 0, 1) 
    return blurred, blur_type, kernel_size


def minmax(v): #for Floyd-Steinberg dithering noise
    if v > 255:
        v = 255
    if v < 0:
        v = 0
    return v

def noise_img(img_LR, noise_types=['clean']):
    noise_type = random.choice(noise_types) 
    
    if noise_type == 'poisson': #note: Poisson noise is not additive like Gaussian, it's dependant on the image values: https://tomroelandts.com/articles/gaussian-noise-is-added-poisson-noise-is-applied
        vals = len(np.unique(img_LR))
        vals = 2 ** np.ceil(np.log2(vals))
        noise_img = np.random.poisson(img_LR * vals) / float(vals)
    
    elif noise_type == 's&p': 
        amount = np.random.uniform(0.02, 0.15) 
        s_vs_p = np.random.uniform(0.3, 0.7) # average = 50% salt, 50% pepper #q
        out = np.copy(img_LR)
        
        flipped = np.random.choice([True, False], size=out.shape, p=[amount, 1 - amount])
        
        # Salted mode
        salted = np.random.choice([True, False], size=out.shape, p=[s_vs_p, 1 - s_vs_p])
        
        # Pepper mode
        peppered = ~salted

        out[flipped & salted] = 1

        noise_img = out
        
    elif noise_type == 'speckle':
        h,w,c = img_LR.shape
        speckle_type = [c, 1] #randomize if colored noise (c) or 1 channel (B&W). #this can be done with other types of noise
        c = random.choice(speckle_type)
        
        mean = 0
        sigma = np.random.uniform(0.04, 0.2)
        noise = np.random.normal(mean, scale=sigma ** 0.5, size=(h,w,c))
        noise_img = img_LR + img_LR * noise
        #noise_img = np.clip(noise_img, 0, 1) 
        
    elif noise_type == 'gaussian': # Gaussian Noise
        h,w,c = img_LR.shape
        
        gauss_type = [c, 1] #randomize if colored gaussian noise or 1 channel (B&W). #this can be done with other types of noise
        c = random.choice(gauss_type)
        
        mean = 0
        sigma = np.random.uniform(4, 200.) 
        gauss = np.random.normal(scale=sigma,size=(h,w,c))
        gauss = gauss.reshape(h,w,c) / 255.
        
        noise_img = img_LR + gauss
        
    elif noise_type == 'JPEG': # JPEG Compression        
        compression = np.random.uniform(10, 50) #randomize quality between 10 and 50%
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression] #encoding parameters
        # encode
        is_success, encimg = cv2.imencode('.jpg', img_LR, encode_param) 
        
        # decode
        noise_img = cv2.imdecode(encimg, 1) 
        noise_img = noise_img.astype(np.uint8)
        
    elif noise_type == 'quantize': # Color quantization / palette
        pixels = np.reshape(img_LR, (img_LR.shape[0]*img_LR.shape[1], 3)) 
        
        N = int(np.random.uniform(2, 8))
        som = MiniSom(2, N, 3, sigma=1.,
                      learning_rate=0.2, neighborhood_function='bubble')  # 2xN final colors
        som.random_weights_init(pixels)
        starting_weights = som.get_weights().copy() 
        som.train_random(pixels, 500, verbose=False)
        
        qnt = som.quantization(pixels) 
        clustered = np.zeros(img_LR.shape)
        for i, q in enumerate(qnt): 
            clustered[np.unravel_index(i, dims=(img_LR.shape[0], img_LR.shape[1]))] = q
            
        noise_img = clustered
        
    elif noise_type == 'dither_bw': #Black and white dithering from color images
        img_gray = cv2.cvtColor(img_LR, cv2.COLOR_RGB2GRAY)
        size = img_gray.shape
        bwdith_types = ['binary', 'average', 'random', 'bayer', 'fs']
        dither_type = random.choice(bwdith_types)
        
        if dither_type == 'binary':
            img_bw = np.zeros(size, dtype=np.uint8)
            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    if img_gray[i, j] < 0.5:
                        img_bw[i, j] = 0
                    else:
                        img_bw[i, j] = 1
                        
            backtorgb = cv2.cvtColor(img_bw,cv2.COLOR_GRAY2RGB)
            noise_img = backtorgb 
        
        elif dither_type == 'average':
            re_aver = np.zeros(size, dtype=np.uint8)
            threshold = 0
            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    threshold = threshold + img_gray[i, j]/(size[0]*size[1])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    if img_gray[i, j] < threshold:
                        re_aver[i, j] = 0
                    else:
                        re_aver[i, j] = 1
            
            backtorgb = cv2.cvtColor(re_aver,cv2.COLOR_GRAY2RGB)
            noise_img = backtorgb
        
        elif dither_type == 'random':
            re_rand = np.zeros(size, dtype=np.uint8)
            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    if img_gray[i, j] < np.random.uniform(0, 1):
                        re_rand[i, j] = 0
                    else:
                        re_rand[i, j] = 1
            backtorgb = cv2.cvtColor(re_rand,cv2.COLOR_GRAY2RGB)
            noise_img = backtorgb
            
        elif dither_type == 'bayer':
            re_bayer = np.zeros(size, dtype=np.uint8) 
            bayer_matrix = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])/256 #4x4 Bayer matrix
            
            bayer_matrix = bayer_matrix*16 

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    x = np.mod(i, 4)
                    y = np.mod(j, 4)
                    if img_gray[i, j] > bayer_matrix[x, y]:
                        re_bayer[i, j] = 1
            backtorgb = cv2.cvtColor(re_bayer,cv2.COLOR_GRAY2RGB)
            noise_img = backtorgb
        
        elif dither_type == 'fs':
            re_fs = 255*img_gray 
            samplingF = 1
            for i in range(0, size[0]-1):
                for j in range(1, size[1]-1):
                    oldPixel = re_fs[i, j] #[y, x]
                    newPixel = np.round(samplingF * oldPixel/255.0) * (255/samplingF)
                    
                    re_fs[i, j] = newPixel
                    quant_error = oldPixel - newPixel
                    
                    re_fs[i, j+1] = minmax(re_fs[i, j+1]+(7/16.0)*quant_error)
                    re_fs[i+1, j-1] = minmax(re_fs[i+1, j-1]+(3/16.0)*quant_error)
                    re_fs[i+1, j] = minmax(re_fs[i+1, j]+(5/16.0)*quant_error)
                    re_fs[i+1, j+1] = minmax(re_fs[i+1, j+1]+(1/16.0)*quant_error)
                    
            backtorgb = cv2.cvtColor(re_fs/255.0,cv2.COLOR_GRAY2RGB)
            noise_img = backtorgb
    
    elif noise_type == 'dither': # Color dithering 
        colordith_types = ['bayer', 'fs']
        dither_type = random.choice(colordith_types)
        
        size = img_LR.shape
        
        # Bayer works more or less. I think it's missing a part of the image, the ditherng pattern is apparent, but the quantized (color palette) is not there. Still enough for models to learn dedithering
        if dither_type == 'bayer':
            bayer_matrix = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])/256 #4x4 Bayer matrix
            
            bayer_matrix = bayer_matrix*16 
            
            red = img_LR[:,:,2]
            green = img_LR[:,:,1]
            blue = img_LR[:,:,0]
            
            img_split = np.zeros((img_LR.shape[0], img_LR.shape[1], 3), dtype = red.dtype)
            
            for values, color, channel in zip((red, green, blue), ('red', 'green', 'blue'), (2,1,0)):
                for i in range(0, values.shape[0]):
                    for j in range(0, values.shape[1]):
                        x = np.mod(i, 4)
                        y = np.mod(j, 4)
                        if values[i, j] > bayer_matrix[x, y]:
                            img_split[i,j,channel] = 1 
            
            noise_img = img_split
        
        # Floyd-Steinberg (F-S) Dithering
        elif dither_type == 'fs':
            
            re_fs = 255*img_LR 
            samplingF = 1
            
            for i in range(0, size[0]-1):
                for j in range(1, size[1]-1):
                    oldPixel_b = re_fs[i, j, 0] #[y, x]
                    oldPixel_g = re_fs[i, j, 1] #[y, x]
                    oldPixel_r = re_fs[i, j, 2] #[y, x]
                    
                    newPixel_b = np.round(samplingF * oldPixel_b/255.0) * (255/samplingF) 
                    newPixel_g = np.round(samplingF * oldPixel_g/255.0) * (255/samplingF)
                    newPixel_r = np.round(samplingF * oldPixel_r/255.0) * (255/samplingF)
                    
                    re_fs[i, j, 0] = newPixel_b
                    re_fs[i, j, 1] = newPixel_g
                    re_fs[i, j, 2] = newPixel_r
                    
                    quant_error_b = oldPixel_b - newPixel_b
                    quant_error_g = oldPixel_g - newPixel_g
                    quant_error_r = oldPixel_r - newPixel_r
                    
                    re_fs[i, j+1, 0] = minmax(re_fs[i, j+1, 0]+(7/16.0)*quant_error_b)
                    re_fs[i, j+1, 1] = minmax(re_fs[i, j+1, 1]+(7/16.0)*quant_error_g)
                    re_fs[i, j+1, 2] = minmax(re_fs[i, j+1, 2]+(7/16.0)*quant_error_r)
                    
                    re_fs[i+1, j-1, 0] = minmax(re_fs[i+1, j-1, 0]+(3/16.0)*quant_error_b)
                    re_fs[i+1, j-1, 1] = minmax(re_fs[i+1, j-1, 1]+(3/16.0)*quant_error_g)
                    re_fs[i+1, j-1, 2] = minmax(re_fs[i+1, j-1, 2]+(3/16.0)*quant_error_r)
                    
                    re_fs[i+1, j, 0] = minmax(re_fs[i+1, j, 0]+(5/16.0)*quant_error_b)
                    re_fs[i+1, j, 1] = minmax(re_fs[i+1, j, 1]+(5/16.0)*quant_error_g)
                    re_fs[i+1, j, 2] = minmax(re_fs[i+1, j, 2]+(5/16.0)*quant_error_r)
                    
                    re_fs[i+1, j+1, 0] = minmax(re_fs[i+1, j+1, 0]+(1/16.0)*quant_error_b)
                    re_fs[i+1, j+1, 1] = minmax(re_fs[i+1, j+1, 1]+(1/16.0)*quant_error_g)
                    re_fs[i+1, j+1, 2] = minmax(re_fs[i+1, j+1, 2]+(1/16.0)*quant_error_r)
                    
            noise_img = re_fs/255.0
    
    elif noise_type == 'clean': # Pass clean image, without noise
        noise_img = img_LR
        
    #img_LR = np.clip(noise_img, 0, 1)
    return noise_img, noise_type


def random_pix(size):
    # Amount of pixels to translate
    # Higher probability for 0 shift
    # Caution: It can be very relevant how many pixels are shifted. 
    #"""
    if size <= 64:
        return random.choice([-1,0,0,1]) #pixels_translation
    elif size > 64 and size <= 96:
        return random.choice([-2,-1,0,0,1,2]) #pixels_translation
    elif size > 96:
        return random.choice([-3,-2,-1,0,0,1,2,3]) #pixels_translation
    #"""
    #return random.choice([-3,-2,-1,0,0,1,2,3]) #pixels_translation

# Note: the translate_chan() has limited success in fixing chromatic aberrations,
# because the patterns are very specific. The function works for the models to
# learn how to align fringes, but in this function the displacements are random
# and natural aberrations are more like purple fringing, axial (longitudinal), 
# and transverse (lateral), which are specific cases of these displacements and
# could be modeled here. 
def translate_chan(img_or):
    # Independently translate image channels to create color fringes
    rows, cols, _ = img_or.shape
    
    # Split the image into its BGR components
    (blue, green, red) = cv2.split(img_or)
    
    """ #V1: randomly displace each channel
    new_channels = []
    for values, channel in zip((blue, green, red), (0,1,2)):
        M = np.float32([[1,0,random_pix(rows)],[0,1,random_pix(cols)]])
        dst = cv2.warpAffine(values,M,(cols,rows))
        new_channels.append(dst)
    
    b_channel = new_channels[0]
    g_channel = new_channels[1]
    r_channel = new_channels[2]
    #"""
    
    #""" V2: only displace one channel at a time
    M = np.float32([[1,0,random_pix(rows)],[0,1,random_pix(cols)]])
    color = random.choice(["blue","green","red"])
    if color == "blue":
        b_channel = cv2.warpAffine(blue,M,(cols,rows))
        g_channel = green 
        r_channel = red
    elif color == "green":
        b_channel = blue
        g_channel = cv2.warpAffine(green,M,(cols,rows)) 
        r_channel = red
    else: # color == red:
        b_channel = blue
        g_channel = green
        r_channel = cv2.warpAffine(red,M,(cols,rows))
    #"""
    
    """ V3: only displace a random crop of one channel at a time (INCOMPLETE)
    # randomly crop
    rnd_h = random.randint(0, max(0, rows - rows/2))
    rnd_w = random.randint(0, max(0, cols - cols/2))
    img_crop = img_or[rnd_h:rnd_h + rows/2, rnd_w:rnd_w + cols/2, :]
    
    (blue_c, green_c, red_c) = cv2.split(img_crop)
    rows_c, cols_c, _ = img_crop.shape
    
    M = np.float32([[1,0,random_pix(rows_c)],[0,1,random_pix(cols_c)]])
    color = random.choice(["blue","green","red"])
    if color == "blue":
        b_channel = cv2.warpAffine(blue_c,M,(cols_c,rows_c))
        g_channel = green_c 
        r_channel = red_c
    elif color == "green":
        b_channel = blue_c
        g_channel = cv2.warpAffine(green_c,M,(cols_c,rows_c)) 
        r_channel = red_c
    else: # color == red:
        b_channel = blue_c
        g_channel = green_c
        r_channel = cv2.warpAffine(red_c,M,(cols_c,rows_c))
        
    merged_crop = cv2.merge((b_channel, g_channel, r_channel))
    
    image[rnd_h:rnd_h + rows/2, rnd_w:rnd_w + cols/2, :] = merged_crop
    return image
    #"""
    
    # merge the channels back together and return the image
    return cv2.merge((b_channel, g_channel, r_channel))

# https://www.pyimagesearch.com/2015/09/28/implementing-the-max-rgb-filter-in-opencv/
def max_rgb_filter(image):
	# split the image into its BGR components
	(B, G, R) = cv2.split(image)
 
	# find the maximum pixel intensity values for each
	# (x, y)-coordinate,, then set all pixel values less
	# than M to zero
	M = np.maximum(np.maximum(R, G), B)
	R[R < M] = 0
	G[G < M] = 0
	B[B < M] = 0
 
	# merge the channels back together and return the image
	return cv2.merge([B, G, R])

# Simple color balance algorithm (similar to Photoshop "auto levels")
# https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc#gistcomment-3025656
# http://www.morethantechnical.com/2015/01/14/simplest-color-balance-with-opencv-wcode/
# https://web.stanford.edu/~sujason/ColorBalancing/simplestcb.html
def simplest_cb(img, percent=1, znorm=False):
    
    if znorm == True: # img is znorm'ed in the [-1,1] range, else img in the [0,1] range
        img = (img + 1.0)/2.0
    
    # back to the OpenCV [0,255] range
    img = (img * 255.0).round().astype(np.uint8) # need to add astype(np.uint8) so cv2.LUT doesn't fail later
    
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
        
    img_out = cv2.merge(out_channels)
    
    # Re-normalize
    if znorm == False: # normalize img_or back to the [0,1] range
        img_out = img_out/255.0
    if znorm==True: # normalize images back to range [-1, 1] 
        img_out = (img_out/255.0 - 0.5) * 2 # xi' = (xi - mu)/sigma    
    return img_out



#https://www.idtools.com.au/unsharp-masking-python-opencv/
def unsharp_mask(img, blur_algo='median', kernel_size=None, strength=None, unsharp_algo='laplacian', znorm=False):
    #h,w,c = img.shape
    
    if znorm == True: # img is znorm'ed in the [-1,1] range, else img in the [0,1] range
        img = (img + 1.0)/2.0
    # back to the OpenCV [0,255] range
    img = (img * 255.0).round().astype(np.uint8)
    
    #randomize strenght from 0.5 to 0.8
    if strength is None:
        strength = np.random.uniform(0.3, 0.9)
    
    if unsharp_algo == 'DoG':
        #If using Difference of Gauss (DoG)
        #run a 5x5 gaussian blur then a 3x3 gaussian blr
        blur5 = cv2.GaussianBlur(img,(5,5),0)
        blur3 = cv2.GaussianBlur(img,(3,3),0)
        DoGim = blur5 - blur3
        img_out = img - strength*DoGim
    
    else: # 'laplacian': using LoG (actually, median blur instead of gaussian)
        #randomize kernel_size between 1, 3 and 5
        if kernel_size is None:
            kernel_sizes = [1, 3, 5]
            kernel_size = random.choice(kernel_sizes)
        # Median filtering (could be Gaussian for proper LoG)
        #gray_image_mf = median_filter(gray_image, 1)
        if blur_algo == 'median':
            smooth = cv2.medianBlur(img, kernel_size)
        # Calculate the Laplacian (LoG, or in this case, Laplacian of Median)
        lap = cv2.Laplacian(smooth,cv2.CV_64F)
        # Calculate the sharpened image
        img_out = img - strength*lap
    
     # Saturate the pixels in either direction
    img_out[img_out>255] = 255
    img_out[img_out<0] = 0
    
    # Re-normalize
    if znorm == False: # normalize img_or back to the [0,1] range
        img_out = img_out/255.0
    if znorm==True: # normalize images back to range [-1, 1] 
        img_out = (img_out/255.0 - 0.5) * 2 # xi' = (xi - mu)/sigma
    
    return img_out



def random_img(img_dir, save_path, crop_size=(128, 128), scale=1, blur_algos=['clean'], noise_types=['clean'], noise_types2=['clean']):
    img_list = _get_paths_from_dir(img_dir)
    
    random_img_path = random.choice(img_list)
    
    env = None
    img = util.read_img(env, random_img_path) #read image from path, opens with OpenCV, value ranges from 0 to 1
    
    img_crop = random_crop(img, crop_size)
    print(img_crop.shape)
    #"""
    cv2.imwrite(save_path+'/crop_.png',img_crop*255) 
    #"""
    
    img_resize, _ = resize_img(img, crop_size)
    print(img_resize.shape)
    #"""
    cv2.imwrite(save_path+'/resize_.png',img_resize*255) 
    #"""
    
    img_random_resize, _ = random_resize_img(img, crop_size)
    print(img_random_resize.shape)
    #"""
    cv2.imwrite(save_path+'/random_resize_.png',img_random_resize*255) 
    #"""
    
    img_cutout = cutout(img, img.shape[0] // 2)
    print(img_cutout.shape)
    #"""
    cv2.imwrite(save_path+'/cutout_.png',img_cutout*255) 
    #"""
    
    img_erasing = random_erasing(img)
    print(img_erasing.shape)
    #"""
    cv2.imwrite(save_path+'/erasing_.png',img_erasing*255) 
    #"""
    
    #scale = 4
    img_scale, interpol_algo = scale_img(img, scale)
    print(img_scale.shape)    
    #"""
    cv2.imwrite(save_path+'/scale_'+str(scale)+'_'+str(interpol_algo)+'_.png',img_scale*255) 
    #"""
    
    img_blur, blur_algo, blur_kernel_size = blur_img(img, blur_algos)
    print(img_blur.shape)
    #"""
    cv2.imwrite(save_path+'/blur_'+str(blur_kernel_size)+'_'+str(blur_algo)+'_.png',img_blur*255) 
    #"""
    
    img_noise, noise_algo = noise_img(img, noise_types)
    print(img_noise.shape)
    #"""
    cv2.imwrite(save_path+'/noise_'+str(noise_algo)+'_.png',img_noise*255) 
    #"""
    
    #img_noise2, noise_algo2 = noise_img(img_noise, noise_types2)
    #print(img_noise2.shape)
    #"""
    #cv2.imwrite(save_path+'/noise2_'+str(noise_algo2)+'_.png',img_noise2*255) 
    #"""
    
    img_rrot, angle = random_rotate(img)
    print(img_rrot.shape)
    #"""
    cv2.imwrite(save_path+'/rrot_'+str(angle)+'_.png',img_rrot*255) 
    #"""
    
    print('Finished')
    

def single_image(img_path, save_path, crop_size=(128, 128), scale=1, blur_algos=['clean'], noise_types=['clean'], noise_types2=['clean']):
    env = None
    img = util.read_img(env, img_path) #read image from path, opens with OpenCV, value ranges from 0 to 1
    print(img.shape)
    
    img_crop = random_crop(img, crop_size)
    print(img_crop.shape)
    #"""
    cv2.imwrite(save_path+'/crop_.png',img_crop*255) 
    #"""
    
    img_resize, _ = resize_img(img, crop_size)
    print(img_resize.shape)
    #"""
    cv2.imwrite(save_path+'/resize_.png',img_resize*255) 
    #"""
    
    img_random_resize, _ = random_resize_img(img, crop_size)
    print(img_random_resize.shape)
    #"""
    cv2.imwrite(save_path+'/random_resize_.png',img_random_resize*255) 
    #"""
    
    img_cutout = cutout(img, img.shape[0] // 2)
    print(img_cutout.shape)
    #"""
    cv2.imwrite(save_path+'/cutout_.png',img_cutout*255) 
    #"""
    
    img_erasing = random_erasing(img)
    print(img_erasing.shape)
    #"""
    cv2.imwrite(save_path+'/erasing_.png',img_erasing*255) 
    #"""
    
    #scale = 4
    img_scale, interpol_algo = scale_img(img, scale)
    print(img_scale.shape)    
    #"""
    cv2.imwrite(save_path+'/scale_'+str(scale)+'_'+str(interpol_algo)+'_.png',img_scale*255) 
    #"""
    
    img_blur, blur_algo, blur_kernel_size = blur_img(img, blur_algos)
    print(img_blur.shape)
    #"""
    cv2.imwrite(save_path+'/blur_'+str(blur_kernel_size)+'_'+str(blur_algo)+'_.png',img_blur*255) 
    #"""
    
    img_noise, noise_algo = noise_img(img, noise_types)
    #img_noise, noise_algo = noise_img(img_scale, noise_types)
    print(img_noise.shape)
    #"""
    cv2.imwrite(save_path+'/noise_'+str(noise_algo)+'_.png',img_noise*255) 
    #"""
    
    img_noise2, noise_algo2 = noise_img(img_noise, noise_types2)
    print(img_noise2.shape)
    #"""
    cv2.imwrite(save_path+'/noise2_'+str(noise_algo2)+'_.png',img_noise2*255) 
    #"""
    
    img_rrot, angle = random_rotate(img)
    print(img_rrot.shape)
    #"""
    cv2.imwrite(save_path+'/rrot_'+str(angle)+'_.png',img_rrot*255) 
    #"""
    
    print('Finished')

    
def apply_dir(img_path, save_path, crop_size=(128, 128), scale=1, blur_algos=['clean'], noise_types=['clean'], noise_types2=['clean']):
    img_list = _get_paths_from_dir(img_dir)
    
    for path in img_list:
        rann = ''
        env = None
        img = util.read_img(env, path)
        import uuid
        rann = uuid.uuid4().hex

        img_crop = random_crop(img, crop_size)
        print(img_crop.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'mask_.png',img_crop*255) 
        #"""
        
        img_resize, _ = resize_img(img, crop_size)
        print(img_resize.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'resize_.png',img_resize*255) 
        #"""
        
        img_random_resize, _ = random_resize_img(img, crop_size)
        print(img_random_resize.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'random_resize_.png',img_random_resize*255) 
        #"""
        
        img_cutout = cutout(img, img.shape[0] // 2)
        print(img_cutout.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'cutout_.png',img_cutout*255) 
        #"""
        
        img_erasing = random_erasing(img)
        print(img_erasing.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'erasing_.png',img_erasing*255) 
        #"""
        
        #scale = 4
        img_scale, interpol_algo = scale_img(img, scale)
        print(img_scale.shape)    
        #"""
        cv2.imwrite(save_path+'/'+rann+'scale_'+str(scale)+'_'+str(interpol_algo)+'_.png',img_scale*255) 
        #"""
        
        img_blur, blur_algo, blur_kernel_size = blur_img(img, blur_algos)
        print(img_blur.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'blur_'+str(blur_kernel_size)+'_'+str(blur_algo)+'_.png',img_blur*255) 
        #cv2.imwrite(save_path+'/blur__.png',img_blur*255) 
        #"""
        
        img_noise, noise_algo = noise_img(img, noise_types)
        #img_noise, noise_algo = noise_img(img_scale, noise_types)
        print(img_noise.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'noise_'+str(noise_algo)+'_.png',img_noise*255) 
        #"""
        
        img_noise2, noise_algo2 = noise_img(img_noise, noise_types2)
        print(img_noise2.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'noise2_'+str(noise_algo2)+'_.png',img_noise2*255) 
        #"""
        
        img_rrot, angle = random_rotate(img)
        print(img_rrot.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'rrot_'+str(angle)+'_.png',img_rrot*255) 
        #"""

    print('Finished')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=int, required=False, help='Select mode. 1: Single Image, 2: Random image from directory, 3: Full directory')
    parser.add_argument('-img_dir', type=str, required=False, help='Image directory') 
    parser.add_argument('-img_path', type=str, required=False, help='Single image path') 
    parser.add_argument('-savepath', type=str, required=False, help='Save path') 
    parser.add_argument('-scale', type=int, required=False, help='Scale to resize')
    parser.add_argument('-crop', type=int, required=False, help='Crop size')
    parser.add_argument('-blur', type=str, required=False, help='Blur algorithm') 
    parser.add_argument('-noise', type=str, required=False, help='Noise algorithm') 
    parser.add_argument('-noise2', type=str, required=False, help='Secondary noise algorithm') 
    args = parser.parse_args()
    print(args)
    
    if args.img_dir:
        img_dir = args.img_dir
        #print(": " + args.-)
    else:
        img_dir = "../../datasets"

    if args.img_path:
        img_path = args.img_path
    else:
        img_path = "../../datasets/0318.png"
    
    if args.savepath:
        savepath = args.savepath
    else:
        savepath = "../../datasets/tests_otf/"
    
    if args.crop:
        crop_size=(args.crop,args.crop)
    else:
        crop_size=(128, 128)
    
    if args.scale:
        crop_size = args.scale
    else:
        scale = 4
        
    if args.blur:
        blur_algos = [args.blur]
    else:
        blur_algos = ["average","box","gaussian","bilateral", "clean", "clean", "clean", "clean"]
        
    if args.noise:
        noise_types = [args.noise]
    else:
        noise_types = ["gaussian", "gaussian", "JPEG", "JPEG", "quantize", "poisson", "dither", "s&p", "speckle", "clean", "clean", "clean", "clean"]
    
    if args.noise2:
        noise_types2 = [args.noise2]
    else:    
        noise_types2 = ["JPEG", "clean", "clean", "clean"]
    
    #Select mode. 1: Single Image, 2: Random image from directory, 3: Full directory
    if args.mode == 1:
        single_image(img_path, savepath, crop_size, scale, blur_algos, noise_types, noise_types2) #take an image path and apply augmentations
    elif args.mode == 2:
        random_img(img_dir, savepath, crop_size, scale, blur_algos, noise_types, noise_types2) #take a random image from the directory and apply augmentations    
    elif args.mode == 3:
        apply_dir(img_dir, savepath, crop_size, scale, blur_algos, noise_types, noise_types2) #take a random image from the directory and apply augmentations
    else:
        print("Mode not selected, please select one of 1: Single Image, 2: Random image from directory, 3: Full directory")
    
