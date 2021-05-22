import random

import os
# import os.path
import glob

import numpy as np
import cv2
import dataops.common as util
from dataops.common import fix_img_channels, get_image_paths, read_img, np2tensor
from dataops.minisom import MiniSom
from dataops.debug import *
from dataops.imresize import resize as imresize  # resize # imresize_np

# import dataops.opencv_transforms.opencv_transforms as transforms
from dataops.opencv_transforms.opencv_transforms.common import wrap_cv2_function, wrap_pil_function, _cv2_interpolation2str
from torch.utils.data.dataset import Dataset #TODO TMP, move NoisePatches to a separate dataloader



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

def Scale(img=None, scale: int = None, algo=None, 
    ds_kernel=None, resize_type=None, img_type=None):

    if img_type == 'pil':
        # for now using scale_() as interface for pil images
        #TODO: pil images will only use default method, not 'algo' yet
        def_pil = get_default_imethod('pil')
        return scale_(img, scale, method=def_pil), def_pil
    
    ow, oh = image_size(img)
    ## original rounds down:
    # h = int(oh/scale)
    # w = int(ow/scale)
    ## rounded_up = -(-numerator // denominator)     # math.ceil(scale_factor * in_sz)
    h = int(-(-oh//scale))
    w = int(-(-ow//scale))

    if (h == oh) and (w == ow):
        return img, None

    resize_fn, resize_type = get_resize(size=(w, h), scale=scale, ds_algo=algo, ds_kernel=ds_kernel, resize_type=resize_type)
    return resize_fn(np.copy(img)), resize_type


class MLResize:
    """Resize the input numpy ndarray to the given size using the Matlab-like
    algorithm (warning: an order of magnitude slower than OpenCV).

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

    resize_fn = None
    if resize_type == 0:
        pass
    elif not resize_type:
        resize_type = random.choice(ds_algo)

    # print(resize_type)
    if resize_type in custom_ktypes:
        # print('matlab')
        resize_fn = MLResize(scale=scale, interpolation=_n_interpolation_to_str[resize_type]) #(np.copy(img_LR))
    elif resize_type == 999: # use realistic downscale kernels
        # print('kernelgan')
        if ds_kernel:
            resize_fn = ds_kernel
    else: # use the provided OpenCV2 algorithms
        resize_fn = transforms.Resize(size, 
                interpolation=_cv2_interpolation2str.get(resize_type, 'BICUBIC'))
        # print('cv2')

    return resize_fn, resize_type


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
    w, h = image_size(img)

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



class RandomQuantize:
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
    hrrot = random.random() > 0.5
    angle = int(random.uniform(-90, 90))

    return {'load_size': load_size,
            'crop_pos': (x, y),
            'flip': flip,
            'rot': rot,
            'vflip': vflip,
            'hrrot': hrrot,
            'angle': angle,}


# TODO: could use the hasattr to set a value for all future calls
# TODO: need something similar to use the other PIL interpolation methods
def get_default_imethod(img_type='cv2'):
    if img_type == 'pil':
        return Image.BICUBIC
    else:
        return "BICUBIC"


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

    # hrrot and regular rotation are mutually exclusive
    if opt.get('use_hrrot', None) and params.get('hrrot', None):
        if params['angle']:
            if preprocess_mode == 'crop' or 'and_crop' in preprocess_mode:
                cs = crop_size
            else:
                cs = None
            transform_list.append(transforms.Lambda(
                lambda img: rotateHR(
                    img, crop_size=cs, rescale=1/4,
                    angle=params['angle'], img_type=img_type,
                    method=method)))
    elif opt.get('use_rot', None):
        if params is None:
            if random.random() < 0.5:
                transform_list.append(
                    transforms.RandomRotation(degrees=(90,90)))
        elif params['rot']:
            transform_list.append(transforms.Lambda(
                lambda img: rotate90(
                    img, params['rot'], params['vflip'], img_type=img_type)))

    return transforms.Compose(transform_list)


def resize(img, w, h, method=None):
    img_type = image_type(img)
    if not method:
        method = get_default_imethod(img_type)

    return transforms.Resize((h,w), interpolation=method)(img)


def scale_(img, scale, mul=False, method=None):
    """
    Returns a rescaled image by a specific factor given in parameter.
    Works with :py:class:`~PIL.Image.Image` or 
    :py:class:`~np.ndarray` objects.
    :param scale: The scale factor, as a float.
    :param mul: If true, a scale factor greater than 1 expands
        the image, between 0 and 1 contracts the image, else it's
        inverted.
    :param method: An optional resampling filter. Same values
        possible as in the PIL.Image.resize function or CV2
        equivalents.
    :returns: the scaled image.
    """
    if scale <= 0:
        raise ValueError("the scale factor must be greater than 0")

    if not method:
        method = get_default_imethod(image_type(img))

    ow, oh = image_size(img)
    if mul:
        h = int(round(scale * oh))
        w = int(round(scale * ow))
    else:
        h = int(-(-oh//scale))
        w = int(-(-ow//scale))

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
    if ph == oh and pw == ow:
        return img

    print_size_warning(ow, oh, pw, ph, base)
    if img_type == 'pil':
        # Note: with PIL if crop sizes > sizes, it adds black padding
        return img.crop((0, 0, pw, ph))
    else:
        #TODO: test if correct-> # padding = (0, pw - ow, 0, ph - oh)
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


def rotateHR(img, crop_size=None, rescale=1/4, angle=None, center=0,
    img_type=None, crop=True, method=None):
    """Rotate the given image with the given rotation degree
        and crop the black edges
    Args:
        img: an image of arbitrary size.
        crop_size: the size of the image after processing.
        rescale: upscaling factor (4x) to use like HD Mode7 to reduce
            jaggy lines after rotating, more accurate underlying
            "sub-pixel" data.
        angle: the degree of rotation on the image.
        crop: crop image to remove black edges if it is True. Will
            also expand image bounding box during rotation if True.
    Returns:
        A rotated image.
    """
    if not img_type:
        img_type = image_type(img)

    if not method:
        method = get_default_imethod(image_type(img))

    if not angle or angle == 0:
        if crop and crop_size:
            return transforms.CenterCrop(crop_size)(img)
        else:
            return img

    #TODO: add @preserve_shape wrapper to cv2 RandomRotation's function
    rrot = transforms.RandomRotation(
        degrees=(angle,angle), expand=crop, resample=method)

    if rescale < 1:
        # TODO: 'pil' images will use default method, adjust algo to take in method
        img, _ = Scale(img=img, scale=rescale, algo=cv2.INTER_CUBIC, img_type=img_type) #INTER_AREA #  cv2.INTER_LANCZOS4?

    w, h = image_size(img)
    img = rrot(img)
    if img_type == 'cv2' and len(img.shape) == 2:
        img = img[..., np.newaxis]

    if crop:
        x_A, y_A = get_crop_pos_rot(h, w, angle)
        tw, th = image_size(img)

        # for sin / cos option
        y1 = (th+2)//2 - int(y_A/2)
        y2 = y1 + int(y_A)
        x1 = (tw+2)//2 - int(x_A/2)
        x2 = x1 + int(x_A)

        if img_type == 'pil':
            img = img.crop((x1, y1, x2, y2))
        else:
            #image = image[top:bottom, left:right, ...]
            img = img[y1:y2, x1:x2, ...]

    if rescale < 1:
        # back to original size
        if crop_size:
            size = (crop_size, crop_size)
        else:
            size = (hr, wr)
        img = transforms.Resize(
            size, interpolation=method)(img)

    return img

def get_crop_pos_rot(h, w, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle',
    computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    angle = angle * np.pi / 180  # to radians

    width_is_longer = w >= h
    long_side, short_side = (w,h) if width_is_longer else (h,w)
    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if short_side <= 2.*sin_a*cos_a*long_side or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * short_side
        wr, hr = (x/sin_a, x/cos_a) if width_is_longer else (x/cos_a, x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr, hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
    return (wr), (hr)


def print_size_warning(ow, oh, w, h, base=4):
    """Print warning information about image size (only print once)"""
    if not hasattr(print_size_warning, 'has_printed'):
        if ow != w or oh != h:
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
    img_type = image_type(img_B)
    default_int_method = get_default_imethod(img_type)

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
            #TODO: pil images will only use default method, not 'algo' yet
            img_B, _ = Scale(img=img_B, scale=hr_downscale_amt,
                            algo=ds_algo, img_type=img_type)
            img_B = make_power_2(img_B, base=4*scale)

            # Downscales LR to match new size of HR if scale does not match after
            w, h = image_size(img_B)
            if img_A is not None:  # and (h // scale != h_A or w // scale != w_A):
                #TODO: pil images will only use default method, not 'algo' yet
                img_A, _ = Scale(img=img_A, scale=hr_downscale_amt,
                                algo=ds_algo, img_type=img_type)
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

    same_s = True if w == w_A and h == h_A else False
    if same_s:
        img_type = image_type(img_A)

    # TODO: validate, to fix potential cases with rounding errors (hr_downscale, etc), should not be needed
    # if abs(h_A*scale - h) <= 40 or abs(w_A*scale - w) <= 40:
    #     img_A = transforms.Resize((h//scale, w//scale), 
    #             interpolation=default_int_method)(img_A)
    #     w_A, h_A = image_size(img_A)

    # if h_A != A_crop_size or w_A != A_crop_size:
    if h_A != h//scale or w_A != w//scale:
        # make B and A power2 or padbase before calculating, else dimensions won't fit
        img_B = make_power_2(img_B, base=scale)
        img_A = make_power_2(img_A, base=scale)

        if opt.get('pre_crop', False) and (h > crop_size) and (w > crop_size):
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
            ds_kernel = ds_kernels
        else:
            ds_kernel = None

        if same_s:
            #TODO: pil images will only use default method, not 'algo' yet
            img_A, _ = Scale(img=img_A, scale=scale, algo=ds_algo, ds_kernel=ds_kernel, img_type=img_type)
        else:
            img_A = transforms.Resize(
                (h//scale, w//scale), interpolation=default_int_method)(img_A)

    return img_A, img_B


def get_ds_kernels(opt):
    """ 
    Use the previously extracted realistic estimated kernels 
        (kernelGAN, etc) to downscale images with.
    Ref:
        https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Kernel_Modeling_Super-Resolution_on_Real_Low-Resolution_Images_ICCV_2019_paper.pdf
        https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf
    """
    if opt.get('dataroot_kernels', None) and 999 in opt["lr_downscale_types"]:
        #TODO: could make this class into a pytorch dataloader, as an iterator can load all the kernels in init
        ds_kernels = transforms.ApplyKernel(
            scale=opt.get('scale', 4), kernels_path=opt['dataroot_kernels'], pattern='kernelgan')
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
    img_type = image_type(img_A)
    default_int_method = get_default_imethod(img_type)

    # Validate there's an img_A, if not, use img_B
    if img_A is None:
        img_A = img_B
        # print("Image LR: ", LR_path, ("was not loaded correctly, using HR pair to downscale on the fly."))
        print("Image was not loaded correctly, using pair to generate on the fly.")

    # check that B and A have the same dimensions ratio
    img_A, img_B = shape_change_fn(
        img_A=img_A, img_B=img_B, opt=opt, scale=scale,
        default_int_method=default_int_method)

    # if the B images are too small, Resize to the crop_size size and fit A pair to A_crop_size too
    img_A, img_B = dim_change_fn(
        img_A=img_A, img_B=img_B, opt=opt, scale=scale,
        default_int_method=default_int_method,
        crop_size=crop_size, A_crop_size=A_crop_size,
        ds_kernels=ds_kernels)

    # randomly scale A (ie. from B) if needed
    img_A, img_B = generate_A_fn(
        img_A=img_A, img_B=img_B, opt=opt, scale=scale,
        default_int_method=default_int_method,
        crop_size=crop_size, A_crop_size=A_crop_size,
        ds_kernels=ds_kernels)

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


