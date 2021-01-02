# from __future__ import division
import torch
import math
# from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
try:
    import accimage
except ImportError:
    accimage = None

import numpy as np
import cv2
import numbers
import collections
import warnings


#For alt 1:
#PAD_MOD
_cv2_pad_to_str = {'constant':cv2.BORDER_CONSTANT,
                   'edge':cv2.BORDER_REPLICATE,
                   'reflect':cv2.BORDER_REFLECT_101,
                   'reflect1':cv2.BORDER_DEFAULT,
                   'symmetric':cv2.BORDER_REFLECT
                  }
#INTER_MODE
_cv2_interpolation_to_str = {'nearest':cv2.INTER_NEAREST,
                         'NEAREST':cv2.INTER_NEAREST, 
                         'bilinear':cv2.INTER_LINEAR,
                         'BILINEAR':cv2.INTER_LINEAR, 
                         'area':cv2.INTER_AREA,
                         'AREA':cv2.INTER_AREA,
                         'bicubic':cv2.INTER_CUBIC,
                         'BICUBIC':cv2.INTER_CUBIC, 
                         'lanczos':cv2.INTER_LANCZOS4,
                         'LANCZOS':cv2.INTER_LANCZOS4,}
_cv2_interpolation_from_str= {v:k for k,v in _cv2_interpolation_to_str.items()}


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
    r"""Convert a ``numpy.ndarray`` to tensor. (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    See ``ToTensor`` for more details.
    Args:
        pic (np.ndarray, torch.Tensor): Image to be converted to tensor, (H x W x C[RGB]).
    Returns:
        Tensor: Converted image.
    """
    # if not(_is_numpy_image(pic)):
        # raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))

    # handle numpy array
    # img = torch.from_numpy(pic.transpose((2, 0, 1)))
    if _is_numpy_image(pic):
        if len(pic.shape) == 2:
            pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor) or img.max() > 1 or img.dtype==torch.uint8:
            return img.float().div(255)
        else:
            try:
                return to_tensor(np.array(pic))
                #return img
            except Exception:
                raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))
    elif _is_tensor_image(pic):
        return pic


def to_cv_image(pic, mode=None):
    r"""Convert a tensor to an ndarray.

        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
            mode (str): color space and pixel depth of input data (optional)
                        for example: cv2.COLOR_RGB2BGR.

        Returns:
            np.array: Image converted to PIL Image.
        """
    if not (_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.squeeze(np.transpose(pic.numpy(), (1, 2, 0)))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))
    if mode is None:
        return npimg

    else:
        return cv2.cvtColor(npimg, mode)


def normalize(tensor, mean, std):
    r"""Normalize a tensor image with mean and standard deviation.
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.
    Returns:
        Tensor: Normalized Tensor image.
    """
    # if not _is_tensor_image(tensor):
        # raise TypeError('tensor is not a torch image.')

    if _is_tensor_image(tensor):
        # This is faster than using broadcasting, don't change without benchmarking
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor
    elif _is_numpy_image(tensor):
        return (tensor.astype(np.float32) - 255.0 * np.array(mean))/np.array(std)
    else:
        raise RuntimeError('Undefined type. Must be a numpy or a torch image.')


def resize(img, size, interpolation='BILINEAR'):
    r"""Resize the input numpy ndarray to the given size.
    Args:
        img (numpy ndarray): Image to be resized.
        size (sequence/tuple or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
            ((size * height / width, size))
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR`` (BILINEAR)
    Returns:
        CV Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))
    
    w, h, =  size
    if isinstance(size, int):
        # h, w, c = img.shape #this would defeat the purpose of "size"
        
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            output = cv2.resize(img, dsize=(ow, oh), interpolation=_cv2_interpolation_to_str[interpolation])
        else:
            oh = size
            ow = int(size * w / h)
            output = cv2.resize(img, dsize=(ow, oh), interpolation=_cv2_interpolation_to_str[interpolation])
    else:
        output = cv2.resize(img, dsize=(size[1], size[0]), interpolation=_cv2_interpolation_to_str[interpolation])
    if img.shape[2]==1:
        return output[:, :, np.newaxis]
    else:
        return output


def scale(*args, **kwargs):
    warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                  "please use transforms.Resize instead.")
    return resize(*args, **kwargs)


def to_rgb_bgr(pic):
    r"""Converts a color image stored in BGR sequence to RGB (BGR to RGB)
    or stored in RGB sequence to BGR (RGB to BGR).

    Args:
        pic (np.ndarray, torch.Tensor): Image to be converted, (H x W x 3).

    Returns:
        Tensor: Converted image.
    """

    if _is_numpy_image(pic) or _is_tensor_image(pic):
        img = pic[:, :, [2, 1, 0]]
        return img
    else:
        try:
            return to_rgb_bgr(np.array(pic))
        except Exception:
            raise TypeError('pic should be numpy.ndarray or torch.Tensor. Got {}'.format(type(pic)))


def pad(img, padding, fill=0, padding_mode='constant'):
# def pad(img, padding, fill=(0, 0, 0), padding_mode='constant'):
    r"""Pad the given numpy ndarray on all sides with specified padding mode and fill value.
    Args:
        img (numpy ndarray): image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int, tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value on the edge of the image
            - reflect: pads with reflection of image (without repeating the last value on the edge)
                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image (repeating the last value on the edge)
                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]
    Returns:
        Numpy image: padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))
    if not isinstance(padding, (numbers.Number, tuple, list)):   
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')
    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
        # pad_left, pad_top, pad_right, pad_bottom = padding
    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_top = pad_bottom = padding[0]
        pad_left = pad_right = padding[1]
    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_top = padding[0]
        pad_bottom = padding[1]
        pad_left = padding[2]
        pad_right = padding[3]
    
    # if fill == 'random':
    #     fill_list = []
    #     for _ in range(len(img.shape)):
    #         fill_list.append(random.randint(0, 255))
    #     fill = tuple(fill_list)

    if isinstance(fill, numbers.Number):
        fill = (fill,) * (2 * len(img.shape) - 3)
    
    if padding_mode == 'constant':
        assert (len(fill) == 3 and len(img.shape) == 3) or (len(fill) == 1 and len(img.shape) == 2), \
            'channel of image is {} but length of fill is {}'.format(img.shape[-1], len(fill))

    if img.shape[2]==1:
        return(cv2.copyMakeBorder(src=img, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                                 borderType=_cv2_pad_to_str[padding_mode], value=fill)[:,:,np.newaxis])
    else:
        return(cv2.copyMakeBorder(src=img, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                                     borderType=_cv2_pad_to_str[padding_mode], value=fill))


#def crop(img, i, j, h, w):
def crop(img, x, y, h, w):
    r"""Crop the given PIL Image.
    Args:
        img (numpy ndarray): Image to be cropped.
        x: Upper pixel coordinate.
        y: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        numpy ndarray: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    # assert _is_numpy_image(img), 'img should be CV Image. Got {}'.format(type(img))
    assert h > 0 and w > 0, 'h={} and w={} should greater than 0'.format(h, w)

    # return img[i:i+h, j:j+w, :]

    x1, y1, x2, y2 = round(x), round(y), round(x+h), round(y+w)

    #try: #doesn't work
        #check_point1 = img[x1, y1, ...]
        #check_point2 = img[x2-1, y2-1, ...]

    #except IndexError:
    if x1<0 or y1<0:
        # warnings.warn('crop region is {} but image size is {}'.format((x1, y1, x2, y2), img.shape))
        img = cv2.copyMakeBorder(img, - min(0, x1), max(x2 - img.shape[0], 0),
                                 -min(0, y1), max(y2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=[0, 0, 0])
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)

    # finally:
        # return img[x1:x2, y1:y2, ...].copy()
    return img[x1:x2, y1:y2, ...].copy()
    

def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = img.shape[0:2] # h, w, _ = img.shape
    th, tw = output_size
    i = int(round((h - th) / 2.)) # i = int(round((h - th) * 0.5))
    j = int(round((w - tw) / 2.)) # j = int(round((w - tw) * 0.5))
    return crop(img, i, j, th, tw)


def resized_crop(img, i, j, h, w, size, interpolation=cv2.INTER_LINEAR):
# def resized_crop(img, i, j, h, w, size, interpolation='BILINEAR'):
    r"""Crop the given numpy ndarray and resize it to desired size.
    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.
    Args:
        img (numpy ndarray): Image to be cropped.        
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``.
    Returns:
        numpy ndarray Image: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be numpy image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation=interpolation)
    return img


def hflip(img):
    r"""Horizontally flip the given numpy ndarray.
    Args:
        img (numpy ndarray): image to be flipped.
    Returns:
        numpy ndarray:  Horizontally flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    # img[:,::-1] is much faster, but doesn't work with torch.from_numpy()!
    if img.shape[2]==1:
        return cv2.flip(img,1)[:,:,np.newaxis]
    else:
        #return img[:, ::-1, :] # test, appears to be faster 
        return cv2.flip(img, 1)


def vflip(img):
    r"""Vertically flip the given numpy ndarray.
    Args:
        img (numpy ndarray): Image to be flipped.
    Returns:
        numpy ndarray:  Vertically flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    if img.shape[2]==1:
        return cv2.flip(img, 0)[:,:,np.newaxis]
    else:
        #return img[::-1, :, :] 
        ##img[::-1] is much faster, but doesn't work with torch.from_numpy()!
        return cv2.flip(img, 0)


def five_crop(img, size):
    r"""Crop the given numpy ndarray into four corners and the central crop.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    h, w = img.shape[0:2] # h, w, _ = img.shape
    
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    tl = crop(img, 0, 0, crop_h, crop_w)
    # tr = crop(img, w - crop_w, 0, crop_h, w)
    tr = crop(img, 0, w - crop_w, crop_h, crop_w)
    # bl = crop(img, 0, h - crop_h, crop_w, h)
    bl = crop(img, h - crop_h, 0, crop_h, crop_w)
    # br = crop(img, w - crop_w, h - crop_h,  h,w)
    br = crop(img, h - crop_h, w - crop_w, crop_h, crop_w)
        
    center = center_crop(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center)


def ten_crop(img, size, vertical_flip=False):
    r"""Crop the given numpy ndarray into four corners and the central crop plus the
        flipped version of these (horizontal flipping is used by default).
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
       vertical_flip (bool): Use vertical flipping instead of horizontal
    Returns:
       tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
                Corresponding top left, top right, bottom left, bottom right and center crop
                and same for the flipped image.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)
    return first_five + second_five


def adjust_brightness(img, brightness_factor):
    r"""Adjust brightness of an Image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        numpy ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    
    # much faster to use the LUT construction than anything else I've tried
    # it's because you have to change dtypes multiple times
    # test the alternatives if necessary
    # alt 1:
    table = np.array([ i*brightness_factor for i in range (0,256)]).clip(0,255).astype('uint8')
    if img.shape[2]==1:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv2.LUT(img, table)
    
    # alt 2:
    # same thing but a bit slower
    # return cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

    # alt 3:
    # same results
    # im = img.astype(np.float32) * brightness_factor
    # im = im.clip(min=0, max=255)
    # return im.astype(img.dtype)


def adjust_contrast(img, contrast_factor):
    r"""Adjust contrast of an image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy ndarray: Contrast adjusted image.
    """
    
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    
    # much faster to use the LUT construction than anything else I've tried
    # it's because you have to change dtypes multiple times
    # test the alternatives if necessary
    # alt 1:
    table = np.array([ (i-74)*contrast_factor+74 for i in range (0,256)]).clip(0,255).astype('uint8')
    # enhancer = ImageEnhance.Contrast(img) #PIL
    # img = enhancer.enhance(contrast_factor) #PIL
    if img.shape[2]==1:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv2.LUT(img,table)

    # alt 2:
    # same results
    # im = img.astype(np.float32)
    # mean = round(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean())
    # im = (1-contrast_factor)*mean + contrast_factor * im
    # im = im.clip(min=0, max=255)
    # return im.astype(img.dtype)


def adjust_saturation(img, saturation_factor):
    r"""Adjust color saturation of an image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white (gray) image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        numpy ndarray: Saturation adjusted image.
    """
    
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    
    # ~10ms slower than PIL!
    # img = Image.fromarray(img) #PIL
    # enhancer = ImageEnhance.Color(img) #PIL
    # img = enhancer.enhance(saturation_factor)
    # return np.array(img)

    im = img.astype(np.float32)
    degenerate = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    im = (1-saturation_factor) * degenerate + saturation_factor * im
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_hue(img, hue_factor):
    r"""Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        numpy ndarray: Hue adjusted image.
    """
    # After testing, found that OpenCV calculates the Hue in a call to 
    # cv2.cvtColor(..., cv2.COLOR_BGR2HSV) differently from PIL

    
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].')
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    
    # alt 1:
    # This function takes 160ms! should be avoided
    # img = Image.fromarray(img) #PIL
    # input_mode = img.mode
    # if input_mode in {'L', '1', 'I', 'F'}:
        # return np.array(img)

    # h, s, v = img.convert('HSV').split()

    # np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    # with np.errstate(over='ignore'):
        # np_h += np.uint8(hue_factor * 255)
    # h = Image.fromarray(np_h, 'L')

    # img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    # return np.array(img)

    # alt 2:
    im = img.astype(np.uint8)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL)
    hsv[..., 0] += np.uint8(hue_factor * 255)

    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
    return im.astype(img.dtype)


def adjust_gamma(img, gamma, gain=1):
    r"""Perform gamma correction on an image.
    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:
    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}
    (I_out = 255 * gain * ((I_in / 255) ** gamma))
    See `Gamma Correction`_ for more details.
    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float): The constant multiplier.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')
    
    # alt 1:
    # from here
    # https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python/41061351
    # table = np.array([((i / 255.0) ** gamma) * 255 * gain 
                      # for i in np.arange(0, 256)]).astype('uint8')
    # if img.shape[2]==1:
        # return cv2.LUT(img, table)[:,:,np.newaxis]
    # else:
        # return cv2.LUT(img,table)

    # alt 2:
    im = img.astype(np.float32)
    im = 255. * gain * np.power(im / 255., gamma)
    im = im.clip(min=0., max=255.)
    return im.astype(img.dtype)


def rotate(img, angle, resample=cv2.INTER_LINEAR, expand=False, center=None, border_value=0, scale=1.0):
# def rotate(img, angle, resample='BILINEAR', expand=False, center=None):
    r"""Rotate the image by angle.
    Args:
        img (numpy ndarray): numpy ndarray to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        # angle ({float, int}): In degrees clockwise order.
        resample (``cv2.INTER_NEAREST` or ``cv2.INTER_LINEAR`` or ``cv2.INTER_AREA`` or ``cv2.INTER_CUBIC``, optional):
            An optional resampling filter. See `filters`_ for more information.
            .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to ``cv2.INTER_NEAREST``.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        border_value (int, optional): Border value.
        scale (float, optional): Isotropic scale factor.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))    

    h,w = img.shape[0:2] # h, w, _ = img.shape

    # alt 1
    # if center is not None and expand:
        # raise ValueError('`expand` conflicts with `center`')
    # if center is None:
        # center = (w/2, h/2) # center = ((w - 1) * 0.5, (h - 1) * 0.5)
    # M = cv2.getRotationMatrix2D(center,angle,1)
    # if img.shape[2]==1:
        # return cv2.warpAffine(img,M,(w,h))[:,:,np.newaxis]
    # else:
        # return cv2.warpAffine(img,M,(w,h))

    # alt 2
    imgtype = img.dtype
    point = center or ((w - 1)*0.5, (h - 1)*0.5) #((w/2, h/2)
    M = cv2.getRotationMatrix2D(point, angle=-angle, scale=scale)

    if expand:
        if center is None:
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos)) # new_w = h * sin + w * cos #may want to cast after next step
            nH = int((h * cos) + (w * sin)) # new_h = h * cos + w * sin #may want to cast after next step

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - point[0] # M[0, 2] += (new_w - w) * 0.5 #alternative
            M[1, 2] += (nH / 2) - point[1] # M[1, 2] += (new_h - h) * 0.5 #alternative

            #alternative casting and rounding after calc
            #nW = int(np.round(nW))
            #nH = int(np.round(nH))

            # perform the actual rotation and return the image
            dst = cv2.warpAffine(img, M, (nW, nH), borderValue=border_value)

        else:
            xx = []
            yy = []
            for point in (np.array([0, 0, 1]), np.array([w-1, 0, 1]), np.array([w-1, h-1, 1]), np.array([0, h-1, 1])):
                target = M@point
                xx.append(target[0])
                yy.append(target[1])
            nh = int(math.ceil(max(yy)) - math.floor(min(yy)))
            nw = int(math.ceil(max(xx)) - math.floor(min(xx)))
            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nw - w)/2
            M[1, 2] += (nh - h)/2
            dst = cv2.warpAffine(img, M, (nw, nh), flags=_cv2_interpolation_to_str[resample], borderValue=border_value)
    else:
        dst = cv2.warpAffine(img, M, (w, h), flags=_cv2_interpolation_to_str[resample], borderValue=border_value)
    return dst.astype(imgtype)


def _get_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute matrix for affine transformation
    # We need compute affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
    #                              [ sin(a)*scale    cos(a + shear)*scale     0]
    #                              [     0                  0          1]
    
    angle = math.radians(angle)
    shear = math.radians(shear)
    # scale = 1.0 / scale
    
    #alt 1:
    T = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0,0,1]])
    C = np.array([[1, 0, center[0]], [0, 1, center[1]], [0,0,1]])
    RSS = np.array([[math.cos(angle)*scale, -math.sin(angle+shear)*scale, 0],
                   [math.sin(angle)*scale, math.cos(angle+shear)*scale, 0],
                   [0,0,1]])
    matrix = T @ C @ RSS @ np.linalg.inv(C)
    
    #alt 2: #Basically, the same
    # M00 = math.cos(angle)*scale
    # M01 = -math.sin(angle+shear)*scale
    # M10 = math.sin(angle)*scale
    # M11 = math.cos(angle+shear)*scale
    # M02 = center[0] - center[0]*M00 - center[1]*M01 + translate[0]
    # M12 = center[1] - center[0]*M10 - center[1]*M11 + translate[1]
    # matrix = np.array([[M00, M01, M02], [M10, M11, M12]], dtype=np.float32)
    
    return matrix[:2,:]


def affine6(img, anglez=0, shear=0, translate=(0, 0), scale=(1, 1), resample='BILINEAR', fillcolor=(0, 0, 0)):
    r"""Apply affine transformation on the image keeping image center invariant
    Args:
        img (np.ndarray): PIL Image to be rotated.
        anglez (float): rotation angle in degrees around Z between -180 and 180, clockwise direction.
        shear (float): rotation angle in degrees around Z between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float, or tuple): overall scale
        resample ({NEAREST, BILINEAR, BICUBIC}, optional):
        fillcolor (int or tuple): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    imgtype = img.dtype
    gray_scale = False

    if len(img.shape) == 2:
        gray_scale = True
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    rows, cols, _ = img.shape
    centery = rows * 0.5
    centerx = cols * 0.5

    alpha = math.radians(shear)
    beta = math.radians(anglez)

    lambda1 = scale[0]
    lambda2 = scale[1]

    tx = translate[0]
    ty = translate[1]

    sina = math.sin(alpha)
    cosa = math.cos(alpha)
    sinb = math.sin(beta)
    cosb = math.cos(beta)

    M00 = cosb * (lambda1 * cosa ** 2 + lambda2 * sina ** 2) - sinb * (lambda2 - lambda1) * sina * cosa
    M01 = - sinb * (lambda1 * sina ** 2 + lambda2 * cosa ** 2) + cosb * (lambda2 - lambda1) * sina * cosa

    M10 = sinb * (lambda1 * cosa ** 2 + lambda2 * sina ** 2) + cosb * (lambda2 - lambda1) * sina * cosa
    M11 = + cosb * (lambda1 * sina ** 2 + lambda2 * cosa ** 2) + sinb * (lambda2 - lambda1) * sina * cosa
    M02 = centerx - M00 * centerx - M01 * centery + tx
    M12 = centery - M10 * centerx - M11 * centery + ty
    affine_matrix = np.array([[M00, M01, M02], [M10, M11, M12]], dtype=np.float32)

    dst_img = cv2.warpAffine(img, affine_matrix, (cols, rows), flags=_cv2_interpolation_to_str[resample],
                             borderMode=cv2.BORDER_CONSTANT, borderValue=fillcolor)
    if gray_scale:
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2GRAY)
    return dst_img.astype(imgtype)


def affine(img, angle, translate, scale, shear, interpolation=cv2.INTER_LINEAR, mode=cv2.BORDER_CONSTANT, fillcolor=0):
# def affine(img, angle=0, translate=(0, 0), scale=1, shear=0, resample='BILINEAR', fillcolor=(0,0,0)):
    r"""Apply affine transformation on the image keeping image center invariant
    Args:
        img (numpy ndarray): numpy ndarray to be transformed.
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float): shear angle value in degrees between -180 to 180, clockwise direction.
        interpolation (``cv2.INTER_NEAREST` or ``cv2.INTER_LINEAR`` or ``cv2.INTER_AREA``, ``cv2.INTER_CUBIC``):
            An optional resampling filter.
            See `filters`_ for more information.
            If omitted, it is set to ``cv2.INTER_LINEAR``, for bicubic interpolation.
        mode (``cv2.BORDER_CONSTANT`` or ``cv2.BORDER_REPLICATE`` or ``cv2.BORDER_REFLECT`` or ``cv2.BORDER_REFLECT_101``)
            Method for filling in border regions. 
            Defaults to cv2.BORDER_CONSTANT, meaning areas outside the image are filled with a value (val, default 0)
        fillcolor (int or tuple): Optional fill color for the area outside the transform in the output image. Default: 0
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    imgtype = img.dtype

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    # alt 1: 
    output_size = img.shape[0:2]
    center = (img.shape[1] * 0.5 + 0.5, img.shape[0] * 0.5 + 0.5) 
    matrix = _get_affine_matrix(center, angle, translate, scale, shear)

    if img.shape[2]==1:
        return cv2.warpAffine(img, matrix, output_size[::-1],interpolation, borderMode=mode, borderValue=fillcolor)[:,:,np.newaxis]
    else:
        return cv2.warpAffine(img, matrix, output_size[::-1],interpolation, borderMode=mode, borderValue=fillcolor)

    # alt 2: 
    # gray_scale = False
    # if len(img.shape) == 2:
        # gray_scale = True
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # h, w, _ = img.shape
    # center = (w * 0.5, h * 0.5)
    # matrix = _get_affine_matrix(center, angle, translate, scale, shear)
    # dst_img = cv2.warpAffine(img, matrix, (w, h), flags=INTER_MODE[resample],
                             # borderMode=cv2.BORDER_CONSTANT, borderValue=fillcolor)
    
    # if gray_scale:
        # dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2GRAY)
    # return dst_img.astype(imgtype)


def to_grayscale(img, num_output_channels=1):
    r"""Convert image to grayscale version of image.
    Args:
        img (numpy ndarray): Image to be converted to grayscale.
    Returns:
        numpy ndarray: Grayscale version of the image.
            if num_output_channels == 1 : returned image is single channel
            if num_output_channels == 3 : returned image is 3 channel with r == g == b
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))

    if num_output_channels == 1:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else: #cv2 fails to convert from 1 channel to 1 channel
            return img
    elif num_output_channels == 3:
        # img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        # much faster than doing cvtColor to go back to gray:
        img = np.broadcast_to(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis], img.shape) 
    else:
        raise ValueError('num_output_channels should be either 1 or 3')
    return img


def erase(img, i, j, h, w, v=None, inplace=False):
    r""" Erase the input Tensor Image with given value.
    Args:
        img (Tensor Image): Tensor image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner (top).
        j (int): j in (i,j) i.e coordinates of the upper left corner (left).
        h (int): Height of the erased region. (i + h = bottom)
        w (int): Width of the erased region. (j + w = right)
        v: Erasing value. If is a number, it's applied to all three channels,
            if is a list of lenght 3 (for example, the ImageNet mean pixel 
            values for each channel are v=[0.4465, 0.4822, 0.4914]*255, where 
            OpenCV follows BGR convention and PIL follows RGB color convention), 
            then each is applied to each channel.
        inplace(bool, optional): For in-place operations. By default is set False.
    Returns:
        Tensor Image: Erased image.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError('img should be np.ndarray Image. Got {}'.format(type(img)))

    if not inplace:
        img = np.copy(img)

    if type(v) is list: 
        if len(v) == 3 and len(img.shape) == 3: 
            #img[i:i + h, j:j + w, 0].fill(v[0])
            img[i:i + h, j:j + w, 0] = v[0]
            #img[i:i + h, j:j + w, 1].fill(v[1])
            img[i:i + h, j:j + w, 1] = v[1]
            #img[i:i + h, j:j + w, 2].fill(v[2])
            img[i:i + h, j:j + w, 2] = v[2]
        else: # == 1
            img[i:i + h, j:j + w, :].fill(v[0])
    elif isinstance(v, numbers.Number): 
        #img[i:i + h, j:j + w, :] = v
        img[i:i + h, j:j + w, :].fill(v)
    elif isinstance(v, np.ndarray):
        img[i:i + h, j:j + w, :] = v
    return img
