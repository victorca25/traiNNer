"""
Functions for color operations on tensors.
If needed, there are more conversions that can be used:
https://github.com/kornia/kornia/tree/master/kornia/color
https://github.com/R08UST/Color_Conversion_pytorch/blob/master/differentiable_color_conversion/basic_op.py
"""


import torch
import torch.nn as nn
import math
import cv2
import numpy as np


def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    # flip image channels
    out: torch.Tensor = image.flip(-3)  # https://github.com/pytorch/pytorch/issues/229
    # out: torch.Tensor = image[[2, 1, 0], :, :] #RGB to BGR #may be faster
    return out


def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    # same operation as bgr_to_rgb(), flip image channels
    return bgr_to_rgb(image)


def bgra_to_rgba(image: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = image[[2, 1, 0, 3], :, :]
    return out


def rgba_to_bgra(image: torch.Tensor) -> torch.Tensor:
    # same operation as bgra_to_rgba(), flip image channels
    return bgra_to_rgba(image)


def rgb_to_grayscale(input: torch.Tensor) -> torch.Tensor:
    r, g, b = torch.chunk(input, chunks=3, dim=-3)
    gray: torch.Tensor = 0.2989 * r + 0.587 * g + 0.114 * b
    # gray = rgb_to_yuv(input,consts='y')
    return gray


def bgr_to_grayscale(input: torch.Tensor) -> torch.Tensor:
    input_rgb = bgr_to_rgb(input)
    gray: torch.Tensor = rgb_to_grayscale(input_rgb)
    # gray = rgb_to_yuv(input_rgb,consts='y')
    return gray


def grayscale_to_rgb(input: torch.Tensor) -> torch.Tensor:
    # repeat the gray image to the three channels
    rgb: torch.Tensor = input.repeat(3, *[1] * (input.dim() - 1))
    return rgb


def grayscale_to_bgr(input: torch.Tensor) -> torch.Tensor:
    return grayscale_to_rgb(input)


def rgb_to_ycbcr(input: torch.Tensor, consts='yuv'):
    return rgb_to_yuv(input, consts == 'ycbcr')


def rgb_to_yuv(input: torch.Tensor, consts='yuv'):
    """Converts one or more images from RGB to YUV.
    Outputs a tensor of the same shape as the `input`
    image tensor, containing the YUV value of the pixels.
    The output is only well defined if the value in images
    are in [0,1]. Y′CbCr is often confused with the YUV color
    space, and typically the terms YCbCr and YUV are used
    interchangeably, leading to some confusion. The main difference
    is that YUV is analog and YCbCr is digital:
    https://en.wikipedia.org/wiki/YCbCr
    Args:
        input: 2-D or higher rank. Image data to convert. Last dimension
            must be size 3. (Could add additional channels, ie,
            AlphaRGB = AlphaYUV)
        consts: YUV constant parameters to use. BT.601 or BT.709.
            Could add YCbCr https://en.wikipedia.org/wiki/YUV
    Returns:
        images: images tensor with the same shape as `input`.
    """

    #channels = input.shape[0]

    if consts == 'BT.709':
        # HDTV YUV
        Wr = 0.2126
        Wb = 0.0722
        Wg = 1 - Wr - Wb  # 0.7152
        Uc = 0.539
        Vc = 0.635
        delta: float = 0.5  # 128 if image range in [0,255]
    elif consts == 'ycbcr':
        # Alt. BT.601 from Kornia YCbCr values, from JPEG conversion
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb  # 0.587
        Uc = 0.564  # (b-y) #cb
        Vc = 0.713  # (r-y) #cr
        delta: float = .5  # 128 if image range in [0,255]
    elif consts == 'yuvK':
        # Alt. yuv from Kornia YUV values:
        # https://github.com/kornia/kornia/blob/master/kornia/color/yuv.py
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb  # 0.587
        Ur = -0.147
        Ug = -0.289
        Ub = 0.436
        Vr = 0.615
        Vg = -0.515
        Vb = -0.100
        # delta: float = 0.0
    elif consts == 'y':
        # returns only Y channel, same as rgb_to_grayscale()
        # Note: torchvision uses ITU-R 601-2: Wr = 0.2989, Wg = 0.5870, Wb = 0.1140
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb  # 0.587
    else:
        # Default to 'BT.601', SDTV YUV
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb  # 0.587
        Uc = 0.493  # 0.492
        Vc = 0.877
        delta: float = 0.5  # 128 if image range in [0,255]

    r: torch.Tensor = input[..., 0, :, :]
    g: torch.Tensor = input[..., 1, :, :]
    b: torch.Tensor = input[..., 2, :, :]
    # TODO: Alt. Which one is faster? Appear to be the same.
    # Differentiable? Kornia uses both in different places
    # r, g, b = torch.chunk(input, chunks=3, dim=-3)

    if consts == 'y':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        # (0.2989 * input[0] + 0.5870 * input[1] + 0.1140 * input[2]).to(img.dtype)
        return y
    elif consts == 'yuvK':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        u: torch.Tensor = Ur * r + Ug * g + Ub * b
        v: torch.Tensor = Vr * r + Vg * g + Vb * b
    else:
        # if consts == 'ycbcr' or consts == 'yuv' or consts == 'BT.709':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        u: torch.Tensor = (b - y) * Uc + delta  # cb
        v: torch.Tensor = (r - y) * Vc + delta  # cr

    if consts == 'uv':
        # returns only UV channels
        return torch.stack((u, v), -3)
    else:
        return torch.stack((y, u, v), -3)


def ycbcr_to_rgb(input: torch.Tensor):
    return yuv_to_rgb(input, consts = 'ycbcr')


def yuv_to_rgb(input: torch.Tensor, consts='yuv') -> torch.Tensor:
    if consts == 'yuvK':
        # Alt. yuv from Kornia YUV values:
        # https://github.com/kornia/kornia/blob/master/kornia/color/yuv.py
        Wr = 1.14  # 1.402
        Wb = 2.029  # 1.772
        Wgu = 0.396  # .344136
        Wgv = 0.581  # .714136
        delta: float = 0.0
    elif consts == 'yuv' or consts == 'ycbcr':
        # BT.601 from Kornia YCbCr values, from JPEG conversion
        Wr = 1.403  # 1.402
        Wb = 1.773  # 1.772
        Wgu = .344  # .344136
        Wgv = .714  # .714136
        delta: float = .5  # 128 if image range in [0,255]

    y: torch.Tensor = input[..., 0, :, :]
    u: torch.Tensor = input[..., 1, :, :]  # cb
    v: torch.Tensor = input[..., 2, :, :]  # cr
    # TODO: Alt. Which one is faster? Appear to be the same.
    # Differentiable? Kornia uses both in different places
    # y, u, v = torch.chunk(input, chunks=3, dim=-3)

    u_shifted: torch.Tensor = u - delta  # cb
    v_shifted: torch.Tensor = v - delta  # cr

    r: torch.Tensor = y + Wr * v_shifted
    g: torch.Tensor = y - Wgv * v_shifted - Wgu * u_shifted
    b: torch.Tensor = y + Wb * u_shifted
    return torch.stack((r, g, b), -3)


def srgb2linear(img):
    """Convert sRGB images to linear RGB color space.
    Tensors are left as f32 in the range [0, 1].
    Uint8 numpy arrays are converted from uint8 in the range [0, 255]
    to f32 in the range [0, 1].
    F32 numpy arrays are assumed to be already be linear RGB.
    Always returns a new array.
    All values are exact as per the sRGB spec.
    """
    a = 0.055
    att = 12.92
    gamma = 2.4
    th = 0.04045

    if isinstance(img, torch.Tensor):
        return torch.where(
                img <= th, img / att, torch.pow((img + a)/(1 + a), gamma))

    if img.dtype == np.uint8:
        linear = np.float32(img) / 255.0

        return np.where(
            linear <= th, linear / att, np.power((linear + a) / (1 + a), gamma))

    return img.copy()


def linear2srgb(img):
    """Convert linear RGB to the sRGB colour space.
    Tensors are left as f32 in the range [0, 1].
    F32 numpy arrays are converted back to the expected uint8 format
    in the range [0, 255].
    Uint8 numpy arrays are assumed to already be sRGB.
    Always returns a new array.
    All values are exact as per the sRGB spec.
    """
    a = 0.055
    att = 12.92
    gamma = 2.4
    th = 0.0031308

    if isinstance(img, torch.Tensor):
        return torch.where(
                img <= th,
                img * att, (1 + a) * torch.pow((img), 1 / gamma) - a)

    if img.dtype == np.float32:
        srgb = np.clip(img, 0.0, 1.0)

        srgb = np.where(
            srgb <= th, srgb * att, (1 + a) * np.power(srgb, 1.0 / gamma) - a)

        np.clip(srgb * 255, 0.0, 255, out=srgb)
        np.around(srgb, out=srgb)

        return srgb.astype(np.uint8)

    return img.copy()


def color_shift(image: torch.Tensor, mode:str='uniform',
    rgb_weights=None, alpha:float=0.8, Y:bool=False,
    channels:str='single') -> torch.Tensor:
    """Random color shift transformation.
    Applies color shift to an image to decrease the influence of
    color and luminance (for texture extraction).
    Arguments:
        image: image to transform
        mode: choose between 'normal' or 'uniform' random weights
        rgb_weights (tensor): the precalculated random shift weights to apply
        alpha: weight to combine the random shift with standard grayscale
        Y: choose if results will be calculated with random values centered
            around the grayscale conversion constants with sigma=0.1
            (Y=false) or around 0.0 with sigma=1.0 ('β1, β2 and β3 ∼ U(−1,1)')
            and the combined with the grayscale image converted from RGB
            color image with interpolation factor 'alpha' (Y=true) for more
            control ('(1−α)(β1∗Ir+β2∗Ig+β3∗Ib)+α∗Y)'). With alpha=0.8,
            both options are equivalent.
    """
    if not isinstance(rgb_weights, torch.Tensor):
        rgb_weights = get_colorshift_weights(mode=mode, Y=Y)

    rgb_weights = rgb_weights.to(image.device)
    if channels == 'multi':
        # returns 3 chanel image
        output = (image * rgb_weights[None, :, None, None]) / rgb_weights.sum()
    else:
        # returns single channel gray image
        r: torch.Tensor = image[..., 0:1, :, :]
        g: torch.Tensor = image[..., 1:2, :, :]
        b: torch.Tensor = image[..., 2:3, :, :]
        output = (rgb_weights[0]*r+rgb_weights[1]*g+rgb_weights[2]*b)/(rgb_weights.sum())

    if Y and channels == 'single':
        output = (1-alpha)*output + alpha*rgb_to_grayscale(image)

    return output

def get_colorshift_weights(mode='uniform', Y=False):
    if Y:
        if mode == 'normal':
            r_weight = np.random.normal(loc=0.0, scale=1.0)
            g_weight = np.random.normal(loc=0.0, scale=1.0)
            b_weight = np.random.normal(loc=0.0, scale=1.0)
        elif mode == 'uniform':
            r_weight = np.random.uniform(low=-1.0, high=1.0)
            g_weight = np.random.uniform(low=-1.0, high=1.0)
            b_weight = np.random.uniform(low=-1.0, high=1.0)
    else:
        if mode == 'normal':
            r_weight = np.random.normal(loc=0.299, scale=0.1)
            g_weight = np.random.normal(loc=0.587, scale=0.1)
            b_weight = np.random.normal(loc=0.114, scale=0.1)
        elif mode == 'uniform':
            r_weight = np.random.uniform(low=0.199, high=0.399)
            g_weight = np.random.uniform(low=0.487, high=0.687)
            b_weight = np.random.uniform(low=0.014, high=0.214)

    return torch.Tensor([r_weight, g_weight, b_weight])


class ColorShift(nn.Module):
    """ Color shift class"""
    def __init__(self, mode='uniform', alpha=0.8, Y=False):
        super(ColorShift, self).__init__()
        self.mode = mode
        self.alpha = alpha
        self.Y = Y

    def forward(self, *img: torch.Tensor):
        rgb_weights = get_colorshift_weights(mode=self.mode, Y=self.Y)
        return (
            color_shift(im, self.mode, rgb_weights, self.alpha, self.Y) for im in img)

