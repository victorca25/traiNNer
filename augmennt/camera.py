# Workaround to disable Intel Fortran Control+C console event handler installed by scipy
from os import environ as os_env
os_env['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'

import numpy as np
import cv2

try:
    from scipy.ndimage.filters import convolve, convolve1d
    scipy_available = True
except ImportError:
    from .common import convolve
    scipy_available = False

from .common import preserve_channel_dim, merge_channels


DEFAULT_FLOAT_DTYPE = 'float64'



# TODO: move to common
def tstack(a, dtype=None) -> np.ndarray:
    """ Stacks arrays in sequence along the last axis (tail).
        Rebuilds arrays divided by :func:`tsplit`.
    Args:
        a: Array to perform the stacking.
    dtype: Type to use for initial conversion to *ndarray*,
        default to the type defined by :attr:`DEFAULT_FLOAT_DTYPE`
        attribute.
    Returns
        ndarray
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    a = np.asarray(a, dtype)

    return np.concatenate([x[..., np.newaxis] for x in a], axis=-1)


# TODO: move to common
def tsplit(a, dtype=None) -> np.ndarray:
    """ Splits arrays in sequence along the last axis (tail).
    Args:
    a: Array to perform the splitting.
    dtype: Type to use for initial conversion to *ndarray*,
        default to the type defined by :attr:`DEFAULT_FLOAT_DTYPE`
        attribute.
    Returns
        ndarray
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    a = np.asarray(a, dtype)

    return np.array([a[..., x] for x in range(a.shape[-1])])


######################
# Mosaic and Demosaic
######################


def masks_CFA_Bayer(shape, pattern:str='RGGB') -> tuple:
    """ Returns the *Bayer* color filter array (CFA) red, green 
        and blue masks for given pattern.
    Args:
        shape: Dimensions of the *Bayer* CFA.
        pattern: Arrangement of the color filters on the pixel
            array, in: {'RGGB', 'BGGR', 'GRBG', 'GBRG'}
    Returns
        tuple: *Bayer* CFA red, green and blue masks.
    """
    pattern = pattern.upper()

    channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].astype(bool) for c in 'RGB')


def make_img_even(img:np.ndarray,
    border=cv2.BORDER_REFLECT101) -> np.ndarray:
    """ Extend image in order to make it even sized """

    h, w = img.shape[0:2]
    top = 0
    bottom = (h % 2 == 1)
    left = 0
    right = (w % 2 == 1)

    if bottom > 0 or right > 0:
        return cv2.copyMakeBorder(img, top, bottom, left, right, border)
    return img


def mosaic(RGB:np.ndarray) -> np.ndarray:
    """ Extracts RGGB *Bayer* planes from an RGB image
        as an array with each plane on a separate channel.
        Args:
            RGB: *RGB* colorspace array.
        Note: only supports 'RGGB' arrangement and images
            with even H and W dimensions at the moment,
            so images are forced to expected dimensions.
    """
    RGB = make_img_even(RGB)
    shape = RGB.shape
    red = RGB[0::2, 0::2, 0]
    green_red = RGB[0::2, 1::2, 1]
    green_blue = RGB[1::2, 0::2, 1]
    blue = RGB[1::2, 1::2, 2]

    out = merge_channels([red, green_red, green_blue, blue])
    out = np.reshape(out, (shape[0] // 2, shape[1] // 2, 4))

    return out


def mosaic_CFA_Bayer(RGB, pattern:str='RGGB') -> np.ndarray:
    """ Returns the *Bayer* color filter array (CFA) mosaic for a
        given *RGB* colorspace array as a single channel image.
    Args:
        RGB: *RGB* colorspace array.
    pattern: Arrangement of the color filters on the pixel
        array, in: {'RGGB', 'BGGR', 'GRBG', 'GBRG'}
    Returns
        ndarray: *Bayer* CFA mosaic.
    """
    RGB = np.asarray(RGB, dtype=DEFAULT_FLOAT_DTYPE)

    R, G, B = tsplit(RGB)
    R_m, G_m, B_m = masks_CFA_Bayer(RGB.shape[0:2], pattern)

    CFA = R * R_m + G * G_m + B * B_m
    del R_m, G_m, B_m

    return CFA


def demosaic(bayer_images:np.ndarray, dmscfn='malvar') -> np.ndarray:
    """Demosaic method selector."""
    if dmscfn == 'pixelshuffle':
        return demosaic_pixelshuffle(bayer_images)

    return cfa_demosaic(bayer_images, fn=dmscfn)


def cfa_demosaic(bayer_images:np.ndarray, fn:str='bilinear',
    pattern:str='RGGB') -> np.ndarray:
    """ Utiliy function to convert RGGB images with separate
        channels to a CFA and apply the selected demosaic method.
    """
    def _fill(im, base):
        base[0::2, 0::2] = im[:,:,0]
        base[0::2, 1::2] = im[:,:,1]
        base[1::2, 0::2] = im[:,:,2]
        base[1::2, 1::2] = im[:,:,3]
        return base

    fn_dict = {
        'bilinear': demosaic_CFA_bilinear,
        'malvar': demosaic_CFA_malvar,
        'menon': demosaic_CFA_menon,
        }

    dem_fn = fn_dict[fn]

    dem_list = []
    for i in bayer_images:
        h, w, _ = i.shape
        empty = np.zeros((h*2, w*2), dtype=bayer_images.dtype)
        cfa = _fill(i, empty)
        dem_list.append(dem_fn(cfa, pattern))
    
    dem_batch = merge_channels(dem_list, axis=0)
    return dem_batch


def demosaic_CFA_bilinear(CFA:np.ndarray,
    pattern:str='RGGB') -> np.ndarray:
    """ Returns the demosaiced *RGB* colorspace array from
        given *Bayer* CFA using bilinear interpolation.
    Args:
        CFA: *Bayer* CFA as single channel image.
        pattern: Arrangement of the color filters on the pixel
            array, in: {'RGGB', 'BGGR', 'GRBG', 'GBRG'}
    Returns
        ndarray: *RGB* colorspace array.
    References
        Losson, O., Macaire, L., & Yang, Y. (2010).
        Comparison of Color Demosaicing Methods.
        In Advances in Imaging and Electron Physics (Vol. 162, pp. 173-265).
        doi:10.1016/S1076-5670(10)62005-8
    """

    CFA = np.asarray(CFA, dtype=DEFAULT_FLOAT_DTYPE)
    R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

    H_G = np.asarray(
        [[0, 1, 0],
         [1, 4, 1],
         [0, 1, 0]]) / 4

    H_RB = np.asarray(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]]) / 4

    R = convolve(CFA * R_m, H_RB)
    G = convolve(CFA * G_m, H_G)
    B = convolve(CFA * B_m, H_RB)

    del R_m, G_m, B_m, H_RB, H_G

    return tstack([R, G, B])


def demosaic_CFA_malvar(CFA:np.ndarray, pattern:str='RGGB') -> np.ndarray:
    """ Returns the demosaiced *RGB* colorspace array from
    given *Bayer* CFA using *Malvar (2004)* demosaicing algorithm.
    Args:
        CFA: *Bayer* CFA as single channel image.
        pattern: Arrangement of the color filters on the pixel
            array, in: {'RGGB', 'BGGR', 'GRBG', 'GBRG'}
    Returns
        ndarray: *RGB* colorspace array.
    References
        Malvar, H. S., He, L.-W., Cutler, R., & Way, O. M. (2004).
        High-Quality Linear Interpolation for Demosaicing of
        Bayer-Patterned Color Images.
        International Conference of Acoustic, Speech and Signal Processing, 5-8.
        http://research.microsoft.com/apps/pubs/default.aspx?id=102068
    """

    CFA = np.asarray(CFA, dtype=DEFAULT_FLOAT_DTYPE)
    R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

    GR_GB = np.asarray(
        [[0, 0, -1, 0, 0],
         [0, 0, 2, 0, 0],
         [-1, 2, 4, 2, -1],
         [0, 0, 2, 0, 0],
         [0, 0, -1, 0, 0]]) / 8

    Rg_RB_Bg_BR = np.asarray(
        [[0, 0, 0.5, 0, 0],
         [0, -1, 0, -1, 0],
         [-1, 4, 5, 4, - 1],
         [0, -1, 0, -1, 0],
         [0, 0, 0.5, 0, 0]]) / 8

    Rg_BR_Bg_RB = np.transpose(Rg_RB_Bg_BR)

    Rb_BB_Br_RR = np.asarray(
        [[0, 0, -1.5, 0, 0],
         [0, 2, 0, 2, 0],
         [-1.5, 0, 6, 0, -1.5],
         [0, 2, 0, 2, 0],
         [0, 0, -1.5, 0, 0]]) / 8

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    del G_m

    G = np.where(np.logical_or(R_m == 1, B_m == 1), convolve(CFA, GR_GB), G)

    RBg_RBBR = convolve(CFA, Rg_RB_Bg_BR)
    RBg_BRRB = convolve(CFA, Rg_BR_Bg_RB)
    RBgr_BBRR = convolve(CFA, Rb_BB_Br_RR)

    del GR_GB, Rg_RB_Bg_BR, Rg_BR_Bg_RB, Rb_BB_Br_RR

    # red rows
    R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R.shape)
    # red columns
    R_c = np.any(R_m == 1, axis=0)[np.newaxis] * np.ones(R.shape)
    # blue rows
    B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B.shape)
    # blue columns
    B_c = np.any(B_m == 1, axis=0)[np.newaxis] * np.ones(B.shape)

    del R_m, B_m

    R = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_RBBR, R)
    R = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_BRRB, R)

    B = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_RBBR, B)
    B = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_BRRB, B)

    R = np.where(np.logical_and(B_r == 1, B_c == 1), RBgr_BBRR, R)
    B = np.where(np.logical_and(R_r == 1, R_c == 1), RBgr_BBRR, B)

    del RBg_RBBR, RBg_BRRB, RBgr_BBRR, R_r, R_c, B_r, B_c

    return tstack([R, G, B])


def _cnv_h(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    """Helper function for horizontal convolution."""
    if scipy_available:
        return convolve1d(x, y, mode='mirror')
    else:
        raise ValueError("To use Menon method, scipy must be available")

def _cnv_v(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    """Helper function for vertical convolution."""
    if scipy_available:
        return convolve1d(x, y, mode='mirror', axis=0)
    else:
        raise ValueError("To use Menon method, scipy must be available")


def demosaic_CFA_menon(CFA:np.ndarray, pattern:str='RGGB',
    refining_step:bool=True) -> np.ndarray:
    """ Returns the demosaiced *RGB* colorspace array from given
        *Bayer* CFA using DDFAPD demosaicing algorithm.
    Args:
        CFA: *Bayer* CFA as single channel image.
        pattern: Arrangement of the color filters on the pixel
            array, in: {'RGGB', 'BGGR', 'GRBG', 'GBRG'}
        refining_step: Perform refining step.
    Returns
        ndarray: *RGB* colorspace array.

    References
        Menon, D., Andriani, S., & Calvagno, G. (2007).
        Demosaicing With Directional Filtering and a posteriori Decision. IEEE
        Transactions on Image Processing, 16(1), 132-141.
        doi:10.1109/TIP.2006.884928
    """

    CFA = np.asarray(CFA, dtype=DEFAULT_FLOAT_DTYPE)
    R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

    h_0 = np.array([0, 0.5, 0, 0.5, 0])
    h_1 = np.array([-0.25, 0, 0.5, 0, -0.25])

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    G_H = np.where(G_m == 0, _cnv_h(CFA, h_0) + _cnv_h(CFA, h_1), G)
    G_V = np.where(G_m == 0, _cnv_v(CFA, h_0) + _cnv_v(CFA, h_1), G)

    C_H = np.where(R_m == 1, R - G_H, 0)
    C_H = np.where(B_m == 1, B - G_H, C_H)

    C_V = np.where(R_m == 1, R - G_V, 0)
    C_V = np.where(B_m == 1, B - G_V, C_V)

    D_H = np.abs(C_H - np.pad(C_H, ((0, 0),
                                    (0, 2)), mode=str('reflect'))[:, 2:])
    D_V = np.abs(C_V - np.pad(C_V, ((0, 2),
                                    (0, 0)), mode=str('reflect'))[2:, :])

    del h_0, h_1, CFA, C_V, C_H

    k = np.array(
        [[0, 0, 1, 0, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 3, 0, 3],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 0, 1]])

    d_H = convolve(D_H, k, mode='constant')
    d_V = convolve(D_V, np.transpose(k), mode='constant')

    del D_H, D_V

    mask = d_V >= d_H
    G = np.where(mask, G_H, G_V)
    M = np.where(mask, 1, 0)

    del d_H, d_V, G_H, G_V

    # red rows
    R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R.shape)
    # blue rows
    B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B.shape)

    k_b = np.array([0.5, 0, 0.5])

    R = np.where(
        np.logical_and(G_m == 1, R_r == 1),
        G + _cnv_h(R, k_b) - _cnv_h(G, k_b),
        R,
    )

    R = np.where(
        np.logical_and(G_m == 1, B_r == 1) == 1,
        G + _cnv_v(R, k_b) - _cnv_v(G, k_b),
        R,
    )

    B = np.where(
        np.logical_and(G_m == 1, B_r == 1),
        G + _cnv_h(B, k_b) - _cnv_h(G, k_b),
        B,
    )

    B = np.where(
        np.logical_and(G_m == 1, R_r == 1) == 1,
        G + _cnv_v(B, k_b) - _cnv_v(G, k_b),
        B,
    )

    R = np.where(
        np.logical_and(B_r == 1, B_m == 1),
        np.where(
            M == 1,
            B + _cnv_h(R, k_b) - _cnv_h(B, k_b),
            B + _cnv_v(R, k_b) - _cnv_v(B, k_b),
        ),
        R,
    )

    B = np.where(
        np.logical_and(R_r == 1, R_m == 1),
        np.where(
            M == 1,
            R + _cnv_h(B, k_b) - _cnv_h(R, k_b),
            R + _cnv_v(B, k_b) - _cnv_v(R, k_b),
        ),
        B,
    )

    RGB = tstack([R, G, B])

    del R, G, B, k_b, R_r, B_r

    if refining_step:
        RGB = refining_step_menon(RGB, tstack([R_m, G_m, B_m]), M)

    del M, R_m, G_m, B_m

    return RGB


def refining_step_menon(RGB: np.ndarray, RGB_m: np.ndarray,
    M:np.ndarray) -> np.ndarray:
    """ Performs the refining step on given *RGB* colorspace array.
    Args
        RGB: *RGB* colorspace array.
        RGB_m: *Bayer* CFA red, green and blue masks.
        M: Estimation for the best directional reconstruction.
    Returns
        ndarray: Refined *RGB* colorspace array.
    """

    R, G, B = tsplit(RGB)
    R_m, G_m, B_m = tsplit(RGB_m)
    M = np.asarray(M)

    del RGB, RGB_m

    # updating of the green component
    R_G = R - G
    B_G = B - G

    FIR = np.ones(3) / 3

    B_G_m = np.where(
        B_m == 1,
        np.where(M == 1, _cnv_h(B_G, FIR), _cnv_v(B_G, FIR)),
        0,
    )
    R_G_m = np.where(
        R_m == 1,
        np.where(M == 1, _cnv_h(R_G, FIR), _cnv_v(R_G, FIR)),
        0,
    )

    del B_G, R_G

    G = np.where(R_m == 1, R - R_G_m, G)
    G = np.where(B_m == 1, B - B_G_m, G)

    # updating of the red and blue components in the green locations
    # red rows
    R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R.shape)
    # red columns
    R_c = np.any(R_m == 1, axis=0)[np.newaxis] * np.ones(R.shape)
    # blue rows
    B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B.shape)
    # blue columns
    B_c = np.any(B_m == 1, axis=0)[np.newaxis] * np.ones(B.shape)

    R_G = R - G
    B_G = B - G

    k_b = np.array([0.5, 0, 0.5])

    R_G_m = np.where(
        np.logical_and(G_m == 1, B_r == 1),
        _cnv_v(R_G, k_b),
        R_G_m,
    )
    R = np.where(np.logical_and(G_m == 1, B_r == 1), G + R_G_m, R)
    R_G_m = np.where(
        np.logical_and(G_m == 1, B_c == 1),
        _cnv_h(R_G, k_b),
        R_G_m,
    )
    R = np.where(np.logical_and(G_m == 1, B_c == 1), G + R_G_m, R)

    del B_r, R_G_m, B_c, R_G

    B_G_m = np.where(
        np.logical_and(G_m == 1, R_r == 1),
        _cnv_v(B_G, k_b),
        B_G_m,
    )
    B = np.where(np.logical_and(G_m == 1, R_r == 1), G + B_G_m, B)
    B_G_m = np.where(
        np.logical_and(G_m == 1, R_c == 1),
        _cnv_h(B_G, k_b),
        B_G_m,
    )
    B = np.where(np.logical_and(G_m == 1, R_c == 1), G + B_G_m, B)

    del B_G_m, R_r, R_c, G_m, B_G

    # updating of the red (blue) component in the blue (red) locations
    R_B = R - B
    R_B_m = np.where(
        B_m == 1,
        np.where(M == 1, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)),
        0,
    )
    R = np.where(B_m == 1, B + R_B_m, R)

    R_B_m = np.where(
        R_m == 1,
        np.where(M == 1, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)),
        0,
    )
    B = np.where(R_m == 1, R - R_B_m, B)

    del R_B, R_B_m, R_m

    return tstack([R, G, B])


def resize_bimg(img:np.ndarray, shape) -> np.ndarray:
    """Utility function to resize a batch of images"""
    @preserve_channel_dim
    def _resize_image(image, shape):
        return cv2.resize(image, dsize=shape, interpolation=cv2.INTER_LINEAR)

    image = [_resize_image(i, shape) for i in img]
    img_batch = merge_channels(image, axis=0)
    return img_batch


def nchw2nhwc(img:np.ndarray) -> np.ndarray:
    """Convert tensor from [N, C, H, W] to [N, H, W, C]"""
    return img.transpose([0, 2, 3, 1])


def nhwc2nchw(img:np.ndarray) -> np.ndarray:
    """Convert tensor from [N, H, W, C] to [N, C, H, W]"""
    return img.transpose([0, 3, 1, 2])


def space_to_depth(x:np.ndarray, block_size:int=2,
    shape:str='NHWC', mode:str='tf'):
    """
    Inverted PixelShuffle.
    Args:
        x: input tensor
        block_size: scale factor to down-sample tensor
        shape: shape of the array, in: 'NHWC', 'NCHW'.
        mode: select tensorflow ('tf') or pytorch ('pt')
            style inverse PixelShuffle.

    Returns:
        x: tensor after pixel shuffle. Shape is
            [N, H/r, W/r, (r*r)*C] if input shape is 'NHWC' or
            [N, (r*r)*C, H/r, W/r] if input shape is 'NCHW',
            where r refers to scale factor
    """
    # d = depth = channels
    if mode == 'pt':
        x = nhwc2nchw(x) if shape == 'NHWC' else x
        b, d, h, w = x.shape
    else:
        x = nchw2nhwc(x) if shape == 'NCHW' else x
        b, h, w, d = x.shape

    if h % block_size != 0 or w % block_size != 0:
        raise ValueError('Height and width of tensor must '
                         'be divisible by block_size.')

    new_d = -1  # d * (block_size**2)
    new_h = h // block_size
    new_w = w // block_size

    if mode == 'pt':
        # (N, C, H//bs, bs, W//bs, bs)
        x = x.reshape([b, d, new_h, block_size, new_w, block_size])
        # (N, bs, bs, C, H//bs, W//bs)
        x = np.ascontiguousarray(x.transpose([0, 1, 3, 5, 2, 4]))
        # (N, C*bs^2, H//bs, W//bs)
        x = x.reshape([b, new_d, new_h, new_w])
        x = nchw2nhwc(x) if shape == 'NHWC' else x
        return x

    # (N, H//bs, bs, W//bs, bs, C)
    x = x.reshape([b, new_h, block_size, new_w, block_size, d])
    # (N, H//bs, W//bs, bs, bs, C)
    x = np.ascontiguousarray(x.transpose([0, 1, 3, 2, 4, 5]))
    # (N, H//bs, W//bs, C*bs^2)
    x = x.reshape([b, new_h, new_w, new_d])
    x = nhwc2nchw(x) if shape == 'NCHW' else x
    return x


def depth_to_space(x:np.ndarray, block_size:int=2,
    shape:str='NHWC', mode:str='tf'):
    """
    PixelShuffle.
    Args:
        x: input tensor
        block_size: scale factor to down-sample tensor
        shape: shape of the array, in: 'NHWC', 'NCHW'.
        mode: select tensorflow ('tf') or pytorch ('pt')
            style PixelShuffle.

    Returns:
        x: tensor after pixel shuffle. Shape is
            [N, r*H, r*W, C/(r*r)] if input shape is 'NHWC' or
            [N, C/(r*r), r*H, r*W] if input shape is 'NCHW',
            where r refers to scale factor
    """
    # d = depth = channels
    if mode == 'pt':
        x = nhwc2nchw(x) if shape == 'NHWC' else x
        b, d, h, w = x.shape
    else:
        x = nchw2nhwc(x) if shape == 'NCHW' else x
        b, h, w, d = x.shape

    if d % (block_size ** 2) != 0:
        raise ValueError('The tensor channels must be divisible by '
                         '(block_size ** 2).')

    new_d = -1  # d // (block_size ** 2)
    new_h = h * block_size
    new_w = w * block_size

    if mode == 'pt':
        # (N, C//bs^2, bs, bs, H, W)
        x = x.reshape([b, new_d, block_size, block_size, h, w])
        # (N, C//bs^2, H, bs, W, bs)
        x = np.ascontiguousarray(x.transpose([0, 1, 4, 2, 5, 3]))
        # (N, C//bs^2, H*bs, W*bs)
        x = x.reshape([b, new_d, new_h, new_w])
        x = nchw2nhwc(x) if shape == 'NHWC' else x
        return x

    # (N, H, W, bs, bs, C//bs^2)
    x = x.reshape([b, h, w, block_size, block_size, new_d])
    # (N, H, bs, W, bs, C//bs^2)
    x = np.ascontiguousarray(x.transpose([0, 1, 3, 2, 4, 5]))
    # (N, H*bs, W*bs, C//bs^2)
    x = x.reshape([b, new_h, new_w, new_d])
    x = nhwc2nchw(x) if shape == 'NCHW' else x
    return x


def demosaic_pixelshuffle(bayer_images:np.ndarray,
    tshape:str='NHWC', psmode:str='tf') -> np.ndarray:
    """ Bilinearly demosaics a batch of RGGB Bayer images,
        using PixelShuffle (depth_to_space) and it's inverse
        (space_to_depth) to demosaic.
        Args:
            bayer_images: array of *Bayer* images with separate
                RGGB channels. Supports batches.
            tshape: tensor shape, in: 'NHWC', 'NCHW'
            psmode: PixelShuffle mode, either tensorflow ('tf') or
                pytorch ('pt'). TF follows the OpenCV convention
                'NHWC', but results are equivalent.
        References:
            Brooks, Tim and Mildenhall, Ben and Xue, Tianfan and Chen,
            Jiawen and Sharlet, Dillon and Barron, Jonathan T. (2019)
            Unprocessing Images for Learned Raw Denoising.
            IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
            https://www.timothybrooks.com/tech/unprocessing/
    """
    shape = bayer_images.shape
    shape = (shape[1] * 2, shape[2] * 2)

    # TODO:
    # need to test if cv2 performs similarly enough to tf align_corners=False
    # PyTorch's align_corners=False is equivalent to cv2 resize
    # https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/

    red = bayer_images[..., 0:1]
    red = resize_bimg(red, shape)

    green_red = bayer_images[..., 1:2]
    green_red = np.fliplr(green_red)  # flip left-right
    green_red = resize_bimg(green_red, shape)
    green_red = np.fliplr(green_red)  # flip left-right
    green_red = space_to_depth(
        green_red, 2, shape=tshape, mode=psmode)

    green_blue = bayer_images[..., 2:3]
    green_blue = np.flipud(green_blue)  # flip up-down
    green_blue = resize_bimg(green_blue, shape)
    green_blue = np.flipud(green_blue)  # flip up-down
    green_blue = space_to_depth(
        green_blue, 2, shape=tshape, mode=psmode)

    green_at_red = (green_red[..., 0] + green_blue[..., 0]) / 2
    green_at_green_red = green_red[..., 1]
    green_at_green_blue = green_blue[..., 2]
    green_at_blue = (green_red[..., 3] + green_blue[..., 3]) / 2

    green_planes = [
        green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
    ]
    green = depth_to_space(
        merge_channels(green_planes), 2, shape=tshape, mode=psmode)

    blue = bayer_images[..., 3:4]
    blue = np.flipud(np.fliplr(blue))
    blue = resize_bimg(blue, shape)
    blue = np.flipud(np.fliplr(blue))

    rgb_images = merge_channels([red, green, blue])
    return rgb_images


######################
# Unprocess
######################


def get_rgb2xyz_array(kind:str='D65') -> np.ndarray:
    if kind=='D50':
        xyz_array = np.array(
            [[0.4360747, 0.3850649, 0.1430804],
            [0.2225045, 0.7168786, 0.0606169],
            [0.0139322, 0.0971045, 0.7141733]])
    elif kind=='D65a':
        xyz_array = np.array(
            [[0.412391, 0.357584, 0.180481],
             [0.212639, 0.715169, 0.072192],
             [0.019331, 0.119195, 0.950532]])
    else:  # D65
        xyz_array = np.array(
            [[0.4124564, 0.3575761, 0.1804375],
             [0.2126729, 0.7151522, 0.0721750],
             [0.0193339, 0.1191920, 0.9503041]])
    return xyz_array


def get_xyz2rgb_array(kind:str='D65') -> np.ndarray:
    if kind=='D50':
        xyz_array = np.array(
            [[3.1338561, -1.6168667, -0.4906146],
            [-0.9787684, 1.9161415, 0.0334540],
            [0.0719453, -0.2289914, 1.4052427]])
    elif kind=='D65a':
        xyz_array = np.array(
            [[3.240970, -1.537383, -0.498611],
             [-0.969244, 1.875968, 0.041555],
             [0.055630, -0.203977, 1.056972]])
    else:  # D65
        xyz_array = np.array(
            [[3.2404542, -1.5371385, -0.4985314],
             [-0.9692660, 1.8760108, 0.0415560],
             [0.0556434, -0.2040259, 1.0572252]])
    return xyz_array


def random_ccm(xyz_arr:str='D65') -> np.ndarray:
    """Generates random RGB -> Camera color correction matrices.
    Ref:
        https://doi.org/10.1117/1.OE.59.11.110801
    """
    # takes a random convex combination of XYZ -> Camera CCMs
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
                [-0.5625, 1.6328, -0.0469],
                [-0.0703, 0.2188, 0.6406]],
                [[0.4913, -0.0541, -0.0202],
                [-0.613, 1.3513, 0.2906],
                [-0.1564, 0.2151, 0.7183]],
                [[0.838, -0.263, -0.0639],
                [-0.2887, 1.0725, 0.2496],
                [-0.0627, 0.1427, 0.5438]],
                [[0.6596, -0.2079, -0.0562],
                [-0.4782, 1.3016, 0.1933],
                [-0.097, 0.1581, 0.5181]]]
    num_ccms = len(xyz2cams)
    xyz2cams = np.array(xyz2cams, dtype=DEFAULT_FLOAT_DTYPE)
    weights = np.random.uniform(1e-8, 1e8, size=(num_ccms, 1, 1))
    weights_sum = np.sum(weights, axis=0)
    xyz2cam = np.sum(xyz2cams * weights, axis=0) / weights_sum

    # multiplies with RGB -> XYZ to get RGB -> Camera CCM
    rgb2xyz = get_rgb2xyz_array(xyz_arr)
    rgb2cam = np.matmul(xyz2cam, rgb2xyz)

    # normalizes each row
    rgb2cam = rgb2cam / np.sum(rgb2cam, axis=-1, keepdims=True)
    return rgb2cam


def random_gains(rg_range=(1.9, 2.4), bg_range=(1.5, 1.9)) -> tuple:
    """Generates random gains for brightening and white balance."""
    # RGB gain represents brightening
    rgb_gain = 1.0 / np.random.normal(loc=0.8, scale=0.1)

    # red and blue gains represent white balance
    red_gain = np.random.uniform(rg_range[0], rg_range[1])
    blue_gain = np.random.uniform(bg_range[0], bg_range[1])
    return rgb_gain, red_gain, blue_gain


def inverse_smoothstep(image:np.ndarray) -> np.ndarray:
    """Approximately inverts a global tone mapping curve."""
    image = np.clip(image, 0.0, 1.0)
    out = 0.5 - np.sin(np.arcsin(1.0 - 2.0 * image) / 3.0)  
    return out


def gamma_expansion(image:np.ndarray) -> np.ndarray:
    """Converts from gamma to linear space."""
    # clamps to prevent numerical instability of gradients near zero
    return np.maximum(image, 1e-8) ** 2.2


def apply_ccm(image:np.ndarray, ccm:np.ndarray) -> np.ndarray:
    """Applies a color correction matrix."""  
    shape = image.shape
    image = np.reshape(image, [-1, 3])
    image = np.tensordot(image, ccm, axes=[[-1], [-1]])
    return np.reshape(image, shape)


def safe_invert_gains(image:np.ndarray, rgb_gain:float, red_gain:float,
    blue_gain:float) -> np.ndarray:
    """Inverts gains while safely handling saturated pixels."""
    gains = merge_channels(
        [1.0 / red_gain, 1.0, 1.0 / blue_gain], axis=0) / rgb_gain
    gains = gains[np.newaxis, np.newaxis, :]

    # prevents dimming of saturated pixels by smoothly masking gains near white
    gray = np.mean(image, axis=-1, keepdims=True)
    inflection = 0.9
    mask = (np.maximum(gray - inflection, 0.0) / (1.0 - inflection)) ** 2.0
    safe_gains = np.maximum(mask + (1.0 - mask) * gains, gains)
    return image * safe_gains


def unprocess(image:np.ndarray, xyz_arr:str='D50',
    rg_range:tuple=(1.2, 2.4), bg_range:tuple=(1.2, 2.4)) -> tuple:
    """Unprocesses an image from sRGB to realistic raw data."""

    # randomly creates image metadata
    rgb2cam = random_ccm(xyz_arr=xyz_arr)
    cam2rgb = np.linalg.inv(rgb2cam)
    rgb_gain, red_gain, blue_gain = random_gains()

    # approximately inverts global tone mapping
    image = inverse_smoothstep(image)
    # inverts gamma compression
    image = gamma_expansion(image)
    # inverts color correction
    image = apply_ccm(image, rgb2cam)
    # approximately inverts white balance and brightening
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    # clips saturated pixels
    image = np.clip(image, 0.0, 1.0)
    # applies a Bayer mosaic
    image = mosaic(image)

    metadata = {
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
    }
    return image, metadata


def random_noise_levels() -> tuple:
    """ Generates random noise levels from a log-log 
        linear distribution.
    """
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = np.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + np.random.normal(scale=0.26)
    read_noise = np.exp(log_read_noise)
    return shot_noise, read_noise


def add_noise(image:np.ndarray, shot_noise=0.01,
    read_noise=0.0005) -> np.ndarray:
    """ Adds random shot (proportional to image) and read
        (independent) noise.
    """
    variance = image * shot_noise + read_noise
    noise = np.random.normal(scale=np.sqrt(variance), size=image.shape)
    return image + noise


######################
# Process
######################


def apply_gains(bayer_images:np.ndarray, red_gains:float,
    blue_gains:float) -> np.ndarray:
    """Applies white balance gains to a batch of Bayer images."""
    green_gains = np.ones_like(red_gains)
    gains = merge_channels([red_gains, green_gains, green_gains, blue_gains])
    gains = gains[:, np.newaxis, np.newaxis, :]
    return bayer_images * gains


def apply_ccms(images:np.ndarray, ccms:np.ndarray) -> np.ndarray:
    """Applies color correction matrices."""
    images = images[:, :, :, np.newaxis, :]
    ccms = ccms[:, np.newaxis, np.newaxis, :, :]
    return np.sum(images * ccms, axis=-1)


def gamma_compression(images:np.ndarray, gamma:float=2.2) -> np.ndarray:
    """Converts from linear to gamma space."""
    # clamps to prevent numerical instability of gradients near zero
    return np.maximum(images, 1e-8) ** (1.0 / gamma)


def smoothstep(image:np.ndarray) -> np.ndarray:
    """A global tone mapping curve."""
    image = np.clip(image, 0.0, 1.0)
    return 3.0 * image**2 - 2.0 * image**3


def process(bayer_images:np.ndarray, red_gains:float, blue_gains:float,
    cam2rgbs:np.ndarray, dmscfn:str='malvar') -> np.ndarray:
    """Processes a batch of Bayer RGGB images into sRGB images."""
    # white balance
    bayer_images = apply_gains(bayer_images, red_gains, blue_gains)
    # demosaic
    bayer_images = np.clip(bayer_images, 0.0, 1.0)
    images = demosaic(bayer_images, dmscfn)
    # color correction
    images = apply_ccms(images, cam2rgbs)
    # gamma compression
    images = np.clip(images, 0.0, 1.0)
    images = gamma_compression(images)
    images = smoothstep(images)
    return images
