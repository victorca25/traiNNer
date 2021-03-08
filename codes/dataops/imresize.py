import os
import math
import numpy as np
import torch
import cv2

from dataops.colors import *
from dataops.debug import tmp_vis, describe_numpy, describe_tensor


####################
# Matlab imresize
####################

def get_img_types(x):
    if isinstance(x, np.ndarray):
        to_dtype = lambda a: a
        fw = np
    else:
        to_dtype = lambda a: a.to(x.dtype)
        fw = torch
    eps = fw.finfo(fw.float32).eps
    return fw, to_dtype, eps


def kernel_width(kw):
    """ Wrapper to add the kernel support to the interpolation 
    functions, i.e length of non-zero segment over its 1d input 
    domain. 
    This is a characteristic of the function. eg. for bicubic 4, 
    linear 2, laczos2 4, lanczos3 6, box 1.
    """
    def wrapper(f):
        f.kernel_width = kw
        return f
    return wrapper


# these next functions are all interpolation methods.
# x is the distance from the left pixel center
@kernel_width(4)
def cubic(x, a=-0.5):
    """Parametrized cubic interpolant (B-Spline order 3).
    Reasonably good quality and faster than Lanczos3, 
    particularly when upsampling. 'ideal filter'.
    Convolution kernel weight function:
             |  |x| ≤ 1:      (a+2)|x|^3 - (a+3)|x|^2 + 1
      k(x) = |  1 < |x| ≤ 2:  a|x|^3 - 5a|x|^2 +8a|x| - 4a
             |  otherwise:    0
      where typically a=-0.5
    """
    fw, to_dtype, eps = get_img_types(x)
    absx = fw.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (((a+2)*absx3 - (a+3)*absx2 + 1.) * to_dtype(absx <= 1.) +
            (a * absx3 - (5*a)*absx2 + (8*a)*absx - 4*a) *
            to_dtype((absx > 1.) & (absx <= 2.)))

@kernel_width(1)
def box(x):
    """Box kernel, also known as Nearest Neighbor kernel
    Convolution kernel weight function:
             |  |x| < 0.5:   1
      k(x) = |  otherwise:   0
    """
    fw, to_dtype, eps = get_img_types(x)
    return to_dtype((-1 <= x) & (x < 0)) + to_dtype((0 <= x) & (x <= 1))
    # return to_dtype((-0.5 <= x) & (x < 0.5)) * 1.0

@kernel_width(2)
def linear(x):
    """Bilinear kernel, also known as linear or triangle kernel
    (B-Spline order 1).
    Convolution kernel weight function:
             |  |x| < 1.0:   1-|x|
      k(x) = |  otherwise:   0
    """
    fw, to_dtype, eps = get_img_types(x)
    return ((x + 1) * to_dtype((-1 <= x) & (x < 0)) + (1 - x) *
            to_dtype((0 <= x) & (x <= 1)))

def lanczos(x, a=3):
    """Lanczos kernel with radius 'a'.
    With a=3: High-quality practical filter but may have some 
        ringing, especially on synthetic images.
    With a=5: Very-high-quality filter but may have stronger 
        ringing.
    Convolution kernel weight function:
             |  |x| < a:   sinc(x) * sinc(x/a)
      k(x) = |  otherwise:   0
    """
    fw, to_dtype, eps = get_img_types(x)
    xp = math.pi * x

    return (((fw.sin(xp) * fw.sin(xp / a) + eps) /
            (((xp)**2 / a) + eps)) * to_dtype(fw.abs(x) <= a))

@kernel_width(4)
def lanczos2(x): return lanczos(x=x, a=2)
@kernel_width(6)
def lanczos3(x): return lanczos(x=x, a=3)
@kernel_width(8)
def lanczos4(x): return lanczos(x=x, a=4)
@kernel_width(10)
def lanczos5(x): return lanczos(x=x, a=5)

def sinc(x, a=2):
    """Sinc kernel with radius 'a'.
    Convolution kernel weight function:
             |  |x| != 0:    sin(x) / x
      k(x) = |  otherwise:   1.
    """
    fw, to_dtype, eps = get_img_types(x)
    xp = math.pi * x
    return (((fw.sin(xp) + eps) / ((xp) + eps)) * 
             to_dtype(fw.abs(x) !=0) + 
             (1.0) * to_dtype(fw.abs(x) == 0)
             )

@kernel_width(4)
def sinc2(x): return sinc(x=x, a=2)
@kernel_width(6)
def sinc3(x): return sinc(x=x, a=3)
@kernel_width(8)
def sinc4(x): return sinc(x=x, a=4)
@kernel_width(10)
def sinc5(x): return sinc(x=x, a=5)

def blackman(x, a=2):
    """Blackman kernel.
    Convolution kernel weight function:
             |  |x| != 0:    .42 - .5*cos(pi*x/a) + .08*cos(2*pi*x/a)
      k(x) = |  x == 0:      1.
             |  otherwise:   0
    """
    fw, to_dtype, eps = get_img_types(x)
    xp = math.pi * x
    # TODO: check if need to multiply by sinc(x) * => (fw.sin(xp + eps) / xp + eps) *
    return (( 
             (0.42 + 0.5*fw.cos((xp/a)) + 0.08*fw.cos(2*(xp/a)))) * 
             to_dtype(fw.abs(x) <= a) + 
             (1.0) * to_dtype(fw.abs(x) == 0)
             )

@kernel_width(4)
def blackman2(x): return blackman(x=x, a=2)
@kernel_width(6)
def blackman3(x): return blackman(x=x, a=3)
@kernel_width(8)
def blackman4(x): return blackman(x=x, a=4)
@kernel_width(10)
def blackman5(x): return blackman(x=x, a=5)

@kernel_width(2)
def hermite(x):
    """Hermite filter (B-Spline order 3). 
    Hermite is a particular case of the bicubic algorithm, 
        where a=0.
    Convolution kernel weight function:
             |  |x| ≤ 1.0:   2|x|^3 - 3|x|^2 + 1
      k(x) = |  otherwise:   0
    """
    fw, to_dtype, eps = get_img_types(x)
    absx = fw.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((2.0 * absx3) - (3.0 * absx2) + 1.) * to_dtype(absx <= 1.)

@kernel_width(2) #1
def bell(x):
    """Bell kernel (B-spline order 2).
    Convolution kernel weight function:
             |  |x| ≤ 0.5:        0.75-|x|^2
      k(x) = |  0.5 < |x| ≤ 1.5:  0.5 * (|x|-1.5)^2
             |  otherwise:        0
    """
    fw, to_dtype, eps = get_img_types(x)
    absx = fw.abs(x)
    absx2 = absx ** 2
    return ((0.75 - absx2) * to_dtype(absx <= 0.5) +
            (0.5 * ((absx - 1.5)**2)) *
            to_dtype((absx > 0.5) & (absx <= 1.5)))

@kernel_width(4)
def mitchell(x):
    """Mitchell-Netravali Filter (B-Spline order 3)
    Convolution kernel weight function:
             |  |x| ≤ 1.0:        1/6.*[ ((12-9B-6C)|x|^3 + ((-18+12B+6C)|x|^2 + (6-2B)) ]
      k(x) = |  1.0 < |x| ≤ 2.0:  1/6.*[ ((-B-6C)|x|^3 + (6B+30C)|x|^2 + (-12B-48C)|x| + (8B+24C) ]
             |  otherwise:        0
    Constants:
        B = 1,   C = 0   - cubic B-spline
        B = 1/3, C = 1/3 - Mitchell-Netravali or just “Mitchell” (recommended)
        B = 0,   C = 1/2 - Catmull-Rom spline
        B = 0,   C = 0   - Hermite
        B = 0,   C = 1   - Sharp Bicubic
    """
   
    A = 1.0 / 6.0
    B = 1.0 / 3.0
    C = 1.0 / 3.0
    p0 = (6.0 - 2.0 * B)
    p2 = (-18.0 + 12.0 * B + 6.0 * C)
    p3 = (12.0 - 9.0 * B - 6.0 * C)
    q0 = (8.0 * B + 24.0 * C)
    q1 = (-12.0 * B - 48.0 * C)
    q2 = (6.0 * B + 30.0 * C)
    q3 = (-B - 6.0 * C)

    fw, to_dtype, eps = get_img_types(x)
    absx = fw.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3

    return ((A * ((p3)*absx3 + (p2)*absx2 + (p0)) * to_dtype(absx < 1.)) +
            (A * ((q3)*absx3 + (q2)*absx2 + (q1)*absx + (q0)) *
            to_dtype((absx >= 1.) & (absx < 2.)))
            )

@kernel_width(4)
def catrom(x):
    """Catmull-Rom filter.
    Convolution kernel weight function:
             |  0 ≤ x < 1:   0.5*(2 + x^2*(-5+x*3))
      k(x) = |  1 ≤ x < 2:   0.5*(4 + x*(-8+x*(5-x)))
             |  2 ≤ x    :   0
    """
    fw, to_dtype, eps = get_img_types(x)
    absx = fw.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (
            (1.5*absx3 - 2.5*absx2 + 1.) * to_dtype(absx < 1.) +
            (-0.5*absx3 + 2.5*absx2 -4.0*absx + 2.0) *
            to_dtype((absx >= 1.) & (absx <= 2.))
            )

@kernel_width(2)
def hanning(x):
    """Hanning filter.
    Convolution kernel weight function:
      k(x) = 0.5 + 0.5 * cos(pi * x)
    """
    fw, to_dtype, eps = get_img_types(x)
    return (
            (0.5 + (0.5 * fw.cos(math.pi * x))) * to_dtype(fw.abs(x) < 5)
            )

@kernel_width(2)
def hamming(x):
    """Hamming filter.
    Convolution kernel weight function:
      k(x) = 0.54 + 0.46 * cos(pi * x)
    """
    fw, to_dtype, eps = get_img_types(x)
    return (
            (0.54 + (0.46 * fw.cos(math.pi * x))) * to_dtype(fw.abs(x) < 5)
            )

@kernel_width(4)
def gaussian(x):
    """Gaussian filter.
    Convolution kernel weight function:
      k(x) = exp(-2.0 * x^2) * sqrt(2.0 / pi)
    """
    fw, to_dtype, eps = get_img_types(x)
    x2 = x ** 2
    return (fw.exp(-2.0 * x2) * fw.sqrt(2.0 / math.pi))

def get_imresize_kernel(interpolation=None):
    if isinstance(interpolation, str):
        return {
            "cubic": cubic,
            "lanczos2": lanczos2,
            "lanczos3": lanczos3,
            "lanczos4": lanczos4,
            "lanczos5": lanczos5,
            "box": box,
            "linear": linear,
            "hermite": hermite,
            "bell": bell,
            "mitchell": mitchell,
            "catrom": catrom,
            "hanning": hanning,
            "hamming": hamming,
            "gaussian": gaussian,
            "sinc2": sinc2,
            "sinc3": sinc3,
            "sinc4": sinc4,
            "sinc5": sinc5,
            "blackman2": blackman2,
            "blackman3": blackman3,
            "blackman4": blackman4,
            "blackman5": blackman5,
            None: cubic,  # set default interpolation method as cubic
        }.get(interpolation)
    elif 'function' in str(type(interpolation)):
        return interpolation
    elif not interpolation:
        return cubic


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    """Calculate weights and indices, used for imresize function.
    Args:
        in_length (int): Input length.
        out_length (int): Output length.
        scale (float): Scale factor.
        kernel: the selected interpolation kernel.
        kernel_width (int): Kernel width (diameter).
        antialisaing (bool): Whether to apply anti-aliasing when downsampling.
    """
    if (scale < 1) and antialiasing:
        # Use a modified kernel with larger kernel width to simultaneously 
        # interpolate and antialias
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    p = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(
        0, p - 1, p).view(1, p).expand(out_length, p)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices
    
    # apply kernel
    if (scale < 1) and antialiasing:
        weights = scale * kernel(distance_to_center * scale)
    else:
        weights = kernel(distance_to_center)
    
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)

    # If a column in weights is all zero, get rid of it. 
    # only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


# @torch.no_grad()
def imresize(img, scale, antialiasing=True, interpolation=None, 
             kernel_width=None, out_shape=None):
    """imresize function same as MATLAB.
    It now only supports bicubic.
    The same scale applies for both height and width.
    Args:
        img (Tensor | Numpy array):
            Tensor: Input image with shape (c, h, w), [0, 1] range.
            Numpy: Input image with shape (h, w, c), [0, 1] range.
        scale (float): Scale factor. The same scale applies for both height
            and width.
        antialisaing (bool): Whether to apply anti-aliasing when downsampling.
            if true the kernel is stretched with 1/scale_factor to prevent 
            aliasing (low-pass filtering). Default: True.
        interpolation: the interpolation method to use for resizing. Can 
            be one of the following strings: "cubic", "box", "linear", 
            "lanczos2", "lanczos3", either as literal string values or
            direct methods. Default: "cubic"
        kernel_width: kernel support size (diameter). If not provided, 
            will use the default value for the selected interpolation 
            method.
    Returns:
        Tensor: Output image with shape (c, h, w), [0, 1] range, w/o round.
    """
    # The scale should be the same for H and W
    # input: img: tensor CHW RGB [0,1] or numpy HWC BGR
    # output: tensor CHW RGB [0,1] w/o round or numpy HWC BGR

    # if isinstance(img, np.ndarray):
    if type(img).__module__ == np.__name__:  # numpy type
        numpy_type = True
        change_range = False
        if img.max() > 1:
            #TODO Note: this adds a bit of latency
            img_type = img.dtype
            if np.issubdtype(img_type, np.integer):
                info = np.iinfo
            elif np.issubdtype(img_type, np.floating):
                info = np.finfo
            img = img/info(img_type).max
            change_range = True
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    else:
        numpy_type = False

    in_c, in_h, in_w = img.size()
    out_h, out_w = math.ceil(in_h * scale), math.ceil(in_w * scale)

    # get interpolation method, each method has the matching kernel size
    kernel = get_imresize_kernel(interpolation)
    
    # unless the kernel support size is specified by the user, use the 
    # attribute of the interpolation method
    if not kernel_width:
        kernel_width = kernel.kernel_width

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = calculate_weights_indices(
        in_h, out_h, scale, kernel, kernel_width, antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = calculate_weights_indices(
        in_w, out_w, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(img)

    sym_patch = img[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(
                0, 1).mv(weights_h[i])
    
    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :,
                                       idx:idx + kernel_width].mv(weights_w[i])

    if numpy_type:
        out_2 = out_2.numpy().transpose(1, 2, 0).clip(0,1)
        if change_range:
            out_2 = out_2*info(img_type).max #uint8 = 255
            out_2 = out_2.astype(img_type)
    return out_2


imresize_np = imresize



if __name__ == '__main__':
    # test imresize function
    # read images
    img = cv2.imread('test.png')
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # imresize
    scale = 1 / 4
    import time
    total_time = 0
    for i in range(10):
        start_time = time.time()
        rlt = imresize(img, scale, antialiasing=True)
        use_time = time.time() - start_time
        total_time += use_time
    print('average time: {}'.format(total_time / 10))

    import torchvision.utils
    torchvision.utils.save_image(
        (rlt * 255).round() / 255, 'rlt.png', nrow=1, padding=0, normalize=False)
