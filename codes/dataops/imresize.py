import os
import math
import numpy as np
import torch
import cv2

from dataops.colors import *
from dataops.debug import tmp_vis, describe_numpy, describe_tensor
from dataops.opencv_transforms.opencv_transforms.common import preserve_range_float


# from dataops.debug import timefn, timefn_100


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


# these functions are all interpolation methods.
# x is the distance from the left pixel center

@kernel_width(4)
def cubic(x, a=-0.5):
    """Parametrized cubic kernel (B-Spline order 3).
    Reasonably good quality and faster than Lanczos3, 
    particularly when upsampling.
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
    Note: raising B causes blurring and raising C causes ringing.
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

'''
# @torch.no_grad()
@preserve_range_float
def imresize(img, scale, antialiasing=True, interpolation=None, 
             kernel_width=None, out_shape=None):
    """imresize function same as MATLAB.
    The same scale applies for both height and width.
    Args:
        img (Tensor | Numpy array):
            Tensor: Input image with shape (c, h, w), [0, 1] range.
            Numpy: Input image with shape (h, w, c), any range.
        scale (float): Scale factor. The same scale applies for both height
            and width.
        antialiasing (bool): Whether to apply anti-aliasing when downsampling.
            if true the kernel is stretched with 1/scale_factor to prevent 
            aliasing (low-pass filtering). Default: True.
        interpolation: the interpolation method to use for resizing. The
            following methods are available: "cubic", "box", "linear", 
            "lanczos2", "lanczos3", "lanczos4", "lanczos5", "hermite",
            "bell", "mitchell", "catrom", "hanning", "hamming", "gaussian",
            "sinc2", "sinc3", "sinc4", "sinc5", "blackman2", "blackman3",
            "blackman4" and "blackman5" either as literal string values or
            direct methods. Default: "cubic"
        kernel_width: kernel support size (diameter). If not provided, 
            will use the default value for the selected interpolation 
            method.
    Returns:
        Tensor: Output image with shape (c, h, w), original range, w/o round.
    """
    # if isinstance(img, np.ndarray):
    if type(img).__module__ == np.__name__:  # numpy type
        numpy_type = True
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

    return out_2


imresize_np = imresize
'''


@preserve_range_float
def resize(img, scale_factors=None, out_shape=None,
           interpolation=None, kernel_width=None, 
           antialiasing=True, clip=True):
    """imresize function that produces results identical (PSNR>60dB) 
        to MATLAB for the simple cases (scale_factor * in_size is 
        integer). 
        Adapted from: https://github.com/assafshocher/ResizeRight
    Args:
        img (Tensor | Numpy array):
            Tensor: Input image with shape (c, h, w), [0, 1] range.
            Numpy: Input image with shape (h, w, c), any range.
        scale_factors: Scale factors. can be specified as: 
            - one scalar: scale (assumes that the first two dims will be 
            resized with this scale for Numpy or last two dims for PyTorch) 
            - a list or tupple of scales, one for each dimension you want 
            to resize. Note: if length of the list is L then first L dims 
            will be rescaled for Numpy and last L for PyTorch.
            - not specified - then it will be calculated using output_size.
        out_shape (list or tuple): if shorter than input.shape then only 
            the first/last (depending np/torch) dims are resized. If not 
            specified, can be calcualated from scale_factor.
        interpolation: the interpolation method to use for resizing. The
            following methods are available: "cubic", "box", "linear", 
            "lanczos2", "lanczos3", "lanczos4", "lanczos5", "hermite",
            "bell", "mitchell", "catrom", "hanning", "hamming", "gaussian",
            "sinc2", "sinc3", "sinc4", "sinc5", "blackman2", "blackman3",
            "blackman4" and "blackman5" either as literal string values or
            direct methods. Default: "cubic".
        antialiasing (bool): Whether to apply anti-aliasing when downsampling.
            if true the kernel is stretched with 1/scale_factor to prevent 
            aliasing (low-pass filtering). Default: True.
        kernel_width: kernel support size (diameter). This is the support 
            of the interpolation function, i.e length of non-zero segment 
            over its 1d input domain. this is a characteristic of the 
            function. If not provided, will use the default value for 
            the selected interpolation method, ie. for bicubic:4, linear:2,
            lanczos2:4, lanczos3:6, box:1, etc
    Returns:
        Tensor: Output image with shape (c, h, w), original range, w/o round.
    """

    # get properties of the input img tensor
    in_shape, n_dims = img.shape, img.ndim

    # fw stands for framework that can be either numpy or torch,
    # determined by the input img type
    fw = np if isinstance(img, np.ndarray) else torch
    eps = fw.finfo(fw.float32).eps

    # get interpolation method, each method has the matching kernel size
    kernel = get_imresize_kernel(interpolation)

    # set missing scale factors or output shape one according to another,
    # fail if both missing
    scale_factors, out_shape = set_scale_and_out_shape(in_shape, out_shape,
                                                    scale_factors, fw)

    # sort indices of dimensions according to scale of each dimension.
    # since it goes dim by dim, this is efficient
    sorted_filtered_dims_and_scales = [(dim, scale_factors[dim])
                                       for dim in sorted(range(n_dims),
                                       key=lambda ind: scale_factors[ind])
                                       if scale_factors[dim] != 1.]
    
    # unless the kernel support size is specified by the user, use the 
    # attribute of the interpolation method
    if kernel_width is None:
        kernel_width = kernel.kernel_width
        
    # when using pytorch, need to know what is the input tensor device
    device = None
    if fw is torch:
        device = img.device

    # output begins identical to input img and changes with each iteration
    output = img

    # iterate over dims
    for dim, scale_factor in sorted_filtered_dims_and_scales:

        # get 1d set of weights and fields of view for each output location
        # along this dim
        field_of_view, weights = prepare_weights_and_field_of_view_1d(
            dim, scale_factor, in_shape[dim], out_shape[dim], kernel,
            kernel_width, antialiasing, fw, eps, device)

        # multiply the weights by the values in the field of view and
        # aggreagate
        output = apply_weights(output, field_of_view, weights, dim, n_dims,
                               fw)
    
    output = fw_clip(output, fw) if clip else output

    return output





def apply_antialiasing(kernel, kernel_width, scale_factor,
                                 antialiasing):
    """
    Antialiasing is "stretching" the field of view according to the scale
    factor (only for downscaling). This is produces the low-pass filtering. 
    This requires modifying both the interpolation (stretching the 1d
    function and multiplying by the scale-factor) and the window size.
    """
    if scale_factor >= 1.0 or not antialiasing:
        return kernel, kernel_width
    cur_kernel = (lambda arg: scale_factor *
                         kernel(scale_factor * arg))
    cur_kernel_width = kernel_width / scale_factor
    return cur_kernel, cur_kernel_width


def prepare_weights_and_field_of_view_1d(dim, scale_factor, in_sz, out_sz,
                                         kernel, kernel_width, 
                                         antialiasing, fw, eps, device=None):
    # If antialiasing is taking place, modify the kernel size and the
    # interpolation method
    kernel, cur_kernel_width = apply_antialiasing(
                                                kernel,
                                                kernel_width,
                                                scale_factor,
                                                antialiasing)

    # STEP 1- PROJECTED GRID: The non-integer locations of the projection of
    # output pixel locations to the input tensor
    projected_grid = get_projected_grid(in_sz, out_sz, scale_factor, fw, device)

    # STEP 2- FIELDS OF VIEW: for each output pixels, map the input pixels
    # that influence it
    field_of_view = get_field_of_view(projected_grid, cur_kernel_width, 
                                      in_sz, fw, eps)

    # STEP 3- CALCULATE WEIGHTS: Match a set of weights to the pixels in the
    # field of view for each output pixel
    weights = get_weights(kernel, projected_grid, field_of_view)

    return field_of_view, weights


def apply_weights(img, field_of_view, weights, dim, n_dims, fw):
    # STEP 4- APPLY WEIGHTS: Each output pixel is calculated by multiplying
    # its set of weights with the pixel values in its field of view.
    # We now multiply the fields of view with their matching weights.
    # We do this by tensor multiplication and broadcasting.
    # this step is separated to a different function, so that it can be
    # repeated with the same calculated weights and fields.

    # for these operations we assume the resized dim is the first one.
    # so we transpose and will transpose back after multiplying
    tmp_img = fw_swapaxes(img, dim, 0, fw)

    # field_of_view is a tensor of order 2: for each output (1d location
    # along cur dim)- a list of 1d neighbors locations.
    # note that this whole operations is applied to each dim separately,
    # this is why it is all in 1d.
    # neighbors = tmp_img[field_of_view] is a tensor of order image_dims+1:
    # for each output pixel (this time indicated in all dims), these are the
    # values of the neighbors in the 1d field of view. note that we only
    # consider neighbors along the current dim, but such set exists for every
    # multi-dim location, hence the final tensor order is image_dims+1.
    neighbors = tmp_img[field_of_view]

    # weights is an order 2 tensor: for each output location along 1d- a list
    # of weighs matching the field of view. we augment it with ones, for
    # broadcasting, so that when multiplies some tensor the weights affect
    # only its first dim.
    tmp_weights = fw.reshape(weights, (*weights.shape, * [1] * (n_dims - 1)))

    # now we simply multiply the weights with the neighbors, and then sum
    # along the field of view, to get a single value per out pixel
    tmp_output = (neighbors * tmp_weights).sum(1)

    # we transpose back the resized dim to its original position
    return fw_swapaxes(tmp_output, 0, dim, fw)


def set_scale_and_out_shape(in_shape, out_shape, scale_factors, fw):
    # eventually we must have both scale-factors and out-sizes for all in/out
    # dims. however, we support many possible partial arguments
    if scale_factors is None and out_shape is None:
        raise ValueError("either scale_factors or out_shape should be "
                         "provided")
    if out_shape is not None:
        # if out_shape has less dims than in_shape, we defaultly resize the
        # first dims for numpy and last dims for torch
        out_shape = (list(out_shape) + list(in_shape[:-len(out_shape)])
                     if fw is np
                     else list(in_shape[:-len(out_shape)]) + list(out_shape))
        if scale_factors is None:
            # if no scale given, we calculate it as the out to in ratio
            # (not recomended)
            scale_factors = [out_sz / in_sz for out_sz, in_sz
                             in zip(out_shape, in_shape)]
    if scale_factors is not None:
        # by default, if a single number is given as scale, we assume resizing
        # two dims (most common are images with 2 spatial dims)
        scale_factors = (scale_factors
                         if isinstance(scale_factors, (list, tuple))
                         else [scale_factors, scale_factors])
        # if less scale_factors than in_shape dims, we defaultly resize the
        # first dims for numpy and last dims for torch
        scale_factors = (list(scale_factors) + [1] *
                         (len(in_shape) - len(scale_factors)) if fw is np
                         else [1] * (len(in_shape) - len(scale_factors)) +
                         list(scale_factors))
        if out_shape is None:
            # when no out_shape given, it is calculated by multiplying the
            # scale by the in_shape (not recomended)
            out_shape = [math.ceil(scale_factor * in_sz)
                         for scale_factor, in_sz in
                         zip(scale_factors, in_shape)]
        # next line intentionally after out_shape determined for stability
        scale_factors = [float(sf) for sf in scale_factors]
    return scale_factors, out_shape


def get_projected_grid(in_sz, out_sz, scale_factor, fw, device=None):
    # we start by having the ouput coordinates which are just integer locations
    out_coordinates = fw.arange(out_sz)
    
    # if using torch we need to match the grid tensor device to the input device
    out_coordinates = fw_set_device(out_coordinates, device, fw)
        
    # This is projecting the ouput pixel locations in 1d to the input tensor,
    # as non-integer locations.
    # the following formula is derived in the paper
    # "From Discrete to Continuous Convolutions" by Shocher et al.
    return (out_coordinates / scale_factor +
            (in_sz - 1) / 2 - (out_sz - 1) / (2 * scale_factor))


def get_field_of_view(projected_grid, cur_kernel_width, in_sz, fw, eps):
    # for each output pixel, map which input pixels influence it, in 1d.
    # we start by calculating the leftmost neighbor, using half of the window
    # size (eps is for when boundary is exact int).
    # What is the left-most pixel that can be involved in the computation?
    left_boundaries = fw_ceil(projected_grid - cur_kernel_width / 2 - eps, fw)

    # then we simply take all the pixel centers in the field by counting
    # window size pixels from the left boundary
    ordinal_numbers = fw.arange(math.ceil(cur_kernel_width - eps))
    
    # in case using torch we need to match the device    
    if fw is torch:
        ordinal_numbers = fw_set_device(ordinal_numbers, projected_grid.device, fw)
    else:
        ordinal_numbers = fw_set_device(ordinal_numbers, projected_grid, fw)
    field_of_view = left_boundaries[:, None] + ordinal_numbers

    # next we do a trick instead of padding, we map the field of view so that
    # it would be like mirror padding, without actually padding
    # (which would require enlarging the input tensor)
    mirror = fw_cat((fw.arange(in_sz), fw.arange(in_sz - 1, -1, step=-1)), fw)
    field_of_view = mirror[fw.remainder(field_of_view, mirror.shape[0])]
    if fw is torch:
        field_of_view = fw_set_device(field_of_view,projected_grid.device, fw)
    else:
        field_of_view = fw_set_device(field_of_view,projected_grid, fw)
    return field_of_view


def get_weights(kernel, projected_grid, field_of_view):
    # the set of weights per each output pixels is the result of the chosen
    # interpolation method applied to the distances between projected grid
    # locations and the pixel-centers in the field of view (distances are
    # directed, can be positive or negative)
    weights = kernel(projected_grid[:, None] - field_of_view)

    # we now carefully normalize the weights to sum to 1 per each output pixel
    weights_sum = weights.sum(1, keepdims=True)
    weights_sum[weights_sum == 0] = 1
    return weights / weights_sum


def fw_ceil(x, fw):
    if fw is np:
        return fw.int_(fw.ceil(x))
    else:
        return x.ceil().long()


def fw_cat(x, fw):
    if fw is np:
        return fw.concatenate(x)
    else:
        return fw.cat(x)


def fw_swapaxes(x, ax_1, ax_2, fw):
    if fw is np:
        return fw.swapaxes(x, ax_1, ax_2)
    else:
        return x.transpose(ax_1, ax_2)

def fw_set_device(x, device, fw):
    if fw is np:
        return x
    else:
        return x.to(device)

def fw_clip(x, fw):
    if fw is np:
        return fw.clip(x, 0, 1)
    else:
        return fw.clamp(x, 0, 1)




"""
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

"""
