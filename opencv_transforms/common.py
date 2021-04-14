from glob import glob
from os.path import join as pjoin

import cv2
import numpy as np
from functools import wraps

try:
    from PIL import Image
    pil_available = True
except ImportError:
    pil_available = False

# try:
#     import cv2
#     cv2_available =  True
# except ImportError:
#     cv2_available = False

#PAD_MOD
_cv2_str2pad = {'constant':cv2.BORDER_CONSTANT,
                   'edge':cv2.BORDER_REPLICATE,
                   'reflect':cv2.BORDER_REFLECT_101,
                   'reflect1':cv2.BORDER_DEFAULT,
                   'symmetric':cv2.BORDER_REFLECT
                  }
#INTER_MODE
_cv2_str2interpolation = {'nearest':cv2.INTER_NEAREST,
                         'NEAREST':cv2.INTER_NEAREST, 
                         'bilinear':cv2.INTER_LINEAR,
                         'BILINEAR':cv2.INTER_LINEAR, 
                         'area':cv2.INTER_AREA,
                         'AREA':cv2.INTER_AREA,
                         'bicubic':cv2.INTER_CUBIC,
                         'BICUBIC':cv2.INTER_CUBIC, 
                         'lanczos':cv2.INTER_LANCZOS4,
                         'LANCZOS':cv2.INTER_LANCZOS4,}
_cv2_interpolation2str={v:k for k,v in _cv2_str2interpolation.items()}


# much faster than iinfo and finfo
MAX_VALUES_BY_DTYPE = {
    np.dtype("int8"): 127,
    np.dtype("uint8"): 255,
    np.dtype("int16"): 32767,
    np.dtype("uint16"): 65535,
    np.dtype("int32"): 2147483647,
    np.dtype("uint32"): 4294967295,
    np.dtype("int64"): 9223372036854775807,
    np.dtype("uint64"): 18446744073709551615,
    np.dtype("float32"): 1.0,
    np.dtype("float64"): 1.0,
}

# TODO: needed?
# MIN_VALUES_BY_DTYPE = {
#     np.dtype("int8"): -128,
#     np.dtype("uint8"): 0,
#     np.dtype("int16"): 32768,
#     np.dtype("uint16"): 0,
#     np.dtype("int32"): 2147483648,
#     np.dtype("uint32"): 0,
#     np.dtype("int64"): 9223372036854775808,
#     np.dtype("uint64"): 0,
#     np.dtype("float32"): -1.0,  # depends on normalization
#     np.dtype("float64"): -1.0,  # depends on normalization
# }


def pil2cv(pil_image):
    open_cv_image = np.array(pil_image)
    if len(open_cv_image.shape) == 2:
        open_cv_image = np.expand_dims(open_cv_image, axis=-1)
    # Convert RGB to BGR
    return open_cv_image[:, :, ::-1].copy()

def cv2pil(open_cv_image):
    if pil_available:
        shape = open_cv_image.shape
        if len(shape) == 3 and shape[-1] == 1: # len(shape) == 2:
            open_cv_image = np.squeeze(open_cv_image, axis=-1)
        if len(shape) == 3 and shape[-1] == 3:
            # Convert BGR to RGB
            open_cv_image = cv2.cvtColor(open_cv_image.copy(), cv2.COLOR_BGR2RGB)
        return Image.fromarray(open_cv_image)
    else:
        raise Exception("PIL not available")

def wrap_cv2_function(func):
    """
    Ensure the image input to the function is a cv2 image
    """
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        if isinstance(img, np.ndarray):
            result = func(img, *args, **kwargs)
        elif pil_available and isinstance(img, Image.Image):
            result = cv2pil(func(pil2cv(img), *args, **kwargs))
        else:
            raise TypeError("Image type not recognized")
        return result
    return wrapped_function

def wrap_pil_function(func):
    """
    Ensure the image input to the function is a pil image
    """
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        if isinstance(img, np.ndarray):
            result = pil2cv(func(cv2pil(img), *args, **kwargs))
        elif pil_available and isinstance(img, Image.Image):
            result = func(img, *args, **kwargs)
        else:
            raise TypeError("Image type not recognized")
        return result
    return wrapped_function

def preserve_shape(func):
    """
    Wrapper to preserve shape of the image
    """
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        # numpy reshape:
        result = result.reshape(shape)
        return result
    return wrapped_function

def preserve_type(func):
    """
    Wrapper to preserve type of the image
    """
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        result = func(img, *args, **kwargs)
        return result.astype(dtype)
    return wrapped_function

def preserve_channel_dim(func):
    """
    Preserve dummy channel dimension for grayscale images
    """
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        if len(shape) == 3 and shape[-1] == 1 and len(result.shape) == 2:
            result = np.expand_dims(result, axis=-1)
            # result = result[:, :, np.newaxis]
        return result
    return wrapped_function

def get_num_channels(img):
    return img.shape[2] if len(img.shape) == 3 else 1

def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more
    than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and 
        rest must be sent via named arguments.
    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.
    Returns:
        numpy.ndarray: Transformed image.
    """
    @wraps(process_fn)
    def __process_fn(img):
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 
                    # 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img
    return __process_fn

def clip(img, dtype, maxval, minval=0):
    return np.clip(img, minval, maxval).astype(dtype)

def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return clip(func(img, *args, **kwargs), dtype, maxval)
    return wrapped_function

def preserve_range_float(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        # if isinstance(img, np.ndarray):
        if type(img).__module__ == np.__name__:
            t_dtype = np.dtype("float32")
            dtype = img.dtype
            if dtype == t_dtype:
                return func(img, *args, **kwargs)

            t_maxval = MAX_VALUES_BY_DTYPE.get(t_dtype, None)
            maxval = MAX_VALUES_BY_DTYPE.get(dtype, None)
            if not maxval:
                if np.issubdtype(dtype, np.integer):
                    info = np.iinfo
                elif np.issubdtype(dtype, np.floating):
                    info = np.finfo
                maxval = info(dtype).max
            img = img.astype(t_dtype)*t_maxval/maxval
            return (func(img, *args, **kwargs)*maxval).astype(dtype)
        else:
            return func(img, *args, **kwargs)
    return wrapped_function







@preserve_shape
def convolve(img, kernel, per_channel=False):
    if per_channel:
        def channel_conv(img, kernel):
            if len(img.shape) < 3:
                img = fix_img_channels(img, 1)
            output = []
            for channel_num in range(img.shape[2]):
                output.append(
                    cv2.filter2D(img[:,:,channel_num], ddepth=-1, kernel=kernel))
            return np.squeeze(np.stack(output,-1))
        return channel_conv(img, kernel)
    else:
        conv_fn = _maybe_process_in_chunks(
            cv2.filter2D, ddepth=-1, kernel=kernel)
        return conv_fn(img)

def norm_kernel(kernel):
    # normalize kernel, so it suns up to 1
    return kernel.astype(np.float32) / np.sum(kernel)

def fetch_kernels(kernels_path, pattern:str='', scale=None, kformat:str='npy'):
    if pattern == 'kernelgan':
        # using the modified kernelGAN file structure.
        kernels = glob(pjoin(kernels_path, '*/kernel_x{}.{}'.format(scale, kformat)))
        if not kernels:
            # try using the original kernelGAN file structure.
            kernels = glob(pjoin(kernels_path, '*/*_kernel_x{}.{}'.format(scale, kformat)))
        # assert kernels, "No kernels found for scale {} in path {}.".format(scale, kernels_path)
    elif pattern == 'matmotion':
        kernels = glob(pjoin(kernels_path, 'm_??.{}'.format(kformat)))
    else:
        kernels = glob(pjoin(kernels_path, '*.{}'.format(kformat)))
    return kernels

def polar2z(r: np.ndarray, θ: np.ndarray) -> np.ndarray:
    """
    Converts a list of radii and angles (radians) and
    into a corresponding list of complex numbers x + yi.
    Arguments:
        r (np.ndarray): radius
        θ (np.ndarray): angle
    Returns:
        [np.ndarray]: list of complex numbers r e^(i theta) as x + iy
    """
    return r * np.exp(1j * θ)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, (list, tuple)):
        param = tuple(param)
    else:
        raise TypeError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)
