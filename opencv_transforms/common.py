import cv2
import numpy as np
from functools import wraps

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
