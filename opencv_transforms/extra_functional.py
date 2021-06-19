# from __future__ import division
#import torch
import math
import random
import warnings

import numpy as np
import cv2
#import numbers
#import types
#import collections
#import warnings

from .common import preserve_shape, preserve_type, preserve_channel_dim, _maybe_process_in_chunks, polar2z, norm_kernel
from .common import _cv2_str2interpolation, _cv2_interpolation2str, MAX_VALUES_BY_DTYPE, from_float, to_float

## Below are new augmentations not available in the original ~torchvision.transforms


@preserve_type
def perspective(img, fov=45, anglex=0, angley=0, anglez=0, shear=0,
                translate=(0, 0), scale=(1, 1), resample='BILINEAR', fillcolor=(0, 0, 0)):
    r"""
    This function is partly referred to in 
    https://blog.csdn.net/dcrmg/article/details/80273818
    """
    gray_scale = False

    if len(img.shape) == 2:
        gray_scale = True
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w, _ = img.shape
    centery = h * 0.5
    centerx = w * 0.5

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
    affine_matrix = np.array([[M00, M01, M02], [M10, M11, M12], [0, 0, 1]], dtype=np.float32)
    # -------------------------------------------------------------------------------
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(math.radians(fov / 2))

    radx = math.radians(anglex)
    rady = math.radians(angley)

    sinx = math.sin(radx)
    cosx = math.cos(radx)
    siny = math.sin(rady)
    cosy = math.cos(rady)

    r = np.array([[cosy, 0, -siny, 0],
                  [-siny * sinx, cosx, -sinx * cosy, 0],
                  [cosx * siny, sinx, cosx * cosy, 0],
                  [0, 0, 0, 1]])

    pcenter = np.array([centerx, centery, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    perspective_matrix = cv2.getPerspectiveTransform(org, dst)
    total_matrix = perspective_matrix @ affine_matrix

    result_img = cv2.warpPerspective(img, total_matrix, (w, h), flags=_cv2_str2interpolation[resample],
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=fillcolor)
    if gray_scale:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)
    return result_img


@preserve_type
def noise_gaussian(img: np.ndarray, mean=0.0, std=1.0, gtype='color'):
    r"""Add OpenCV Gaussian noise (Additive) to the image.
    Args:
        img (numpy ndarray): Image to be augmented.
        mean (float): Mean (“centre”) of the Gaussian distribution. Default=0.0
        std (float): Standard deviation (spread or “width”) of the Gaussian distribution. Default=1.0
        gtype ('str': ``color`` or ``bw``): Type of Gaussian noise to add, either colored or black and white. 
            Default='color' (Note: can introduce color noise during training)
    Returns:
        numpy ndarray: version of the image with Gaussian noise added.
    """
    h,w,c = img.shape
    
    if gtype == 'bw':
        c = 1

    gauss = np.random.normal(loc=mean, scale=std, size=(h,w,c)).astype(np.float32)
    noisy = np.clip((1 + gauss) * img.astype(np.float32), 0, 255) 
    
    return noisy


@preserve_type
def noise_poisson(img):
    r"""Add OpenCV Poisson noise to the image.
        Important: Poisson noise is not additive like Gaussian, it's dependant on 
        the image values. Read: https://tomroelandts.com/articles/gaussian-noise-is-added-poisson-noise-is-applied
    Args:
        img (numpy ndarray): Image to be augmented.
    Returns:
        numpy ndarray: version of the image with Poisson noise added.
    """
    img = img.astype(np.float32)/255.0

    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = 255 * np.clip(np.random.poisson(img.astype(np.float32) * vals) / float(vals), 0, 1)
    return noisy


@preserve_type
def noise_salt_and_pepper(img, prob=0.01):
    r"""Adds "Salt & Pepper" noise to an image.
    Args:
        img (numpy ndarray): Image to be augmented.
        prob (float): probability (threshold) that controls level of noise
    Returns:
        numpy ndarray: version of the image with Poisson noise added.
    """
    #alt 1: black and white s&p
    rnd = np.random.rand(img.shape[0], img.shape[1])
    noisy = img.copy()
    noisy[rnd < prob/2] = 0.0
    noisy[rnd > 1 - prob/2] = 255.0

    #alt 2: coming up as colored s&p
    #randomize the amount of noise and the ratio between salt and pepper
    # amount = np.random.uniform(0.02, 0.15) 
    # s_vs_p = np.random.uniform(0.3, 0.7) # average = 50% salt, 50% pepper #q
    # noisy = np.copy(img)
    # flipped = np.random.choice([True, False], size=noisy.shape, p=[amount, 1 - amount])
    # # Salted mode
    # salted = np.random.choice([True, False], size=noisy.shape, p=[s_vs_p, 1 - s_vs_p])
    # # Pepper mode
    # peppered = ~salted
    # noisy[flipped & salted] = 1

    return noisy


@preserve_type
def noise_speckle(img: np.ndarray, mean=0.0, std=1.0, gtype='color'):
    r"""Add Speckle noise to the image.
    Args:
        img (numpy ndarray): Image to be augmented.
        mean (float): Mean (“centre”) of the distribution. Default=0.0
        std (float): Standard deviation (spread or “width”) of the distribution. Default=1.0
        type ('str': ``color`` or ``bw``): Type of noise to add, either colored or black and white. 
            Default='color' (Note: can introduce color noise during training)
    Returns:
        numpy ndarray: version of the image with Speckle noise added.
    """
    h,w,c = img.shape
    
    if gtype == 'bw':
        c = 1

    speckle = np.random.normal(loc=mean, scale=std ** 0.5, size=(h,w,c)).astype(np.float32)

    noisy = img + img * speckle
    noisy = np.clip(noisy, 0, 255) 
    
    return noisy


@preserve_shape
@preserve_type
def compression(img: np.ndarray, quality=20, image_type='.jpeg'):
    r"""Compress the image using OpenCV.
    Args:
        img (numpy ndarray): Image to be compressed.
        quality (int: [0,100]): Compression quality for the image. 
            Lower values represent higher compression and lower 
            quality. Default=20
        image_type (str): select between '.jpeg' or '.webp'
            compression. Default='.jpeg'.
    Returns:
        numpy ndarray: version of the image with compression.
    """
    if image_type in [".jpeg", ".jpg"]:
        quality_flag = cv2.IMWRITE_JPEG_QUALITY
    elif image_type == ".webp":
        quality_flag = cv2.IMWRITE_WEBP_QUALITY
    else:
        NotImplementedError("Only '.jpg' and '.webp' compression transforms are implemented. ")

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        warnings.warn(
            "Image compression augmentation "
            "is most effective with uint8 inputs, "
            "{} is used as input.".format(input_dtype),
            UserWarning,
        )
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise TypeError("Unexpected dtype {} for compression augmentation".format(input_dtype))

    #encoding parameters
    encode_param = [int(quality_flag), quality]
    # encode
    is_success, encimg = cv2.imencode(image_type, img, encode_param) 
    
    # decode
    compressed_img = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)

    if needs_float:
        compressed_img = to_float(compressed_img, max_value=255)

    return compressed_img


#Get a valid kernel for the blur operations
def valid_kernel(h: int, w: int, kernel_size: int):
    #make sure the kernel size is smaller than the image dimensions
    kernel_size = min(kernel_size,h,w)

    #round up and cast to int
    kernel_size = int(np.ceil(kernel_size))
    #kernel size has to be an odd number
    if kernel_size % 2 == 0:
        kernel_size+=1 
    
    return kernel_size

@preserve_shape
@preserve_type
def average_blur(img: np.ndarray, kernel_size: int = 3):
    r"""Blurs an image using OpenCV Gaussian Blur.
    Args:
        img (numpy ndarray): Image to be augmented.
        kernel_size (int): size of the blur filter to use. Default: 3.
    Returns:
        numpy ndarray: version of the image with blur applied.
    """
    h, w = img.shape[0:2]

    #Get a valid kernel size
    kernel_size = valid_kernel(h,w,kernel_size)
    
    #Averaging Filter Blur (Homogeneous filter)
    # blurred = cv2.blur(img, (kernel_size,kernel_size))
    blur_fn = _maybe_process_in_chunks(cv2.blur, ksize=(kernel_size,kernel_size))
    return blur_fn(img)

#Box blur and average blur should be the same
@preserve_shape
@preserve_type
def box_blur(img: np.ndarray, kernel_size: int = 3):
    r"""Blurs an image using OpenCV Gaussian Blur.
    Args:
        img (numpy ndarray): Image to be augmented.
        kernel_size (int): size of the blur filter to use. Default: 3.
    Returns:
        numpy ndarray: version of the image with blur applied.
    """
    h, w = img.shape[0:2]

    #Get a valid kernel size
    kernel_size = valid_kernel(h,w,kernel_size)
    
    #Box Filter Blur 
    # blurred = cv2.boxFilter(img,ddepth=-1,ksize=(kernel_size,kernel_size))
    blur_fn = _maybe_process_in_chunks(cv2.boxFilter, ddepth=-1, ksize=(kernel_size,kernel_size))
    return blur_fn(img)

@preserve_shape
@preserve_type
def gaussian_blur(img: np.ndarray, kernel_size: int = 3, sigma=0.0):
    r"""Blurs an image using OpenCV Gaussian Blur.
    Args:
        img (numpy ndarray): Image to be augmented.
        kernel_size (int): size of the blur filter to use. Default: 3.
    Returns:
        numpy ndarray: version of the image with blur applied.
    Note: When sigma=0, it is computed as `sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8`
    """
    h, w = img.shape[0:2]

    #Get a valid kernel size
    kernel_size = valid_kernel(h,w,kernel_size)
    
    #Gaussian Filter Blur
    # blurred = cv2.GaussianBlur(img,(kernel_size,kernel_size),0)
    blur_fn = _maybe_process_in_chunks(cv2.GaussianBlur, ksize=(kernel_size,kernel_size), sigmaX=sigma)
    return blur_fn(img)

@preserve_shape
@preserve_type
def median_blur(img: np.ndarray, kernel_size: int = 3):
    r"""Blurs an image using OpenCV Median Blur.
    Args:
        img (numpy ndarray): Image to be augmented.
        kernel_size (int): size of the blur filter to use. Default: 3.
    Returns:
        numpy ndarray: version of the image with blur applied.
    """
    h, w = img.shape[0:2]

    #Get a valid kernel size
    kernel_size = valid_kernel(h,w,kernel_size)
    
    #Median Filter Blur
    blur_fn = _maybe_process_in_chunks(cv2.medianBlur, ksize=kernel_size)
    return blur_fn(img)

#Needs testing
@preserve_shape
@preserve_type
def bilateral_blur(img: np.ndarray, kernel_size: int = 3, sigmaColor: int = 5, sigmaSpace: int = 5):
    r"""Blurs an image using OpenCV Gaussian Blur.
    Args:
        img (numpy ndarray): Image to be augmented.
        kernel_size (int): size of the blur filter to use. Default: 3. Large filters 
            (d > 5) are very slow, so it is recommended to use d=5 for real-time 
            applications, and perhaps d=9 for offline applications that need heavy 
            noise filtering.
        Sigma values: For simplicity, you can set the 2 sigma values to be the same. 
            If they are small (< 10), the filter will not have much effect, whereas 
            if they are large (> 150), they will have a very strong effect, making 
            the image look "cartoonish".
        sigmaColor	Filter sigma in the color space. A larger value of the parameter 
            means that farther colors within the pixel neighborhood (see sigmaSpace) 
            will be mixed together, resulting in larger areas of semi-equal color.
        sigmaSpace	Filter sigma in the coordinate space. A larger value of the parameter 
            means that farther pixels will influence each other as long as their colors 
            are close enough (see sigmaColor ). When d>0, it specifies the neighborhood 
            size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
        borderType	border mode used to extrapolate pixels outside of the image
    Returns:
        numpy ndarray: version of the image with blur applied.
    """
    h, w = img.shape[0:2]

    #Get a valid kernel size
    kernel_size = valid_kernel(h,w,kernel_size)
    #Bilateral filter doesn't appear to work with kernel_size > 9, check
    # if kernel_size > 9:
    #     kernel_size = 9
    
    #Bilateral Filter
    # blurred = cv2.bilateralFilter(img,kernel_size,sigmaColor,sigmaSpace)
    blur_fn = _maybe_process_in_chunks(
        cv2.bilateralFilter,
        d=kernel_size,
        sigmaColor=sigmaColor,
        sigmaSpace=sigmaSpace)
    return blur_fn(img)


def apply_kmeans(Z, K=8):
    """ Utility function to apply cv2 k-means.
    Defines criteria, uses number of clusters (K) and
    applies kmeans() algorithm
    """
    K = len(Z) if K > len(Z) else K
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centroids = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return ret, labels, centroids


@preserve_type
def km_quantize(img, K=8, single_rnd_color=False):
    r""" Color quantization with CV2 K-Means clustering.
    Color quantization is the process of reducing number of colors
        in an image. Here we use k-means clustering for color
        quantization. There are 3 features (R,G,B) in the images,
        so they are reshaped to an array of Px3 size (P is number
        of pixels in an image, M*N, where M=img.shape[1] and
        N=img.shape[0]). And after the clustering, we apply centroid
        values (it is also R,G,B) to all pixels, such that resulting
        image will have specified number of colors. Finally, it's
        reshaped back to the shape of the original image.
    Args:
        img (numpy ndarray): Image to be quantized.
    Returns:
        numpy ndarray: the quantized image.
    """

    # reshape to (M*N, 3)
    Z = img.reshape((-1,3)) 
    # convert image to np.float32
    Z = np.float32(Z)

    _, labels, centroids = apply_kmeans(Z, K=8)

    res = centroids[labels.flatten()]
    return res.reshape((img.shape))


def simple_quantize(image, rgb_range):
    r""" Simple image quantization nased on color ranges.
    """
    pixel_range = 255. / rgb_range
    image = image.astype(np.float32)
    return (255.*(image*pixel_range/255.).clip(0, 255).round()/(pixel_range)).astype(np.uint8)


@preserve_type
def noise_dither_bayer(img: np.ndarray):
    r"""Adds colored bayer dithering noise to the image.
    Args:
        img (numpy ndarray): Image to be dithered.
    Returns:
        numpy ndarray: version of the image with dithering applied.
    """    
    imgtype = img.dtype
    size = img.shape

    #Note: these are very slow for large images, must crop first before applying.
    # Bayer works more or less. I think it's missing a part of the image, the
    # dithering pattern is apparent, but the quantized (color palette) is not there. 
    # Still enough for models to learn dedithering
    bayer_matrix = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) #/256 #4x4 Bayer matrix
    
    bayer_matrix = bayer_matrix*16
    
    red = img[:,:,2] #/255.
    green = img[:,:,1] #/255.
    blue = img[:,:,0] #/255.
    
    img_split = np.zeros((img.shape[0], img.shape[1], 3), dtype = imgtype)
    
    for values, color, channel in zip((red, green, blue), ('red', 'green', 'blue'), (2,1,0)):
        for i in range(0, values.shape[0]):
            for j in range(0, values.shape[1]):
                x = np.mod(i, 4)
                y = np.mod(j, 4)
                if values[i, j] > bayer_matrix[x, y]:
                    img_split[i,j,channel] = 255 #1
    dithered = img_split #*255.
    
    return dithered

@preserve_type
def noise_dither_fs(img: np.ndarray, samplingF = 1):
    r"""Adds colored Floyd-Steinberg dithering noise to the image.

    Floyd–Steinberg dithering is an image dithering algorithm first published in
    1976 by Robert W. Floyd and Louis Steinberg. It is commonly used by image 
    manipulation software, for example when an image is converted into GIF format 
    that is restricted to a maximum of 256 colors.

    The algorithm achieves dithering using error diffusion, meaning it pushes 
    (adds) the residual quantization error of a pixel onto its neighboring 
    pixels, to be dealt with later.

    https://en.wikipedia.org/wiki/Floyd–Steinberg_dithering
    Pseudocode:
        for each y from top to bottom
           for each x from left to right
              oldpixel  := pixel[x][y]
              newpixel  := find_closest_palette_color(oldpixel)
              pixel[x][y]  := newpixel
              quant_error  := oldpixel - newpixel
              pixel[x+1][y  ] := pixel[x+1][y  ] + quant_error * 7/16
              pixel[x-1][y+1] := pixel[x-1][y+1] + quant_error * 3/16
              pixel[x  ][y+1] := pixel[x  ][y+1] + quant_error * 5/16
              pixel[x+1][y+1] := pixel[x+1][y+1] + quant_error * 1/16
        find_closest_palette_color(oldpixel) = floor(oldpixel / 256)

    Args:
        img (numpy ndarray): Image to be dithered.
        samplingF: controls the amount of dithering 
    Returns:
        numpy ndarray: version of the image with dithering applied.
    """
    #for Floyd-Steinberg dithering noise
    def minmax(v):
        v = min(v, 255)
        v = max(v, 0)
        return v
    
    size = img.shape

    #Note: these are very slow for large images, must crop first before applying.    
    #Floyd-Steinberg
    re_fs = img.copy()
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
    dithered = re_fs

    return dithered

def noise_dither_avg_bw(img):
    """
        https://github.com/QunixZ/Image_Dithering_Implements/blob/master/HW1.py
    """
    if len(img.shape) > 2 and img.shape[2] != 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    threshold = np.average(img)
    re_aver = np.where(img < threshold, 0, 255).astype(np.uint8)
    #re_aver = cv2.cvtColor(re_aver,cv2.COLOR_GRAY2RGB)

    return re_aver

def noise_dither_bayer_bw(img):
    """
        https://github.com/QunixZ/Image_Dithering_Implements/blob/master/HW1.py
    """
    if len(img.shape) > 2 and img.shape[2] != 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    size = img.shape

    re_bayer = np.zeros(size, dtype=np.uint8) #this dtype may be wrong if images in range (0,1)
    bayer_matrix = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) #4x4 Bayer matrix
    
    bayer_matrix = bayer_matrix*16 

    for i in range(0, size[0]):
        for j in range(0, size[1]):
            x = np.mod(i, 4)
            y = np.mod(j, 4)
            if img[i, j] > bayer_matrix[x, y]:
                re_bayer[i, j] = 255

    #re_bayer = cv2.cvtColor(re_bayer,cv2.COLOR_GRAY2RGB)
    return re_bayer

def noise_dither_bin_bw(img):
    """
        https://github.com/QunixZ/Image_Dithering_Implements/blob/master/HW1.py
    """
    if len(img.shape) > 2 and img.shape[2] != 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_bw = np.where(img < 127, 0, 255).astype(np.uint8)
    #img_bw = cv2.cvtColor(img_bw,cv2.COLOR_GRAY2RGB)
    return img_bw


def noise_dither_fs_bw(img, samplingF = 1):
    """
        https://github.com/QunixZ/Image_Dithering_Implements/blob/master/HW1.py
    """
    #for Floyd-Steinberg dithering noise
    def minmax(v): 
        v = min(v, 255)
        v = max(v, 0)
        return v

    if len(img.shape) > 2 and img.shape[2] != 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    size = img.shape

    re_fs = img 
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
            
    #re_fs = cv2.cvtColor(re_fs,cv2.COLOR_GRAY2RGB)
    return re_fs

def noise_dither_random_bw(img):
    if len(img.shape) > 2 and img.shape[2] != 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    size = img.shape

    re_rand = np.zeros(size, dtype=np.uint8) #this dtype may be wrong if images in range (0,1)

    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if img[i, j] < np.random.uniform(0, 256):
                re_rand[i, j] = 0
            else:
                re_rand[i, j] = 255

    #re_rand = cv2.cvtColor(re_rand,cv2.COLOR_GRAY2RGB)
    return re_rand


#translate_chan()
#TBD


def filter_max_rgb(img: np.ndarray):
    r"""The Max RGB filter is used to visualize which channel 
        contributes most to a given area of an image. 
        Can be used for simple color-based segmentation.
        More infotmation on: https://www.pyimagesearch.com/2015/09/28/implementing-the-max-rgb-filter-in-opencv/
    Args:
        img (numpy ndarray): Image to be filtered.
    Returns:
        numpy ndarray: version of the image after Max RGB filter.
    """
    # split the image into its BGR components
    (B, G, R) = cv2.split(img)

    # find the maximum pixel intensity values for each
    # (x, y)-coordinate,, then set all pixel values less
    # than M to zero
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0

    # merge the channels back together and return the image
    return cv2.merge([B, G, R])

@preserve_type
def filter_colorbalance(img: np.ndarray, percent=1):
    r"""Simple color balance algorithm (similar to Photoshop "auto levels")
        More infotmation on: 
        https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc#gistcomment-3025656
        http://www.morethantechnical.com/2015/01/14/simplest-color-balance-with-opencv-wcode/
        https://web.stanford.edu/~sujason/ColorBalancing/simplestcb.html
    Args:
        img (numpy ndarray): Image to be filtered.
        percent (int): amount of balance to apply
    Returns:
        numpy ndarray: version of the image after Simple Color Balance filter.
    """

    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        channel = channel.astype(np.uint8)
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))

    return cv2.merge(out_channels)


@preserve_shape
@preserve_type
def filter_unsharp(img: np.ndarray, blur_algo='median', kernel_size=None, strength=0.3, unsharp_algo='laplacian'):
    r"""Unsharp mask filter, used to sharpen images to make edges and interfaces look crisper.
        More infotmation on: 
        https://www.idtools.com.au/unsharp-masking-python-opencv/
    Args:
        img (numpy ndarray): Image to be filtered.
        blur_algo (str: 'median' or None): blur algorithm to use if using laplacian (LoG) filter. Default: 'median'
        strength (float: [0,1]): strength of the filter to be applied. Default: 0.3 (30%)
        unsharp_algo (str: 'DoG' or 'laplacian'): selection of algorithm between LoG and DoG. Default: 'laplacian'
    Returns:
        numpy ndarray: version of the image after Unsharp Mask.
    """
    #can randomize strength from 0.5 to 0.8
    # if strength is None:
    #     strength = np.random.uniform(0.3, 0.9)

    if unsharp_algo == 'DoG':
        # If using Difference of Gauss (DoG)
        # run a 5x5 gaussian blur then a 3x3 gaussian blur
        blur5 = gaussian_blur(img.astype(np.float32), 5)
        blur3 = gaussian_blur(img.astype(np.float32), 3)
        DoGim = blur5 - blur3
        img_out = img - strength*DoGim
        img_out = img_out.astype(np.uint8)
    
    else: 
        # 'laplacian': using LoG (actually, median blur instead of gaussian)
        #randomize kernel_size between 1, 3 and 5
        if kernel_size is None:
            kernel_sizes = [1, 3, 5] #TODO: ks 5 is causing errors
            kernel_size = random.choice(kernel_sizes)
        # Median filtering (could be Gaussian for proper LoG)
        #gray_image_mf = median_filter(gray_image, 1)
        if blur_algo == 'median':
            smooth = median_blur(img.astype(np.uint8), kernel_size)

        # Calculate the Laplacian
        # (LoG, or in this case, Laplacian of Median)
        lap = cv2.Laplacian(smooth, cv2.CV_64F)
        if len(lap.shape) == 2:
            lap = lap.reshape(lap.shape[0], lap.shape[1], 1)

        # Calculate the sharpened image
        img_out = img - strength*lap

    # Saturate the pixels in either direction
    img_out[img_out>255] = 255
    img_out[img_out<0] = 0
    
    return img_out


def binarize(img, threshold):
    r"""Binarize operation (ie. for edge detectors)
    Args:
        threshold: threshold value for binarize option
    """
    #img = img > threshold
    img[img < threshold] = 0.
    return img

@preserve_shape
@preserve_type
def filter_canny(img: np.ndarray, sigma:float=0.33, 
            bin_thresh:bool=False, threshold:int=127, to_rgb:bool=False):
    r"""Automatic Canny filter for edge detection
    Args:
        img: Image to be filtered.
        sigma: standard deviation from the median to automatically calculate minimun 
            values and maximum values thresholds. Default: 0.33.
        bin_thresh: flag to apply binarize (threshold) operation

    Returns:
        numpy ndarray: version of the image after Canny filter.
    """
    if len(img.shape) > 2 and img.shape[2] != 1:
        to_rgb = True

    # compute the median of the single channel pixel intensities
    median = np.median(img)

    # apply automatic Canny edge detection using the computed median
    minVal = int(max(0, (1.0 - sigma) * median))
    maxVal = int(min(255, (1.0 + sigma) * median))
    edged = cv2.Canny(img, minVal, maxVal)

    if bin_thresh:
        edged = binarize(edged, threshold)

    if to_rgb:
        edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)

    # return the edged image
    return edged







def simple_motion_kernel(kernel_size):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)

    # get random points to draw
    xs, xe = random.randint(0, kernel_size - 1), random.randint(0, kernel_size - 1)
    if xs == xe:
        ys, ye = random.sample(range(kernel_size), 2)
    else:
        ys, ye = random.randint(0, kernel_size - 1), random.randint(0, kernel_size - 1)

    # draw motion path
    cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)

    # normalize kernel
    return norm_kernel(kernel)

def complex_motion_kernel(SIZE, SIZEx2, DIAGONAL, 
            COMPLEXITY: float=0, eps: float=0.1):
    """
    Get a kernel (psf) of given complexity.
    Adapted from: https://github.com/LeviBorodenko/motionblur
    """
    
    # generate the path
    motion_path = create_motion_path(
        DIAGONAL, SIZEx2, COMPLEXITY, eps)

    # initialize an array with super-sized dimensions
    kernel = np.zeros(SIZEx2, dtype=np.uint8)

    # convert path values to int32 NumPy array (needed for cv2.polylines)
    pts = np.array(motion_path).astype(np.int32)
    pts = pts.reshape((-1,1,2))

    # draw the path using polygon lines
    kernel = cv2.polylines(kernel,
                        [pts], #motion_path, #
                        isClosed=False,
                        color=(64, 64, 64),
                        thickness=int(DIAGONAL / 150), #=3,
                        lineType=cv2.LINE_AA)

    # applying gaussian blur for realism
    # kernel_size = (2*radius)-1
    # for now added 2* that and sigmas = 30, lines are coming up aliased
    kernel_size = 2*(int(DIAGONAL * 0.01)*2)-1
    kernel = cv2.GaussianBlur(
                kernel,
                (kernel_size, kernel_size),
                sigmaX=30.0,
                sigmaY=30.0, 
                borderType=0)

    # resize to actual size
    # Note: CV2 resize is faster, but has no antialias
    # kernel = resize(kernel, 
    #                 out_shape=SIZE, 
    #                 interpolation="gaussian", #"lanczos2", #lanczos3
    #                 antialiasing=True)
    kernel = cv2.resize(kernel,
                        dsize=SIZE,
                        #fx=scale,
                        #fy=scale,
                        interpolation=cv2.INTER_CUBIC)

    # normalize kernel, so it suns up to 1
    return norm_kernel(kernel)

def create_motion_path(DIAGONAL, SIZEx2, COMPLEXITY, eps):
    """
    creates a motion blur path with the given complexity.
    Proceed in 5 steps:
        1. get a random number of random step sizes
        2. for each step get a random angle
        3. combine steps and angles into a sequence of increments
        4. create path out of increments
        5. translate path to fit the kernel dimensions
    NOTE: "random" means random but might depend on the given
        complexity
    """

    # first find the lengths of the motion blur steps
    def getSteps():
        """
        Calculate the length of the steps taken by the motion blur
        A higher complexity lead to a longer total motion
        blur path and more varied steps along the way.
        Hence we sample:
            MAX_PATH_LEN =[U(0,1) + U(0, complexity^2)] * diagonal * 0.75
        and each step is: 
            beta(1, 30) * (1 - COMPLEXITY + eps) * diagonal)
        """
        # getting max length of blur motion
        MAX_PATH_LEN = 0.75 * DIAGONAL * \
            (np.random.uniform() + np.random.uniform(0, COMPLEXITY**2))

        # getting step
        steps = []

        while sum(steps) < MAX_PATH_LEN:
            # sample next step
            step = np.random.beta(1, 30) * (1 - COMPLEXITY + eps) * DIAGONAL
            if step < MAX_PATH_LEN:
                steps.append(step)

        # return the total number of steps and the steps
        return len(steps), np.asarray(steps)

    def getAngles(NUM_STEPS):
        """
        Gets an angle for each step.
        The maximal angle should be larger the more intense
            the motion is, so it's sampled from a
            U(0, complexity * pi).
            Sample "jitter" from a beta(2,20) which is the 
            probability that the next angle has a different
            sign than the previous one.
        """

        # first get the max angle in radians
        MAX_ANGLE = np.random.uniform(0, COMPLEXITY * math.pi)

        # now sample "jitter" which is the probability that the
        # next angle has a different sign than the previous one
        JITTER = np.random.beta(2, 20)

        # initialising angles (and sign of angle)
        angles = [np.random.uniform(low=-MAX_ANGLE, high=MAX_ANGLE)]

        while len(angles) < NUM_STEPS:
            # sample next angle (absolute value)
            angle = np.random.triangular(0, COMPLEXITY *
                                MAX_ANGLE, MAX_ANGLE + eps)

            # with jitter probability change sign wrt previous angle
            if np.random.uniform() < JITTER:
                angle *= -np.sign(angles[-1])
            else:
                angle *= np.sign(angles[-1])

            angles.append(angle)

        # save angles
        return np.asarray(angles)

    # Get steps and angles
    NUM_STEPS, STEPS = getSteps()
    ANGLES = getAngles(NUM_STEPS)

    # Turn them into a path
    ####

    # turn angles and steps into complex numbers
    complex_increments = polar2z(STEPS, ANGLES)

    # generate path as the cumsum of these increments
    path_complex = np.cumsum(complex_increments)

    # find center of mass of path
    com_complex = sum(path_complex) / NUM_STEPS

    # shift path s.t. center of mass lies in the middle of
    # the kernel and a apply a random rotation
    ###

    # center it on center of mass
    # center_of_kernel = (x + 1j * y) / 2
    center_of_kernel = (SIZEx2[0] + 1j * SIZEx2[1]) / 2
    path_complex -= com_complex

    # randomly rotate path by an angle a in (0, pi)
    path_complex *= np.exp(1j * np.random.uniform(0, math.pi))

    # center COM on center of kernel
    path_complex += center_of_kernel

    # convert complex path to final list of coordinate tuples
    return [(i.real, i.imag) for i in path_complex]


@preserve_channel_dim
def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img

