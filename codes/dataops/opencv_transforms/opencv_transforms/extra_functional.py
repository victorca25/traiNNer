# from __future__ import division
#import torch
import math
import random

import numpy as np
import cv2
#import numbers
#import types
#import collections
#import warnings


## Below are new augmentations not available in the original ~torchvision.transforms

_cv2_interpolation_to_str= {'nearest':cv2.INTER_NEAREST,
                         'NEAREST':cv2.INTER_NEAREST,
                         'bilinear':cv2.INTER_LINEAR,
                         'BILINEAR':cv2.INTER_LINEAR,
                         'area':cv2.INTER_AREA,
                         'AREA':cv2.INTER_AREA,
                         'bicubic':cv2.INTER_CUBIC,
                         'BICUBIC':cv2.INTER_CUBIC,
                         'lanczos':cv2.INTER_LANCZOS4,
                         'LANCZOS':cv2.INTER_LANCZOS4}
_cv2_interpolation_from_str= {v:k for k,v in _cv2_interpolation_to_str.items()}

def perspective(img, fov=45, anglex=0, angley=0, anglez=0, shear=0,
                translate=(0, 0), scale=(1, 1), resample='BILINEAR', fillcolor=(0, 0, 0)):
    r"""

    This function is partly referred to https://blog.csdn.net/dcrmg/article/details/80273818

    """

    imgtype = img.dtype
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

    result_img = cv2.warpPerspective(img, total_matrix, (w, h), flags=_cv2_interpolation_to_str[resample],
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=fillcolor)
    if gray_scale:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)
    return result_img.astype(imgtype)


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
    imgtype = img.dtype
    h,w,c = img.shape
    
    if gtype == 'bw':
        c = 1

    gauss = np.random.normal(loc=mean, scale=std, size=(h,w,c)).astype(np.float32)
    noisy = np.clip((1 + gauss) * img.astype(np.float32), 0, 255) 
    
    return noisy.astype(imgtype)


def noise_poisson(img):
    r"""Add OpenCV Poisson noise to the image.
        Important: Poisson noise is not additive like Gaussian, it's dependant on 
        the image values. Read: https://tomroelandts.com/articles/gaussian-noise-is-added-poisson-noise-is-applied
    Args:
        img (numpy ndarray): Image to be augmented.
    Returns:
        numpy ndarray: version of the image with Poisson noise added.
    """
    imgtype = img.dtype
    img = img.astype(np.float32)/255.0

    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = 255 * np.clip(np.random.poisson(img.astype(np.float32) * vals) / float(vals), 0, 1)
    return noisy.astype(imgtype)


def noise_salt_and_pepper(img, prob=0.01):
    r"""Adds "Salt & Pepper" noise to an image.
    Args:
        img (numpy ndarray): Image to be augmented.
        prob (float): probability (threshold) that controls level of noise
    Returns:
        numpy ndarray: version of the image with Poisson noise added.
    """

    imgtype = img.dtype
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

    return noisy.astype(imgtype)


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
    imgtype = img.dtype
    h,w,c = img.shape
    
    if gtype == 'bw':
        c = 1

    speckle = np.random.normal(loc=mean, scale=std ** 0.5, size=(h,w,c)).astype(np.float32)

    noisy = img + img * speckle
    noisy = np.clip(noisy, 0, 255) 
    
    return noisy.astype(imgtype)

def compression_jpeg(img: np.ndarray, quality=20):
    r"""Compress the image as JPEG using OpenCV.
    Args:
        img (numpy ndarray): Image to be compressed.
        quality (int: [0,100]): Compression quality for the image. Lower values represent 
            higher compression and lower quality. Default=20
    Returns:
        numpy ndarray: version of the image with JPEG compression.
    """
    
    imgtype = img.dtype
    if img.ndim >= 3:
        img_channels = img.shape[2]
    elif img.ndim == 2:
        img_channels = 1

    #encoding parameters
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    # encode
    is_success, encimg = cv2.imencode('.jpg', img, encode_param) 
    
    # decode
    jpeg_img = cv2.imdecode(encimg, 1)
    
    # fix for grayscale images
    if jpeg_img.ndim == 3 and jpeg_img.shape[2] != img_channels:
        jpeg_img = jpeg_img[:,:,1]

    if jpeg_img.ndim != 3:
        jpeg_img = jpeg_img[..., np.newaxis]
    
    return jpeg_img.astype(imgtype)


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


def average_blur(img: np.ndarray, kernel_size: int = 3):
    r"""Blurs an image using OpenCV Gaussian Blur.
    Args:
        img (numpy ndarray): Image to be augmented.
        kernel_size (int): size of the blur filter to use. Default: 3.
    Returns:
        numpy ndarray: version of the image with blur applied.
    """
    imgtype = img.dtype
    h,w,c = img.shape

    #Get a valid kernel size
    kernel_size = valid_kernel(h,w,kernel_size)
    
    #Averaging Filter Blur (Homogeneous filter)
    blurred = cv2.blur(img,(kernel_size,kernel_size))

    return blurred.astype(imgtype)

#Needs testing
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
    imgtype = img.dtype
    h,w,c = img.shape

    #Get a valid kernel size
    kernel_size = valid_kernel(h,w,kernel_size)
    #Bilateral filter doesn't appear to work with kernel_size > 9, check
    if kernel_size > 9:
        kernel_size = 9
    
    #Bilateral Filter
    blurred = cv2.bilateralFilter(img,kernel_size,sigmaColor,sigmaSpace)

    return blurred.astype(imgtype)

#Box blur and average blur should be the same
def box_blur(img: np.ndarray, kernel_size: int = 3):
    r"""Blurs an image using OpenCV Gaussian Blur.
    Args:
        img (numpy ndarray): Image to be augmented.
        kernel_size (int): size of the blur filter to use. Default: 3.
    Returns:
        numpy ndarray: version of the image with blur applied.
    """
    imgtype = img.dtype
    h,w,c = img.shape

    #Get a valid kernel size
    kernel_size = valid_kernel(h,w,kernel_size)
    
    #Box Filter Blur 
    blurred = cv2.boxFilter(img,-1,(kernel_size,kernel_size))

    return blurred.astype(imgtype)

def gaussian_blur(img: np.ndarray, kernel_size: int = 3):
    r"""Blurs an image using OpenCV Gaussian Blur.
    Args:
        img (numpy ndarray): Image to be augmented.
        kernel_size (int): size of the blur filter to use. Default: 3.
    Returns:
        numpy ndarray: version of the image with blur applied.
    """
    imgtype = img.dtype
    h,w,c = img.shape

    #Get a valid kernel size
    kernel_size = valid_kernel(h,w,kernel_size)
    
    #Gaussian Filter Blur
    blurred = cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

    return blurred.astype(imgtype)


#with Minisom
#def quantize():
#TBD


def simple_quantize(image, rgb_range):
    pixel_range = 255 / rgb_range
    return 255*(image*pixel_range/255).clip(0, 255).round()/(pixel_range)



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
    
    return dithered.astype(imgtype)


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
        if v > 255:
            v = 255
        if v < 0:
            v = 0
        return v
    
    imgtype = img.dtype
    size = img.shape

    #Note: these are very slow for large images, must crop first before applying.    
    #Floyd-Steinberg
    re_fs = img.copy()
    
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

    return dithered.astype(imgtype)

def noise_dither_avg_bw(img):
    """
        https://github.com/QunixZ/Image_Dithering_Implements/blob/master/HW1.py
    """
    if len(img.shape) > 2 and img.shape[2] != 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)/255.
    
    threshold = np.average(img)
    re_aver = np.where(img < threshold, 0, 1).astype(np.float32)
    #re_aver = cv2.cvtColor(re_aver,cv2.COLOR_GRAY2RGB)

    return re_aver*255.

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
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)/255.

    img_bw = np.where(img < 0.5, 0, 1).astype(np.float32)
    #img_bw = cv2.cvtColor(img_bw,cv2.COLOR_GRAY2RGB)
    return img_bw*255.


def noise_dither_fs_bw(img, samplingF = 1):
    """
        https://github.com/QunixZ/Image_Dithering_Implements/blob/master/HW1.py
    """
    #for Floyd-Steinberg dithering noise
    def minmax(v): 
        if v > 255:
            v = 255
        if v < 0:
            v = 0
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
    r"""The Max RGB filter is used to visualize which channel contributes most to a given area of an image. 
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
    #h,w,c = img.shape
    imgtype = img.dtype
    
    #can randomize strength from 0.5 to 0.8
    # if strength is None:
    #     strength = np.random.uniform(0.3, 0.9)
    
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
            kernel_sizes = [1, 3, 5] #TODO: ks 5 is causing errors
            kernel_size = random.choice(kernel_sizes)
        # Median filtering (could be Gaussian for proper LoG)
        #gray_image_mf = median_filter(gray_image, 1)
        if blur_algo == 'median':
            smooth = cv2.medianBlur(img.astype(np.uint8), kernel_size)
        # Calculate the Laplacian (LoG, or in this case, Laplacian of Median)
        lap = cv2.Laplacian(smooth,cv2.CV_64F)
        # Calculate the sharpened image
        img_out = img - strength*lap
    
     # Saturate the pixels in either direction
    img_out[img_out>255] = 255
    img_out[img_out<0] = 0
    
    return img_out.astype(imgtype)


def filter_canny(img: np.ndarray, sigma=0.33):
    r"""Automatic Canny filter for edge detection
    Args:
        img (numpy ndarray): Image to be filtered.
        sigma (float): standard deviation from the median to automatically calculate minimun 
            values and maximum values thresholds. Default: 0.33.
    Returns:
        numpy ndarray: version of the image after Canny filter.
    """
    imgtype = img.dtype

    # compute the median of the single channel pixel intensities
    median = np.median(img)

    # apply automatic Canny edge detection using the computed median
    minVal = int(max(0, (1.0 - sigma) * median))
    maxVal = int(min(255, (1.0 + sigma) * median))
    edged = cv2.Canny(img, minVal, maxVal)

    # return the edged image
    return edged.astype(imgtype)