import random
import argparse

import os
import os.path
import sys

import numpy as np
import cv2
import dataops.common as util
from dataops.minisom import MiniSom
from dataops.debug import *

IMAGE_EXTENSIONS = ['.png', '.jpg']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMAGE_EXTENSIONS)

def _get_paths_from_dir(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    image_list = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                image_path = os.path.join(dirpath, fname)
                image_list.append(image_path)
    assert image_list, '{:s} has no valid image file'.format(path)
    return image_list

def horizontal_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, ::-1, :]
    return image

def vertical_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[::-1, :, :]
    return image

def random_crop(image, crop_size=(224, 224)): 
    h, w, _ = image.shape
    
    if h > crop_size[0] and w > crop_size[1]:
        top = np.random.randint(0, h - crop_size[0])
        left = np.random.randint(0, w - crop_size[1])

        bottom = top + crop_size[0]
        right = left + crop_size[1]

        image = image[top:bottom, left:right, :]
    else:
        image, _ = resize_img(image, crop_size, algo=[4])
    
    return image
    
def random_crop_pairs(img_HR, img_LR, HR_size, scale):
    H, W, C = img_LR.shape
    
    LR_size = int(HR_size/scale)
    
    # randomly crop
    rnd_h = random.randint(0, max(0, H - LR_size))
    rnd_w = random.randint(0, max(0, W - LR_size))
    #print ("LR rnd: ",rnd_h, rnd_w)
    img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, ...]
    rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
    #print ("HR rnd: ",rnd_h_HR, rnd_w_HR)
    img_HR = img_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, ...]
    
    return img_HR, img_LR

def cutout(image_origin, mask_size, p=0.5):
    if np.random.rand() > p:
        return image_origin
    image = np.copy(image_origin)
    mask_value = image.mean()

    h, w, _ = image.shape
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    if top < 0:
        top = 0
    if left < 0:
        left = 0

    image[top:bottom, left:right, :].fill(mask_value)
    return image

def random_erasing(image_origin, p=0.5, s=(0.02, 0.4), r=(0.3, 3), modes=[0,1,2]): # erasing probability p, the area ratio range of erasing region sl and sh, and the aspect ratio range of erasing region r1 and r2.
    if np.random.rand() > p:
        return image_origin
    image = np.copy(image_origin)

    h, w, _ = image.shape
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])

    mask_aspect_ratio = np.random.rand() * r[1] + r[0]

    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1

    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width
    
    mode = random.choice(modes) #mode 0 fills with a random number, mode 1 fills with ImageNet mean values, mode 2 fills with random pixel values (noise)
    
    if mode == 0: # original code 
        mask_value = np.random.uniform(0., 1.)
        image[top:bottom, left:right, :].fill(mask_value) 
    elif mode == 1: # use ImageNet mean pixel values for each channel 
        if image.shape[2] == 3:
            mean=[0.4465, 0.4822, 0.4914] #OpenCV follows BGR convention and PIL follows RGB color convention
            image[top:bottom, left:right, 0] = mean[0]
            image[top:bottom, left:right, 1] = mean[1]
            image[top:bottom, left:right, 2] = mean[2]
        else: 
            mean=[0.4914]
            image[top:bottom, left:right, :].fill(mean[0])
    else: # replace with random pixel values (noise) (With the selected erasing region Ie, each pixel in Ie is assigned to a random value in [0, 1], respectively.)
        image[top:bottom, left:right, :] = np.random.rand(mask_height, mask_width, image.shape[2])
    return image

# scale image
def scale_img(image, scale, algo=None):
    h, w = image.shape[0], image.shape[1]
    newdim = (int(w/scale), int(h/scale))
    
    # randomly use OpenCV2 algorithms if none are provided
    if algo is None:
        scale_algos = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT] #scaling interpolation options
        interpol = random.choice(scale_algos)
        resized = cv2.resize(image, newdim, interpolation = interpol)
    # using matlab imresize
    else:
        if isinstance(algo, list):
            interpol = random.choice(algo)
        elif isinstance(algo, int):
            interpol = algo
        
        if interpol == 777: #'matlab_bicubic'
            resized = util.imresize_np(image, 1 / scale, True)
            # force to 3 channels
            # if resized.ndim == 2:
                # resized = np.expand_dims(resized, axis=2)
        else:
            # use the provided OpenCV2 algorithms
            resized = cv2.resize(image, newdim, interpolation = interpol)
    
    #resized = np.clip(resized, 0, 1)
    return resized, interpol

# resize image to a defined size 
def resize_img(image, newdim=(128, 128), algo=None):
    if algo is None:
        scale_algos = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT] #scaling interpolation options
        interpol = random.choice(scale_algos)
    else:
        if isinstance(algo, list):
            interpol = random.choice(algo)
        elif isinstance(algo, int):
            interpol = algo
            
    if interpol == 777: #'matlab_bicubic'
        resized = util.imresize_np(image, 1 / scale, True)
        # force to 3 channels
        # if resized.ndim == 2:
            # resized = np.expand_dims(resized, axis=2)
    else:
        # use the provided OpenCV2 algorithms
        resized = cv2.resize(image, newdim, interpolation = interpol)
    
    return resized, interpol

def random_resize_img(img_LR, crop_size=(128, 128), new_scale=0, algo=None):
    mu, sigma = 0, 0.3
    if new_scale == 0:
        new_scale = np.random.uniform(-0.5, 0.) #resize randomly from 0 to 2x
        scale = 1+new_scale
    else:
        scale = new_scale
    
    if algo is None:
        scale_algos = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT] #scaling interpolation options
        interpol = random.choice(scale_algos)
    else:
        interpol = random.choice(algo)
    
    img_LR, interpol = scale_img(img_LR, scale, interpol) 
    img_LR = random_crop(img_LR, crop_size)
    
    return img_LR, interpol

def rotate(image, angle, center=None, scale=1.0, border_value=0, auto_bound=False):
    """Rotate an image.
    Args:
        image (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple): Center of the rotation in the source image, by default
            it is the center of the image.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        auto_bound (bool): Whether to increase the image size to contain the 
            whole rotated image.
    Returns:
        ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    #(h, w) = image.shape[:2]
    h, w = image.shape[:2]
    if center is None:
        #center = (w // 2, h // 2)
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)
     
    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))

    #rotated = cv2.warpAffine(image, matrix, (w, h))
    rotated = cv2.warpAffine(image, matrix, (w, h), borderValue=border_value)

    return rotated

def random_rotate(image, angle=0, center=None, scale=1.0):    
    if angle == 0:
        angle = int(np.random.uniform(-45, 45))
    
    h,w,c = image.shape
    
    if np.random.rand() > 0.5: #randomly upscaling 2x like HD Mode7 to reduce jaggy lines after rotating, more accurate underlying "sub-pixel" data. cv2.INTER_LANCZOS4?
        image, _ = scale_img(image, 1/2, cv2.INTER_CUBIC)
    
    rotated = rotate(image, angle) #will change image shape unless auto_bound is used
    rotated = random_crop(image, crop_size=(w, h)) #crop back to the original size
    return rotated, angle 

def random_rotate_pairs(img_HR, img_LR, HR_size, scale, angle=0, center=0):
    if angle == 0:
        angle = int(np.random.uniform(-45, 45))
    
    if np.random.rand() > 0.5: #randomly upscaling 2x like HD Mode7 to reduce jaggy lines after rotating, more accurate underlying "sub-pixel" data. cv2.INTER_LANCZOS4?
        img_HR, _ = scale_img(img_HR, 1/2, cv2.INTER_CUBIC) 
        img_LR, _ = scale_img(img_LR, 1/2, cv2.INTER_CUBIC) 
    
    img_HR = rotate(img_HR, angle) #will change image shape if auto_bound is used    
    img_LR = rotate(img_LR, angle) #will change image shape if auto_bound is used
    
    img_HR, img_LR = random_crop_pairs(img_HR, img_LR, HR_size, scale) #crop back to the original size if needed 
    return img_HR, img_LR

def blur_img(img_LR, blur_algos=['clean'], kernel_size = 0):
    h,w,c = img_LR.shape
    blur_type = random.choice(blur_algos)
    
    if blur_type == 'average':
        #Averaging Filter Blur (Homogeneous filter)
        if kernel_size == 0:
            kernel_size = int(np.random.uniform(3, 11))
        if kernel_size > h:
            kernel_size = int(np.random.uniform(3, h/2))
        
        kernel_size = int(np.ceil(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size+=1 
        
        blurred = cv2.blur(img_LR,(kernel_size,kernel_size))
        
    elif blur_type == 'box':
        #Box Filter Blur 
        if kernel_size == 0:
            kernel_size = int(np.random.uniform(3, 11))
        if kernel_size > h:
            kernel_size = int(np.random.uniform(3, h/2))
            
        kernel_size = int(np.ceil(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size+=1 
            
        blurred = cv2.boxFilter(img_LR,-1,(kernel_size,kernel_size))
    
    elif blur_type == 'gaussian':
        #Gaussian Filter Blur
        if kernel_size == 0:
            kernel_size = int(np.random.uniform(3, 11))
        if kernel_size > h:
            kernel_size = int(np.random.uniform(3, h/2))
        
        kernel_size = int(np.ceil(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size+=1 
            
        blurred = cv2.GaussianBlur(img_LR,(kernel_size,kernel_size),0)
    
    elif blur_type == 'bilateral':
        #Bilateral Filter
        sigma_bil = int(np.random.uniform(20, 120))
        if kernel_size == 0:
            kernel_size = int(np.random.uniform(3, 9))
        if kernel_size > h:
            kernel_size = int(np.random.uniform(3, h/2))
        
        kernel_size = int(np.ceil(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size+=1 
        
        blurred = cv2.bilateralFilter(img_LR,kernel_size,sigma_bil,sigma_bil)

    elif blur_type == 'clean': # Pass clean image, without blur
        blurred = img_LR

    #img_LR = np.clip(blurred, 0, 1) 
    return blurred, blur_type, kernel_size


def minmax(v): #for Floyd-Steinberg dithering noise
    if v > 255:
        v = 255
    if v < 0:
        v = 0
    return v

def noise_img(img_LR, noise_types=['clean']):
    noise_type = random.choice(noise_types) 
    
    if noise_type == 'poisson': #note: Poisson noise is not additive like Gaussian, it's dependant on the image values: https://tomroelandts.com/articles/gaussian-noise-is-added-poisson-noise-is-applied
        vals = len(np.unique(img_LR))
        vals = 2 ** np.ceil(np.log2(vals))
        noise_img = np.random.poisson(img_LR * vals) / float(vals)
    
    elif noise_type == 's&p': 
        amount = np.random.uniform(0.02, 0.15) 
        s_vs_p = np.random.uniform(0.3, 0.7) # average = 50% salt, 50% pepper #q
        out = np.copy(img_LR)
        
        flipped = np.random.choice([True, False], size=out.shape, p=[amount, 1 - amount])
        
        # Salted mode
        salted = np.random.choice([True, False], size=out.shape, p=[s_vs_p, 1 - s_vs_p])
        
        # Pepper mode
        peppered = ~salted

        out[flipped & salted] = 1

        noise_img = out
        
    elif noise_type == 'speckle':
        h,w,c = img_LR.shape
        speckle_type = [c, 1] #randomize if colored noise (c) or 1 channel (B&W). #this can be done with other types of noise
        c = random.choice(speckle_type)
        
        mean = 0
        sigma = np.random.uniform(0.04, 0.2)
        noise = np.random.normal(mean, scale=sigma ** 0.5, size=(h,w,c))
        noise_img = img_LR + img_LR * noise
        #noise_img = np.clip(noise_img, 0, 1) 
        
    elif noise_type == 'gaussian': # Gaussian Noise
        h,w,c = img_LR.shape
        
        gauss_type = [c, 1] #randomize if colored gaussian noise or 1 channel (B&W). #this can be done with other types of noise
        c = random.choice(gauss_type)
        
        mean = 0
        sigma = np.random.uniform(4, 200.) 
        gauss = np.random.normal(scale=sigma,size=(h,w,c))
        gauss = gauss.reshape(h,w,c) / 255.
        
        noise_img = img_LR + gauss
        
    elif noise_type == 'JPEG': # JPEG Compression        
        compression = np.random.uniform(10, 50) #randomize quality between 10 and 50%
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression] #encoding parameters
        # encode
        is_success, encimg = cv2.imencode('.jpg', img_LR*255, encode_param) 
        
        # decode
        noise_img = cv2.imdecode(encimg, 1) 
        noise_img = noise_img.astype(np.float32) / 255.
        
    elif noise_type == 'quantize': # Color quantization / palette
        pixels = np.reshape(img_LR, (img_LR.shape[0]*img_LR.shape[1], 3)) 
        
        N = int(np.random.uniform(2, 8))
        som = MiniSom(2, N, 3, sigma=1.,
                      learning_rate=0.2, neighborhood_function='bubble')  # 2xN final colors
        som.random_weights_init(pixels)
        starting_weights = som.get_weights().copy() 
        som.train_random(pixels, 500, verbose=False)
        
        qnt = som.quantization(pixels) 
        clustered = np.zeros(img_LR.shape)
        for i, q in enumerate(qnt): 
            clustered[np.unravel_index(i, dims=(img_LR.shape[0], img_LR.shape[1]))] = q
            
        noise_img = clustered
        
    elif noise_type == 'dither_bw': #Black and white dithering from color images
        img_gray = cv2.cvtColor(img_LR, cv2.COLOR_RGB2GRAY)
        size = img_gray.shape
        bwdith_types = ['binary', 'average', 'random', 'bayer', 'fs']
        dither_type = random.choice(bwdith_types)
        
        if dither_type == 'binary':
            img_bw = np.zeros(size, dtype=np.uint8)
            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    if img_gray[i, j] < 0.5:
                        img_bw[i, j] = 0
                    else:
                        img_bw[i, j] = 1
                        
            backtorgb = cv2.cvtColor(img_bw,cv2.COLOR_GRAY2RGB)
            noise_img = backtorgb 
        
        elif dither_type == 'average':
            re_aver = np.zeros(size, dtype=np.uint8)
            threshold = 0
            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    threshold = threshold + img_gray[i, j]/(size[0]*size[1])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    if img_gray[i, j] < threshold:
                        re_aver[i, j] = 0
                    else:
                        re_aver[i, j] = 1
            
            backtorgb = cv2.cvtColor(re_aver,cv2.COLOR_GRAY2RGB)
            noise_img = backtorgb
        
        elif dither_type == 'random':
            re_rand = np.zeros(size, dtype=np.uint8)
            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    if img_gray[i, j] < np.random.uniform(0, 1):
                        re_rand[i, j] = 0
                    else:
                        re_rand[i, j] = 1
            backtorgb = cv2.cvtColor(re_rand,cv2.COLOR_GRAY2RGB)
            noise_img = backtorgb
            
        elif dither_type == 'bayer':
            re_bayer = np.zeros(size, dtype=np.uint8) 
            bayer_matrix = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])/256 #4x4 Bayer matrix
            
            bayer_matrix = bayer_matrix*16 

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    x = np.mod(i, 4)
                    y = np.mod(j, 4)
                    if img_gray[i, j] > bayer_matrix[x, y]:
                        re_bayer[i, j] = 1
            backtorgb = cv2.cvtColor(re_bayer,cv2.COLOR_GRAY2RGB)
            noise_img = backtorgb
        
        elif dither_type == 'fs':
            re_fs = 255*img_gray 
            samplingF = 1
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
                    
            backtorgb = cv2.cvtColor(re_fs/255.0,cv2.COLOR_GRAY2RGB)
            noise_img = backtorgb
    
    elif noise_type == 'dither': # Color dithering 
        colordith_types = ['bayer', 'fs']
        dither_type = random.choice(colordith_types)
        
        size = img_LR.shape
        
        # Bayer works more or less. I think it's missing a part of the image, the ditherng pattern is apparent, but the quantized (color palette) is not there. Still enough for models to learn dedithering
        if dither_type == 'bayer':
            bayer_matrix = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])/256 #4x4 Bayer matrix
            
            bayer_matrix = bayer_matrix*16 
            
            red = img_LR[:,:,2]
            green = img_LR[:,:,1]
            blue = img_LR[:,:,0]
            
            img_split = np.zeros((img_LR.shape[0], img_LR.shape[1], 3), dtype = red.dtype)
            
            for values, color, channel in zip((red, green, blue), ('red', 'green', 'blue'), (2,1,0)):
                for i in range(0, values.shape[0]):
                    for j in range(0, values.shape[1]):
                        x = np.mod(i, 4)
                        y = np.mod(j, 4)
                        if values[i, j] > bayer_matrix[x, y]:
                            img_split[i,j,channel] = 1 
            
            noise_img = img_split
        
        # Floyd-Steinberg (F-S) Dithering
        elif dither_type == 'fs':
            
            re_fs = 255*img_LR 
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
                    
            noise_img = re_fs/255.0
    
    elif noise_type == 'clean': # Pass clean image, without noise
        noise_img = img_LR
        
    #img_LR = np.clip(noise_img, 0, 1)
    return noise_img, noise_type


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

# https://www.pyimagesearch.com/2015/09/28/implementing-the-max-rgb-filter-in-opencv/
def max_rgb_filter(image):
	# split the image into its BGR components
	(B, G, R) = cv2.split(image)
 
	# find the maximum pixel intensity values for each
	# (x, y)-coordinate,, then set all pixel values less
	# than M to zero
	M = np.maximum(np.maximum(R, G), B)
	R[R < M] = 0
	G[G < M] = 0
	B[B < M] = 0
 
	# merge the channels back together and return the image
	return cv2.merge([B, G, R])

# Simple color balance algorithm (similar to Photoshop "auto levels")
# https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc#gistcomment-3025656
# http://www.morethantechnical.com/2015/01/14/simplest-color-balance-with-opencv-wcode/
# https://web.stanford.edu/~sujason/ColorBalancing/simplestcb.html
def simplest_cb(img, percent=1, znorm=False):
    
    if znorm == True: # img is znorm'ed in the [-1,1] range, else img in the [0,1] range
        img = (img + 1.0)/2.0
    
    # back to the OpenCV [0,255] range
    img = (img * 255.0).round().astype(np.uint8) # need to add astype(np.uint8) so cv2.LUT doesn't fail later
    
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
        
    img_out = cv2.merge(out_channels)
    
    # Re-normalize
    if znorm == False: # normalize img_or back to the [0,1] range
        img_out = img_out/255.0
    if znorm==True: # normalize images back to range [-1, 1] 
        img_out = (img_out/255.0 - 0.5) * 2 # xi' = (xi - mu)/sigma    
    return img_out



#https://www.idtools.com.au/unsharp-masking-python-opencv/
def unsharp_mask(img, blur_algo='median', kernel_size=None, strength=None, unsharp_algo='laplacian', znorm=False):
    #h,w,c = img.shape
    
    if znorm == True: # img is znorm'ed in the [-1,1] range, else img in the [0,1] range
        img = (img + 1.0)/2.0
    # back to the OpenCV [0,255] range
    img = (img * 255.0).round().astype(np.uint8)
    
    #randomize strenght from 0.5 to 0.8
    if strength is None:
        strength = np.random.uniform(0.3, 0.9)
    
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
            kernel_sizes = [1, 3, 5]
            kernel_size = random.choice(kernel_sizes)
        # Median filtering (could be Gaussian for proper LoG)
        #gray_image_mf = median_filter(gray_image, 1)
        if blur_algo == 'median':
            smooth = cv2.medianBlur(img, kernel_size)
        # Calculate the Laplacian (LoG, or in this case, Laplacian of Median)
        lap = cv2.Laplacian(smooth,cv2.CV_64F)
        # Calculate the sharpened image
        img_out = img - strength*lap
    
     # Saturate the pixels in either direction
    img_out[img_out>255] = 255
    img_out[img_out<0] = 0
    
    # Re-normalize
    if znorm == False: # normalize img_or back to the [0,1] range
        img_out = img_out/255.0
    if znorm==True: # normalize images back to range [-1, 1] 
        img_out = (img_out/255.0 - 0.5) * 2 # xi' = (xi - mu)/sigma
    
    return img_out



def random_img(img_dir, save_path, crop_size=(128, 128), scale=1, blur_algos=['clean'], noise_types=['clean'], noise_types2=['clean']):
    img_list = _get_paths_from_dir(img_dir)
    
    random_img_path = random.choice(img_list)
    
    env = None
    img = util.read_img(env, random_img_path) #read image from path, opens with OpenCV, value ranges from 0 to 1
    
    img_crop = random_crop(img, crop_size)
    print(img_crop.shape)
    #"""
    cv2.imwrite(save_path+'/crop_.png',img_crop*255) 
    #"""
    
    img_resize, _ = resize_img(img, crop_size)
    print(img_resize.shape)
    #"""
    cv2.imwrite(save_path+'/resize_.png',img_resize*255) 
    #"""
    
    img_random_resize, _ = random_resize_img(img, crop_size)
    print(img_random_resize.shape)
    #"""
    cv2.imwrite(save_path+'/random_resize_.png',img_random_resize*255) 
    #"""
    
    img_cutout = cutout(img, img.shape[0] // 2)
    print(img_cutout.shape)
    #"""
    cv2.imwrite(save_path+'/cutout_.png',img_cutout*255) 
    #"""
    
    img_erasing = random_erasing(img)
    print(img_erasing.shape)
    #"""
    cv2.imwrite(save_path+'/erasing_.png',img_erasing*255) 
    #"""
    
    #scale = 4
    img_scale, interpol_algo = scale_img(img, scale)
    print(img_scale.shape)    
    #"""
    cv2.imwrite(save_path+'/scale_'+str(scale)+'_'+str(interpol_algo)+'_.png',img_scale*255) 
    #"""
    
    img_blur, blur_algo, blur_kernel_size = blur_img(img, blur_algos)
    print(img_blur.shape)
    #"""
    cv2.imwrite(save_path+'/blur_'+str(blur_kernel_size)+'_'+str(blur_algo)+'_.png',img_blur*255) 
    #"""
    
    img_noise, noise_algo = noise_img(img, noise_types)
    print(img_noise.shape)
    #"""
    cv2.imwrite(save_path+'/noise_'+str(noise_algo)+'_.png',img_noise*255) 
    #"""
    
    #img_noise2, noise_algo2 = noise_img(img_noise, noise_types2)
    #print(img_noise2.shape)
    #"""
    #cv2.imwrite(save_path+'/noise2_'+str(noise_algo2)+'_.png',img_noise2*255) 
    #"""
    
    img_rrot, angle = random_rotate(img)
    print(img_rrot.shape)
    #"""
    cv2.imwrite(save_path+'/rrot_'+str(angle)+'_.png',img_rrot*255) 
    #"""
    
    print('Finished')
    

def single_image(img_path, save_path, crop_size=(128, 128), scale=1, blur_algos=['clean'], noise_types=['clean'], noise_types2=['clean']):
    env = None
    img = util.read_img(env, img_path) #read image from path, opens with OpenCV, value ranges from 0 to 1
    print(img.shape)
    
    img_crop = random_crop(img, crop_size)
    print(img_crop.shape)
    #"""
    cv2.imwrite(save_path+'/crop_.png',img_crop*255) 
    #"""
    
    img_resize, _ = resize_img(img, crop_size)
    print(img_resize.shape)
    #"""
    cv2.imwrite(save_path+'/resize_.png',img_resize*255) 
    #"""
    
    img_random_resize, _ = random_resize_img(img, crop_size)
    print(img_random_resize.shape)
    #"""
    cv2.imwrite(save_path+'/random_resize_.png',img_random_resize*255) 
    #"""
    
    img_cutout = cutout(img, img.shape[0] // 2)
    print(img_cutout.shape)
    #"""
    cv2.imwrite(save_path+'/cutout_.png',img_cutout*255) 
    #"""
    
    img_erasing = random_erasing(img)
    print(img_erasing.shape)
    #"""
    cv2.imwrite(save_path+'/erasing_.png',img_erasing*255) 
    #"""
    
    #scale = 4
    img_scale, interpol_algo = scale_img(img, scale)
    print(img_scale.shape)    
    #"""
    cv2.imwrite(save_path+'/scale_'+str(scale)+'_'+str(interpol_algo)+'_.png',img_scale*255) 
    #"""
    
    img_blur, blur_algo, blur_kernel_size = blur_img(img, blur_algos)
    print(img_blur.shape)
    #"""
    cv2.imwrite(save_path+'/blur_'+str(blur_kernel_size)+'_'+str(blur_algo)+'_.png',img_blur*255) 
    #"""
    
    img_noise, noise_algo = noise_img(img, noise_types)
    #img_noise, noise_algo = noise_img(img_scale, noise_types)
    print(img_noise.shape)
    #"""
    cv2.imwrite(save_path+'/noise_'+str(noise_algo)+'_.png',img_noise*255) 
    #"""
    
    img_noise2, noise_algo2 = noise_img(img_noise, noise_types2)
    print(img_noise2.shape)
    #"""
    cv2.imwrite(save_path+'/noise2_'+str(noise_algo2)+'_.png',img_noise2*255) 
    #"""
    
    img_rrot, angle = random_rotate(img)
    print(img_rrot.shape)
    #"""
    cv2.imwrite(save_path+'/rrot_'+str(angle)+'_.png',img_rrot*255) 
    #"""
    
    print('Finished')

    
def apply_dir(img_path, save_path, crop_size=(128, 128), scale=1, blur_algos=['clean'], noise_types=['clean'], noise_types2=['clean']):
    img_list = _get_paths_from_dir(img_dir)
    
    for path in img_list:
        rann = ''
        env = None
        img = util.read_img(env, path)
        import uuid
        rann = uuid.uuid4().hex

        img_crop = random_crop(img, crop_size)
        print(img_crop.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'mask_.png',img_crop*255) 
        #"""
        
        img_resize, _ = resize_img(img, crop_size)
        print(img_resize.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'resize_.png',img_resize*255) 
        #"""
        
        img_random_resize, _ = random_resize_img(img, crop_size)
        print(img_random_resize.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'random_resize_.png',img_random_resize*255) 
        #"""
        
        img_cutout = cutout(img, img.shape[0] // 2)
        print(img_cutout.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'cutout_.png',img_cutout*255) 
        #"""
        
        img_erasing = random_erasing(img)
        print(img_erasing.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'erasing_.png',img_erasing*255) 
        #"""
        
        #scale = 4
        img_scale, interpol_algo = scale_img(img, scale)
        print(img_scale.shape)    
        #"""
        cv2.imwrite(save_path+'/'+rann+'scale_'+str(scale)+'_'+str(interpol_algo)+'_.png',img_scale*255) 
        #"""
        
        img_blur, blur_algo, blur_kernel_size = blur_img(img, blur_algos)
        print(img_blur.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'blur_'+str(blur_kernel_size)+'_'+str(blur_algo)+'_.png',img_blur*255) 
        #cv2.imwrite(save_path+'/blur__.png',img_blur*255) 
        #"""
        
        img_noise, noise_algo = noise_img(img, noise_types)
        #img_noise, noise_algo = noise_img(img_scale, noise_types)
        print(img_noise.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'noise_'+str(noise_algo)+'_.png',img_noise*255) 
        #"""
        
        img_noise2, noise_algo2 = noise_img(img_noise, noise_types2)
        print(img_noise2.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'noise2_'+str(noise_algo2)+'_.png',img_noise2*255) 
        #"""
        
        img_rrot, angle = random_rotate(img)
        print(img_rrot.shape)
        #"""
        cv2.imwrite(save_path+'/'+rann+'rrot_'+str(angle)+'_.png',img_rrot*255) 
        #"""

    print('Finished')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=int, required=False, help='Select mode. 1: Single Image, 2: Random image from directory, 3: Full directory')
    parser.add_argument('-img_dir', type=str, required=False, help='Image directory') 
    parser.add_argument('-img_path', type=str, required=False, help='Single image path') 
    parser.add_argument('-savepath', type=str, required=False, help='Save path') 
    parser.add_argument('-scale', type=int, required=False, help='Scale to resize')
    parser.add_argument('-crop', type=int, required=False, help='Crop size')
    parser.add_argument('-blur', type=str, required=False, help='Blur algorithm') 
    parser.add_argument('-noise', type=str, required=False, help='Noise algorithm') 
    parser.add_argument('-noise2', type=str, required=False, help='Secondary noise algorithm') 
    args = parser.parse_args()
    print(args)
    
    if args.img_dir:
        img_dir = args.img_dir
        #print(": " + args.-)
    else:
        img_dir = "../../datasets"

    if args.img_path:
        img_path = args.img_path
    else:
        img_path = "../../datasets/0318.png"
    
    if args.savepath:
        savepath = args.savepath
    else:
        savepath = "../../datasets/tests_otf/"
    
    if args.crop:
        crop_size=(args.crop,args.crop)
    else:
        crop_size=(128, 128)
    
    if args.scale:
        crop_size = args.scale
    else:
        scale = 4
        
    if args.blur:
        blur_algos = [args.blur]
    else:
        blur_algos = ["average","box","gaussian","bilateral", "clean", "clean", "clean", "clean"]
        
    if args.noise:
        noise_types = [args.noise]
    else:
        noise_types = ["gaussian", "gaussian", "JPEG", "JPEG", "quantize", "poisson", "dither", "s&p", "speckle", "clean", "clean", "clean", "clean"]
    
    if args.noise2:
        noise_types2 = [args.noise2]
    else:    
        noise_types2 = ["JPEG", "clean", "clean", "clean"]
    
    #Select mode. 1: Single Image, 2: Random image from directory, 3: Full directory
    if args.mode == 1:
        single_image(img_path, savepath, crop_size, scale, blur_algos, noise_types, noise_types2) #take an image path and apply augmentations
    elif args.mode == 2:
        random_img(img_dir, savepath, crop_size, scale, blur_algos, noise_types, noise_types2) #take a random image from the directory and apply augmentations    
    elif args.mode == 3:
        apply_dir(img_dir, savepath, crop_size, scale, blur_algos, noise_types, noise_types2) #take a random image from the directory and apply augmentations
    else:
        print("Mode not selected, please select one of 1: Single Image, 2: Random image from directory, 3: Full directory")
    
