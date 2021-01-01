import os
import math
import pickle
import random
import numpy as np
import torch
import cv2
import logging

import copy
from torchvision.utils import make_grid

from dataops.colors import *
from dataops.debug import tmp_vis, describe_numpy, describe_tensor

####################
# Files & IO
####################

###################### get image path list ######################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.dng', '.DNG', '.webp','.npy', '.NPY']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    '''get image path list from lmdb'''
    import lmdb
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    keys_cache_file = os.path.join(dataroot, '_keys_cache.p')
    logger = logging.getLogger('base')
    if os.path.isfile(keys_cache_file):
        logger.info('Read lmdb keys from cache: {}'.format(keys_cache_file))
        keys = pickle.load(open(keys_cache_file, "rb"))
    else:
        with env.begin(write=False) as txn:
            logger.info('Creating lmdb keys cache: {}'.format(keys_cache_file))
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, 'wb'))
    paths = sorted([key for key in keys if not key.endswith('.meta')])
    return env, paths


def get_image_paths(data_type, dataroot):
    '''get image path list
    support lmdb or image files'''
    env, paths = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            env, paths = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return env, paths


###################### read images ######################
def _read_lmdb_img(env, key, size=None):
    '''read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple'''
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
        buf_meta = txn.get((key + '.meta').encode('ascii')).decode('ascii')
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    if size:
        C, H, W = size
    else:
        H, W, C = [int(s) for s in buf_meta.split(',')]
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path, out_nc=3, fix_channels=True, size=None):
    '''
        Reads image using cv2 (rawpy if dng), from lmdb bor from a buffer 
        (path=buffer). Could also use using PIL instead of cv2.
    Arguments:
        path: image path or buffer to read
        out_nc: Desired number of channels
        fix_channels: changes the images to the desired number of channels
        size: fixed size to use for lmbg (optional)
    Output:
        Numpy HWC, BGR, [0,255] by default 
    '''

    img = None
    if env is None or env is 'img':  # img
        if(path[-3:].lower() == 'dng'): # if image is a DNG
            import rawpy
            with rawpy.imread(path) as raw:
                img = raw.postprocess()
        if(path[-3:].lower() == 'npy'): # if image is a NPY numpy array
            with open(path, 'rb') as f:
                img = np.load(f)
        else: # else, if image can be read by cv2 
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        #TODO: add variable detecting if cv2 is not available and try PIL instead
        # elif: # using PIL instead of OpenCV
            # img = Image.open(path).convert('RGB')
        # else: # For other images unrecognized by cv2
            # import matplotlib.pyplot as plt
            # img = (255*plt.imread(path)[:,:,:3]).astype('uint8')
    elif env is 'lmdb':
        img = _read_lmdb_img(env, path, size)
    elif env is 'buffer':
        img = cv2.imdecode(path, cv2.IMREAD_UNCHANGED)
    else:
        raise NotImplementedError("Unsupported env: %s" % (env,))

    # if not img:
    #     raise ValueError(f"Failed to read image: {path}")

    if fix_channels:
        img = fix_img_channels(img, out_nc)

    return img

def fix_img_channels(img, out_nc):
    '''
        fix image channels to the expected number
    '''

    # if image has only 2 dimensions, add "channel" dimension (1)
    if img.ndim == 2:
        #img = img[..., np.newaxis] #alt
        #img = np.expand_dims(img, axis=2)
        img = np.tile(np.expand_dims(img, axis=2), (1, 1, 3))
    # special case: properly remove alpha channel 
    if out_nc == 3 and img.shape[2] == 4: 
        img = bgra2rgb(img)
    # remove all extra channels 
    elif img.shape[2] > out_nc: 
        img = img[:, :, :out_nc] 
    # if alpha is expected, add solid alpha channel
    elif img.shape[2] == 3 and out_nc == 4:
        img = np.dstack((img, np.full(img.shape[:-1], 255, dtype=np.uint8)))
    return img


####################
# image processing
# process on numpy image
####################

def bgra2rgb(img):
    '''
        cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) has an issue removing the alpha channel,
        this gets rid of wrong transparent colors that can harm training
    '''
    if img.shape[2] == 4:
        #b, g, r, a = cv2.split((img*255).astype(np.uint8))
        b, g, r, a = cv2.split((img.astype(np.uint8)))
        b = cv2.bitwise_and(b, b, mask=a)
        g = cv2.bitwise_and(g, g, mask=a)
        r = cv2.bitwise_and(r, r, mask=a)
        #return cv2.merge([b, g, r]).astype(np.float32)/255.
        return cv2.merge([b, g, r])
    return img

def channel_convert(in_c, tar_type, img_list):
    # conversion among BGR, gray and y
    # Note: OpenCV uses inverted channels BGR, instead of RGB.
    #  If images are loaded with something other than OpenCV,
    #  check that the channels are in the correct order and use
    #  the alternative conversion functions.
    #if in_c == 4 and tar_type == 'RGB-A':  # BGRA to BGR, remove alpha channel
        #return [cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) for img in img_list]
        #return [bgra2rgb(img) for img in img_list]
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'RGB-LAB': #RGB to LAB
        return [cv2.cvtColor(img, cv2.COLOR_BGR2LAB) for img in img_list]
    elif in_c == 3 and tar_type == 'LAB-RGB': #RGB to LAB
        return [cv2.cvtColor(img, cv2.COLOR_LAB2BGR) for img in img_list]
    elif in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img_ = img.astype(np.float32)
    if in_img_type != np.uint8:
        img_  *= 255.
    # convert
    if only_y:
        rlt = np.dot(img_ , [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img_ , [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def bgr2ycbcr(img, only_y=True, separate=False):
    '''bgr version of matlab rgb2ycbcr
    Python opencv library (cv2) cv2.COLOR_BGR2YCrCb has 
    different parameters with MATLAB color convertion.
    only_y: only return Y channel
    separate: if true, will returng the channels as 
        separate images
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img_ = img.astype(np.float32)
    if in_img_type != np.uint8:
        img_  *= 255.
    # convert
    if only_y:
        rlt = np.dot(img_ , [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img_ , [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
        # to make ycrcb like cv2
        # rlt = rlt[:, :, (0, 2, 1)]
    
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    
    if separate:
        rlt = rlt.astype(in_img_type)
        # y, cb, cr
        return rlt[:, :, 0], rlt[:, :, 1], rlt[:, :, 2]
    else:
        return rlt.astype(in_img_type)

'''
def ycbcr2rgb_(img, only_y=True):
    """same as matlab ycbcr2rgb
    (Note: this implementation is the original from BasicSR, but 
    appears to be for ycrcb, like cv2)
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img_ = img.astype(np.float32)
    if in_img_type != np.uint8:
        img_  *= 255.
    
    # to make ycrcb like cv2
    # rlt = rlt[:, :, (0, 2, 1)]

    # convert
    # original (for ycrcb):
    rlt = np.matmul(img_ , [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]

    #alternative conversion:
    # xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    # img_[:, :, [1, 2]] -= 128
    # rlt = img_.dot(xform.T)
    np.putmask(rlt, rlt > 255, 255)
    np.putmask(rlt, rlt < 0, 0)
    
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
'''

def ycbcr2rgb(img, only_y=True):
    '''
    bgr version of matlab ycbcr2rgb
    Python opencv library (cv2) cv2.COLOR_YCrCb2BGR has 
    different parameters to MATLAB color convertion.

    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img_ = img.astype(np.float32)
    if in_img_type != np.uint8:
        img_  *= 255.
    
    # to make ycrcb like cv2
    # rlt = rlt[:, :, (0, 2, 1)]

    # convert
    mat = np.array([[24.966, 128.553, 65.481],[112, -74.203, -37.797], [-18.214, -93.786, 112.0]])
    mat = np.linalg.inv(mat.T) * 255
    offset = np.array([[[16, 128, 128]]])

    rlt = np.dot((img_ - offset), mat)
    rlt = np.clip(rlt, 0, 255)
    ## rlt = np.rint(rlt).astype('uint8')
    
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

'''
#TODO: TMP RGB version, to check (PIL)
def rgb2ycbcr(img_rgb):
    ## the range of img_rgb should be (0, 1)
    img_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] + 16 / 255.0
    img_cb = -0.148 * img_rgb[:, :, 0] - 0.291 * img_rgb[:, :, 1] + 0.439 * img_rgb[:, :, 2] + 128 / 255.0
    img_cr = 0.439 * img_rgb[:, :, 0] - 0.368 * img_rgb[:, :, 1] - 0.071 * img_rgb[:, :, 2] + 128 / 255.0
    return img_y, img_cb, img_cr

#TODO: TMP RGB version, to check (PIL)
def ycbcr2rgb(img_ycbcr):
    ## the range of img_ycbcr should be (0, 1)
    img_r = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 1.596 * (img_ycbcr[:, :, 2] - 128 / 255.0)
    img_g = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) - 0.392 * (img_ycbcr[:, :, 1] - 128 / 255.0) - 0.813 * (img_ycbcr[:, :, 2] - 128 / 255.0)
    img_b = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 2.017 * (img_ycbcr[:, :, 1] - 128 / 255.0)
    img_r = img_r[:, :, np.newaxis]
    img_g = img_g[:, :, np.newaxis]
    img_b = img_b[:, :, np.newaxis]
    img_rgb = np.concatenate((img_r, img_g, img_b), 2)
    return img_rgb
'''

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

#TODO: this should probably be elsewhere (augmentations.py)
def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    #rot90n = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = np.flip(img, axis=1) #img[:, ::-1, :]
        if vflip: img = np.flip(img, axis=0) #img[::-1, :, :]
        #if rot90: img = img.transpose(1, 0, 2)
        if rot90: img = np.rot90(img, 1) #90 degrees # In PIL: img.transpose(Image.ROTATE_90)
        #if rot90n: img = np.rot90(img, -1) #-90 degrees
        return img

    return [_augment(img) for img in img_list]



####################
# Normalization functions
####################


#TODO: Could also automatically detect the possible range with min and max, like in def ssim()
def denorm(x, min_max=(-1.0, 1.0)):
    '''
        Denormalize from [-1,1] range to [0,1]
        formula: xi' = (xi - mu)/sigma
        Example: "out = (x + 1.0) / 2.0" for denorm 
            range (-1,1) to (0,1)
        for use with proper act in Generator output (ie. tanh)
    '''
    out = (x - min_max[0]) / (min_max[1] - min_max[0])
    if isinstance(x, torch.Tensor):
        return out.clamp(0, 1)
    elif isinstance(x, np.ndarray):
        return np.clip(out, 0, 1)
    else:
        raise TypeError("Got unexpected object type, expected torch.Tensor or \
        np.ndarray")

def norm(x): 
    #Normalize (z-norm) from [0,1] range to [-1,1]
    out = (x - 0.5) * 2.0
    if isinstance(x, torch.Tensor):
        return out.clamp(-1, 1)
    elif isinstance(x, np.ndarray):
        return np.clip(out, -1, 1)
    else:
        raise TypeError("Got unexpected object type, expected torch.Tensor or \
        np.ndarray")


####################
# np and tensor conversions
####################


#2tensor
def np2tensor(img, bgr2rgb=True, data_range=1., normalize=False, change_range=True, add_batch=True):
    """
    Converts a numpy image array into a Tensor array.
    Parameters:
        img (numpy array): the input image numpy array
        add_batch (bool): choose if new tensor needs batch dimension added 
    """
    if not isinstance(img, np.ndarray): #images expected to be uint8 -> 255
        raise TypeError("Got unexpected object type, expected np.ndarray")
    #check how many channels the image has, then condition, like in my BasicSR. ie. RGB, RGBA, Gray
    #if bgr2rgb:
        #img = img[:, :, [2, 1, 0]] #BGR to RGB -> in numpy, if using OpenCV, else not needed. Only if image has colors.
    if change_range:
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo
        elif np.issubdtype(img.dtype, np.floating):
            info = np.finfo
        img = img*data_range/info(img.dtype).max #uint8 = /255
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float() #"HWC to CHW" and "numpy to tensor"
    if bgr2rgb:
        #BGR to RGB -> in tensor, if using OpenCV, else not needed. Only if image has colors.)
        if img.shape[0] % 3 == 0: #RGB or MultixRGB (3xRGB, 5xRGB, etc. For video tensors.)
            img = bgr_to_rgb(img)
        elif img.shape[0] == 4: #RGBA
            img = bgra_to_rgba(img)
    if add_batch:
        img.unsqueeze_(0) # Add fake batch dimension = 1 . squeeze() will remove the dimensions of size 1
    if normalize:
        img = norm(img)
    return img

#2np
def tensor2np(img, rgb2bgr=True, remove_batch=True, data_range=255, 
              denormalize=False, change_range=True, imtype=np.uint8):
    """
    Converts a Tensor array into a numpy image array.
    Parameters:
        img (tensor): the input image tensor array
            4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
        remove_batch (bool): choose if tensor of shape BCHW needs to be squeezed 
        denormalize (bool): Used to denormalize from [-1,1] range back to [0,1]
        imtype (type): the desired type of the converted numpy array (np.uint8 
            default)
    Output: 
        img (np array): 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("Got unexpected object type, expected torch.Tensor")
    n_dim = img.dim()

    #TODO: Check: could denormalize here in tensor form instead, but end result is the same
    
    img = img.float().cpu()  
    
    if n_dim == 4 or n_dim == 3:
        #if n_dim == 4, has to convert to 3 dimensions, either removing batch or by creating a grid
        if n_dim == 4 and remove_batch:
            if img.shape[0] > 1:
                # leave only the first image in the batch and remove batch dimension
                img = img[0,...] 
            else:
                # remove a fake batch dimension
                img = img.squeeze(dim=0)
                #TODO: the following 'if' should not be required
                ## squeeze removes batch and channel of grayscale images (dimensions = 1)
                # if len(img.shape) < 3: 
                #     #add back the lost channel dimension
                #     img = img.unsqueeze(dim=0)
        elif n_dim == 4 and not remove_batch:
            # convert images in batch (BCHW) to a grid of all images (C B*H B*W)
            n_img = len(img)
            img = make_grid(img, nrow=int(math.sqrt(n_img)), normalize=False)
        
        if img.shape[0] == 3 and rgb2bgr: #RGB
            #RGB to BGR -> in tensor, if using OpenCV, else not needed. Only if image has colors.
            img_np = rgb_to_bgr(img).numpy()
        elif img.shape[0] == 4 and rgb2bgr: #RGBA
            #RGBA to BGRA -> in tensor, if using OpenCV, else not needed. Only if image has colors.
            img_np = rgba_to_bgra(img).numpy()
        else:
            img_np = img.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
    elif n_dim == 2:
        img_np = img.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    #if rgb2bgr:
        #img_np = img_np[[2, 1, 0], :, :] #RGB to BGR -> in numpy, if using OpenCV, else not needed. Only if image has colors.
    #TODO: Check: could denormalize in the begining in tensor form instead
    if denormalize:
        img_np = denorm(img_np) #denormalize if needed
    if change_range:
        img_np = np.clip(data_range*img_np,0,data_range).round() #clip to the data_range
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    #has to be in range (0,255) before changing to np.uint8, else np.float32
    return img_np.astype(imtype)




####################
# Prepare Images
####################
# https://github.com/sunreef/BlindSR/blob/master/src/image_utils.py
def patchify_tensor(features, patch_size, overlap=10):
    batch_size, channels, height, width = features.size()

    effective_patch_size = patch_size - overlap
    n_patches_height = (height // effective_patch_size)
    n_patches_width = (width // effective_patch_size)

    if n_patches_height * effective_patch_size < height:
        n_patches_height += 1
    if n_patches_width * effective_patch_size < width:
        n_patches_width += 1

    patches = []
    for b in range(batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(h * effective_patch_size, height - patch_size)
                patch_start_width = min(w * effective_patch_size, width - patch_size)
                patches.append(features[b:b+1, :,
                               patch_start_height: patch_start_height + patch_size,
                               patch_start_width: patch_start_width + patch_size])
    return torch.cat(patches, 0)

def recompose_tensor(patches, full_height, full_width, overlap=10):

    batch_size, channels, patch_size, _ = patches.size()
    effective_patch_size = patch_size - overlap
    n_patches_height = (full_height // effective_patch_size)
    n_patches_width = (full_width // effective_patch_size)

    if n_patches_height * effective_patch_size < full_height:
        n_patches_height += 1
    if n_patches_width * effective_patch_size < full_width:
        n_patches_width += 1

    n_patches = n_patches_height * n_patches_width
    if batch_size % n_patches != 0:
        print("Error: The number of patches provided to the recompose function does not match the number of patches in each image.")
    final_batch_size = batch_size // n_patches

    blending_in = torch.linspace(0.1, 1.0, overlap)
    blending_out = torch.linspace(1.0, 0.1, overlap)
    middle_part = torch.ones(patch_size - 2 * overlap)
    blending_profile = torch.cat([blending_in, middle_part, blending_out], 0)

    horizontal_blending = blending_profile[None].repeat(patch_size, 1)
    vertical_blending = blending_profile[:, None].repeat(1, patch_size)
    blending_patch = horizontal_blending * vertical_blending

    blending_image = torch.zeros(1, channels, full_height, full_width)
    for h in range(n_patches_height):
        for w in range(n_patches_width):
            patch_start_height = min(h * effective_patch_size, full_height - patch_size)
            patch_start_width = min(w * effective_patch_size, full_width - patch_size)
            blending_image[0, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += blending_patch[None]

    recomposed_tensor = torch.zeros(final_batch_size, channels, full_height, full_width)
    if patches.is_cuda:
        blending_patch = blending_patch.cuda()
        blending_image = blending_image.cuda()
        recomposed_tensor = recomposed_tensor.cuda()
    patch_index = 0
    for b in range(final_batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(h * effective_patch_size, full_height - patch_size)
                patch_start_width = min(w * effective_patch_size, full_width - patch_size)
                recomposed_tensor[b, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += patches[patch_index] * blending_patch
                patch_index += 1
    recomposed_tensor /= blending_image

    return recomposed_tensor


















#TODO: imresize could be an independent file (imresize.py)
####################
# Matlab imresize
####################


# These next functions are all interpolation methods. x is the distance from the left pixel center
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5*absx3 - 2.5*absx2 + 1) * ((absx <= 1).type_as(absx)) + \
        (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * (((absx > 1)*(absx <= 2)).type_as(absx))

def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0

def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))

def lanczos2(x):
    return (((torch.sin(math.pi*x) * torch.sin(math.pi*x/2) + torch.finfo(torch.float32).eps) /
             ((math.pi**2 * x**2 / 2) + torch.finfo(torch.float32).eps))
            * (torch.abs(x) < 2))

def lanczos3(x):
    return (((torch.sin(math.pi*x) * torch.sin(math.pi*x/3) + torch.finfo(torch.float32).eps) /
            ((math.pi**2 * x**2 / 3) + torch.finfo(torch.float32).eps))
            * (torch.abs(x) < 3))


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
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
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply kernel
    if (scale < 1) and (antialiasing):
        weights = scale * kernel(distance_to_center * scale)
    else:
        weights = kernel(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True, interpolation=None):
    # The scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round

    in_C, in_H, in_W = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)

    # Choose interpolation method, each method has the matching kernel size
    kernel, kernel_width = {
        "cubic": (cubic, 4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "box": (box, 1.0),
        "linear": (linear, 2.0),
        None: (cubic, 4.0)  # set default interpolation method as cubic
    }.get(interpolation)

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = img_aug[0, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[1, i, :] = img_aug[1, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[2, i, :] = img_aug[2, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :, idx:idx + kernel_width].mv(weights_W[i])

    return out_2


def imresize_np(img, scale, antialiasing=True, interpolation=None):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC BGR [0,1]
    # output: HWC BGR [0,1] w/o round
    
    change_range = False
    if img.max() > 1:
        img_type = img.dtype
        if np.issubdtype(img_type, np.integer):
            info = np.iinfo
        elif np.issubdtype(img_type, np.floating):
            info = np.finfo
        img = img/info(img_type).max
        change_range = True

    img = torch.from_numpy(img)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)

    # Choose interpolation method, each method has the matching kernel size
    kernel, kernel_width = {
        "cubic": (cubic, 4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "box": (box, 1.0),
        "linear": (linear, 2.0),
        None: (cubic, 4.0)  # set default interpolation method as cubic
    }.get(interpolation)

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = img_aug[idx:idx + kernel_width, :, 0].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 1] = img_aug[idx:idx + kernel_width, :, 1].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 2] = img_aug[idx:idx + kernel_width, :, 2].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx:idx + kernel_width, 0].mv(weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx:idx + kernel_width, 1].mv(weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx:idx + kernel_width, 2].mv(weights_W[i])

    out_2 = out_2.numpy().clip(0,1)
    
    if change_range:
        out_2 = out_2*info(img_type).max #uint8 = 255
        out_2 = out_2.astype(img_type)

    return out_2


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
