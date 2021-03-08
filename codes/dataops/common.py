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

try:
    from PIL import Image
except:
    pass

####################
# Files & IO
####################

###################### get image path list ######################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', 
                  '.PPM', '.bmp', '.BMP', '.dng', '.DNG', '.webp',
                  '.tif', '.TIF', '.tiff', '.TIFF', '.npy', '.NPY']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path, max_dataset_size=float("inf")):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images[:min(max_dataset_size, len(images))]


def _get_paths_from_lmdb(dataroot):
    """Get image path list from lmdb meta info.
    Args:
        dataroot (str): dataroot path.
    Returns:
        list[str]: Returned path list.
    """
    if not dataroot.endswith('.lmdb'):
        raise ValueError(f'Folder {dataroot} should in lmdb format.')
    with open(os.path.join(dataroot, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]
    return paths


def _init_lmdb(dataroot, readonly=True, lock=False, readahead=False, meminit=False):
    """ initializes lmbd env from dataroot """
    try:
        import lmdb
    except ImportError:
        raise ImportError('Please install lmdb to enable use.')
    
    assert isinstance(dataroot, str), 'lmdb is only supported using a single lmdb database per dataroot.'

    #lmdb_env
    return lmdb.open(dataroot, readonly=readonly, lock=lock, readahead=readahead, meminit=meminit)


def get_image_paths(data_type, dataroot, max_dataset_size=float("inf")):
    '''get image path list
    support lmdb or image files'''
    paths = None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot, max_dataset_size=max_dataset_size))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return paths


###################### read images ######################

def _read_lmdb_img(key, lmdb_env):
    """ Read image from lmdb with key.
    Args:
        key (str | obj:`Path`): the lmdb key / image path in lmdb.
        lmdb_env: lmdb environment initialized with _init_lmdb()
    Returns:
        Decoded image from buffer in bytes
    """
    
    with lmdb_env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))

    # return buf
    img = imfrombytes(buf)  # float32=True
    return img


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.
    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.
    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def read_img(env=None, path=None, out_nc=3, fix_channels=True, lmdb_env=None, loader='cv2'):
    '''
        Reads image using cv2 or PIL (rawpy if dng), from lmdb or from a 
        buffer (path=buffer).
    Arguments:
        path: image path or buffer to read
        out_nc: Desired number of channels
        fix_channels: changes the images to the desired number of channels
        lmdb_env: lmdb environment to use (for lmdb)
        loader: select a library to open the images with: 'cv2', 'pil',
         'plt' (optional)
    Output:
        Numpy HWC, BGR, [0,255] by default 
    '''

    img = None
    if env is None or env == 'img':  # img
        if(path[-3:].lower() == 'dng'): # if image is a DNG
            import rawpy
            with rawpy.imread(path) as raw:
                img = raw.postprocess()
        elif(path[-3:].lower() == 'npy'): # if image is a NPY numpy array
            with open(path, 'rb') as f:
                img = np.load(f)
        elif loader == 'pil': # using PIL instead of OpenCV
            img = Image.open(path).convert('RGB')
        elif loader == 'plt' : # For other images unrecognized by cv2
            import matplotlib.pyplot as plt
            img = (255*plt.imread(path)[:,:,:3]).astype('uint8')
        else: # else, if image can be read by cv2
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    elif env is 'lmdb':
        img = _read_lmdb_img(path, lmdb_env)
    elif env is 'buffer':
        img = cv2.imdecode(path, cv2.IMREAD_UNCHANGED)
    else:
        raise NotImplementedError("Unsupported env: {}".format(env))

    # if (not isinstance(img, np.ndarray)): # or (not isinstance(img, Image.Image)):
    #     raise ValueError(f"Failed to read image: {path}")

    if fix_channels and loader == 'cv2':
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

#TODO: will be unused, but move to augmentations.py
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
# Convert (Tensor) Images to Patches
####################

def extract_patches_2d(img, patch_shape, step=[1.0,1.0], batch_first=False):
    """
    Convert a 4D tensor into a 5D tensor of patches (crops) of the original tensor.
    Uses unfold to extract sliding local blocks from an batched input tensor.
    
    Arguments:
        img: the image batch to crop
        patch_shape: tuple with the shape of the last two dimensions (H,W) 
            after crop
        step: the size of the step used to slide the blocks in each dimension.
            If each value 0.0 < step < 1.0, the overlap will be relative to the
            patch size * step
        batch_first: return tensor with batch as the first dimension or the 
            second

    Reference: 
    https://gist.github.com/dem123456789/23f18fd78ac8da9615c347905e64fc78
    """
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    
    # pad to fit patch dimensions
    if(img.size(2) < patch_H):
        num_padded_H_Top = (patch_H - img.size(2)) // 2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0, 0, num_padded_H_Top, num_padded_H_Bottom), 0)
        img = padding_H(img)
    if(img.size(3) < patch_W):
        num_padded_W_Left = (patch_W - img.size(3)) // 2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left, num_padded_W_Right, 0, 0), 0)
        img = padding_W(img)
    
    # steps to overlay crops of the images
    step_int = [0, 0]
    step_int[0] = int(patch_H * step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W * step[1]) if(isinstance(step[1], float)) else step[1]
    
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0):
        patches_fold_H = torch.cat((patches_fold_H,
                                    img[:, :, -patch_H:, :].permute(0, 1, 3, 2).unsqueeze(2)),dim=2)
    
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])
    if((img.size(3) - patch_W) % step_int[1] != 0):
        patches_fold_HW = torch.cat((patches_fold_HW,
                                     patches_fold_H[:, :, :, -patch_W:, :].permute(0, 1, 2, 4, 3).unsqueeze(3)), dim=3)
    
    patches = patches_fold_HW.permute(2, 3, 0, 1, 4, 5)
    patches = patches.reshape(-1, img.size(0), img.size(1), patch_H, patch_W)
    
    if(batch_first):
        patches = patches.permute(1, 0, 2, 3, 4)
    return patches


def reconstruct_from_patches_2d(patches, img_shape, step=[1.0,1.0], batch_first=False):
    """
    Reconstruct a batch of images from the cropped patches. 
    Inverse operation from extract_patches_2d(). 

    Arguments: 
        patches: the patches tensor
        img_shape: tuple with the sizes of the last two dimensions (H,W)
            of the resulting images. If the patches dimensions have
            changed (scaled), must provide the final dimensions
            i.e. [H * scale, W * scale]
        step: same step used in extract_patches_2d
        batch_first: if the incoming tensor has batch as the first
            dimension or not
    """
    if(batch_first):
        patches = patches.permute(1, 0, 2, 3, 4)
    
    patch_H, patch_W = patches.size(3), patches.size(4)
    img_size = (patches.size(1), patches.size(2),max(img_shape[0], patch_H), max(img_shape[1], patch_W))
    
    step_int = [0, 0]
    step_int[0] = int(patch_H * step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W * step[1]) if(isinstance(step[1], float)) else step[1]
    
    nrow, ncol = 1 + (img_size[-2] - patch_H)//step_int[0], 1 + (img_size[-1] - patch_W)//step_int[1]
    r_nrow = nrow + 1 if((img_size[2] - patch_H) % step_int[0] != 0) else nrow
    r_ncol = ncol + 1 if((img_size[3] - patch_W) % step_int[1] != 0) else ncol
    
    patches = patches.reshape(r_nrow, r_ncol, img_size[0], img_size[1], patch_H, patch_W)
    
    img = torch.zeros(img_size, device = patches.device)
    
    overlap_counter = torch.zeros(img_size, device = patches.device)
    
    for i in range(nrow):
        for j in range(ncol):
            img[:, :, i * step_int[0]:i * step_int[0] + patch_H, j * step_int[1]:j * step_int[1] + patch_W] += patches[i, j,]
            overlap_counter[:, :, i * step_int[0]:i * step_int[0] + patch_H, j * step_int[1]:j * step_int[1] + patch_W] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0):
        for j in range(ncol):
            img[:, :, -patch_H:, j * step_int[1]:j * step_int[1] + patch_W] += patches[-1, j,]
            overlap_counter[:, :, -patch_H:, j * step_int[1]:j * step_int[1] + patch_W] += 1
    if((img_size[3] - patch_W) % step_int[1] != 0):
        for i in range(nrow):
            img[:, :, i * step_int[0]:i*step_int[0] + patch_H, -patch_W:] += patches[i, -1,]
            overlap_counter[:, :, i * step_int[0]:i * step_int[0] + patch_H, -patch_W:] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0 and (img_size[3] - patch_W) % step_int[1] != 0):
        img[:, :, -patch_H:, -patch_W:] += patches[-1, -1,]
        overlap_counter[:, :, -patch_H:, -patch_W:] += 1
    img /= overlap_counter
    
    # remove the extra padding if image is smaller than the patch sizes
    if(img_shape[0] < patch_H):
        num_padded_H_Top = (patch_H - img_shape[0])//2
        num_padded_H_Bottom = patch_H - img_shape[0] - num_padded_H_Top
        img = img[:, :, num_padded_H_Top:-num_padded_H_Bottom,]
    if(img_shape[1] < patch_W):
        num_padded_W_Left = (patch_W - img_shape[1])//2
        num_padded_W_Right = patch_W - img_shape[1] - num_padded_W_Left
        img = img[:, :, :, num_padded_W_Left:-num_padded_W_Right]
    
    return img


def recompose_tensor(patches, height, width, step=[1.0,1.0], scale=1):
    """ Reconstruct images that have been cropped to patches.
    Unlike reconstruct_from_patches_2d(), this function allows to 
    use blending between the patches if they were generated a 
    step between 0.5 (50% overlap) and 1.0 (0% overlap), 
    relative to the original patch size

    Arguments:
        patches: the image patches
        height: the original image height
        width: the original image width
        step: the overlap step factor, from 0.5 to 1.0
        scale: the scale at which the patches are in relation to the 
            original image

    References:
    https://github.com/sunreef/BlindSR/blob/master/src/image_utils.py
    https://gist.github.com/dem123456789/23f18fd78ac8da9615c347905e64fc78

    """

    assert isinstance(step, float) and step >= 0.5 and step <= 1.0

    full_height = scale * height
    full_width = scale * width
    batch_size, channels, patch_size, _ = patches.size()
    overlap = scale * int(round((1.0 - step) * (patch_size / scale)))
    # print("patch_size:", patch_size) #TODO
    # print("overlap:", overlap) #TODO

    effective_patch_size = int(step * patch_size)
    patch_H, patch_W = patches.size(2), patches.size(3)
    img_size = (patches.size(0), patches.size(1), max(full_height, patch_H), max(full_width, patch_W))

    step = [step, step]
    step_int = [0, 0]
    step_int[0] = int(patch_H * step[0])
    step_int[1] = int(patch_W * step[1])

    nrow, ncol = 1 + (img_size[-2] - patch_H)//step_int[0], 1 + (img_size[-1] - patch_W)//step_int[1]
    n_patches_height = nrow + 1 if((img_size[2] - patch_H) % step_int[0] != 0) else nrow
    n_patches_width = ncol + 1 if((img_size[3] - patch_W) % step_int[1] != 0) else ncol

    final_batch_size = batch_size // (n_patches_height * n_patches_width)

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
