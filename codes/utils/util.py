import os
import math
from datetime import datetime
import numpy as np
import cv2
from torchvision.utils import make_grid
import random
import torch
import logging

import re


####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

#TODO: reuse in other cases where files have to be searched
def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the defined files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the found files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def save_img(img, img_path, mode='RGB'):
    '''
    Save a single image to the defined path
    '''
    cv2.imwrite(img_path, img)

def merge_imgs(img_list):
    '''
    Auxiliary function to horizontally concatenate images in
        a list using cv2.hconcat
    '''
    if isinstance(img_list, list):
        img_h = 0
        img_v = 0
        for img in img_list:
            if img.shape[0] > img_v:
                img_h = img.shape[0]
            if img.shape[1] > img_v:
                img_v = img.shape[1]

        img_list_res = []
        for img in img_list:
            if img.shape[1] < img_v or img.shape[0] < img_h:
                img_res = cv2.resize(img, (img_v, img_h), interpolation=cv2.INTER_NEAREST)
                img_list_res.append(img_res)
            else:
                img_list_res.append(img)
        
        return cv2.hconcat(img_list_res)
    elif isinstance(img_list, np.ndarray):
        return img_list
    else:
        raise NotImplementedError('To merge images img_list should be a list of cv2 images.')

def save_img_comp(img_list, img_path, mode='RGB'):
    '''
    Create a side by side comparison of multiple images in a list
        to save to a defined path
    '''
    # lr_resized = cv2.resize(lr_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # comparison = cv2.hconcat([lr_resized, sr_img])
    comparison = merge_imgs(img_list)
    save_img(img=comparison, img_path=img_path, mode=mode)
