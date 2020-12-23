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

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)

def merge_imgs(img_list):
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
            if img.shape[1] < img_v or img.shape[0] > img_v:
                img_res = cv2.resize(img, (img_v, img_h), interpolation=cv2.INTER_NEAREST)
                img_list_res.append(img_res)
            else:
                img_list_res.append(img)
        
        return cv2.hconcat(img_list_res)

def save_img_comp(img_list, img_path, mode='RGB'):
    # lr_resized = cv2.resize(lr_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # comparison = cv2.hconcat([lr_resized, sr_img])
    comparison = merge_imgs(img_list)
    save_img(img=comparison, img_path=img_path, mode=mode)
