import numpy as np
import torch
import dataops.common as util
from data.base_dataset import BaseDataset, get_dataroots_paths


class LRDataset(BaseDataset):
    '''Read LR images only in the test phase.'''

    def __init__(self, opt):
        super(LRDataset, self).__init__(opt, keys_ds=['LR','HR'])
        # self.opt = opt
        self.paths_LR = None
        self.LR_env = None  # environment for lmdb
        self.znorm = opt.get('znorm', False) # Alternative: images are z-normalized to the [-1,1] range

        # read image list from lmdb or image files
        self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])
        if self.opt.get('data_type') == 'lmdb':
            self.LR_env = util._init_lmdb(opt.get('dataroot_'+self.keys_ds[0]))
        
        assert self.paths_LR, 'Error: LR paths are empty.'

    def __getitem__(self, index):
        LR_path = None
        data_type = self.opt.get('data_type', 'img')
        # get LR image
        LR_path = self.paths_LR[index]
        img_LR = util.read_img(env=data_type, path=LR_path, lmdb_env=self.LR_env)
        H, W, C = img_LR.shape

        # change color space if necessary
        if self.opt['color']:
            img_LR = util.channel_convert(C, self.opt['color'], [img_LR])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LR = util.np2tensor(img_LR, normalize=self.znorm, add_batch=False)

        return {'LR': img_LR, 'LR_path': LR_path}

    def __len__(self):
        return len(self.paths_LR)
