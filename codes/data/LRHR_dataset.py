import os.path
import random
import numpy as np
import cv2
import torch
import dataops.common as util
from data.base_dataset import BaseDataset, get_dataroots_paths
from dataops.imresize import resize as imresize


class LRHRDataset(BaseDataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LRHRDataset, self).__init__(opt, keys_ds=['LR','HR'])
        # self.opt = opt
        # self.paths_LR = None
        # self.paths_HR = None
        self.LR_env = None  # environment for lmdb
        self.HR_env = None
        self.znorm = opt.get('znorm', False)

        # get images paths (and optional environments for lmdb) from dataroots
        self.paths_LR, self.paths_HR = get_dataroots_paths(opt, strict=False, keys_ds=self.keys_ds)

        if self.opt.get('data_type') == 'lmdb':
            self.LR_env = util._init_lmdb(opt.get('dataroot_'+self.keys_ds[0]))
            self.HR_env = util._init_lmdb(opt.get('dataroot_'+self.keys_ds[1]))

        self.random_scale_list = [1]

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        data_type = self.opt.get('data_type', 'img')
        scale = self.opt['scale']
        HR_size = self.opt['HR_size']

        # get HR image
        HR_path = self.paths_HR[index]
        img_HR = util.read_img(env=data_type, path=HR_path, lmdb_env=self.HR_env)
        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_HR = util.modcrop(img_HR, scale)
        # change color space if necessary
        if self.opt['color']:
            img_HR = util.channel_convert(img_HR.shape[2], self.opt['color'], [img_HR])[0]

        # get LR image
        if self.paths_LR:
            LR_path = self.paths_LR[index]
            img_LR = util.read_img(env=data_type, path=LR_path, lmdb_env=self.LR_env)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_HR.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, HR_size)
                W_s = _mod(W_s, random_scale, scale, HR_size)
                img_HR = cv2.resize(np.copy(img_HR), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_HR.ndim == 2:
                    img_HR = cv2.cvtColor(img_HR, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_HR.shape
            # using matlab imresize
            img_LR = imresize(img_HR, 1 / scale, antialiasing=True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_HR.shape
            if H < HR_size or W < HR_size:
                img_HR = cv2.resize(
                    np.copy(img_HR), (HR_size, HR_size), interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LR = imresize(img_HR, 1 / scale, antialiasing=True)
                if img_LR.ndim == 2:
                    img_LR = np.expand_dims(img_LR, axis=2)

            H, W, C = img_LR.shape
            LR_size = HR_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            img_HR = img_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]

            # augmentation - flip, rotate
            img_LR, img_HR = util.augment([img_LR, img_HR], self.opt['use_flip'], \
                self.opt['use_rot'])

        # change color space if necessary
        if self.opt['color']:
            img_LR = util.channel_convert(C, self.opt['color'], [img_LR])[0] # TODO during val no definetion

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_HR = util.np2tensor(img_HR, normalize=self.znorm, add_batch=False)
        img_LR = util.np2tensor(img_LR, normalize=self.znorm, add_batch=False)

        if LR_path is None:
            LR_path = HR_path
        return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path}

    def __len__(self):
        return len(self.paths_HR)
