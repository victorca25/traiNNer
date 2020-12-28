import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import dataops.common as util
import dataops.augmentations as augmentations


class DVDDataset(data.Dataset):
    '''
    Read interlaced and progressive frame triplets (pairs of three).
    Interlaced frame is expected to be "combed" from the progressive pair.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(DVDDataset, self).__init__()
        self.opt = opt
        self.paths_in, self.paths_top, self.paths_bot, self.paths_prog = None, None, None, None
        self.output_sample_imgs = None
        
        _, self.paths_in = util.get_image_paths('img', opt['dataroot_in'])
        _, self.paths_top = util.get_image_paths('img', opt['dataroot_top'])
        _, self.paths_bot = util.get_image_paths('img', opt['dataroot_bottom'])
        _, self.paths_prog = util.get_image_paths('img', opt['dataroot_progressive'])

        if self.paths_in and self.paths_top and self.paths_bot:
            assert len(self.paths_top) >= len(self.paths_in), \
                'Top dataset contains fewer images than interlaced dataset  - {}, {}.'.format(
                len(self.paths_top), len(self.paths_in))
            assert len(self.paths_bot) >= len(self.paths_in), \
                'Bottom dataset contains fewer images than interlaced dataset  - {}, {}.'.format(
                len(self.paths_bot), len(self.paths_in))

    def __getitem__(self, index):
        in_path, top_path, bot_path = None, None, None
        patch_size = self.opt['HR_size']

        image_channels = self.opt.get('image_channels', 3)

        if self.paths_prog:
            if index+1 != len(self):
                top_path = self.paths_prog[index]
                bot_path = self.paths_prog[index+1]
            else:
                top_path = self.paths_prog[index-1]
                bot_path = self.paths_prog[index]
        else:
            top_path = self.paths_top[index]
            bot_path = self.paths_bot[index]

        img_top = util.read_img(None, top_path, out_nc=image_channels)
        img_bot = util.read_img(None, bot_path, out_nc=image_channels)

        # Read interlaced frame or create interlaced image from top/bottom frames
        if self.paths_in is None:
            img_in = img_top.copy()
            img_in[1::2, :, :] = img_bot[1::2, :, :]
        else:
            in_path = self.paths_in[index]
            img_in = util.read_img(None, in_path, out_nc=image_channels)
        
        if self.opt['phase'] == 'train':
            # Random Crop (reduce computing cost and adjust images to correct size first)
            for img_hr in img_top, img_bot:
                if img_hr.shape[0] > patch_size or img_hr.shape[1] > patch_size:
                    img_top, img_bot, img_in = augmentations.random_crop_dvd(
                        img_top, img_bot, img_in, patch_size)
            
        # Debug #TODO: use the debugging functions to visualize or save images instead
        # Save img_in, img_top, and img_bot images to a directory to visualize what is the result of the on the fly augmentations
        # DO NOT LEAVE ON DURING REAL TRAINING
        # self.output_sample_imgs = True
        if self.opt['phase'] == 'train':
            if self.output_sample_imgs:
                import os
                _, im_name = os.path.split(top_path)
                debugpath = os.path.join('C:/tmp_test', 'sampleOTFimgs')
                if not os.path.exists(debugpath):
                    os.makedirs(debugpath)
                import uuid
                hex = uuid.uuid4().hex
                cv2.imwrite(debugpath+"\\"+im_name+hex+'_interlaced.png',
                            img_in)  # random name to save
                cv2.imwrite(debugpath+"\\"+im_name+hex+'_top.png',
                            img_top)  # random name to save
                cv2.imwrite(debugpath+"\\"+im_name+hex+'_bottom.png',
                            img_bot)  # random name to save

        img_in = util.np2tensor(img_in, add_batch=False)
        img_top = util.np2tensor(img_top, add_batch=False)
        img_bot = util.np2tensor(img_bot, add_batch=False)

        if in_path is None:
            in_path = 'OTF'
        return {'in': img_in, 'top': img_top, 'bottom': img_bot, 'in_path': in_path, 'top_path': top_path, 'bot_path': bot_path}

    def __len__(self):
        return len(self.paths_top) if self.paths_top else len(self.paths_prog)


class DVDIDataset(data.Dataset):
    '''Read interlaced images only in the test phase.'''

    def __init__(self, opt):
        super(DVDIDataset, self).__init__()
        self.opt = opt
        self.paths_in = None

        # read image list from lmdb or image files
        _, self.paths_in = util.get_image_paths('img', opt['dataroot_in'])
        assert self.paths_in, 'Error: Interlaced paths are empty.'

    def __getitem__(self, index):
        in_path = None

        # get LR image
        in_path = self.paths_in[index]
        #img_LR = util.read_img(self.LR_env, LR_path)
        img_in = util.read_img(None, in_path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in = util.np2tensor(img_in, add_batch=False)

        return {'in': img_in, 'in_path': in_path}

    def __len__(self):
        return len(self.paths_in)
