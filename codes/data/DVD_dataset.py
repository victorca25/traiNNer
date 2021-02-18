import os
import uuid

import cv2
import torch
import torch.utils.data as data

from data.Vid_dataset import get_crop_params, apply_crop_params
from dataops.common import get_image_paths, read_img, np2tensor


class DVDDataset(data.Dataset):
    """
    Read interlaced and progressive frame triplets (pairs of three).
    Interlaced frame is expected to be "combed" from the progressive pair.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super(DVDDataset, self).__init__()
        self.opt = opt
        self.debug = None  # path to a directory to save debug files, None = Disable

        self.paths_in = get_image_paths('img', opt['dataroot_in'])
        self.paths_top = get_image_paths('img', opt['dataroot_top'])
        self.paths_bot = get_image_paths('img', opt['dataroot_bottom'])
        self.paths_progressive = get_image_paths('img', opt['dataroot_progressive'])

        if self.paths_in and self.paths_top and self.paths_bot:
            assert len(self.paths_top) >= len(self.paths_in), \	
                'Top dataset contains fewer images than interlaced dataset  - {}, {}.'.format(
                len(self.paths_top), len(self.paths_in))
            
            assert len(self.paths_bot) >= len(self.paths_in), \	
                'Bottom dataset contains fewer images than interlaced dataset  - {}, {}.'.format(
                len(self.paths_bot), len(self.paths_in))

    def __getitem__(self, index):
        patch_size = self.opt['HR_size']
        image_channels = self.opt.get('image_channels', 3)

        if self.paths_progressive:
            if index + 1 != len(self):
                top_path = self.paths_progressive[index]
                bot_path = self.paths_progressive[index + 1]
            else:
                top_path = self.paths_progressive[index - 1]
                bot_path = self.paths_progressive[index]
        else:
            top_path = self.paths_top[index]
            bot_path = self.paths_bot[index]

        img_top = read_img(None, top_path, out_nc=image_channels)
        img_bot = read_img(None, bot_path, out_nc=image_channels)

        # Read interlaced frame or create interlaced image from top/bottom frames
        if self.paths_in:
            in_path = self.paths_in[index]
            img_in = read_img(None, in_path, out_nc=image_channels)
        else:
            in_path = "OTF"
            img_in = img_top.copy()
            img_in[1::2, :, :] = img_bot[1::2, :, :]

        if self.opt['phase'] == 'train':
            # Random Crop (reduce computing cost and adjust images to correct size first)
            crop_params, _ = get_crop_params(img_top, patch_size, 1)
            # for img in img_in, img_top, img_bot:
            #     if img.shape[0] > patch_size or img.shape[1] > patch_size:
            #         img, _ = vd.apply_crop_params(HR=img, hr_crop_params=crop_params)
            if img_in.shape[0] > patch_size or img_in.shape[1] > patch_size:
                img_in, _ = apply_crop_params(HR=img_in, hr_crop_params=crop_params)
            if img_top.shape[0] > patch_size or img_top.shape[1] > patch_size:
                img_top, _ = apply_crop_params(HR=img_top, hr_crop_params=crop_params)
            if img_bot.shape[0] > patch_size or img_bot.shape[1] > patch_size:
                img_bot, _ = apply_crop_params(HR=img_bot, hr_crop_params=crop_params)

        # Debug
        # TODO: use the debugging functions to visualize or save images instead
        # Save img_in, img_top, and img_bot images to a directory to visualize the OTF augmentations
        # DO NOT LEAVE ON DURING REAL TRAINING
        if self.opt['phase'] == 'train':
            if self.debug:
                _, im_name = os.path.split(top_path)
                if not os.path.exists(self.debug):
                    os.makedirs(self.debug)
                rand = uuid.uuid4().hex
                cv2.imwrite(os.path.join(self.debug, im_name + rand + '_interlaced.png'), img_in)
                cv2.imwrite(os.path.join(self.debug, im_name + rand + '_top.png'), img_top)
                cv2.imwrite(os.path.join(self.debug, im_name + rand + '_bottom.png'), img_bot)

        img_in = np2tensor(img_in, add_batch=False)
        img_top = np2tensor(img_top, add_batch=False)
        img_bot = np2tensor(img_bot, add_batch=False)

        return {
            'in': img_in,
            'top': img_top,
            'bottom': img_bot,
            'in_path': in_path,
            'top_path': top_path,
            'bot_path': bot_path
        }

    def __len__(self):
        return len(self.paths_top or self.paths_progressive)


class DVDIDataset(data.Dataset):
    """Read interlaced images only in the test phase."""

    def __init__(self, opt):
        super(DVDIDataset, self).__init__()
        self.opt = opt
        self.paths_in = None

        # read image list from lmdb or image files
        self.paths_in = get_image_paths('img', opt['dataroot_in'])
        assert self.paths_in, 'Error: Interlaced paths are empty.'

    def __getitem__(self, index):
        # get LR image
        in_path = self.paths_in[index]
        # img_LR = util.read_img(self.LR_env, LR_path)
        img_in = read_img(None, in_path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in = np2tensor(img_in, add_batch=False)

        return {'in': img_in, 'in_path': in_path}

    def __len__(self):
        return len(self.paths_in)
