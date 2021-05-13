import os.path
import random
import numpy as np
import cv2
import torch
import dataops.common as util

# from dataops.imresize import resize as imresize




from data.base_dataset import BaseDataset, get_dataroots_paths, read_imgs_from_path

from dataops.augmentations import generate_A_fn, image_type, get_default_imethod, dim_change_fn, shape_change_fn, random_downscale_B
from dataops.augmentations import get_unpaired_params, get_augmentations, get_totensor_params, get_totensor
from dataops.augmentations import get_ds_kernels, get_noise_patches
from dataops.augmentations import get_params, image_size, image_channels, scale_params, scale_opt, get_transform
from dataops.augmentations import random_rotate_pairs  # TMP


class LRHRDataset(BaseDataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LRHRDataset, self).__init__(opt, keys_ds=['LR','HR'])
        # self.opt = opt
        # self.paths_LR, self.paths_HR = None, None
        self.LR_env, self.HR_env = None, None  # environment for lmdb
        # self.znorm = opt.get('znorm', False)
        self.vars = opt.get('outputs', 'LRHR')  #'AB'
        self.ds_kernels = get_ds_kernels(opt)
        self.noise_patches = get_noise_patches(opt)

        # get images paths (and optional environments for lmdb) from dataroots
        self.paths_LR, self.paths_HR = get_dataroots_paths(opt, strict=False, keys_ds=self.keys_ds)

        if self.opt.get('data_type') == 'lmdb':
            self.LR_env = util._init_lmdb(opt.get('dataroot_'+self.keys_ds[0]))
            self.HR_env = util._init_lmdb(opt.get('dataroot_'+self.keys_ds[1]))

        # self.random_scale_list = [1]

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.opt['scale']
        HR_size = self.opt['HR_size']
        # data_type = self.opt.get('data_type', 'img')
        if HR_size:
            LR_size = HR_size // scale

        """
        # get HR image
        HR_path = self.paths_HR[index]
        img_HR = util.read_img(env=data_type, path=HR_path, lmdb_env=self.HR_env)

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
        """

        ######## Read the images ########
        img_LR, img_HR, LR_path, HR_path = read_imgs_from_path(
            self.opt, index, self.paths_LR, self.paths_HR, self.LR_env, self.HR_env)

        ######## Modify the images ########

        # HR modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_HR = util.modcrop(img_HR, scale)

        # change color space if necessary
        color_HR = self.opt.get('color', None) or self.opt.get('color_HR', None)
        if color_HR:
            img_HR = util.channel_convert(image_channels(img_HR), color_HR, [img_HR])[0]

        if self.opt['phase'] == 'train':
            """
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
            """

            default_int_method = get_default_imethod(image_type(img_LR))

            # self.opt['crop_size'] = HR_size  # TODO: TMP
            # # self.opt['preprocess'] = 'crop'  # TODO: TMP
            # self.opt['load_size'] = 512  # TODO: TMP
            # self.opt['center_crop_size'] = 256  # TODO: TMP
            # self.opt['aspect_ratio'] = 2  # TODO: TMP

            # random HR downscale
            img_LR, img_HR = random_downscale_B(img_A=img_LR, img_B=img_HR, 
                                opt=self.opt)

            # Validate there's an img_LR, if not, use img_HR
            if img_LR is None:
                img_LR = img_HR
                print("Image LR: ", LR_path, ("was not loaded correctly, using HR pair to downscale on the fly."))

            # check that HR and LR have the same dimensions ratio
            img_LR, img_HR = shape_change_fn(
                img_A=img_LR, img_B=img_HR, opt=self.opt, scale=scale,
                default_int_method=default_int_method)

            # if the HR images are too small, Resize to the HR_size size and fit LR pair to LR_size too
            img_LR, img_HR = dim_change_fn(
                img_A=img_LR, img_B=img_HR, opt=self.opt, scale=scale,
                default_int_method=default_int_method, 
                crop_size=HR_size, A_crop_size=LR_size,
                ds_kernels=self.ds_kernels)

            # randomly scale LR (from HR) if needed
            img_LR, img_HR = generate_A_fn(img_A=img_LR, img_B=img_HR,
                            opt=self.opt, scale=scale,
                            default_int_method=default_int_method, 
                            crop_size=HR_size, A_crop_size=LR_size,
                            ds_kernels=self.ds_kernels)

            # get and apply the paired transformations below
            transform_params = get_params(
                scale_opt(self.opt, scale), image_size(img_LR))
            A_transform = get_transform(
                scale_opt(self.opt, scale),
                transform_params,
                # grayscale=(input_nc == 1),
                method=default_int_method)
            B_transform = get_transform(
                self.opt,
                scale_params(transform_params, scale),
                # grayscale=(output_nc == 1),
                method=default_int_method)
            img_LR = A_transform(img_LR)
            img_HR = B_transform(img_HR)

            #TODO: Remove TMP
            if self.opt.get('hr_rrot', None):
                if random.random() > 0.5: # randomize the random rotations, so half the images are the original
                    img_HR, img_LR = random_rotate_pairs(img_HR, img_LR, HR_size, scale)


            # get and apply the unpaired transformations below
            lr_aug_params, hr_aug_params = get_unpaired_params(self.opt)

            lr_augmentations = get_augmentations(
                self.opt, 
                params=lr_aug_params,
                noise_patches=self.noise_patches,
                )
            hr_augmentations = get_augmentations(
                self.opt, 
                params=hr_aug_params,
                noise_patches=self.noise_patches,
                )

            img_LR = lr_augmentations(img_LR)
            img_HR = hr_augmentations(img_HR)


        # For testing and validation
        if self.opt['phase'] != 'train':
            # Randomly downscale LR if enabled 
            if self.opt['lr_downscale']:
                # if self.opt['lr_downscale_types']:
                #     img_LR, scale_interpol_algo = augmentations.scale_img(img_LR, scale, algo=self.opt['lr_downscale_types'])
                # else: # Default to matlab-like bicubic downscale
                #     img_LR, scale_interpol_algo = augmentations.scale_img(img_LR, scale, algo=777)
                img_LR, _ = Scale(img_LR, scale, algo=self.opt.get('lr_downscale_types', 777))


        # change color space if necessary
        color_LR = self.opt.get('color', None) or self.opt.get('color_LR', None)
        if color_LR:
            img_LR = util.channel_convert(image_channels(img_LR), color_LR, [img_LR])[0]  # TODO during val no definetion



        ######## Convert images to PyTorch Tensors ########

        """
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_HR = util.np2tensor(img_HR, normalize=self.znorm, add_batch=False)
        img_LR = util.np2tensor(img_LR, normalize=self.znorm, add_batch=False)
        """

        totensor_params = get_totensor_params(self.opt)
        tensor_transform = get_totensor(self.opt, params=totensor_params, toTensor=True, grayscale=False)
        img_LR = tensor_transform(img_LR)
        img_HR = tensor_transform(img_HR)

        if LR_path is None:
            LR_path = HR_path
        if self.vars == 'AB':
            return {'A': img_LR, 'B': img_HR, 'A_paths': LR_path, 'B_paths': HR_path}
        else:
            return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path}

    def __len__(self):
        return len(self.paths_HR)
