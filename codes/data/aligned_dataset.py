import os.path
import random
import numpy as np
import cv2
import torch
import dataops.common as util

# import dataops.augmentations as augmentations #TMP
# from dataops.augmentations import Scale, MLResize, RandomQuantize, KernelDownscale, NoisePatches, RandomNoisePatches, get_resize, get_blur, get_noise, get_pad
# from dataops.debug import tmp_vis, describe_numpy, describe_tensor

# import dataops.opencv_transforms.opencv_transforms as transforms
from data.base_dataset import BaseDataset, get_dataroots_paths, read_imgs_from_path

from dataops.augmentations import generate_A_fn, image_type, get_default_imethod, dim_change_fn, shape_change_fn, random_downscale_B
from dataops.augmentations import get_unpaired_params, get_augmentations, get_totensor_params, get_totensor
from dataops.augmentations import set_transforms, get_ds_kernels, get_noise_patches
from dataops.augmentations import get_params, image_size, image_channels, scale_params, scale_opt, get_transform
from dataops.augmentations import random_rotate_pairs  # TMP


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.
    It can work with either a single dataroot directory that contains single images 
    pairs in the form of {A,B} or one directory for images in the A domain (dataroot_A
    or dataroot_LR) and another for images in the B domain (dataroot_B or dataroot_HR). 
    In the second case, the A-B image pairs in each directory have to have the same name.
    The pair is ensured by 'sorted' function, so please check the name convention.
    If only target image is provided, generate source image on-the-fly.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option dictionary) -- stores all the experiment flags
        """
        super(AlignedDataset, self).__init__(opt, keys_ds=['LR','HR'])
        # self.opt = opt
        # self.paths_LR, self.paths_HR = None, None
        self.LR_env, self.HR_env = None, None  # environment for lmdb
        self.vars = opt.get('outputs', 'LRHR')  #'AB'
        self.ds_kernels = get_ds_kernels(opt)
        self.noise_patches = get_noise_patches(opt)
        set_transforms(opt.get('img_loader', 'cv2'))

        # get images paths (and optional environments for lmdb) from dataroots
        self.paths_LR, self.paths_HR = get_dataroots_paths(opt, strict=False, keys_ds=self.keys_ds)

        if self.opt.get('data_type') == 'lmdb':
            self.LR_env = util._init_lmdb(opt.get('dataroot_'+self.keys_ds[0]))
            self.HR_env = util._init_lmdb(opt.get('dataroot_'+self.keys_ds[1]))

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index: a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            (or LR, HR, LR_paths and HR_paths)
            A (tensor): an image in the input domain
            B (tensor): its corresponding image in the target domain
            A_paths (str): paths A images
            B_paths (str): paths B images (can be same as A_paths if 
                using single images)
        """
        scale = self.opt.get('scale')
        crop_size = self.opt.get('crop_size', 128)
        A_crop_size = crop_size // scale if crop_size else None

        ######## Read the images ########
        img_LR, img_HR, LR_path, HR_path = read_imgs_from_path(
            self.opt, index, self.paths_LR, self.paths_HR, self.LR_env, self.HR_env)

        ######## Modify the images ########

        # HR modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_HR = util.modcrop(img_HR, scale)

        # change color space if necessary
        # TODO: move to get_transform()
        color_HR = self.opt.get('color', None) or self.opt.get('color_HR', None)
        if color_HR:
            img_HR = util.channel_convert(image_channels(img_HR), color_HR, [img_HR])[0]
        color_LR = self.opt.get('color', None) or self.opt.get('color_LR', None)
        if color_LR:
            img_LR = util.channel_convert(image_channels(img_LR), color_LR, [img_LR])[0]

        ######## Augmentations ########

        #Augmentations during training
        if self.opt['phase'] == 'train':

            default_int_method = get_default_imethod(image_type(img_LR))

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

            # if the HR images are too small, Resize to the crop_size size and fit LR pair to A_crop_size too
            img_LR, img_HR = dim_change_fn(
                img_A=img_LR, img_B=img_HR, opt=self.opt, scale=scale,
                default_int_method=default_int_method, 
                crop_size=crop_size, A_crop_size=A_crop_size,
                ds_kernels=self.ds_kernels)

            # randomly scale LR (from HR) if needed
            img_LR, img_HR = generate_A_fn(img_A=img_LR, img_B=img_HR,
                            opt=self.opt, scale=scale,
                            default_int_method=default_int_method, 
                            crop_size=crop_size, A_crop_size=A_crop_size,
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

            """
            # Final sizes checks
            # if the resulting HR image size so far is too large or too small, resize HR to the correct size and downscale to generate a new LR on the fly
            # if the resulting LR so far does not have the correct dimensions, also generate a new HR-LR image pair on the fly
            if img_HR.shape[0] != crop_size or img_HR.shape[1] != crop_size or img_LR.shape[0] != A_crop_size or img_LR.shape[1] != A_crop_size:

                #if img_HR.shape[0] != crop_size or img_HR.shape[1] != crop_size:
                    #TODO: temp disabled to test
                    #print("Image: ", HR_path, " size does not match HR size: (", crop_size,"). The image size is: ", img_HR.shape)
                #if img_LR.shape[0] != A_crop_size or img_LR.shape[0] != A_crop_size:
                    #TODO: temp disabled to test
                    #print("Image: ", LR_path, " size does not match LR size: (", crop_size//scale,"). The image size is: ", img_LR.shape)

                # rescale HR image to the crop_size (should not be needed in LR case, but something went wrong before, just for sanity)
                #img_HR, _ = augmentations.resize_img(np.copy(img_HR), newdim=(crop_size,crop_size), algo=cv2.INTER_LINEAR)
                img_HR = transforms.Resize((crop_size, crop_size), interpolation="BILINEAR")(np.copy(img_HR))
                # if manually provided and scale algorithms are provided, then use it, else use matlab imresize to generate LR pair
                ds_algo  = self.opt.get('lr_downscale_types', 777) 
                #img_LR, _ = augmentations.scale_img(img_HR, scale, algo=ds_algo)
                img_LR, _ = Scale(img_HR, scale, algo=ds_algo)
            """

            # Below are the On The Fly augmentations

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
                img_LR, _ = Scale(img_LR, scale, algo=self.opt.get('lr_downscale_types', 777))

        # Alternative position for changing the colorspace of LR. 
        # color_LR = self.opt.get('color', None) or self.opt.get('color_LR', None)
        # if color_LR:
        #     img_LR = util.channel_convert(image_channels(img_LR), color_LR, [img_LR])[0]

        ######## Convert images to PyTorch Tensors ########

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

