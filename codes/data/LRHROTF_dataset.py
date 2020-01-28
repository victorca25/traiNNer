import os
import sys
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import logging

import data.util as util
from scripts import augmentations

class LRHROTFDataset(data.Dataset):
    """
    Read LR and HR image pairs.
    If no LR image exist's for a HR image, one will be generated on-the-fly.
    Make sure filenames and it's relative to root locations are kept the same between LR and HR!
    
    Augmentations can be enabled/disabled in the training json:
    - flip
    - rotate
    - downscale
    - blur
    - noise
    - color fringing
    - auto levels
    - unsharpening masks
    - (test) cutout/erasing
    """

    def __init__(self, opt):
        super(LRHROTFDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.LR_env = None  # environment for lmdb
        self.HR_env = None
        self.output_sample_imgs = False  # debugging
    
        self.logger = logging.getLogger('base')

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) \
                        for line in f])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            # Note: lmdb will not currently work
            if type(opt['dataroot_HR']) is str:
                opt['dataroot_HR'] = [opt['dataroot_HR']]
            if type(opt['dataroot_LR']) is str:
                opt['dataroot_LR'] = [opt['dataroot_LR']]
            generated_lr_imgs = 0
            for i, hr_path in enumerate(opt['dataroot_HR']):
                hr_env, hr_imgs = util.get_image_paths(opt['data_type'], hr_path)
                if type(hr_env) is list:
                    if type(self.HR_env) is not list:
                        self.HR_env = []
                    self.HR_env += hr_env
                if type(hr_imgs) is list:
                    if type(self.paths_HR) is not list:
                        self.paths_HR = []
                    self.paths_HR += hr_imgs
                    for hr_img in hr_imgs:
                        lr_img = os.path.join(opt['dataroot_LR'][i], os.path.relpath(hr_img, hr_path))
                        if not os.path.exists(lr_img):
                            self.logger.info(f"LR not found, generating on-the-fly: {lr_img}")
                            self.paths_LR.append(None)
                        else:
                            if type(self.paths_LR) is not list:
                                self.paths_LR = []
                            self.paths_LR.append(lr_img)
            self.logger.info(f"Missing but generated on-the-fly LR images: {self.paths_LR.count(None)}")
            # sort images and env's if any
            if self.HR_env:
                self.HR_env = sorted(self.HR_env)
            if self.LR_env:
                self.LR_env = sorted(self.LR_env)
            if self.paths_HR:
                self.paths_HR = sorted(self.paths_HR)
            if self.paths_LR:
                self.paths_LR = sorted(self.paths_LR)

        assert self.paths_HR, "Error: No HR images found in specified directories."
        assert (
            self.paths_HR and len(self.paths_HR) >= len(self.paths_LR)
        ), "Error: HR dataset contains less images than LR dataset - {}, {}.".format(
            len(self.paths_LR), len(self.paths_HR)
        )
        
        #self.random_scale_list = [1]

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.opt["scale"]
        HR_size = self.opt["HR_size"]
        if HR_size:
            LR_size = HR_size // scale
        
        self.znorm = False # Default case: images are in the [0,1] range
        if self.opt["znorm"]:
            self.znorm = True # Alternative: images are z-normalized to the [-1,1] range
        
        ######## Read the images ########
        
        if self.paths_LR:
            # FLIP
            LRHRchance = 0.
            flip_chance = 0.
            if self.opt["rand_flip_LR_HR"] and self.opt["phase"] == "train":
                LRHRchance = random.uniform(0, 1)
                if self.opt["flip_chance"]:
                    flip_chance = self.opt["flip_chance"]
                else:
                    flip_chance = 0.05
            if LRHRchance < (1 - flip_chance):
                HR_path = self.paths_HR[index]
                LR_path = self.paths_LR[index]
                if LR_path is None:
                    LR_path = HR_path
            else:  # flip
                HR_path = self.paths_LR[index]
                LR_path = self.paths_HR[index]
                if HR_path is None:
                    HR_path = LR_path
            # Read the LR and HR images from the provided paths
            img_LR = util.read_img(self.LR_env, LR_path, znorm=self.znorm)
            img_HR = util.read_img(self.HR_env, HR_path, znorm=self.znorm)
            # Even if LR dataset is provided, force to generate aug_downscale % of downscales OTF from HR
            # The code will later make sure img_LR has the correct size
            if self.opt["aug_downscale"]:
                aug_downscale = self.opt["aug_downscale"]
                if np.random.rand() < aug_downscale:
                    img_LR = img_HR
        else:
            # No LR, let's use HR
            HR_path = self.paths_HR[index]
            img_HR = util.read_img(self.HR_env, HR_path, znorm=self.znorm)
            img_LR = img_HR
        
        ######## Modify the images ########
        
        # HR modcrop in the validation / test phase
        if self.opt["phase"] != "train":
            img_HR = util.modcrop(img_HR, scale)
        
        # change color space if necessary
        # Note: Changing the LR colorspace here could make it so some colors are introduced when 
        #  doing the augmentations later (ie: with Gaussian or Speckle noise), may be good if the
        #  model can learn to remove color noise in grayscale images, otherwise move to before
        #  converting to tensors
        if self.opt["color_tar_HR"]:
            img_HR = util.channel_convert(img_HR.shape[2], self.opt["color_tar_HR"], [img_HR])[0]
        if self.opt["color_tar_LR"]:
            img_LR = util.channel_convert(img_LR.shape[2], self.opt["color_tar_LR"], [img_LR])[0]
        
        ######## Augmentations ########
        
        #Augmentations during training
        if self.opt["phase"] == "train":
            
            # Validate there's an img_LR, if not, use img_HR
            if img_LR is None:
                img_LR = img_HR
            
            # Check that HR and LR have the same dimensions ratio, else, generate new LR from HR
            if img_HR.shape[0] // img_LR.shape[0] != img_HR.shape[1] // img_LR.shape[1]:
                self.logger.info(f"img_LR dimensions ratio does not match img_HR dimensions ratio for: {HR_path}")
                img_LR = img_HR
            
            # Random Crop (reduce computing cost and adjust images to correct size first)
            if img_HR.shape[0] > HR_size or img_HR.shape[1] > HR_size:
                # Here the scale should be in respect to the images, not to the training scale (in case they are being scaled on the fly)
                img_HR, img_LR = augmentations.random_crop_pairs(
                    img_HR, img_LR, HR_size, img_HR.shape[0] // img_LR.shape[0]
                )
            
            # Or if the HR images are too small, Resize to the HR_size size and fit LR pair to LR_size too
            if img_HR.shape[0] < HR_size or img_HR.shape[1] < HR_size:
                self.logger.warning(f"Image: {HR_path} size does not match HR size: ({HR_size}). The image size is: {img_HR.shape}")
                # rescale HR image to the HR_size
                img_HR, _ = augmentations.resize_img(np.copy(img_HR), crop_size=(HR_size,HR_size), algo=cv2.INTER_LINEAR)
                # rescale LR image to the LR_size (The original code discarded the img_LR and generated a new one on the fly from img_HR)
                img_LR, _ = augmentations.resize_img(np.copy(img_LR), crop_size=(LR_size,LR_size), algo=cv2.INTER_LINEAR)
            
            # Randomly scale LR from HR during training if:
            # - LR dataset is not provided
            # - LR dataset is not in the correct scale
            # - Also to check if LR is not at the correct scale already (if img_LR was changed to img_HR)
            if img_LR.shape[0] != LR_size or img_LR.shape[1] != LR_size:
                ds_algo = 777  # matlab-like bicubic downscale
                if self.opt["lr_downscale"]:
                    if self.opt["lr_downscale_types"]:
                        ds_algo = self.opt["lr_downscale_types"]
                else:
                    # else, if for some reason img_LR is too large, default to matlab-like bicubic downscale
                    if not self.opt["aug_downscale"]: # hr images being forced are probably a cause
                        self.logger.info(f"LR image is too large, auto generating new LR for: {LR_path}")
                img_LR, _ = augmentations.scale_img(img_LR, scale, algo=ds_algo)
                # The generated LR sometimes get slightly out of the range
                if self.znorm:
                    np.clip(img_LR, -1., 1., out=img_LR)
                else:
                    np.clip(img_LR, 0., 1., out=img_LR)
            
            # Rotations. 'use_flip' = 180 or 270 degrees (mirror), 'use_rot' = 90 degrees, 'HR_rrot' = random rotations +-45 degrees
            if self.opt["hr_rrot"] and (self.opt["use_flip"] or self.opt["use_rot"]):
                # 50% chance for flip+specific rotation and another 50% chance for random rotation
                # results in 50% chance of untouched image
                if np.random.rand() > 0.5:  # 50% chance for flip and specified rotation
                    img_LR, img_HR = util.augment(
                        [img_LR, img_HR], self.opt["use_flip"], self.opt["use_rot"]
                    )
                elif np.random.rand() > 0.5:  # 50% chance for using unspecified random rotations
                    img_HR, img_LR = augmentations.random_rotate_pairs(img_HR, img_LR, HR_size, scale)
            elif not self.opt["hr_rrot"] and (self.opt["use_flip"] or self.opt["use_rot"]):
                # augmentation - flip, rotate
                img_LR, img_HR = util.augment(
                    [img_LR, img_HR], self.opt["use_flip"], self.opt["use_rot"]
                )
            elif self.opt["hr_rrot"] and np.random.rand() > 0.5:
                img_HR, img_LR = augmentations.random_rotate_pairs(img_HR, img_LR, HR_size, scale)
            
            # Final checks
            # if the resulting HR image size so far is too large or too small, resize HR to the correct size and downscale to generate a new LR on the fly
            if img_HR.shape[0] != HR_size or img_HR.shape[1] != HR_size:
                self.logger.warning(f"Image: {HR_path} size does not match HR size: ({HR_size}). The image size is: {img_HR.shape}")
                img_HR, _ = augmentations.resize_img(np.copy(img_HR), crop_size=(HR_size,HR_size), algo=cv2.INTER_LINEAR)
                if self.opt["lr_downscale_types"]:
                    ds_algo = self.opt["lr_downscale_types"]
                else:
                    ds_algo = 777
                img_LR, _ = augmentations.scale_img(img_HR, scale, algo=ds_algo)
            # if the resulting LR so far does not have the correct dimensions, also generate a new HR-LR image pair on the fly
            if img_LR.shape[0] != LR_size or img_LR.shape[0] != LR_size:
                self.logger.warning(f"Image: {LR_path} size does not match LR size: ({HR_size//scale}). The image size is: {img_LR.shape}")
                # rescale HR image to the HR_size (should not be needed, but something went wrong before, just for sanity)
                img_HR, _ = augmentations.resize_img(np.copy(img_HR), crop_size=(HR_size,HR_size), algo=cv2.INTER_LINEAR)
                if self.opt["lr_downscale_types"]:
                    ds_algo = self.opt["lr_downscale_types"]
                else:
                    ds_algo = 777
                img_LR, _ = augmentations.scale_img(img_HR, scale, algo=ds_algo)
            
            # Below are the LR On The Fly augmentations
            if self.opt["hr_noise"]:
                if self.opt["hr_noise_types"]:
                    img_HR, _ = augmentations.noise_img(img_HR, noise_types=self.opt["hr_noise_types"])
                else:
                    self.logger.info("Noise types 'hr_noise_types' is not defined. Skipping OTF noise for HR.")
            
            # Create color fringes
            if self.opt["lr_fringes"]:
                # cation: can cause destabilization, 20-50% seems stable
                lr_fringes_chance = 0.4
                if self.opt["lr_fringes_chance"]:
                    lr_fringes_chance = self.opt["lr_fringes_chance"]
                if np.random.rand() > (1.- lr_fringes_chance):
                    img_LR = augmentations.translate_chan(img_LR)
            
            # Create blur
            if self.opt["lr_blur"]:
                if self.opt["lr_blur_types"]:
                    img_LR, _, _ = augmentations.blur_img(img_LR, blur_algos=self.opt["lr_blur_types"])
                else:
                    self.logger.info("Blur types 'lr_blur_types' is not defined. Skipping OTF blur.")
            
            # Primary Noise
            if self.opt["lr_noise"]:
                if self.opt["lr_noise_types"]:
                    img_LR, _ = augmentations.noise_img(img_LR, noise_types=self.opt["lr_noise_types"])
                else:
                    self.logger.info("Noise types 'lr_noise_types' is not defined. Skipping OTF primary noise.")
            # Secondary Noise
            if self.opt['lr_noise2']:
                if self.opt['lr_noise_types2']:
                    img_LR, _ = augmentations.noise_img(img_LR, noise_types=self.opt['lr_noise_types2'])
                else:
                    self.logger.info("Noise types 'lr_noise_types2' is not defined. Skipping OTF secondary noise.")
            
            # LR cutout / random erasing (for inpainting/classification tests)
            if self.opt["lr_cutout"] and self.opt["lr_erasing"] != True:
                img_LR = augmentations.cutout(img_LR, img_LR.shape[0] // 2)
            elif self.opt["lr_erasing"] and self.opt["lr_cutout"] != True: 
                img_LR = augmentations.random_erasing(img_LR)
            elif self.opt["lr_cutout"] and self.opt["lr_erasing"]:
                # only do cutout or erasing, not both at the same time
                if np.random.rand() > 0.5:
                    img_LR = augmentations.cutout(img_LR, img_LR.shape[0] // 2, p=0.5)
                else:
                    img_LR = augmentations.random_erasing(img_LR, p=0.5, modes=[3])                
            
            # Apply "auto levels" to images
            if self.opt["auto_levels"]:
                chance = (1 - self.opt["rand_auto_levels"]) if self.opt["rand_auto_levels"] else 1
                if np.random.rand() > chance:
                    if self.opt["auto_levels"] == "HR":
                        img_HR = augmentations.simplest_cb(img_HR, znorm=self.znorm)
                    elif self.opt["auto_levels"] == "LR":
                        img_LR = augmentations.simplest_cb(img_LR, znorm=self.znorm)
                    elif self.opt["auto_levels"] == True or self.opt["auto_levels"] == "Both":
                        img_HR = augmentations.simplest_cb(img_HR, znorm=self.znorm)
                        img_LR = augmentations.simplest_cb(img_LR, znorm=self.znorm)
            
            # Apply unsharpening mask to HR images
            if self.opt["unsharp_mask"]:
                # img_HR1 = img_HR
                chance = (1 - self.opt["rand_unsharp"]) if self.opt["rand_unsharp"] else 1
                if np.random.rand() > chance:
                    img_HR = augmentations.unsharp_mask(img_HR, znorm=self.znorm)
        
        # For testing and validation
        if self.opt["phase"] != "train":
            if self.opt["lr_downscale"]:
                algo = 777
                if self.opt["lr_downscale_types"]:
                    algo = 777
                img_LR, _ = augmentations.scale_img(img_LR, scale, algo=algo)
            
        # Debug
        # Save img_LR and img_HR images to a directory to visualize what is the result of the on the fly augmentations
        # DO NOT LEAVE ON DURING REAL TRAINING
        if self.opt["phase"] == "train" and self.output_sample_imgs:
            debugpath = "/home/owner/Images/BasicSR-otf-debug"
            os.makedirs(debugpath, exist_ok=True)

            import uuid
            hex = uuid.uuid4().hex
            im_name, _ = os.path.basename(HR_path)
            cv2.imwrite(
                os.path.join(debugpath, im_name + "-" + hex + "-LR.png"),
                (((img_LR + 1.0) / 2.0) if self.opt["znorm"] else img_LR)*255
            )
            cv2.imwrite(
                os.path.join(debugpath, im_name + "-" + hex + "-HR.png"),
                (((img_HR + 1.0) / 2.0) if self.opt["znorm"] else img_HR)*255
            )
        
        ######## Convert images to PyTorch Tensors ########
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        # BGRA to RGBA, HWC to CHW, numpy to tensor
        elif img_LR.shape[2] == 4:
            img_HR = img_HR[:, :, [2, 1, 0, 3]]
            img_LR = img_LR[:, :, [2, 1, 0, 3]]

        return {
            "LR": torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float(),
            "HR": torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float(),
            "LR_path": HR_path if LR_path is None else LR_path,
            "HR_path": HR_path
        }

    def __len__(self):
        return len(self.paths_HR)
