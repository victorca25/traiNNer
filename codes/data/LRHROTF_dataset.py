import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util

import sys
sys.path.append('../codes/scripts')
sys.path.append('../codes/data')
import augmentations

class LRHRDataset(data.Dataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.LR_env = None  # environment for lmdb
        self.HR_env = None
        #self.HR_crop = None #v
        self.HR_rrot = None #v
        self.LR_scale = None #v
        self.scale_algos = None #v
        self.LR_blur = None #v
        self.HR_noise = None #v
        self.LR_noise = None #v
        self.LR_noise2 = None #v
        self.LR_cutout = None #v
        self.LR_erasing = None #v
        self.output_sample_imgs = None

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) \
                        for line in f])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.HR_env, self.paths_HR = util.get_image_paths(opt['data_type'], opt['dataroot_HR'])
            self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])

        assert self.paths_HR, 'Error: HR path is empty.'
        if self.paths_LR and self.paths_HR:
            # Modify to allow using HR and LR folders with different amount of images
            # - If an LR image pair is not found, downscale HR on the fly, else, use the LR
            # - If all LR are provided and 'lr_downscale' is enabled, randomize use of provided LR and OTF LR for augmentation
            """
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_HR))
            """
            #"""
            assert len(self.paths_HR) >= len(self.paths_LR), \
                'HR dataset contains less images than LR dataset  - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_HR))
            if len(self.paths_LR) < len(self.paths_HR):
                print('LR contains less images than HR dataset  - {}, {}. Will generate missing images on the fly.'.format(len(self.paths_LR), len(self.paths_HR)))
                import os
                i=0
                tmp = []
                for idx in range(0, len(self.paths_HR)):
                    _, HRtail = os.path.split(self.paths_HR[idx])
                    if i < len(self.paths_LR):
                        LRhead, LRtail = os.path.split(self.paths_LR[i])
                        
                        if LRtail == HRtail:
                            LRimg_path = os.path.join(LRhead, LRtail)
                            tmp.append(LRimg_path)
                            i+=1
                        else:
                            LRimg_path = None
                            tmp.append(LRimg_path)
                    else: #if the last image is missing
                        LRimg_path = None
                        tmp.append(LRimg_path)
                self.paths_LR = tmp
            #"""
            
        #v parse on the fly options
        if opt['hr_rrot']: #v variable to activate automatic rotate HR image and generate LR
            self.HR_rrot = True
            print("HR random rotation enabled")
        if opt['hr_noise']: #v  variable to activate adding noise to HR image
            self.HR_noise = True 
            self.hr_noise_types = opt['hr_noise_types']
            print("HR_noise enabled")
            print(self.hr_noise_types)
        if opt['lr_downscale']: #v variable to activate automatic downscale of HR images to LR pair, controlled by the scale of the model
            self.LR_scale = True 
            self.scale_algos = opt['lr_downscale_types']
            print("LR_scale enabled")
            print(self.scale_algos)
        if opt['lr_blur']: #v variable to activate automatic blur of LR images
            self.LR_blur = True 
            self.blur_algos = opt['lr_blur_types']
            print("LR_blur enabled")
            print(self.blur_algos)
        if opt['lr_noise']: #v variable to activate adding noise to LR image
            self.LR_noise = True 
            self.noise_types = opt['lr_noise_types']
            print("LR_noise enabled")
            print(self.noise_types)
        if opt['lr_noise2']: #v variable to activate adding a secondary noise to LR image
            self.LR_noise2 = True 
            self.noise_types2 = opt['lr_noise_types2']
            print("LR_noise 2 enabled")
            print(self.noise_types2)
        if opt['lr_cutout']: #v variable to activate random cutout 
            self.LR_cutout = True
            print("LR cutout enabled")
        if opt['lr_erasing']: #v variable to activate random erasing
            self.LR_erasing = True
            print("LR random erasing enabled")
        #v parse on the fly options     
        
        #self.random_scale_list = [1]

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.opt['scale']
        HR_size = self.opt['HR_size']
        if HR_size:
            LR_size = HR_size // scale
        
        # Check if LR Path is provided
        if self.paths_LR:
            #If LR is provided, check if 'rand_flip_LR_HR' is enabled (Only will work if HR and LR images have the same initial size) during training
            if self.opt['rand_flip_LR_HR'] and (self.LR_scale or scale == 1) and self.opt['phase'] == 'train':
                LRHRchance = random.uniform(0, 1)
                if self.opt['flip_chance']:
                    flip_chance = self.opt['flip_chance']
                else:
                    flip_chance = 0.05
                #print("Random Flip Enabled")
            # Normal case, no flipping:
            else:
                LRHRchance = 0.
                flip_chance = 0.
                #print("No Random Flip")

            # get HR and LR images
            # If enabled, random chance that LR and HR images are flipped
            # Normal case, no flipping
            # If img_LR (LR_path) doesn't exist, use img_HR (HR_path)
            if LRHRchance < (1- flip_chance):
                HR_path = self.paths_HR[index]
                LR_path = self.paths_LR[index]
                if LR_path is None:
                    LR_path = HR_path
                #print("HR kept")
            # Flipped case:
            # If img_HR (LR_path) doesn't exist, use img_HR (LR_path)
            else:
                HR_path = self.paths_LR[index]
                LR_path = self.paths_HR[index]
                if HR_path is None:
                    HR_path = LR_path
                #print("HR flipped")
            
            # Read the LR and HR images from the provided paths
            img_LR = util.read_img(self.LR_env, LR_path)
            img_HR = util.read_img(self.HR_env, HR_path)
            
            # Even if LR dataset is provided, force to generate aug_downscale % of downscales OTF from HR
            # The code will later make sure img_LR has the correct size
            if self.opt['aug_downscale']:
                aug_downscale = self.opt['aug_downscale']
                if np.random.rand() < aug_downscale:
                    img_LR = img_HR
            
        # If LR is not provided, use HR and modify on the fly
        else:
            HR_path = self.paths_HR[index]
            img_HR = util.read_img(self.HR_env, HR_path)
            img_LR = img_HR
        
        # HR modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_HR = util.modcrop(img_HR, scale)
        
        # change color space if necessary
        if self.opt['color']:
            img_HR = util.channel_convert(img_HR.shape[2], self.opt['color'], [img_HR])[0]
            img_LR = util.channel_convert(img_LR.shape[2], self.opt['color'], [img_LR])[0]
                
        #Augmentations during training
        if self.opt['phase'] == 'train':
            
            #Validate there's an img_LR, 
            if img_LR is None:
                img_LR = img_HR
                print("Image LR: ", LR_path, ("was not loaded correctly, using HR pair to downscale on the fly."))
            
            #Random Crop (reduce computing cost and adjust images to correct size first)
            if img_HR.shape[0] > HR_size or img_HR.shape[1] > HR_size:
                #Here the scale should be in respect to the images, not to the training scale (in case they are being scaled on the fly)
                if img_HR.shape[0]//img_LR.shape[0] is not img_HR.shape[1]//img_LR.shape[1]:
                    print("Warning: img_LR dimensions ratio does not match img_HR dimensions ratio for: ", HR_path)
                    img_LR = img_HR
                scaleor = img_HR.shape[0]//img_LR.shape[0]
                img_HR, img_LR = augmentations.random_crop_pairs(img_HR, img_LR, HR_size, scaleor)
            
            #or if the HR images are too small, Resize to the HR_size size and fit LR pair to LR_size too
            if img_HR.shape[0] < HR_size or img_HR.shape[1] < HR_size:
                print("Warning: Image: ", HR_path, " size does not match HR size: (", HR_size,"). The image size is: ", img_HR.shape)
                # rescale HR image to the HR_size 
                img_HR, _ = augmentations.resize_img(np.copy(img_HR), crop_size=(HR_size,HR_size), algo=cv2.INTER_LINEAR)
                # rescale LR image to the LR_size 
                img_LR, _ = augmentations.resize_img(np.copy(img_LR), crop_size=(LR_size,LR_size), algo=cv2.INTER_LINEAR)
            
            #"""
            # randomly scale LR from HR during training if LR dataset is not provided
            # Also check if LR is not at the correct scale already
            if img_LR.shape[0] is not LR_size and img_LR.shape[1] is not LR_size:
                if self.LR_scale: # if manually provided and scale algorithms are provided, then:
                    if self.scale_algos:
                        ds_algo = self.scale_algos
                    else:
                        ds_algo = 777
                else: # else, if for some reason img_LR is too large, default to matlab-like bicubic downscale
                    #if not self.opt['aug_downscale']: #only print the warning if not being forced to use HR images instead of LR dataset (which is a known case)
                    ds_algo = 777
                    print("LR image is too large, auto generating new LR for: ", LR_path)
                img_LR, scale_interpol_algo = augmentations.scale_img(img_LR, scale, algo=ds_algo)
            #"""
            
            # Rotations. 'use_flip' = 180 or 270 degrees (mirror), 'use_rot' = 90 degrees, 'HR_rrot' = random rotations +-45 degrees
            if self.opt['use_flip'] and self.opt['use_rot'] and self.HR_rrot:
                if np.random.rand() > 0.5:
                    img_LR, img_HR = util.augment([img_LR, img_HR], self.opt['use_flip'], \
                        self.opt['use_rot'])
                else:
                    if np.random.rand() > 0.5: # randomize the random rotations, so half the images are the original
                        img_HR, img_LR = augmentations.random_rotate_pairs(img_HR, img_LR, HR_size, scale)
            elif (self.opt['use_flip'] or self.opt['use_rot']) and not self.HR_rrot:
                # augmentation - flip, rotate
                img_LR, img_HR = util.augment([img_LR, img_HR], self.opt['use_flip'], \
                    self.opt['use_rot'])
            elif self.HR_rrot:
                if np.random.rand() > 0.5: # randomize the random rotations, so half the images are the original
                    img_HR, img_LR = augmentations.random_rotate_pairs(img_HR, img_LR, HR_size, scale)
            
            # Final checks
            # if the resulting HR image size so far is too large or too small, resize HR to the correct size and downscale to generate a new LR on the fly
            if img_HR.shape[0] is not HR_size or img_HR.shape[1] is not HR_size:
                print("Image: ", HR_path, " size does not match HR size: (", HR_size,"). The image size is: ", img_HR.shape)
                # rescale HR image to the HR_size 
                img_HR, _ = augmentations.resize_img(np.copy(img_HR), crop_size=(HR_size,HR_size), algo=cv2.INTER_LINEAR)
                if self.scale_algos: # if manually provided and scale algorithms are provided, then:
                    ds_algo = self.scale_algos
                else:
                    ## using matlab imresize to generate LR pair
                    ds_algo = 777
                img_LR, _ = augmentations.scale_img(img_HR, scale, algo=ds_algo)
            
            # Final checks
            # if the resulting LR so far does not have the correct dimensions, also generate a new HR- LR image pair on the fly
            if img_LR.shape[0] is not LR_size or img_LR.shape[0] is not LR_size:
                print("Image: ", LR_path, " size does not match LR size: (", HR_size//scale,"). The image size is: ", img_LR.shape)
                # rescale HR image to the HR_size 
                img_HR, _ = augmentations.resize_img(np.copy(img_HR), crop_size=(HR_size,HR_size), algo=cv2.INTER_LINEAR)
                if self.scale_algos: # if manually provided and scale algorithms are provided, then:
                    ds_algo = self.scale_algos
                else:
                    ## using matlab imresize to generate LR pair
                    ds_algo = 777
                img_LR, _ = augmentations.scale_img(img_HR, scale, algo=ds_algo)
            
            # Add noise to HR if enabled
            if self.HR_noise:
                img_HR, hr_noise_algo = augmentations.noise_img(img_HR, noise_types=self.hr_noise_types)
            
            # Below are the LR On The Fly augmentations
            #"""
            #v LR blur 
            if self.LR_blur:
                img_LR, blur_algo, blur_kernel_size = augmentations.blur_img(img_LR, blur_algos=self.blur_algos) 
            #"""
                
            #"""
            #v LR primary noise
            if self.LR_noise:
                img_LR, noise_algo = augmentations.noise_img(img_LR, noise_types=self.noise_types)
            #v LR secondary noise
            if self.LR_noise2:
                img_LR, noise_algo2 = augmentations.noise_img(img_LR, noise_types=self.noise_types2)
            #"""
                
            #"""
            #v LR cutout / LR random erasing
            if self.LR_cutout and (self.LR_erasing  != True):
                img_LR = augmentations.cutout(img_LR, img_LR.shape[0] // 2)
            elif self.LR_erasing and (self.LR_cutout  != True): #only do cutout or erasing, not both
                img_LR = augmentations.random_erasing(img_LR)
            elif self.LR_cutout and self.LR_erasing:
                if np.random.rand() > 0.5:
                    img_LR = augmentations.cutout(img_LR, img_LR.shape[0] // 2, p=0.5)
                else:
                    img_LR = augmentations.random_erasing(img_LR, p=0.5, modes=[3])                
            #"""
        
        #For testing and validation
        if self.opt['phase'] != 'train':
            #"""
            #v randomly downscale LR if enabled 
            if self.LR_scale:
                img_LR, scale_interpol_algo = augmentations.scale_img(img_LR, scale, algo=self.scale_algos)
            #"""
            
        # Debug
        # Save img_LR and img_HR images to a directory to visualize what is the result of the on the fly augmentations
        # DO NOT LEAVE ON DURING REAL TRAINING
        # self.output_sample_imgs = True
        if self.opt['phase'] == 'train':
            if self.output_sample_imgs:
                import os
                LR_dir, im_name = os.path.split(LR_path)
                #baseHRdir, _ = os.path.split(HR_dir)
                #debugpath = os.path.join(baseHRdir, os.sep, 'sampleOTFimgs')
                
                debugpath = os.path.join(os.path.split(LR_dir)[0], 'sampleOTFimgs')
                #print(debugpath)
                if not os.path.exists(debugpath):
                    os.makedirs(debugpath)
                    
                import uuid
                hex = uuid.uuid4().hex
                cv2.imwrite(debugpath+"\\"+im_name+hex+'_LR.png',img_LR*255) #random name to save + had to multiply by 255, else getting all black image
                cv2.imwrite(debugpath+"\\"+im_name+hex+'_HR.png',img_HR*255) #random name to save + had to multiply by 255, else getting all black image
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]        
        # BGRA to RGBA, HWC to CHW, numpy to tensor
        elif img_LR.shape[2] == 4:
            img_HR = img_HR[:, :, [2, 1, 0, 3]]
            img_LR = img_LR[:, :, [2, 1, 0, 3]]

        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
        
        if LR_path is None:
            LR_path = HR_path
        return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path} 

    def __len__(self):
        return len(self.paths_HR)
        
