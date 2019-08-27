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
        self.HR_crop = None #v
        self.HR_rrot = None #v
        self.LR_scale = None #v
        self.LR_blur = None #v
        self.HR_noise = None #v
        self.LR_noise = None #v
        self.LR_noise2 = None #v
        self.LR_cutout = None #v
        self.LR_erasing = None #v

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
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_HR)) 
        
        #v parse on the fly options
        if opt['hr_crop']: #v variable to activate automatic crop of HR image to correct size and generate LR
            self.HR_crop = True
            print("Automatic crop of HR images enabled")
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
        
        self.random_scale_list = [1]

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.opt['scale']
        HR_size = self.opt['HR_size']
        
        #v
        if self.opt['rand_flip_LR_HR'] and self.LR_scale and self.opt['phase'] == 'train': 
            LRHRchance = random.uniform(0, 1)
            if self.opt['flip_chance']:
                flip_chance = self.opt['flip_chance']
            else:
                flip_chance = 0.05
            #print("Random Flip Enabled")
        else:
            LRHRchance = 0.
            flip_chance = 0.
            #print("No Random Flip")

        # get HR image
        if LRHRchance < (1- flip_chance):
            HR_path = self.paths_HR[index]
            #print("HR kept")
        else:
            HR_path = self.paths_LR[index]
            #print("HR flipped")
        #v
        
        img_HR = util.read_img(self.HR_env, HR_path)
        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_HR = util.modcrop(img_HR, scale)
        # change color space if necessary
        if self.opt['color']:
            img_HR = util.channel_convert(img_HR.shape[2], self.opt['color'], [img_HR])[0]
        
        #v
        if self.HR_crop and (self.HR_rrot != True):
            crop_size = (HR_size, HR_size)
            img_HR, _ = augmentations.random_resize_img(img_HR, crop_size)
        elif self.HR_rrot and (self.HR_crop != True):
            img_HR, _ = augmentations.random_rotate(img_HR)
        elif self.HR_crop and self.HR_rrot:
            if np.random.rand() > 0.5:
                crop_size = (HR_size, HR_size)
                img_HR, _ = augmentations.random_resize_img(img_HR, crop_size)
            else:
                img_HR, _ = augmentations.random_rotate(img_HR)
        #v
            
        #v
        if self.HR_noise:
            img_HR, hr_noise_algo = augmentations.noise_img(img_HR, self.hr_noise_types)
        #v

        # get LR image
        if self.paths_LR:
            if self.HR_crop or self.HR_rrot: #v
                img_LR = img_HR
            else:
                if LRHRchance < (1- flip_chance):
                    LR_path = self.paths_LR[index]
                    #print("LR kept")
                else:
                    LR_path = self.paths_HR[index]
                    #print("LR flipped")
                img_LR = util.read_img(self.LR_env, LR_path)
            
            #"""
            #v scale 
            if self.LR_scale:
                img_LR, scale_interpol_algo = augmentations.scale_img(img_LR, scale)
            #"""
            
            #"""
            #v blur 
            if self.LR_blur:
                img_LR, blur_algo, blur_kernel_size = augmentations.blur_img(img_LR, self.blur_algos) 
            #"""
            
            #"""
            #v noise
            if self.LR_noise:
                img_LR, noise_algo = augmentations.noise_img(img_LR, self.noise_types)
            if self.LR_noise2:
                img_LR, noise_algo2 = augmentations.noise_img(img_LR, self.noise_types2)
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
            img_LR = util.imresize_np(img_HR, 1 / scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_HR.shape
            if H < HR_size or W < HR_size:
                img_HR = cv2.resize(
                    np.copy(img_HR), (HR_size, HR_size), interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LR = util.imresize_np(img_HR, 1 / scale, True)
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
            #img_LR = util.channel_convert(C, self.opt['color'], [img_LR])[0] # TODO during val no definetion
            img_LR = util.channel_convert(img_LR.shape[2], self.opt['color'], [img_LR])[0] # v appears to work ok 

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
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
