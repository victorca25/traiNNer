import os.path
# import glob
import random
import numpy as np
import cv2
import torch
# import torch.utils.data as data
from torch.utils.data.dataset import Dataset
import dataops.common as util

# import dataops.augmentations as augmentations #TMP
# from dataops.augmentations import Scale, MLResize, RandomQuantize, KernelDownscale, NoisePatches, RandomNoisePatches, get_resize, get_blur, get_noise, get_pad
from dataops.augmentations import KernelDownscale, NoisePatches
from dataops.debug import tmp_vis, describe_numpy, describe_tensor

import dataops.opencv_transforms.opencv_transforms as transforms



class LRHRDataset(Dataset):
    '''
    Read PBR images.
    '''

    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.paths_LR, self.paths_HR = None, None
        self.output_sample_imgs = None

        if opt.get('dataroot_kernels', None):
            #TODO: note: use the model scale to get the right kernel 
            scale = opt.get('scale', 4)
            self.ds_kernels = KernelDownscale(scale=scale, kernel_paths=opt['dataroot_kernels'])

        if opt['phase'] == 'train' and opt.get('lr_noise_types', 3) and "patches" in opt['lr_noise_types']:
            assert opt['noise_data']
            self.noise_patches = NoisePatches(opt['noise_data'], opt.get('HR_size', 128)/opt.get('scale', 4))
        else:
            self.noise_patches = None

        # Check if dataroot_HR is a list of directories or a single directory. Note: lmdb will not currently work with a list
        self.paths_HR = opt.get('dataroot_HR', None)
        if self.paths_HR:
            self.pbr_list = os.listdir(self.paths_HR)

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.opt.get('scale', 4)
        HR_size = self.opt.get('HR_size', 128)
        if HR_size:
            LR_size = HR_size // scale
        
        # Default case: tensor will result in the [0,1] range
        # Alternative: tensor will be z-normalized to the [-1,1] range
        znorm  = self.opt.get('znorm', False)

        # get a random pbr directory
        idx_pbr = random.randint(0, len(self.pbr_list)-1)
        pbr_dir = self.pbr_list[idx_pbr]
        # print(pbr_dir)

        #TODO: TMP os problem
        import os
        cur_dir = os.path.join(self.paths_HR, pbr_dir)
        dir_content = os.listdir(cur_dir)
        # print(dir_content)

        ######## Read the images ########
        diffuse_img = None
        ao_img = None
        glossiness_img = None
        height_img = None
        metalness_img = None
        normal_img = None
        reflection_img = None
        roughness_img = None

        for source in dir_content:
            if source.find('_diffuse.') >= 0:
                diffuse_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=3)
            elif source.find('_ao.') >= 0:
                ao_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=1)
            elif source.find('_glossiness.') >= 0:
                glossiness_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=1)
            elif source.find('_height.') >= 0:
                height_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=1)
            elif source.find('_metalness.') >= 0:
                metalness_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=1)
            elif source.find('_normal.') >= 0:
                normal_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=3)
            elif source.find('_reflection.') >= 0:
                reflection_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=3)
            elif source.find('_roughness.') >= 0:
                roughness_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=1)
        
        # if isinstance(diffuse_img, np.ndarray):
        #     tmp_vis(diffuse_img, False)
        # if isinstance(ao_img, np.ndarray):
        #     tmp_vis(ao_img, False)
        # if isinstance(glossiness_img, np.ndarray):
        #     tmp_vis(glossiness_img, False)
        # if isinstance(height_img, np.ndarray):
        #     tmp_vis(height_img, False)
        # if isinstance(metalness_img, np.ndarray):
        #     tmp_vis(metalness_img, False)
        # if isinstance(normal_img, np.ndarray):
        #     tmp_vis(normal_img, False)
        # if isinstance(reflection_img, np.ndarray):
        #     tmp_vis(reflection_img, False)
        # if isinstance(roughness_img, np.ndarray):
        #     tmp_vis(roughness_img, False)
        
        ######## Modify the images ########
        
        # HR modcrop in the validation / test phase
        # if self.opt['phase'] != 'train':
        #     img_HR = util.modcrop(img_HR, scale)
        
        ######## Augmentations ########
        
        #Augmentations during training
        if self.opt['phase'] == 'train':
            
            # Random Crop (reduce computing cost and adjust images to correct size first)
            if diffuse_img.shape[0] > HR_size or diffuse_img.shape[1] > HR_size:
                #Here the scale should be in respect to the images, not to the training scale (in case they are being scaled on the fly)
                crop_params = get_crop_params(diffuse_img, HR_size, scale)

                diffuse_img = apply_crop_params(diffuse_img, crop_params)
                ao_img = apply_crop_params(ao_img, crop_params)
                glossiness_img = apply_crop_params(glossiness_img, crop_params)
                height_img = apply_crop_params(height_img, crop_params)
                metalness_img = apply_crop_params(metalness_img, crop_params)
                normal_img = apply_crop_params(normal_img, crop_params)
                reflection_img = apply_crop_params(reflection_img, crop_params)
                roughness_img= apply_crop_params(roughness_img, crop_params)

        if isinstance(diffuse_img, np.ndarray):
            # tmp_vis(diffuse_img, False)
            diffuse_img = util.np2tensor(diffuse_img, normalize=znorm, add_batch=False)
            # tmp_vis(diffuse_img, True)
        else:
            diffuse_img = []
        if isinstance(ao_img, np.ndarray):
            # tmp_vis(ao_img, False)
            ao_img = util.np2tensor(ao_img, normalize=znorm, add_batch=False)
            # tmp_vis(ao_img, True)
        else:
            ao_img = []
        if isinstance(glossiness_img, np.ndarray):
            # tmp_vis(glossiness_img, False)
            glossiness_img = util.np2tensor(glossiness_img, normalize=znorm, add_batch=False)
        else:
            glossiness_img = []
        if isinstance(height_img, np.ndarray):
            # tmp_vis(height_img, False)
            height_img = util.np2tensor(height_img, normalize=znorm, add_batch=False)
        else:
            height_img = []
        if isinstance(metalness_img, np.ndarray):
            # tmp_vis(metalness_img, False)
            metalness_img = util.np2tensor(metalness_img, normalize=znorm, add_batch=False)
        else:
            metalness_img = []
        if isinstance(normal_img, np.ndarray):
            # tmp_vis(normal_img, False)
            normal_img = util.np2tensor(normal_img, normalize=znorm, add_batch=False)
        else:
            normal_img = []
        if isinstance(reflection_img, np.ndarray):
            # tmp_vis(reflection_img, False)
            reflection_img = util.np2tensor(reflection_img, normalize=znorm, add_batch=False)
        else:
            reflection_img = []
        if isinstance(roughness_img, np.ndarray):
            # tmp_vis(roughness_img, False)
            roughness_img = util.np2tensor(roughness_img, normalize=znorm, add_batch=False)
        else:
            roughness_img = []

        # return {'LR': diffuse_img, 'HR': normal_img, 'LR_path': cur_dir, 'HR_path': cur_dir}
        return {'LR': diffuse_img, 
                'HR': normal_img, 
                'AO': ao_img, 
                'GL': glossiness_img, 
                'HE': height_img, 
                'ME': metalness_img, 
                'RE': reflection_img, 
                'RO': roughness_img, 
                'LR_path': cur_dir, 
                'HR_path': cur_dir}

    def __len__(self):
        return len(self.pbr_list)


def get_crop_params(img, patch_size, scale):
    h_hr, w_hr, _ = img.shape
    h_lr = h_hr // scale
    w_lr = w_hr // scale
    idx_h = random.randint(10, h_lr - patch_size - 10)
    idx_w = random.randint(10, w_lr - patch_size - 10)

    h_start_hr = (idx_h - 1) * scale
    h_end_hr = (idx_h - 1 + patch_size) * scale
    w_start_hr = (idx_w - 1) * scale
    w_end_hr = (idx_w - 1 + patch_size) * scale
    
    crop_params = [h_start_hr, h_end_hr, w_start_hr, w_end_hr]
    
    return crop_params

def apply_crop_params(img=None, crop_params=None):
    if isinstance(img, np.ndarray) and crop_params:
        (h_start_hr, h_end_hr, w_start_hr, w_end_hr) = crop_params
        img = img[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    else:
        img = None
        
    return img