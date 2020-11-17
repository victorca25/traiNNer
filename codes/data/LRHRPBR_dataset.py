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
from dataops.augmentations import Scale, MLResize, RandomQuantize, KernelDownscale, NoisePatches, RandomNoisePatches, get_resize, get_blur, get_noise, get_pad
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

        # Get dataroot_HR
        self.paths_HR = opt.get('dataroot_HR', None)
        if self.paths_HR:
            self.pbr_list = os.listdir(self.paths_HR)
        
        # Get dataroot_LR
        self.paths_LR = opt.get('dataroot_LR', None)

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
        # idx_pbr = random.randint(0, len(self.pbr_list)-1)
        # pbr_dir = self.pbr_list[idx_pbr]
        pbr_dir = self.pbr_list[index]
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
            #TODO: handle uppercase names
            #ref: https://marmoset.co/posts/pbr-texture-conversion/
            if source.find('_diffuse.') >= 0 or source.find('_color.') >= 0:
                diffuse_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=3)
                if self.paths_LR:
                    cur_dir_lr = os.path.join(self.paths_LR, pbr_dir)
                    diffuse_img_lr = util.read_img(None, os.path.join(cur_dir_lr, source), out_nc=3)
                else:
                    diffuse_img_lr = diffuse_img
            elif source.find('_albedo.') >= 0:
                albedo_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=1)
            elif source.find('_ao.') >= 0 or source.find('_occlusion.') >= 0 or source.find('_ambientocclusion.') >= 0:
                ao_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=1)
            elif source.find('_height.') >= 0 or source.find('_displacement.') >= 0 or source.find('_bump.') >= 0:
                height_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=1)
            elif source.find('_metalness.') >= 0:
                metalness_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=1)
            elif source.find('_normal.') >= 0:
                normal_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=3)
            elif source.find('_reflection.') >= 0:
                reflection_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=1)
            elif source.find('_roughness.') >= 0:
                roughness_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=1)
            elif source.find('_glossiness.') >= 0 and not isinstance(roughness_img, np.ndarray):
                # glossiness_img = util.read_img(None, os.path.join(cur_dir, source), out_nc=1)
                roughness_img = 255 - util.read_img(None, os.path.join(cur_dir, source), out_nc=1)
        
        # if isinstance(diffuse_img, np.ndarray):
        #     tmp_vis(diffuse_img, False)
        # if isinstance(diffuse_img_lr, np.ndarray):
        #     tmp_vis(diffuse_img_lr, False)
        # if isinstance(albedo_img, np.ndarray):
        #     tmp_vis(albedo_img, False)
        # if isinstance(ao_img, np.ndarray):
        #     tmp_vis(ao_img, False)
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
            
            # Or if the HR images are too small, fix to the HR_size size and fit LR pair to LR_size too
            dim_change = self.opt.get('dim_change', 'pad')
            if diffuse_img.shape[0] < HR_size or diffuse_img.shape[1] < HR_size:
                if dim_change == "resize":
                    # rescale HR image to the HR_size 
                    diffuse_img = transforms.Resize((HR_size, HR_size), interpolation="BILINEAR")(np.copy(diffuse_img))
                    # rescale LR image to the LR_size (The original code discarded the diffuse_img_lr and generated a new one on the fly from diffuse_img)
                    diffuse_img_lr = transforms.Resize((LR_size, LR_size), interpolation="BILINEAR")(np.copy(diffuse_img_lr))
                elif dim_change == "pad":
                    # if diffuse_img_lr is diffuse_img, padding will be wrong, downscaling LR before padding
                    if diffuse_img_lr.shape[0] != diffuse_img.shape[0]//scale or diffuse_img_lr.shape[1] != diffuse_img.shape[1]//scale:
                        ds_algo = 777 # default to matlab-like bicubic downscale
                        if self.opt.get('lr_downscale', None): # if manually set and scale algorithms are provided, then:
                            ds_algo  = self.opt.get('lr_downscale_types', 777)
                        if self.opt.get('lr_downscale', None) and self.opt.get('dataroot_kernels', None) and 999 in self.opt["lr_downscale_types"]:
                            ds_kernel = self.ds_kernels
                        else:
                            ds_kernel = None
                        diffuse_img_lr, _ = Scale(img=diffuse_img_lr, scale=scale, algo=ds_algo, ds_kernel=ds_kernel)
                    
                    HR_pad, fill = get_pad(diffuse_img, HR_size, fill='random', padding_mode=self.opt.get('pad_mode', 'constant'))
                    diffuse_img = HR_pad(np.copy(diffuse_img))
                    
                    LR_pad, _ = get_pad(diffuse_img_lr, HR_size//scale, fill=fill, padding_mode=self.opt.get('pad_mode', 'constant'))
                    diffuse_img_lr = LR_pad(np.copy(diffuse_img_lr))
            
            # (Randomly) scale LR (from HR) during training if :
            # - LR dataset is not provided
            # - LR dataset is not in the correct scale
            # - Also to check if LR is not at the correct scale already (if img_LR was changed to img_HR)
            if diffuse_img_lr.shape[0] != LR_size or diffuse_img_lr.shape[1] != LR_size:
                ds_algo = 777 # default to matlab-like bicubic downscale
                if self.opt.get('lr_downscale', None): # if manually set and scale algorithms are provided, then:
                    ds_algo  = self.opt.get('lr_downscale_types', 777)
                else: # else, if for some reason diffuse_img_lr is too large, default to matlab-like bicubic downscale
                    #if not self.opt['aug_downscale']: #only print the warning if not being forced to use HR images instead of LR dataset (which is a known case)
                    print("LR image is too large, auto generating new LR for: ", LR_path)
                if self.opt.get('lr_downscale', None) and self.opt.get('dataroot_kernels', None) and 999 in self.opt["lr_downscale_types"]:
                    ds_kernel = self.ds_kernels #KernelDownscale(scale, self.kernel_paths, self.num_kernel)
                else:
                    ds_kernel = None
                diffuse_img_lr, _ = Scale(img=diffuse_img_lr, scale=scale, algo=ds_algo, ds_kernel=ds_kernel)

            # Random Crop (reduce computing cost and adjust images to correct size first)
            if diffuse_img.shape[0] > HR_size or diffuse_img.shape[1] > HR_size:
                #Here the scale should be in respect to the images, not to the training scale (in case they are being scaled on the fly)
                hr_crop_params, lr_crop_params = get_crop_params(diffuse_img, LR_size, scale)
                diffuse_img, _ = apply_crop_params(HR=diffuse_img, LR=None, hr_crop_params=hr_crop_params, lr_crop_params=None)
                _, diffuse_img_lr = apply_crop_params(HR=None, LR=diffuse_img_lr, hr_crop_params=None, lr_crop_params=lr_crop_params)
                albedo_img, _ = apply_crop_params(HR=albedo_img, LR=None, hr_crop_params=hr_crop_params, lr_crop_params=None)
                ao_img, _ = apply_crop_params(HR=ao_img, LR=None, hr_crop_params=hr_crop_params, lr_crop_params=None)
                height_img, _ = apply_crop_params(HR=height_img, LR=None, hr_crop_params=hr_crop_params, lr_crop_params=None)
                metalness_img, _ = apply_crop_params(HR=metalness_img, LR=None, hr_crop_params=hr_crop_params, lr_crop_params=None)
                normal_img, _ = apply_crop_params(HR=normal_img, LR=None, hr_crop_params=hr_crop_params, lr_crop_params=None)
                reflection_img, _ = apply_crop_params(HR=reflection_img, LR=None, hr_crop_params=hr_crop_params, lr_crop_params=None)
                roughness_img, _ = apply_crop_params(HR=roughness_img, LR=None, hr_crop_params=hr_crop_params, lr_crop_params=None)

            # Below are the On The Fly augmentations
            
            # Apply unsharpening mask to HR images
            if self.opt.get('hr_unsharp_mask', None):
                hr_rand_unsharp = self.opt.get('hr_rand_unsharp', 0)
                diffuse_img_lr =  transforms.FilterUnsharp(p=hr_rand_unsharp)(diffuse_img_lr)
            
            # Add blur if LR blur AND blur types are provided, else will skip
            if self.opt.get('lr_blur', None):
                blur_option = get_blur(self.opt.get('lr_blur_types', None))
                if blur_option:
                    diffuse_img_lr = blur_option(diffuse_img_lr)
            
            # LR primary noise: Add noise to LR if enabled AND noise types are provided, else will skip
            if self.opt.get('lr_noise', None):
                noise_option = get_noise(self.opt.get('lr_noise_types', None), self.noise_patches)
                if noise_option:
                    diffuse_img_lr = noise_option(diffuse_img_lr)

            # LR secondary noise: Add additional noise to LR if enabled AND noise types are provided, else will skip
            if self.opt.get('lr_noise2', None):
                noise_option = get_noise(self.opt.get('lr_noise_types2', None), self.noise_patches)
                if noise_option:
                    diffuse_img_lr = noise_option(diffuse_img_lr)

        dataset_out = {}
        if isinstance(diffuse_img, np.ndarray):
            # tmp_vis(diffuse_img, False)
            diffuse_img = util.np2tensor(diffuse_img, normalize=znorm, add_batch=False)
            # tmp_vis(diffuse_img, True)
            dataset_out['HR'] = diffuse_img
            dataset_out['HR_path'] = cur_dir
        if isinstance(diffuse_img_lr, np.ndarray):
            # tmp_vis(diffuse_img, False)
            diffuse_img_lr = util.np2tensor(diffuse_img_lr, normalize=znorm, add_batch=False)
            # tmp_vis(diffuse_img, True)
            dataset_out['LR'] = diffuse_img_lr
            dataset_out['LR_path'] = cur_dir
        if isinstance(albedo_img, np.ndarray):
            # tmp_vis(albedo_img, False)
            albedo_img = util.np2tensor(albedo_img, normalize=znorm, add_batch=False)
            # tmp_vis(ao_img, True)
            dataset_out['AL'] = albedo_img
        if isinstance(ao_img, np.ndarray):
            # tmp_vis(ao_img, False)
            ao_img = util.np2tensor(ao_img, normalize=znorm, add_batch=False)
            # tmp_vis(ao_img, True)
            dataset_out['AO'] = ao_img
        if isinstance(height_img, np.ndarray):
            # tmp_vis(height_img, False)
            height_img = util.np2tensor(height_img, normalize=znorm, add_batch=False)
            dataset_out['HE'] = height_img
        if isinstance(metalness_img, np.ndarray):
            # tmp_vis(metalness_img, False)
            metalness_img = util.np2tensor(metalness_img, normalize=znorm, add_batch=False)
            dataset_out['ME'] = metalness_img
        if isinstance(normal_img, np.ndarray):
            # tmp_vis(normal_img, False)
            normal_img = util.np2tensor(normal_img, normalize=znorm, add_batch=False)
            dataset_out['NO'] = normal_img
        if isinstance(reflection_img, np.ndarray):
            # tmp_vis(reflection_img, False)
            reflection_img = util.np2tensor(reflection_img, normalize=znorm, add_batch=False)
            dataset_out['RE'] = reflection_img
        if isinstance(roughness_img, np.ndarray):
            # tmp_vis(roughness_img, False)
            roughness_img = util.np2tensor(roughness_img, normalize=znorm, add_batch=False)
            dataset_out['RO'] = roughness_img

        return dataset_out

    def __len__(self):
        return len(self.pbr_list)


def get_crop_params(img, patch_size_lr, scale):
    h_hr, w_hr, _ = img.shape
    h_lr = h_hr // scale
    w_lr = w_hr // scale
    idx_h = random.randint(10, h_lr - patch_size_lr - 10)
    idx_w = random.randint(10, w_lr - patch_size_lr - 10)

    h_start_hr = (idx_h - 1) * scale
    h_end_hr = (idx_h - 1 + patch_size_lr) * scale
    w_start_hr = (idx_w - 1) * scale
    w_end_hr = (idx_w - 1 + patch_size_lr) * scale

    h_start_lr = idx_h - 1
    h_end_lr = idx_h - 1 + patch_size_lr
    w_start_lr = idx_w - 1
    w_end_lr = idx_w - 1 + patch_size_lr
    
    hr_crop_params = [h_start_hr, h_end_hr, w_start_hr, w_end_hr]
    lr_crop_params = [h_start_lr, h_end_lr, w_start_lr, w_end_lr]
    
    return hr_crop_params, lr_crop_params

def apply_crop_params(HR=None, LR=None, hr_crop_params=None, lr_crop_params=None):
    if isinstance(HR, np.ndarray) and hr_crop_params:
        (h_start_hr, h_end_hr, w_start_hr, w_end_hr) = hr_crop_params
        HR = HR[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    else:
        HR = None
    
    if isinstance(LR, np.ndarray) and lr_crop_params:
        (h_start_lr, h_end_lr, w_start_lr, w_end_lr) = lr_crop_params
        LR = LR[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    else:
        LR = None
        
    return HR, LR