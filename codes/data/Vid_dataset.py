#TODO: TMP
# import sys
# sys.path.append('../')
# from dataops.colors import ycbcr_to_rgb, yuv_to_rgb

import os #, glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data.dataset import Dataset
import dataops.common as util

from dataops.augmentations import Scale, MLResize, NoisePatches, RandomNoisePatches, get_resize, get_blur, get_noise
from dataops.debug import tmp_vis, describe_numpy, describe_tensor

import dataops.opencv_transforms.opencv_transforms as transforms



class VidTrainsetLoader(Dataset):
    def __init__(self, opt):
        super(VidTrainsetLoader).__init__()
        self.opt = opt
        self.image_channels = opt.get('image_channels', 3)
        self.num_frames  = opt.get('num_frames', 3)
        self.srcolors = opt.get('srcolors', None)
        self.otf_noise = opt.get('lr_noise', None) or opt.get('lr_blur', None)
        self.y_only = opt.get('y_only', True)
        self.shape = opt.get('tensor_shape', 'TCHW')

        assert self.num_frames % 2 == 1, (
            f'num_frame must be an odd number, but got {self.num_frames}')

        if self.opt['phase'] == 'train':
            if opt.get('dataroot_kernels', None) and 999 in opt["lr_downscale_types"]:
                self.ds_kernels = transforms.ApplyKernel(
                    scale=opt.get('scale', 4), kernels_path=opt['dataroot_kernels'], pattern='kernelgan')
            else:
                self.ds_kernels = None

            # if opt['phase'] == 'train' and opt.get('lr_noise_types', None) and "patches" in opt['lr_noise_types']:
            if opt.get('lr_noise_types', None) and "patches" in opt['lr_noise_types']:
                assert opt['noise_data']
                self.noise_patches = NoisePatches(opt['noise_data'], opt.get('HR_size', 128)/opt.get('scale', 4), grayscale=self.y_only)
            else:
                self.noise_patches = None
            
            self.n_iters = opt.get('n_iters', 200000) * opt.get('batch_size', 32) * opt.get('virtual_batch_size', 32) / opt.get('batch_size', 32)

        # Check if dataroot_HR is a list of directories or a single directory. Note: lmdb will not currently work with a list
        self.paths_HR = opt.get('dataroot_HR', None)
        if self.paths_HR:
            self.video_list = os.listdir(self.paths_HR)

        # Check if dataroot_LR is a list of directories or a single directory. Note: lmdb will not currently work with a list
        self.paths_LR = opt.get('dataroot_LR', None)
        if self.paths_LR and not self.paths_HR:
            self.video_list = os.listdir(self.paths_LR)
        

    def __getitem__(self, idx):
        scale = self.opt.get('scale', 4)
        HR_size = self.opt.get('HR_size', 128)
        LR_size = HR_size // scale
        idx_center = (self.num_frames - 1) // 2
        ds_kernel = None
        
        # Default case: tensor will result in the [0,1] range
        # Alternative: tensor will be z-normalized to the [-1,1] range
        znorm  = self.opt.get('znorm', False)

        if self.opt['phase'] == 'train':
            if self.opt.get('lr_downscale', None) and self.opt.get('dataroot_kernels', None) and 999 in self.opt["lr_downscale_types"]:
                ds_kernel = self.ds_kernels #KernelDownscale(scale, self.kernel_paths, self.num_kernel)

            # get a random video directory
            idx_video = random.randint(0, len(self.video_list)-1)
            video_dir = self.video_list[idx_video]
            # print(video_dir)
        else:
            # only one video and paths_LR/paths_HR is already the video dir
            video_dir = ""
        
        # list the frames in the directory 
        # hr_dir = self.trainset_dir + '/' + video_dir + '/hr'
        paths_HR = util.get_image_paths(self.opt['data_type'], os.path.join(self.paths_HR, video_dir))
        # print(paths_HR)

        if self.opt['phase'] == 'train':
            # random reverse augmentation
            random_reverse = self.opt.get('random_reverse', False)
            
            # skipping intermediate frames to learn from low FPS videos augmentation
            # testing random frameskip up to 'max_frameskip' frames
            max_frameskip = self.opt.get('max_frameskip', 0)
            if max_frameskip > 0:
                max_frameskip = min(max_frameskip, len(paths_HR)//(self.num_frames-1))
                frameskip = random.randint(1, max_frameskip)
            else:
                frameskip = 1
            # print("max_frameskip: ", max_frameskip)

            assert ((self.num_frames-1)*frameskip) <= (len(paths_HR)-1), (
                f'num_frame*frameskip must be smaller than the number of frames per video, check {video_dir}')
            
            # if number of frames of training video is for example 31, "max index -num_frames" = 31-3=28
            idx_frame = random.randint(0, (len(paths_HR)-1)-((self.num_frames-1)*frameskip))
            # print('frameskip:', frameskip)
        else:
            frameskip = 1
            idx_frame = idx
        
        '''
        List based frames loading
        '''
        if self.paths_LR:
            paths_LR = util.get_image_paths(self.opt['data_type'], os.path.join(self.paths_LR, video_dir))
        else:
            paths_LR = paths_HR
            ds_algo = 777 # default to matlab-like bicubic downscale
            if self.opt.get('lr_downscale', None): # if manually set and scale algorithms are provided, then:
                ds_algo  = self.opt.get('lr_downscale_types', 777)

        # get the video directory
        HR_dir, _ = os.path.split(paths_HR[idx_frame])
        LR_dir, _ = os.path.split(paths_HR[idx_frame])

        # read HR & LR frames
        HR_list = []
        LR_list = []
        resize_type = None
        LR_bicubic = None
        HR_center = None

        # print('len(paths_HR)', len(paths_HR))
        for i_frame in range(self.num_frames):
            # print('frame path:', paths_HR[int(idx_frame)+(frameskip*i_frame)])
            HR_img = util.read_img(None, paths_HR[int(idx_frame)+(frameskip*i_frame)], out_nc=self.image_channels)
            HR_img = util.modcrop(HR_img, scale)

            if self.opt['phase'] == 'train':
                '''
                If using individual image augmentations, get cropping parameters for reuse
                '''
                if self.otf_noise and i_frame == 0: #only need to calculate once, from the first frame
                    # reuse the cropping parameters for all LR and HR frames
                    hr_crop_params, lr_crop_params = get_crop_params(HR_img, LR_size, scale)
                    if self.opt.get('lr_noise', None):
                        # reuse the same noise type for all the frames
                        noise_option = get_noise(self.opt.get('lr_noise_types', None), self.noise_patches)
                    if self.opt.get('lr_blur', None):
                        # reuse the same blur type for all the frames
                        blur_option = get_blur(self.opt.get('lr_blur_types', None))

            if self.paths_LR:
                # LR images are provided at the correct scale
                LR_img = util.read_img(None, paths_LR[int(idx_frame)+(frameskip*i_frame)], out_nc=self.image_channels)
                if LR_img.shape == HR_img.shape:
                    LR_img, resize_type = Scale(img=HR_img, scale=scale, algo=ds_algo, ds_kernel=ds_kernel, resize_type=resize_type)
            else:
                # generate LR images on the fly
                LR_img, resize_type = Scale(img=HR_img, scale=scale, algo=ds_algo, ds_kernel=ds_kernel, resize_type=resize_type)

            # get the bicubic upscale of the center frame to concatenate for SR
            if self.y_only and self.srcolors and i_frame == idx_center:
                LR_bicubic, _ = Scale(img=LR_img, scale=1/scale, algo=777) # bicubic upscale
                HR_center = HR_img
                # tmp_vis(LR_bicubic, False)
                # tmp_vis(HR_center, False)
            
            if self.y_only:
                # extract Y channel from frames
                # normal path, only Y for both
                HR_img = util.bgr2ycbcr(HR_img, only_y=True)
                LR_img = util.bgr2ycbcr(LR_img, only_y=True)

            # crop patches randomly if using otf noise
            #TODO: make a BasicSR composable random_crop
            #TODO: note the original crop should go here and crop after loading each image, but could also be much simpler
            # to crop after concatenating. Check the speed difference.
            if self.otf_noise and self.opt['phase'] == 'train':
                HR_img, LR_img = apply_crop_params(HR_img, LR_img, hr_crop_params, lr_crop_params)
                if self.y_only and self.srcolors and i_frame == idx_center:
                    LR_bicubic, _ = apply_crop_params(LR_bicubic, None, hr_crop_params, None)
                    HR_center, _ = apply_crop_params(HR_center, None, hr_crop_params, None)

            # expand Y images to add the channel dimension
            # normal path, only Y for both
            if self.y_only:
                HR_img = util.fix_img_channels(HR_img, 1)
                LR_img = util.fix_img_channels(LR_img, 1)

            if self.opt['phase'] == 'train':
                # single frame augmentation (noise, blur, etc). Would only be efficient if patches are cropped in this loop
                if self.opt.get('lr_blur', None):
                    if blur_option:
                        LR_img = blur_option(LR_img)
                if self.opt.get('lr_noise', None):
                    if noise_option:
                        LR_img = noise_option(LR_img)
            
                # expand LR images to add the channel dimension again if needed (blur removes the grayscale channel)
                #TODO: add a if condition, can compare to the ndim before the augs, maybe move inside the aug condition
                # if not fullimgchannels: #TODO: TMP, this should be when using srcolors for HR or when training with 3 channels tests, separatedly
                if self.y_only:
                    LR_img = util.fix_img_channels(LR_img, 1)
            
            # print("HR_img.shape: ", HR_img.shape)
            # print("LR_img.shape", LR_img.shape)

            HR_list.append(HR_img) # h, w, c
            LR_list.append(LR_img) # h, w, c

        # print(len(HR_list))
        # print(len(LR_list))

        if self.opt['phase'] == 'train':
            # random reverse sequence augmentation
            if random_reverse and random.random() < 0.5:
                HR_list.reverse()
                LR_list.reverse()

        if not self.y_only:
            t = self.num_frames
            HR = [np.asarray(GT) for GT in HR_list]  # list -> numpy # input: list (contatin numpy: [H,W,C])
            HR = np.asarray(HR) # numpy, [T,H,W,C]
            h_HR, w_HR, c = HR_img.shape #HR_center.shape #TODO: check, may be risky
            HR = HR.transpose(1,2,3,0).reshape(h_HR, w_HR, -1) # numpy, [H',W',CT]
            LR = [np.asarray(LT) for LT in LR_list]  # list -> numpy # input: list (contatin numpy: [H,W,C])
            LR = np.asarray(LR) # numpy, [T,H,W,C]
            LR = LR.transpose(1,2,3,0).reshape(h_HR//scale, w_HR//scale, -1) # numpy, [Hl',Wl',CT]
        else:
            HR = np.concatenate((HR_list), axis=2) # h, w, t
            LR = np.concatenate((LR_list), axis=2) # h, w, t

        if self.opt['phase'] == 'train':
            '''
            # If not using individual image augmentations, this cropping should be faster, only once 
            '''
            # crop patches randomly. If not using otf noise, crop all concatenated images 
            if not self.otf_noise:
                HR, LR, hr_crop_params, _ = random_crop_mod(HR, LR, LR_size, scale)
                if self.y_only and self.srcolors:
                    LR_bicubic, _, _, _ = random_crop_mod(LR_bicubic, _, LR_size, scale, hr_crop_params)
                    HR_center, _, _, _ = random_crop_mod(HR_center, _, LR_size, scale, hr_crop_params)
                    # tmp_vis(LR_bicubic, False)
                    # tmp_vis(HR_center, False)

            # data augmentation
            #TODO: use BasicSR augmentations
            #TODO: use variables from config
            LR, HR, LR_bicubic, HR_center = augmentation()([LR, HR, LR_bicubic, HR_center])

        # tmp_vis(HR, False)
        # tmp_vis(LR, False)
        # tmp_vis(LR_bicubic, False)
        # tmp_vis(HR_center, False)

        if self.y_only:
            HR = util.np2tensor(HR, normalize=znorm, bgr2rgb=False, add_batch=False) # Tensor, [CT',H',W'] or [T, H, W]
            LR = util.np2tensor(LR, normalize=znorm, bgr2rgb=False, add_batch=False) # Tensor, [CT',H',W'] or [T, H, W]
        else:
            HR = util.np2tensor(HR, normalize=znorm, bgr2rgb=True, add_batch=False) # Tensor, [CT',H',W'] or [T, H, W]
            LR = util.np2tensor(LR, normalize=znorm, bgr2rgb=True, add_batch=False) # Tensor, [CT',H',W'] or [T, H, W]
        
        #TODO: TMP to test generating 3 channel images for SR loss
        # HR = util.np2tensor(HR, normalize=znorm, bgr2rgb=False, add_batch=True) # Tensor, [CT',H',W'] or [T, H, W]
        # LR = util.np2tensor(LR, normalize=znorm, bgr2rgb=False, add_batch=True) # Tensor, [CT',H',W'] or [T, H, W]
        
        # if self.srcolors:
        #     HR = HR.view(c,t,HR_size,HR_size) # Tensor, [C,T,H,W]
        if not self.y_only:
            HR = HR.view(c,t,HR_size,HR_size) # Tensor, [C,T,H,W]
            LR = LR.view(c,t,LR_size,LR_size) # Tensor, [C,T,H,W]
            if self.shape == 'TCHW':
                HR = HR.transpose(0,1) # Tensor, [T,C,H,W]
                LR = LR.transpose(0,1) # Tensor, [T,C,H,W]

        # generate Cr, Cb channels using bicubic interpolation
        #TODO: check, it might be easier to return the whole image and separate later when needed
        if self.y_only and self.srcolors:
            LR_bicubic = util.bgr2ycbcr(LR_bicubic, only_y=False)
            # HR_center = util.bgr2ycbcr(HR_center, only_y=False) #not needed, can directly use rgb image
            ## LR_bicubic = util.ycbcr2rgb(LR_bicubic, only_y=False) #test, looks ok
            ## HR_center = util.ycbcr2rgb(HR_center, only_y=False) #test, looks ok
            ## _, SR_cb, SR_cr = util.bgr2ycbcr(LR_bicubic, only_y=False, separate=True)
            LR_bicubic = util.np2tensor(LR_bicubic, normalize=znorm, bgr2rgb=False, add_batch=False)
            # HR_center = util.np2tensor(HR_center, normalize=znorm, bgr2rgb=False, add_batch=False) # will test using rgb image instead
            HR_center = util.np2tensor(HR_center, normalize=znorm, bgr2rgb=True, add_batch=False)
            #TODO: TMP to test generating 3 channel images for SR loss
            # LR_bicubic = util.np2tensor(LR_bicubic, normalize=znorm, bgr2rgb=False, add_batch=True)
            # HR_center = util.np2tensor(HR_center, normalize=znorm, bgr2rgb=False, add_batch=True)
        elif self.y_only and not self.srcolors:
            LR_bicubic = []
            HR_center = []
        else:
            HR_center = HR[:,idx_center,:,:] if self.shape == 'CTHW' else HR[idx_center,:,:,:]
            LR_bicubic = []

        # return toTensor(LR), toTensor(HR)
        return {'LR': LR, 'HR': HR, 'LR_path': LR_dir, 'HR_path': HR_dir, 'LR_bicubic': LR_bicubic, 'HR_center': HR_center}

    def __len__(self):
        return int(self.n_iters)
        # return len(self.paths_HR)


class VidTestsetLoader(Dataset):
    def __init__(self, opt):
        super(VidTestsetLoader).__init__()
        self.opt = opt
        self.image_channels  = opt.get('image_channels', 3)
        self.num_frames  = opt.get('num_frames', 3)
        self.srcolors = opt.get('srcolors', None)
        self.y_only = opt.get('y_only', True)
        self.shape = opt.get('tensor_shape', 'TCHW')

        assert self.num_frames % 2 == 1, (
            f'num_frame must be an odd number, but got {self.num_frames}')

        # Check if dataroot_HR is a list of directories or a single directory. Note: lmdb will not currently work with a list
        self.paths_HR = opt.get('dataroot_HR', None)
        if self.paths_HR:
            self.video_list = os.listdir(self.paths_HR)

        # Check if dataroot_LR is a list of directories or a single directory. Note: lmdb will not currently work with a list
        self.paths_LR = opt.get('dataroot_LR', None)
        if self.paths_LR and not self.paths_HR:
            self.video_list = os.listdir(self.paths_LR)


    def __getitem__(self, idx):
        scale = self.opt.get('scale', 4)
        idx_center = (self.num_frames - 1) // 2
        h_LR = None
        w_LR = None

        # Default case: tensor will result in the [0,1] range
        # Alternative: tensor will be z-normalized to the [-1,1] range
        znorm  = self.opt.get('znorm', False)

        # only one video and paths_LR/paths_HR is already the video dir
        video_dir = ""
        
        # list the frames in the directory 
        # hr_dir = self.trainset_dir + '/' + video_dir + '/hr'

        '''
        List based frames loading
        '''
        paths_LR = util.get_image_paths(self.opt['data_type'], os.path.join(self.paths_LR, video_dir))

        assert self.num_frames <= len(paths_LR), (
            f'num_frame must be smaller than the number of frames per video, check {video_dir}')

        idx_frame = idx
        LR_name = paths_LR[idx_frame + 1] # center frame
        # print(LR_name)
        # print(len(self.video_list))

        # read LR frames
        # HR_list = []
        LR_list = []
        resize_type = None
        LR_bicubic = None
        for i_frame in range(self.num_frames):
            if idx_frame == len(self.video_list)-2 and self.num_frames == 3:
                # print("second to last frame:", i_frame)
                if i_frame == 0:
                    LR_img = util.read_img(None, paths_LR[int(idx_frame)], out_nc=self.image_channels)
                else:
                    LR_img = util.read_img(None, paths_LR[int(idx_frame)+1], out_nc=self.image_channels)
            elif idx_frame == len(self.video_list)-1 and self.num_frames == 3:
                # print("last frame:", i_frame)
                LR_img = util.read_img(None, paths_LR[int(idx_frame)], out_nc=self.image_channels)
            # every other internal frame
            else:
                # print("normal frame:", idx_frame)
                LR_img = util.read_img(None, paths_LR[int(idx_frame)+(i_frame)], out_nc=self.image_channels)
            #TODO: check if this is necessary
            LR_img = util.modcrop(LR_img, scale)

            # get the bicubic upscale of the center frame to concatenate for SR
            if not self.y_only and self.srcolors and i_frame == idx_center:
                if self.opt.get('denoise_LRbic', False):
                    LR_bicubic = transforms.RandomAverageBlur(p=1, kernel_size=3)(LR_img)
                    # LR_bicubic = transforms.RandomBoxBlur(p=1, kernel_size=3)(LR_img)
                else:
                    LR_bicubic = LR_img
                LR_bicubic, _ = Scale(img=LR_bicubic, scale=1/scale, algo=777) # bicubic upscale
                # HR_center = HR_img
                # tmp_vis(LR_bicubic, False)
                # tmp_vis(HR_center, False)
            
            if self.y_only:
                # extract Y channel from frames
                # normal path, only Y for both
                LR_img = util.bgr2ycbcr(LR_img, only_y=True)

                # expand Y images to add the channel dimension
                # normal path, only Y for both
                LR_img = util.fix_img_channels(LR_img, 1)
                
                # print("HR_img.shape: ", HR_img.shape)
                # print("LR_img.shape", LR_img.shape)

            LR_list.append(LR_img) # h, w, c
            
            if not self.y_only and (not h_LR or not w_LR):
                h_LR, w_LR, c = LR_img.shape
        
        if not self.y_only:
            t = self.num_frames
            LR = [np.asarray(LT) for LT in LR_list]  # list -> numpy # input: list (contatin numpy: [H,W,C])
            LR = np.asarray(LR) # numpy, [T,H,W,C]
            LR = LR.transpose(1,2,3,0).reshape(h_LR, w_LR, -1) # numpy, [Hl',Wl',CT]
        else:
            LR = np.concatenate((LR_list), axis=2) # h, w, t

        if self.y_only:
            LR = util.np2tensor(LR, normalize=znorm, bgr2rgb=False, add_batch=False) # Tensor, [CT',H',W'] or [T, H, W]
        else:
            LR = util.np2tensor(LR, normalize=znorm, bgr2rgb=True, add_batch=False) # Tensor, [CT',H',W'] or [T, H, W]
            LR = LR.view(c,t,h_LR,w_LR) # Tensor, [C,T,H,W]
            if self.shape == 'TCHW':
                LR = LR.transpose(0,1) # Tensor, [T,C,H,W]

        if self.y_only and self.srcolors:
            # generate Cr, Cb channels using bicubic interpolation
            LR_bicubic = util.bgr2ycbcr(LR_bicubic, only_y=False)
            LR_bicubic = util.np2tensor(LR_bicubic, normalize=znorm, bgr2rgb=False, add_batch=False)
            HR_center = []
        else:
            LR_bicubic = []
            HR_center = []

        # return toTensor(LR), toTensor(HR)
        return {'LR': LR, 'LR_path': LR_name, 'LR_bicubic': LR_bicubic, 'HR_center': HR_center}

    def __len__(self):
        return len(self.video_list)-1


class augmentation(object):
    def __call__(self, img_list, hflip=True, vflip=True, rot=True):
        # horizontal flip OR rotate
        hflip = hflip and random.random() < 0.5
        vflip = vflip and random.random() < 0.5
        rot90 = rot and random.random() < 0.5
        #rot90n = rot and random.random() < 0.5

        def _augment(img):
            if isinstance(img, np.ndarray):
                if hflip: img = np.flip(img, axis=1) #img[:, ::-1, :]
                if vflip: img = np.flip(img, axis=0) #img[::-1, :, :]
                #if rot90: img = img.transpose(1, 0, 2)
                if rot90: img = np.rot90(img, 1) #90 degrees # In PIL: img.transpose(Image.ROTATE_90)
                #if rot90n: img = np.rot90(img, -1) #-90 degrees
                return img

        return [np.ascontiguousarray(_augment(img)) for img in img_list]


# TODO: this could be an initialization of a class
def get_crop_params(HR, patch_size_lr, scale):
    h_hr, w_hr, _ = HR.shape
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

# TODO: this could be the call of a class
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

def random_crop_mod(HR=None, LR=None, patch_size_lr=None, 
                scale=None, hr_crop_params=None, lr_crop_params=None):
    h_hr, w_hr, _ = HR.shape
    h_lr = h_hr // scale
    w_lr = w_hr // scale
    idx_h = random.randint(10, h_lr - patch_size_lr - 10)
    idx_w = random.randint(10, w_lr - patch_size_lr - 10)

    if hr_crop_params:
        (h_start_hr, h_end_hr, w_start_hr, w_end_hr) = hr_crop_params
    else:
        h_start_hr = (idx_h - 1) * scale
        h_end_hr = (idx_h - 1 + patch_size_lr) * scale
        w_start_hr = (idx_w - 1) * scale
        w_end_hr = (idx_w - 1 + patch_size_lr) * scale

    HR = HR[h_start_hr:h_end_hr, w_start_hr:w_end_hr, :]
    hr_crop_params = [h_start_hr, h_end_hr, w_start_hr, w_end_hr]
    
    if isinstance(LR, np.ndarray):
        if lr_crop_params:
            (h_start_lr, h_end_lr, w_start_lr, w_end_lr) = lr_crop_params
        else:
            h_start_lr = idx_h - 1
            h_end_lr = idx_h - 1 + patch_size_lr
            w_start_lr = idx_w - 1
            w_end_lr = idx_w - 1 + patch_size_lr

        LR = LR[h_start_lr:h_end_lr, w_start_lr:w_end_lr, :]
        lr_crop_params = [h_start_lr, h_end_lr, w_start_lr, w_end_lr]
    else:
        LR = None
        lr_crop_params = []
    
    return HR, LR, hr_crop_params, lr_crop_params

