import os
import random

import numpy as np
from torch.utils.data.dataset import Dataset

import codes.dataops.opencv_transforms.opencv_transforms as transforms
from codes.dataops.augmentations import Scale, KernelDownscale, NoisePatches, get_blur, get_noise
from codes.dataops.common import get_image_paths, read_img, modcrop, bgr2ycbcr, fix_img_channels, np2tensor


class VidTrainsetLoader(Dataset):
    def __init__(self, opt):
        super(VidTrainsetLoader).__init__()
        self.opt = opt
        self.image_channels = opt.get('image_channels', 3)
        self.num_frames = opt.get('num_frames', 3)
        self.srcolors = opt.get('srcolors', None)
        self.otf_noise = opt.get('lr_noise', None) or opt.get('lr_blur', None)
        self.y_only = opt.get('y_only', True)
        self.shape = opt.get('tensor_shape', 'TCHW')

        if self.num_frames % 2 != 1:
            raise ValueError('num_frame must be an odd number, but got %d' % self.num_frames)

        if self.opt['phase'] == 'train':
            if opt.get('dataroot_kernels', None):
                scale = opt.get('scale', 4)
                self.ds_kernels = KernelDownscale(scale=scale, kernel_paths=opt['dataroot_kernels'])

            self.noise_patches = None
            if "patches" in opt.get('lr_noise_types', ''):
                if not opt.get('noise_data', None):
                    raise ValueError('noise_data is required')
                self.noise_patches = NoisePatches(
                    opt['noise_data'],
                    opt.get('HR_size', 128) / opt.get('scale', 4),
                    grayscale=self.y_only
                )

            self.n_iters = opt.get('n_iters', 200000)
            self.n_iters = self.n_iters * opt.get('batch_size', 32)
            self.n_iters = self.n_iters * opt.get('virtual_batch_size', 32) / opt.get('batch_size', 32)

        # TODO: Check if dataroot_HR is a list of directories or a single directory.
        #       Note: lmdb will not currently work with a list
        self.paths_HR = opt.get('dataroot_HR', None)
        if self.paths_HR:
            self.video_list = os.listdir(self.paths_HR)

        # TODO: Check if dataroot_LR is a list of directories or a single directory.
        #       Note: lmdb will not currently work with a list
        self.paths_LR = opt.get('dataroot_LR', None)
        if self.paths_LR and not self.paths_HR:
            self.video_list = os.listdir(self.paths_LR)

    def __getitem__(self, idx):
        scale = self.opt.get('scale', 4)
        hr_size = self.opt.get('HR_size', 128)
        lr_size = hr_size // scale
        idx_center = (self.num_frames - 1) // 2
        ds_kernel = None

        # Default case: tensor will result in the [0,1] range
        # Alternative: tensor will be z-normalized to the [-1,1] range
        znorm = self.opt.get('znorm', False)

        if self.opt['phase'] == 'train':
            if self.opt.get('lr_downscale', None) and self.opt.get('dataroot_kernels', None) and \
                    999 in self.opt["lr_downscale_types"]:
                ds_kernel = self.ds_kernels  # KernelDownscale(scale, self.kernel_paths, self.num_kernel)

            # get a random video directory
            idx_video = random.randint(0, len(self.video_list) - 1)
            video_dir = self.video_list[idx_video]
        else:
            # only one video and paths_LR/paths_HR is already the video dir
            video_dir = ""

        # list the frames in the directory
        _, paths_hr = get_image_paths(self.opt['data_type'], os.path.join(self.paths_HR, video_dir))

        if self.opt['phase'] == 'train':
            # random reverse augmentation
            random_reverse = self.opt.get('random_reverse', False)

            # skipping intermediate frames to learn from low FPS videos augmentation
            # testing random frameskip up to 'max_frameskip' frames
            max_frameskip = self.opt.get('max_frameskip', 0)
            if max_frameskip > 1:
                frameskip = random.randint(1, min(max_frameskip, len(paths_hr) // (self.num_frames - 1)))
            else:
                frameskip = 1

            if ((self.num_frames - 1) * frameskip) > (len(paths_hr) - 1):
                raise ValueError(
                    'num_frame * frameskip must be smaller than the number of frames per video, check %s' %
                    video_dir
                )

            # if number of frames of training video is for example 31, "max index -num_frames" = 31-3=28
            idx_frame = random.randint(0, (len(paths_hr) - 1) - ((self.num_frames - 1) * frameskip))
        else:
            frameskip = 1
            idx_frame = idx

        # List based frames loading
        if self.paths_LR:
            _, paths_lr = get_image_paths(self.opt['data_type'], os.path.join(self.paths_LR, video_dir))
        else:
            paths_lr = paths_hr
            ds_algo = 777  # default to matlab-like bicubic downscale
            if self.opt.get('lr_downscale', None):  # if manually set and scale algorithms are provided, then:
                ds_algo = self.opt.get('lr_downscale_types', 777)

        # get the video directory
        hr_dir, _ = os.path.split(paths_hr[idx_frame])
        lr_dir, _ = os.path.split(paths_hr[idx_frame])

        # read HR & LR frames
        hr_list = []
        lr_list = []
        resize_type = None
        lr_bicubic = None
        hr_center = None

        for i_frame in range(self.num_frames):
            hr_img = read_img(None, paths_hr[int(idx_frame) + (frameskip * i_frame)], out_nc=self.image_channels)
            hr_img = modcrop(hr_img, scale)

            if self.opt['phase'] == 'train':
                # If using individual image augmentations, get cropping parameters for reuse
                if self.otf_noise and i_frame == 0:  # only need to calculate once, from the first frame
                    # reuse the cropping parameters for all LR and HR frames
                    hr_crop_params, lr_crop_params = get_crop_params(hr_img, lr_size, scale)
                    if self.opt.get('lr_noise', None):
                        # reuse the same noise type for all the frames
                        noise_option = get_noise(self.opt.get('lr_noise_types', None), self.noise_patches)
                    if self.opt.get('lr_blur', None):
                        # reuse the same blur type for all the frames
                        blur_option = get_blur(self.opt.get('lr_blur_types', None))

            if self.paths_LR:
                # LR images are provided at the correct scale
                lr_img = read_img(None, paths_lr[int(idx_frame) + (frameskip * i_frame)], out_nc=self.image_channels)
                if lr_img.shape == hr_img.shape:
                    lr_img, resize_type = Scale(
                        img=hr_img,
                        scale=scale,
                        algo=ds_algo,
                        ds_kernel=ds_kernel,
                        resize_type=resize_type)
            else:
                # generate LR images on the fly
                lr_img, resize_type = Scale(
                    img=hr_img,
                    scale=scale,
                    algo=ds_algo,
                    ds_kernel=ds_kernel,
                    resize_type=resize_type)

            # get the bicubic upscale of the center frame to concatenate for SR
            if self.y_only and self.srcolors and i_frame == idx_center:
                lr_bicubic, _ = Scale(img=lr_img, scale=1 / scale, algo=777)  # bicubic upscale
                hr_center = hr_img

            if self.y_only:
                # extract Y channel from frames
                # normal path, only Y for both
                hr_img = bgr2ycbcr(hr_img, only_y=True)
                lr_img = bgr2ycbcr(lr_img, only_y=True)

            # crop patches randomly if using otf noise
            # TODO: make a BasicSR composable random_crop
            #       the original crop should go here and crop after loading each image, but could also be much simpler
            # to crop after concatenating. Check the speed difference.
            if self.otf_noise and self.opt['phase'] == 'train':
                hr_img, lr_img = apply_crop_params(hr_img, lr_img, hr_crop_params, lr_crop_params)
                if self.y_only and self.srcolors and i_frame == idx_center:
                    lr_bicubic, _ = apply_crop_params(lr_bicubic, None, hr_crop_params, None)
                    hr_center, _ = apply_crop_params(hr_center, None, hr_crop_params, None)

            # expand Y images to add the channel dimension
            # normal path, only Y for both
            if self.y_only:
                hr_img = fix_img_channels(hr_img, 1)
                lr_img = fix_img_channels(lr_img, 1)

            if self.opt['phase'] == 'train':
                # single frame augmentation (noise, blur, etc). Only efficient if patches are cropped in this loop
                if self.opt.get('lr_blur', None):
                    if blur_option:
                        lr_img = blur_option(lr_img)
                if self.opt.get('lr_noise', None):
                    if noise_option:
                        lr_img = noise_option(lr_img)

                # expand LR images to add the channel dimension again if needed (blur removes the grayscale channel)
                # TODO: add a if condition, can compare to the ndim before the augs, maybe move inside the aug condition
                # if not fullimgchannels:
                #   TODO: TMP, this should be when using srcolors for HR or when training with 3 channels tests,
                #         separately
                if self.y_only:
                    lr_img = fix_img_channels(lr_img, 1)

            hr_list.append(hr_img)  # h, w, c
            lr_list.append(lr_img)  # h, w, c

        if self.opt['phase'] == 'train':
            # random reverse sequence augmentation
            if random_reverse and random.random() < 0.5:
                hr_list.reverse()
                lr_list.reverse()

        if not self.y_only:
            t = self.num_frames
            hr = [np.asarray(GT) for GT in hr_list]  # list -> numpy # input: list (contatin numpy: [H,W,C])
            hr = np.asarray(hr)  # numpy, [T,H,W,C]
            h_hr, w_hr, c = hr_img.shape  # HR_center.shape # TODO: check, may be risky
            # TODO: Unexpected argument (-1)
            hr = hr.transpose((1, 2, 3, 0)).reshape(h_hr, w_hr, -1)  # numpy, [H',W',CT]
            lr = [np.asarray(x) for x in lr_list]  # list -> numpy # input: list (contain numpy: [H,W,C])
            lr = np.asarray(lr)  # numpy, [T,H,W,C]
            lr = lr.transpose((1, 2, 3, 0)).reshape(h_hr // scale, w_hr // scale, -1)  # numpy, [Hl',Wl',CT]
        else:
            hr = np.concatenate(hr_list, axis=2)  # h, w, t
            lr = np.concatenate(lr_list, axis=2)  # h, w, t

        if self.opt['phase'] == 'train':
            # If not using individual image augmentations, this cropping should be faster, only once
            # crop patches randomly. If not using otf noise, crop all concatenated images
            if not self.otf_noise:
                hr, lr, hr_crop_params, _ = random_crop_mod(hr, lr, lr_size, scale)
                if self.y_only and self.srcolors:
                    lr_bicubic, _, _, _ = random_crop_mod(lr_bicubic, _, lr_size, scale, hr_crop_params)
                    hr_center, _, _, _ = random_crop_mod(hr_center, _, lr_size, scale, hr_crop_params)

            # data augmentation
            # TODO: use BasicSR augmentations
            # TODO: use variables from config
            lr, hr, lr_bicubic, hr_center = augmentation()([lr, hr, lr_bicubic, hr_center])

        if self.y_only:
            hr = np2tensor(hr, normalize=znorm, bgr2rgb=False, add_batch=False)  # Tensor, [CT',H',W'] or [T, H, W]
            lr = np2tensor(lr, normalize=znorm, bgr2rgb=False, add_batch=False)  # Tensor, [CT',H',W'] or [T, H, W]
        else:
            hr = np2tensor(hr, normalize=znorm, bgr2rgb=True, add_batch=False)  # Tensor, [CT',H',W'] or [T, H, W]
            lr = np2tensor(lr, normalize=znorm, bgr2rgb=True, add_batch=False)  # Tensor, [CT',H',W'] or [T, H, W]

        # TODO: TMP to test generating 3 channel images for SR loss
        # hr = np2tensor(hr, normalize=znorm, bgr2rgb=False, add_batch=True) # Tensor, [CT',H',W'] or [T, H, W]
        # lr = np2tensor(lr, normalize=znorm, bgr2rgb=False, add_batch=True) # Tensor, [CT',H',W'] or [T, H, W]

        # if self.srcolors:
        #     hr = hr.view(c, t, hr_size, hr_size) # Tensor, [C,T,H,W]
        if not self.y_only:
            hr = hr.view(c, t, hr_size, hr_size)  # Tensor, [C,T,H,W]
            lr = lr.view(c, t, lr_size, lr_size)  # Tensor, [C,T,H,W]
            if self.shape == 'TCHW':
                hr = hr.transpose(0, 1)  # Tensor, [T,C,H,W]
                lr = lr.transpose(0, 1)  # Tensor, [T,C,H,W]

        # generate Cr, Cb channels using bicubic interpolation
        # TODO: check, it might be easier to return the whole image and separate later when needed
        if self.y_only and self.srcolors:
            lr_bicubic = bgr2ycbcr(lr_bicubic, only_y=False)
            # hr_center = bgr2ycbcr(hr_center, only_y=False)  # not needed, can directly use rgb image
            # lr_bicubic = ycbcr2rgb(lr_bicubic, only_y=False)  # test, looks ok
            # hr_center = ycbcr2rgb(hr_center, only_y=False)  # test, looks ok
            # _, sr_cb, sr_cr = bgr2ycbcr(lr_bicubic, only_y=False, separate=True)
            lr_bicubic = np2tensor(lr_bicubic, normalize=znorm, bgr2rgb=False, add_batch=False)
            hr_center = np2tensor(hr_center, normalize=znorm, bgr2rgb=True, add_batch=False)
            # TODO: TMP to test generating 3 channel images for SR loss
            # lr_bicubic = np2tensor(lr_bicubic, normalize=znorm, bgr2rgb=False, add_batch=True)
            # hr_center = np2tensor(hr_center, normalize=znorm, bgr2rgb=False, add_batch=True)
        elif self.y_only and not self.srcolors:
            lr_bicubic = []
            hr_center = []
        else:
            hr_center = hr[:, idx_center, :, :] if self.shape == 'CTHW' else hr[idx_center, :, :, :]
            lr_bicubic = []

        # return toTensor(LR), toTensor(HR)
        return {
            'LR': lr,
            'HR': hr,
            'LR_path': lr_dir,
            'HR_path': hr_dir,
            'LR_bicubic': lr_bicubic,
            'HR_center': hr_center
        }

    def __len__(self):
        return int(self.n_iters)
        # return len(self.paths_HR)


class VidTestsetLoader(Dataset):
    def __init__(self, opt):
        super(VidTestsetLoader).__init__()
        self.opt = opt
        self.image_channels = opt.get('image_channels', 3)
        self.num_frames = opt.get('num_frames', 3)
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
        znorm = self.opt.get('znorm', False)

        # only one video and paths_LR/paths_HR is already the video dir
        video_dir = ""

        # list the frames in the directory 
        # hr_dir = self.trainset_dir + '/' + video_dir + '/hr'

        '''
        List based frames loading
        '''
        _, paths_LR = get_image_paths(self.opt['data_type'], os.path.join(self.paths_LR, video_dir))

        assert self.num_frames <= len(paths_LR), (
            f'num_frame must be smaller than the number of frames per video, check {video_dir}')

        idx_frame = idx
        LR_name = paths_LR[idx_frame + 1]  # center frame
        # print(LR_name)
        # print(len(self.video_list))

        # read LR frames
        # HR_list = []
        LR_list = []
        resize_type = None
        LR_bicubic = None
        for i_frame in range(self.num_frames):
            if idx_frame == len(self.video_list) - 2 and self.num_frames == 3:
                # print("second to last frame:", i_frame)
                if i_frame == 0:
                    LR_img = read_img(None, paths_LR[int(idx_frame)], out_nc=self.image_channels)
                else:
                    LR_img = read_img(None, paths_LR[int(idx_frame) + 1], out_nc=self.image_channels)
            elif idx_frame == len(self.video_list) - 1 and self.num_frames == 3:
                # print("last frame:", i_frame)
                LR_img = read_img(None, paths_LR[int(idx_frame)], out_nc=self.image_channels)
            # every other internal frame
            else:
                # print("normal frame:", idx_frame)
                LR_img = read_img(None, paths_LR[int(idx_frame) + (i_frame)], out_nc=self.image_channels)
            # TODO: check if this is necessary
            LR_img = modcrop(LR_img, scale)

            # get the bicubic upscale of the center frame to concatenate for SR
            if not self.y_only and self.srcolors and i_frame == idx_center:
                if self.opt.get('denoise_LRbic', False):
                    LR_bicubic = transforms.RandomAverageBlur(p=1, kernel_size=3)(LR_img)
                    # LR_bicubic = transforms.RandomBoxBlur(p=1, kernel_size=3)(LR_img)
                else:
                    LR_bicubic = LR_img
                LR_bicubic, _ = Scale(img=LR_bicubic, scale=1 / scale, algo=777)  # bicubic upscale
                # HR_center = HR_img
                # tmp_vis(LR_bicubic, False)
                # tmp_vis(HR_center, False)

            if self.y_only:
                # extract Y channel from frames
                # normal path, only Y for both
                LR_img = bgr2ycbcr(LR_img, only_y=True)

                # expand Y images to add the channel dimension
                # normal path, only Y for both
                LR_img = fix_img_channels(LR_img, 1)

                # print("HR_img.shape: ", HR_img.shape)
                # print("LR_img.shape", LR_img.shape)

            LR_list.append(LR_img)  # h, w, c

            if not self.y_only and (not h_LR or not w_LR):
                h_LR, w_LR, c = LR_img.shape

        if not self.y_only:
            t = self.num_frames
            LR = [np.asarray(LT) for LT in LR_list]  # list -> numpy # input: list (contatin numpy: [H,W,C])
            LR = np.asarray(LR)  # numpy, [T,H,W,C]
            LR = LR.transpose(1, 2, 3, 0).reshape(h_LR, w_LR, -1)  # numpy, [Hl',Wl',CT]
        else:
            LR = np.concatenate((LR_list), axis=2)  # h, w, t

        if self.y_only:
            LR = np2tensor(LR, normalize=znorm, bgr2rgb=False, add_batch=False)  # Tensor, [CT',H',W'] or [T, H, W]
        else:
            LR = np2tensor(LR, normalize=znorm, bgr2rgb=True, add_batch=False)  # Tensor, [CT',H',W'] or [T, H, W]
            LR = LR.view(c, t, h_LR, w_LR)  # Tensor, [C,T,H,W]
            if self.shape == 'TCHW':
                LR = LR.transpose(0, 1)  # Tensor, [T,C,H,W]

        if self.y_only and self.srcolors:
            # generate Cr, Cb channels using bicubic interpolation
            LR_bicubic = bgr2ycbcr(LR_bicubic, only_y=False)
            LR_bicubic = np2tensor(LR_bicubic, normalize=znorm, bgr2rgb=False, add_batch=False)
            HR_center = []
        else:
            LR_bicubic = []
            HR_center = []

        # return toTensor(LR), toTensor(HR)
        return {'LR': LR, 'LR_path': LR_name, 'LR_bicubic': LR_bicubic, 'HR_center': HR_center}

    def __len__(self):
        return len(self.video_list) - 1


class augmentation():
    def __call__(self, img_list, hflip=True, vflip=True, rot=True):
        # horizontal flip OR rotate
        hflip = hflip and random.random() < 0.5
        vflip = vflip and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        # rot90n = rot and random.random() < 0.5

        def _augment(img):
            if isinstance(img, np.ndarray):
                if hflip: img = np.flip(img, axis=1)  # img[:, ::-1, :]
                if vflip: img = np.flip(img, axis=0)  # img[::-1, :, :]
                # if rot90: img = img.transpose(1, 0, 2)
                if rot90: img = np.rot90(img, 1)  # 90 degrees # In PIL: img.transpose(Image.ROTATE_90)
                # if rot90n: img = np.rot90(img, -1) #-90 degrees
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
