from dataops.common import _init_lmdb, channel_convert
from data.base_dataset import BaseDataset, get_dataroots_paths, read_imgs_from_path, get_single_dataroot_path, read_split_single_dataset
from dataops.augmentations import (generate_A_fn, image_type, get_default_imethod, dim_change_fn,
                    shape_change_fn, random_downscale_B, paired_imgs_check,
                    get_unpaired_params, get_augmentations, get_totensor_params, get_totensor,
                    set_transforms, get_ds_kernels, get_noise_patches,
                    get_params, image_size, image_channels, scale_params, scale_opt, get_transform,
                    Scale, modcrop)
# from dataops.debug import tmp_vis, describe_numpy, describe_tensor


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
            opt (Option dictionary): stores all the experiment flags
        """
        super(AlignedDataset, self).__init__(opt, keys_ds=['LR','HR'])
        self.vars = self.opt.get('outputs', 'LRHR')  #'AB'
        self.ds_kernels = get_ds_kernels(self.opt)
        self.noise_patches = get_noise_patches(self.opt)
        set_transforms(self.opt.get('img_loader', 'cv2'))

        # get images paths (and optional environments for lmdb) from dataroots
        dir_AB = self.opt.get('dataroot', None) or self.opt.get('dataroot_AB', None)
        if dir_AB:
            self.AB_env = None  # environment for lmdb
            self.AB_paths = get_single_dataroot_path(self.opt, dir_AB)
            if self.opt.get('data_type') == 'lmdb':
                self.AB_env = _init_lmdb(dir_AB)
        else:
            self.A_paths, self.B_paths = get_dataroots_paths(self.opt, strict=False, keys_ds=self.keys_ds)
            self.AB_paths = None
            self.A_env, self.B_env = None, None  # environment for lmdb

            if self.opt.get('data_type') == 'lmdb':
                self.A_env = _init_lmdb(self.opt.get(f'dataroot_{self.keys_ds[0]}'))
                self.B_env = _init_lmdb(self.opt.get(f'dataroot_{self.keys_ds[1]}'))

        # get reusable totensor params
        self.totensor_params = get_totensor_params(self.opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int): a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            (or LR, HR, LR_paths and HR_paths)
            A (tensor): an image in the input domain
            B (tensor): its corresponding image in the target domain
            A_paths (str): paths A images
            B_paths (str): paths B images (can be same as A_paths if 
                using single images)
        """
        scale = self.opt.get('scale')

        ######## Read the images ########
        if self.AB_paths:
            img_A, img_B, A_path, B_path = read_split_single_dataset(
                self.opt, index, self.AB_paths, self.AB_env)
        else:
            img_A, img_B, A_path, B_path = read_imgs_from_path(
                self.opt, index, self.A_paths, self.B_paths, self.A_env, self.B_env)

        ######## Modify the images ########

        # for the validation / test phases
        if self.opt['phase'] != 'train':
            img_type = image_type(img_B)
            # B/HR modcrop
            img_B = modcrop(img_B, scale=scale, img_type=img_type)
            # modcrop and downscale A/LR if enabled
            if self.opt['lr_downscale']:
                img_A = modcrop(img_A, scale=scale, img_type=img_type)
                # TODO: 'pil' images will use default method for scaling
                img_A, _ = Scale(img_A, scale,
                    algo=self.opt.get('lr_downscale_types', 777), img_type=img_type)

        # change color space if necessary
        # TODO: move to get_transform()
        color_B = self.opt.get('color', None) or self.opt.get('color_HR', None)
        if color_B:
            img_B = channel_convert(image_channels(img_B), color_B, [img_B])[0]
        color_A = self.opt.get('color', None) or self.opt.get('color_LR', None)
        if color_A:
            img_A = channel_convert(image_channels(img_A), color_A, [img_A])[0]

        ######## Augmentations ########

        #Augmentations during training
        if self.opt['phase'] == 'train':

            default_int_method = get_default_imethod(image_type(img_A))

            # random HR downscale
            img_A, img_B = random_downscale_B(img_A=img_A, img_B=img_B,
                                opt=self.opt)

            # validate there's an img_A, if not, use img_B
            if img_A is None:
                img_A = img_B
                print(f"Image A: {A_path} was not loaded correctly, using B pair to downscale on the fly.")

            # validate proper dimensions between paired images, generate A if needed
            img_A, img_B = paired_imgs_check(
                img_A, img_B, opt=self.opt, ds_kernels=self.ds_kernels)

            # get and apply the paired transformations below
            transform_params = get_params(
                scale_opt(self.opt, scale), image_size(img_A))
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
            img_A = A_transform(img_A)
            img_B = B_transform(img_B)

            # Below are the On The Fly augmentations

            # get and apply the unpaired transformations below
            a_aug_params, b_aug_params = get_unpaired_params(self.opt)

            a_augmentations = get_augmentations(
                self.opt, 
                params=a_aug_params,
                noise_patches=self.noise_patches,
                )
            b_augmentations = get_augmentations(
                self.opt, 
                params=b_aug_params,
                noise_patches=self.noise_patches,
                )

            img_A = a_augmentations(img_A)
            img_B = b_augmentations(img_B)

        # Alternative position for changing the colorspace of A/LR.
        # color_A = self.opt.get('color', None) or self.opt.get('color_A', None)
        # if color_A:
        #     img_A = channel_convert(image_channels(img_A), color_A, [img_A])[0]

        ######## Convert images to PyTorch Tensors ########

        tensor_transform = get_totensor(
            self.opt, params=self.totensor_params, toTensor=True, grayscale=False)
        img_A = tensor_transform(img_A)
        img_B = tensor_transform(img_B)

        if A_path is None:
            A_path = B_path
        if self.vars == 'AB':
            return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}
        else:
            return {'LR': img_A, 'HR': img_B, 'LR_path': A_path, 'HR_path': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.AB_paths:
            return len(self.AB_paths)
        else:
            return len(self.B_paths)
