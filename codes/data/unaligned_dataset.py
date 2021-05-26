from dataops.common import _init_lmdb, channel_convert
from data.base_dataset import BaseDataset, get_single_dataroot_path, read_single_dataset
from dataops.augmentations import (set_transforms, image_channels, image_type, image_size, 
                get_params, get_transform, get_totensor_params, get_totensor, get_default_imethod,
                scale_opt, scale_params, get_noise_patches, get_unpaired_params, get_augmentations)


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    It requires two directories with the training images from domain A
    '/path/to/data/trainA' and from domain B '/path/to/data/trainB'
    respectively.
    You can train the model with the datasets as 'dataroot_A:
    /path/to/data/trainA' and 'dataroot_B: /path/to/data/trainB'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time
    or use the SingleDataset class for the single image case.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option dictionary): stores all the experiment flags
        """
        super(UnalignedDataset, self).__init__(opt, keys_ds=['A','B'])
        self.vars = self.opt.get('outputs', 'AB')
        self.A_env, self.B_env = None, None  # environment for lmdb
        self.noise_patches = get_noise_patches(self.opt)
        self.idx_case = 'serial' if self.opt.get('serial_batches', None) else 'random'
        set_transforms(self.opt.get('img_loader', 'cv2'))

        # get images paths (and optional environments for lmdb) from dataroots
        dir_A = self.opt.get(f'dataroot_{self.keys_ds[0]}')  # create a path '/path/to/data/trainA'
        self.A_paths = get_single_dataroot_path(self.opt, dir_A)  # load images from '/path/to/data/trainA'
        dir_B = self.opt.get(f'dataroot_{self.keys_ds[1]}')  # create a path '/path/to/data/trainB'
        self.B_paths = get_single_dataroot_path(self.opt, dir_B)  # load images from '/path/to/data/trainB'

        if self.opt.get('data_type') == 'lmdb':
            self.A_env = _init_lmdb(dir_A)
            self.B_env = _init_lmdb(dir_B)

        assert self.A_paths, f'Error: image path {dir_A} empty.'
        assert self.B_paths, f'Error: image path {dir_B} empty.'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        # TODO: fix with change color space, etc
        # BtoA = self.opt.get('direction') == 'BtoA'
        # # get the number of channels of input image
        # input_nc = self.opt.get('output_nc') if BtoA else self.opt.get('input_nc')
        # # get the number of channels of output image
        # output_nc = self.opt.get('input_nc') if BtoA else self.opt.get('output_nc')

        # get reusable totensor params
        self.totensor_params = get_totensor_params(self.opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int): a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor): an image in the input domain
            B (tensor): its corresponding image in the target domain
            A_paths (str): paths A images
            B_paths (str): paths B images
        """
        scale = self.opt.get('scale')

        ######## Read the images ########
        img_A, A_path = read_single_dataset(
                opt=self.opt, index=index, paths=self.A_paths, env=self.A_env,
                idx_case='inrange', d_size=self.A_size)
        img_B, B_path = read_single_dataset(
                opt=self.opt, index=index, paths=self.B_paths, env=self.B_env,
                idx_case=self.idx_case, d_size=self.B_size)

        ######## Modify the images ########
        # change color space if necessary
        # TODO: move to get_transform()
        color_B = self.opt.get('color', None) or self.opt.get('color_B', None)
        if color_B:
            img_B = channel_convert(image_channels(img_B), color_B, [img_B])[0]
        color_A = self.opt.get('color', None) or self.opt.get('color_A', None)
        if color_A:
            img_A = channel_convert(image_channels(img_A), color_A, [img_A])[0]

        # apply image transformation
        default_int_method = get_default_imethod(image_type(img_A))
        # get first set of random params
        transform_params_A = get_params(
                scale_opt(self.opt, scale), image_size(img_A))
        # get second set of random params
        transform_params_B = get_params(
                self.opt, image_size(img_B))

        A_transform = get_transform(
                scale_opt(self.opt, scale),
                transform_params_A,
                # grayscale=(input_nc == 1),
                method=default_int_method)
        B_transform = get_transform(
                self.opt,
                transform_params_B,
                # grayscale=(output_nc == 1),
                method=default_int_method)
        img_A = A_transform(img_A)
        img_B = B_transform(img_B)

        #TODO: not needed initially, but available
        # get and apply the unpaired transformations below
        # a_aug_params, b_aug_params = get_unpaired_params(self.opt)

        # a_augmentations = get_augmentations(
        #     self.opt, 
        #     params=a_aug_params,
        #     noise_patches=self.noise_patches,
        #     )
        # b_augmentations = get_augmentations(
        #     self.opt, 
        #     params=b_aug_params,
        #     noise_patches=self.noise_patches,
        #     )

        # img_A = a_augmentations(img_A)
        # img_B = b_augmentations(img_B)

        ######## Convert images to PyTorch Tensors ########

        tensor_transform = get_totensor(
            self.opt, params=self.totensor_params, toTensor=True, grayscale=False)        
        img_A = tensor_transform(img_A)
        img_B = tensor_transform(img_B)

        if self.vars == 'AB':
            return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}
        else:
            return {'LR': img_A, 'HR': img_B, 'LR_path': A_path, 'HR_path': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of
        images, we take a maximum of the two.
        """
        return max(self.A_size, self.B_size)
