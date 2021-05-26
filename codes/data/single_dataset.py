from dataops.common import _init_lmdb, channel_convert
from data.base_dataset import BaseDataset, get_single_dataroot_path, read_single_dataset
from dataops.augmentations import (set_transforms, image_channels, image_type, image_size,
                get_params, get_transform, get_totensor_params, get_totensor, get_default_imethod)


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by a single
    dataroot path: /path/to/data. It can be used for generating results
    in the testing phase of the models, for example the LR image or for
    CycleGAN only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option dictionary): stores all the experiment flags
        """
        super(SingleDataset, self).__init__(opt, keys_ds=['LR','HR'])
        self.vars = self.opt.get('outputs', 'LR')  #'A'
        self.A_env = None  # environment for lmdb
        set_transforms(self.opt.get('img_loader', 'cv2'))

        # get images paths (and optional environments for lmdb) from dataroots
        dir_A = self.opt.get(f'dataroot_{self.keys_ds[0]}')
        self.A_paths = get_single_dataroot_path(self.opt, dir_A)
        if self.opt.get('data_type') == 'lmdb':
            self.A_env = _init_lmdb(dir_A)

        # get reusable totensor transform
        totensor_params = get_totensor_params(self.opt)
        self.tensor_transform = get_totensor(self.opt,
                params=totensor_params, toTensor=True, grayscale=False)

        assert self.A_paths, f'Error: image path {dir_A} empty.'

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int): a random integer for data indexing
        Returns a dictionary that contains A and A_paths or
            LR and LR_path
            A (tensor): an image in the input domain
            A_paths (str): the path of the image
        """

        # get single image
        img_A, A_path = read_single_dataset(
                self.opt, index, self.A_paths, self.A_env)

        # change color space if necessary
        # TODO: move to get_transform()
        if self.opt.get('color', None):
            img_A = channel_convert(image_channels(img_A), self.opt['color'], [img_A])[0]

        # # apply transforms if any
        # default_int_method = get_default_imethod(image_type(img_A))
        # transform_params = get_params(self.opt, image_size(img_A))
        # A_transform = get_transform(
        #         self.opt,
        #         transform_params,
        #         # grayscale=(input_nc == 1),
        #         method=default_int_method)
        # img_A = A_transform(img_A)

        ######## Convert images to PyTorch Tensors ########

        img_A = self.tensor_transform(img_A)

        if self.vars == 'A':
            return {'A': img_A, 'A_path': A_path}
        else:
            return {'LR': img_A, 'LR_path': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
