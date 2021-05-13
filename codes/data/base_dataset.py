"""This module implements a 'BaseDataset' for datasets.
It also includes common dataroot validation functions and PIL transformation functions 
(e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import os
import random
import numpy as np
import torch.utils.data as data

from dataops.common import get_image_paths, read_img




class BaseDataset(data.Dataset):
    """This class is an base Dataset class for datasets.
    To create a subclass, you need to implement the following four functions:
    -- <__init__>: initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>: return the size of dataset.
    -- <__getitem__>: get a data point.
    -- <name>: returns the dataset name if defined, else returns BaseDataset
    """

    def __init__(self, opt, keys_ds=['A','B']):
        """Initialize the class; save the options in the class
        Parameters:
            opt (Options dictionary): stores all the experiment flags
            keys_ds (list): the paired 'dataroot_' properties names expected in the Dataset. ie:
                dataroot_A-dataroot_B or dataroot_LR-dataroot_HR. Will check for correct names in
                the opt dictionary and modify if needed. For single dataroot cases, will use the
                first element.
        """

        opt = check_data_keys(opt, keys_ds=keys_ds)
        self.opt = opt
        self.keys_ds = keys_ds

    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

    def name(self):
        return self.opt.get('name', 'BaseDataset')



############# Testing below

def process_img_paths(images_paths=None, data_type='img'):
    if not images_paths:
        return images_paths

    # process images_paths
    paths_list = []
    for path in images_paths:
        paths = get_image_paths(data_type, path)
        for imgs in paths:
            paths_list.append(imgs)
    paths_list = sorted(paths_list)
    return paths_list


def read_subset(dataroot, subset_file):
    with open(subset_file) as f:
        paths = sorted([os.path.join(dataroot, line.rstrip('\n')) for line in f])
    return paths


def format_paths(dataroot=None):
    """
    Check if dataroot_HR is a list of directories or a single directory. 
    Note: lmdb will not currently work with a list
    """
    # if receiving a single path in str format, convert to list
    if dataroot:
        if type(dataroot) is str:
            dataroot = os.path.join(dataroot)
            dataroot = [dataroot]
        else:
            assert isinstance(dataroot, list)
    return dataroot


def paired_dataset_validation(A_images_paths, B_images_paths, data_type='img', max_dataset_size=float("inf")):

    if isinstance(A_images_paths, str) and isinstance(B_images_paths, str):
        A_images_paths = [A_images_paths]
        B_images_paths = [B_images_paths]

    paths_A = []
    paths_B = []
    for paths in zip(A_images_paths, B_images_paths):
        A_paths = get_image_paths(data_type, paths[0], max_dataset_size)  # get image paths
        B_paths = get_image_paths(data_type, paths[1], max_dataset_size)  # get image paths
        for imgs in zip(A_paths, B_paths):
            _, A_filename = os.path.split(imgs[0])
            _, B_filename = os.path.split(imgs[1])
            assert A_filename == B_filename, 'Wrong pair of images {} and {}'.format(A_filename, B_filename)
            paths_A.append(imgs[0])
            paths_B.append(imgs[1])
    return paths_A, paths_B


def check_data_keys(opt, keys_ds=['LR', 'HR']):
    keys_A = ['LR', 'A', 'lq']
    keys_B = ['HR', 'B', 'gt']
    
    root_A = 'dataroot_' + keys_ds[0]
    root_B = 'dataroot_' + keys_ds[1]
    for pair_keys in zip(keys_A, keys_B):
        A_el = 'dataroot_' + pair_keys[0]
        B_el = 'dataroot_' + pair_keys[1]
        if opt.get(B_el, None) and B_el != root_B:
            opt[root_B] = opt[B_el]
            opt.pop(B_el)
        if opt.get(A_el, None) and A_el != root_A:
            opt[root_A] = opt[A_el]
            opt.pop(A_el)
    
    return opt



def read_dataroots(opt, keys_ds=['LR','HR']):
    """ Read the dataroots from the options dictionary
    Parameters:
        opt (Options dictionary): stores all the experiment flags
        keys_ds (list): the paired 'dataroot_' properties names expected in the Dataset.
            Note that `LR` dataset corresponds to `A` or `lq` domain, while `HR` 
            corresponds to `B` or `gt`
    """
    paths_A, paths_B = None, None
    root_A = 'dataroot_' + keys_ds[0]
    root_B = 'dataroot_' + keys_ds[1]

    # read image list from subset list txt
    if opt['subset_file'] is not None and opt['phase'] == 'train':
        paths_B = read_subset(opt[root_B], opt['subset_file'])
        if opt[root_A] is not None and opt.get('subset_file_'+keys_ds[0], None):
            paths_A = read_subset(opt[root_A], opt['subset_file'])
        else:
            print('Using subset will generate {}s on-the-fly.').format(keys_ds[0])
    else:  # read image list from lmdb or image files
        A_images_paths = format_paths(opt[root_A])
        B_images_paths = format_paths(opt[root_B])
        
        # special case when dealing with duplicate B_images_paths or A_images_paths
        # lmdb not be supported with this option
        if len(B_images_paths) != len(set(B_images_paths)) or \
            A_images_paths and (len(A_images_paths) != len(set(A_images_paths))):

            # only resolve when the two path lists coincide in the number of elements, 
            # they have to be ordered specifically as they will be used in the options file
            assert len(B_images_paths) == len(A_images_paths), \
                'Error: When using duplicate paths, {} and {} must contain the same number of elements.'.format(
                    root_B, root_A)

            paths_A, paths_B = paired_dataset_validation(A_images_paths, B_images_paths, 
                                        opt['data_type'], opt.get('max_dataset_size', float("inf")))
        else: # for cases with extra HR directories for OTF images or original single directories
            paths_A = process_img_paths(A_images_paths, opt['data_type'])
            paths_B = process_img_paths(B_images_paths, opt['data_type'])

    return paths_A, paths_B


def validate_paths(paths_A, paths_B, strict=True, keys_ds=['LR','HR']):
    """ Validate the constructed images path lists are consistent. 
    Can allow using B/HR and A/LR folders with different amount of images
    Parameters:
        paths_A (str): the path to domain A
        paths_B (str): the path to domain B
        keys_ds (list): the paired 'dataroot_' properties names expected in the Dataset.
        strict (bool): If strict = True, will make sure both lists only contains images
            if properly paired in the other dataset, otherwise will fill missing images 
            paths in LR/A with 'None' to be taken care of later (ie. with on-the-fly 
            generation)
    Examples of OTF usage:
    - If an LR image pair is not found, downscale HR on the fly, else, use the LR
    - If all LR are provided and 'lr_downscale' is enabled, randomize use of provided 
        LR and OTF LR for augmentation
    """
    
    if not strict:
        assert len(paths_B) >= len(paths_A), \
            '{} dataset contains less images than {} dataset  - {}, {}.'.format(\
            keys_ds[1], keys_ds[0], len(paths_B), len(paths_A))
        if len(paths_A) < len(paths_B):
            print('{} contains less images than {} dataset  - {}, {}. Will generate missing images on the fly.'.format(
                keys_ds[0], keys_ds[1], len(paths_A), len(paths_B)))

    i=0
    tmp_A = []
    tmp_B = []
    for idx in range(0, len(paths_B)):
        B_head, B_tail = os.path.split(paths_B[idx])
        if i < len(paths_A):
            A_head, A_tail = os.path.split(paths_A[i])
            
            if A_tail == B_tail:
                A_img_path = os.path.join(A_head, A_tail)
                tmp_A.append(A_img_path)
                i+=1
                if strict:
                    B_img_path = os.path.join(B_head, B_tail)
                    tmp_B.append(B_img_path)
            else:
                if not strict:
                    A_img_path = None
                    tmp_A.append(A_img_path)
        else: #if the last image is missing
            if not strict:
                A_img_path = None
                tmp_A.append(A_img_path)
    paths_A = tmp_A
    paths_B = tmp_B if strict else paths_B

    assert len(paths_A) == len(paths_B)
    return paths_A, paths_B


def get_dataroots_paths(opt, strict=False, keys_ds=['LR', 'HR']):
    paths_A, paths_B = read_dataroots(opt, keys_ds=keys_ds)
    assert paths_B, 'Error: {} path is empty.'.format(keys_ds[1])
    if strict:
        assert paths_A, 'Error: {} path is empty.'.format(keys_ds[0])

    if paths_A and paths_B:
        paths_A, paths_B = validate_paths(paths_A, paths_B, strict=strict, keys_ds=keys_ds)
    return paths_A, paths_B





def read_imgs_from_path(opt, index, paths_A, paths_B, A_env, B_env):
    #TODO: check cases where default of 3 channels will be troublesome
    image_channels  = opt.get('image_channels', 3)
    input_nc = opt.get('input_nc', image_channels)
    output_nc = opt.get('output_nc', image_channels)
    data_type = opt.get('data_type', 'img')
    loader = opt.get('data_loader', 'cv2')

    # Check if A/LR Path is provided
    if paths_A:
        if (opt.get('rand_flip_LR_HR', None) or opt.get('rand_flip_A_B', None)) and opt['phase'] == 'train':
            # If A/LR is provided, check if 'rand_flip_LR_HR' or 'rand_flip_A_B' is enabled
            randomchance = random.uniform(0, 1)
            flip_chance  = opt.get('flip_chance', 0.05)
            # print("Random Flip Enabled")
        elif opt.get('direction', None) == 'BtoA':
            # if flipping domain translation direction
            randomchance = 1.
            flip_chance = 1.
        else:
            # Normal case, no flipping:
            randomchance = 0.
            flip_chance = 0.
            # print("No Random Flip")

        # get B/HR and A/LR images pairs
        # If enabled, random chance that A/LR and B/HR images domains are flipped
        if randomchance < (1- flip_chance):
            # Normal case, no flipping
            # If img_A (A_path) doesn't exist, use img_B (B_path)
            B_path = paths_B[index]
            A_path = paths_A[index]
            if A_path is None:
                A_path = B_path
            # print("HR kept")
        else:
            # Flipped case:
            # If img_B (A_path) doesn't exist, use img_B (A_path)
            t = output_nc
            output_nc = input_nc
            input_nc = t
            B_path = paths_A[index]
            A_path = paths_B[index]
            if B_path is None:
                B_path = A_path
            # print("HR flipped")

        # Read the A/LR and B/HR images from the provided paths
        img_A = read_img(env=data_type, path=A_path, lmdb_env=A_env, out_nc=input_nc, loader=loader)
        img_B = read_img(env=data_type, path=B_path, lmdb_env=B_env, out_nc=output_nc, loader=loader)
        
        # Even if A/LR dataset is provided, force to generate aug_downscale % of downscales OTF from B/HR
        # The code will later make sure img_A has the correct size
        if opt.get('aug_downscale', None):
            # if np.random.rand() < opt['aug_downscale']:
            if random.random() < opt['aug_downscale']:
                img_A = img_B
        
    # If A/LR is not provided, use B/HR and modify on the fly
    else:
        B_path = paths_B[index]
        img_B = read_img(env=data_type, path=B_path, lmdb_env=B_env, out_nc=output_nc, loader=loader)
        img_A = img_B
        A_path = B_path

    return img_A, img_B, A_path, B_path
