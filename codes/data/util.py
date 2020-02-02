import logging
import math
import os
import pickle
import random

import cv2
import numpy as np
import torch

####################
# Files & IO
####################

###################### get image path list ######################
IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".dng",
    ".DNG",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    """get image path list from image folder"""
    if isinstance(path, str):
        # add support for both list and str by just converting str to a list
        path = [path]
    images = []
    for p in path:
        if not os.path.isdir(p):
            raise ValueError(f"Error: {p:s} is not a valid directory")
        for dirpath, _, fnames in sorted(os.walk(p)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    img_path = os.path.join(dirpath, fname)
                    images.append(img_path)
    if not images:
        raise ValueError(f"{p:s} has no valid image file")
    return images


def _get_paths_from_lmdb(dataroot):
    """get image path list from lmdb"""
    import lmdb

    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    keys_cache_file = os.path.join(dataroot, "_keys_cache.p")
    logger = logging.getLogger("base")
    if os.path.isfile(keys_cache_file):
        logger.info("Read lmdb keys from cache: {}".format(keys_cache_file))
        keys = pickle.load(open(keys_cache_file, "rb"))
    else:
        with env.begin(write=False) as txn:
            logger.info("Creating lmdb keys cache: {}".format(keys_cache_file))
            keys = [key.decode("ascii") for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, "wb"))
    paths = sorted([key for key in keys if not key.endswith(".meta")])
    return env, paths


def get_image_paths(data_type, dataroot):
    """get image path list
    support lmdb or image files"""
    env, paths = None, None
    if dataroot is not None:
        if data_type == "lmdb":
            env, paths = _get_paths_from_lmdb(dataroot)
        elif data_type == "img":
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError(
                "data_type [{:s}] is not recognized.".format(data_type)
            )
    return env, paths


###################### read images ######################
def _read_lmdb_img(env, path):
    with env.begin(write=False) as txn:
        buf = txn.get(path.encode("ascii"))
        buf_meta = txn.get((path + ".meta").encode("ascii")).decode("ascii")
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    H, W, C = [int(s) for s in buf_meta.split(",")]
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path, out_nc=3, znorm=False):
    # read image by cv2 (rawpy if dng) or from lmdb
    # (alt: using PIL instead of cv2)
    # out_nc: Desired number of channels
    # return: Numpy float32, HWC, BGR, [0,1] by default
    #   or [-1, 1] if znorm = True (Normalization, z-score)
    #   for use with tanh act as Generator output
    if env is None:  # img
        if path[-3:] == "dng":  # if image is a DNG
            import rawpy

            with rawpy.imread(path) as raw:
                img = raw.postprocess()
        else:  # else, if image can be read by cv2
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # elif: # using PIL instead of OpenCV
        # img = Image.open(path).convert('RGB')
        # else: # For other images unrecognized by cv2
        # import matplotlib.pyplot as plt
        # img = (255*plt.imread(path)[:,:,:3]).astype('uint8')
    else:
        img = _read_lmdb_img(env, path)
    img = img.astype(np.float32) / 255.0
    if znorm:  # normalize images range to [-1, 1] (zero-normalization)
        img = (img - 0.5) * 2  # xi' = (xi - mu)/sigma
    # print("Min. image value:",img.min()) # Debug
    # print("Max. image value:",img.max()) # Debug
    """ og code:
    if img.ndim == 2:	
	        img = np.expand_dims(img, axis=2)	
	    # some images have 4 channels	
	    if img.shape[2] > 3:	
	        img = img[:, :, :3]
    """
    # """ edit:
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.tile(img, (1, 1, 3))
    if img.shape[2] > out_nc:  # remove extra channels
        img = img[:, :, :out_nc]
    elif img.shape[2] == 3 and out_nc == 4:  # pad with solid alpha channel
        img = np.dstack((img, np.full(img.shape[:-1], 1.0, dtype=np.float32)))
    # """
    return img


####################
# image processing
# process on numpy image
####################


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            """ og code:
            img = img[:, ::-1, :]
            """
            # """ edit:
            img = np.flip(img, axis=1)
            # """
        if vflip:
            """ og code:
            img = img[::-1, :, :]
            """
            # """ edit:
            img = np.flip(img, axis=0)
            # """
        if rot90:
            """ og code:
            img = img.transpose(1, 0, 2)
            """
            # """ edit:
            img = np.rot90(img, 1)
            # """
            """ PIL alt:
            img = img.transpose(Image.ROTATE_90)
            """
        return img

    return [_augment(img) for img in img_list]


def channel_convert(in_c, tar_type, img_list):
    # conversion among BGR, gray and y
    # Note: OpenCV uses inverted channels BGR, instead of RGB.
    #  If images are loaded with something other than OpenCV,
    #  check that the channels are in the correct order and use
    #  the alternative conversion functions.
    if in_c == 3 and tar_type == "gray":  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == "y":  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == "RGB":  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    elif in_c == 3 and tar_type == "RGB-LAB":  # RGB to LAB [add]
        return [cv2.cvtColor(img, cv2.COLOR_BGR2LAB) for img in img_list]
    elif in_c == 3 and tar_type == "LAB-RGB":  # RGB to LAB [add]
        return [cv2.cvtColor(img, cv2.COLOR_LAB2BGR) for img in img_list]
    elif in_c == 4 and tar_type == "RGB-A":  # BGRA to BGR, remove alpha channel [add]
        return [cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) for img in img_list]
    else:
        return img_list


def rgb2ycbcr(img, only_y=True):
    """same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    """ og code:
    img.astype(np.float32)
    """
    # """ edit:
    img = img.astype(np.float32)
    # """
    if in_img_type != np.uint8:
        img *= 255.0
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img,
            [
                [65.481, -37.797, 112.0],
                [128.553, -74.203, -93.786],
                [24.966, 112.0, -18.214],
            ],
        ) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    """bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    """ og code:
    img.astype(np.float32)
    """
    # """ edit:
    img = img.astype(np.float32)
    # """
    if in_img_type != np.uint8:
        img *= 255.0
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img,
            [
                [24.966, 112.0, -18.214],
                [128.553, -74.203, -93.786],
                [65.481, -37.797, 112.0],
            ],
        ) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    """same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    """ og code:
    img.astype(np.float32)
    """
    # """ edit:
    img = img.astype(np.float32)
    # """
    if in_img_type != np.uint8:
        img *= 255.0
    # convert
    rlt = np.matmul(
        img,
        [
            [0.00456621, 0.00456621, 0.00456621],
            [0, -0.00153632, 0.00791071],
            [0.00625893, -0.00318811, 0],
        ],
    ) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[: H - H_r, : W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[: H - H_r, : W - W_r, :]
    else:
        raise ValueError("Wrong img ndim: [{:d}].".format(img.ndim))
    return img


####################
# Prepare Images
# None of these were used by original code
####################
# https://github.com/sunreef/BlindSR/blob/master/src/image_utils.py
def patchify_tensor(features, patch_size, overlap=10):
    batch_size, channels, height, width = features.size()

    effective_patch_size = patch_size - overlap
    n_patches_height = height // effective_patch_size
    n_patches_width = width // effective_patch_size

    if n_patches_height * effective_patch_size < height:
        n_patches_height += 1
    if n_patches_width * effective_patch_size < width:
        n_patches_width += 1

    patches = []
    for b in range(batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(h * effective_patch_size, height - patch_size)
                patch_start_width = min(w * effective_patch_size, width - patch_size)
                patches.append(
                    features[
                    b: b + 1,
                    :,
                    patch_start_height: patch_start_height + patch_size,
                    patch_start_width: patch_start_width + patch_size,
                    ]
                )
    return torch.cat(patches, 0)


def recompose_tensor(patches, full_height, full_width, overlap=10):
    batch_size, channels, patch_size, _ = patches.size()
    effective_patch_size = patch_size - overlap
    n_patches_height = full_height // effective_patch_size
    n_patches_width = full_width // effective_patch_size

    if n_patches_height * effective_patch_size < full_height:
        n_patches_height += 1
    if n_patches_width * effective_patch_size < full_width:
        n_patches_width += 1

    n_patches = n_patches_height * n_patches_width
    if batch_size % n_patches != 0:
        print(
            "Error: The number of patches provided to the recompose function does not match the number of patches in each image."
        )
    final_batch_size = batch_size // n_patches

    blending_in = torch.linspace(0.1, 1.0, overlap)
    blending_out = torch.linspace(1.0, 0.1, overlap)
    middle_part = torch.ones(patch_size - 2 * overlap)
    blending_profile = torch.cat([blending_in, middle_part, blending_out], 0)

    horizontal_blending = blending_profile[None].repeat(patch_size, 1)
    vertical_blending = blending_profile[:, None].repeat(1, patch_size)
    blending_patch = horizontal_blending * vertical_blending

    blending_image = torch.zeros(1, channels, full_height, full_width)
    for h in range(n_patches_height):
        for w in range(n_patches_width):
            patch_start_height = min(h * effective_patch_size, full_height - patch_size)
            patch_start_width = min(w * effective_patch_size, full_width - patch_size)
            blending_image[
            0,
            :,
            patch_start_height: patch_start_height + patch_size,
            patch_start_width: patch_start_width + patch_size,
            ] += blending_patch[None]

    recomposed_tensor = torch.zeros(final_batch_size, channels, full_height, full_width)
    if patches.is_cuda:
        blending_patch = blending_patch.cuda()
        blending_image = blending_image.cuda()
        recomposed_tensor = recomposed_tensor.cuda()
    patch_index = 0
    for b in range(final_batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(
                    h * effective_patch_size, full_height - patch_size
                )
                patch_start_width = min(
                    w * effective_patch_size, full_width - patch_size
                )
                recomposed_tensor[
                b,
                :,
                patch_start_height: patch_start_height + patch_size,
                patch_start_width: patch_start_width + patch_size,
                ] += (patches[patch_index] * blending_patch)
                patch_index += 1
    recomposed_tensor /= blending_image

    return recomposed_tensor


####################
# Functions
####################


# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
            -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    ) * (((absx > 1) * (absx <= 2)).type_as(absx))


def calculate_weights_indices(
    in_length, out_length, scale, kernel, kernel_width, antialiasing
):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(
        0, P - 1, P
    ).view(1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round

    in_C, in_H, in_W = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = "cubic"

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing
    )
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing
    )
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = (
            img_aug[0, idx: idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        )
        out_1[1, i, :] = (
            img_aug[1, idx: idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        )
        out_1[2, i, :] = (
            img_aug[2, idx: idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        )

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :, idx: idx + kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :, idx: idx + kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :, idx: idx + kernel_width].mv(weights_W[i])

    return out_2


def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC BGR [0,1]
    # output: HWC BGR [0,1] w/o round
    img = torch.from_numpy(img)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = "cubic"

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing
    )
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing
    )
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = (
            img_aug[idx: idx + kernel_width, :, 0].transpose(0, 1).mv(weights_H[i])
        )
        out_1[i, :, 1] = (
            img_aug[idx: idx + kernel_width, :, 1].transpose(0, 1).mv(weights_H[i])
        )
        out_1[i, :, 2] = (
            img_aug[idx: idx + kernel_width, :, 2].transpose(0, 1).mv(weights_H[i])
        )

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx: idx + kernel_width, 0].mv(weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx: idx + kernel_width, 1].mv(weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx: idx + kernel_width, 2].mv(weights_W[i])

    return out_2.numpy()


if __name__ == "__main__":
    # test imresize function
    # read images
    img = cv2.imread("test.png")
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # imresize
    scale = 1 / 4
    import time

    total_time = 0
    for i in range(10):
        start_time = time.time()
        rlt = imresize(img, scale, antialiasing=True)
        use_time = time.time() - start_time
        total_time += use_time
    print("average time: {}".format(total_time / 10))

    import torchvision.utils

    torchvision.utils.save_image(
        (rlt * 255).round() / 255, "rlt.png", nrow=1, padding=0, normalize=False
    )
