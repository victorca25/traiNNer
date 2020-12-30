import glob
import math
import os

import cv2
import numpy as np


def main():
    """
    Calculate the PSNR and SSIM. Same as MATLAB's results.

    Terms:
        GT: Ground-truth
        Gen: Generated/Restored/Recovered images
    """
    # Config
    folder_gt = '/mnt/SSD/BasicSR_datasets/val_set5/Set5'
    folder_gen = '/home/Projects/BasicSR/results/RRDB_PSNR_x4/set5'
    crop_border = 4
    suffix = ''  # suffix for Gen images
    test_y = False  # True: test Y channel only, will convert to YCbCr color space; False: test RGB channels

    print('Testing [%s] channels.' % ('Y' if test_y else 'R,G,B'))
    psnr_all = []
    ssim_all = []
    for i, img_path in enumerate(sorted(glob.glob(folder_gt + '/*'))):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        im_gt = cv2.imread(img_path) / 255.
        im_gen = cv2.imread(os.path.join(folder_gen, base_name + suffix + '.png')) / 255.

        if test_y and im_gt.shape[2] == 3:
            # evaluate on Y channel in YCbCr color space
            # TODO: Why? Is there some kind of benefit I'm not seeing here?
            #       Was this done JUST because it returns only the Y channel in a convenient matter?
            im_gt = bgr2ycbcr(im_gt)
            im_gen = bgr2ycbcr(im_gen)

        # crop borders
        if im_gt.ndim == 3:
            cropped_gt = im_gt[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_gen = im_gen[crop_border:-crop_border, crop_border:-crop_border, :]
        elif im_gt.ndim == 2:
            cropped_gt = im_gt[crop_border:-crop_border, crop_border:-crop_border]
            cropped_gen = im_gen[crop_border:-crop_border, crop_border:-crop_border]
        else:
            raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_gt.ndim))

        # calculate PSNR and SSIM
        psnr = calculate_psnr(cropped_gt * 255, cropped_gen * 255)
        ssim = calculate_ssim(cropped_gt * 255, cropped_gen * 255)
        print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(i + 1, base_name, psnr, ssim))
        psnr_all.append(psnr)
        ssim_all.append(ssim)
    print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
        sum(psnr_all) / len(psnr_all), sum(ssim_all) / len(ssim_all))
    )


def calculate_psnr(img_1: np.ndarray, img_2: np.ndarray) -> float:
    """
    Calculate PSNR
    :param img_1: Image 1 [0, 255]
    :param img_2: Image 2 [0, 255]
    """
    img_1 = img_1.astype(np.float64)
    img_2 = img_2.astype(np.float64)
    mse = np.mean((img_1 - img_2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate SSIM
    The same outputs as MATLAB's
    :param img1: Image 1 [0, 255]
    :param img2: Image 2 [0, 255]
    """

    def ssim(img_1: np.ndarray, img_2: np.ndarray) -> float:
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        img_1 = img_1.astype(np.float64)
        img_2 = img_2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img_1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img_2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img_1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img_2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img_1 * img_2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim_map.mean()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            return np.array([ssim(img1, img2) for _ in range(3)]).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img: np.ndarray, only_y: bool = True) -> np.ndarray:
    """
    Same as matlab rgb2ycbcr
    Input:
        uint8, [0, 255]
        float, [0, 1]
    :param img: Input image
    :param only_y: Only return Y channel
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':
    main()
