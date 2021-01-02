"""
Calculate the PSNR and SSIM. Same as MATLAB's results.

Terms:
    GT: Ground-truth
    Gen: Generated/Restored/Recovered images
"""
import glob
import os

import cv2

from codes.dataops.common import bgr2ycbcr
from codes.utils.metrics import calculate_psnr, calculate_ssim


def main():
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


if __name__ == '__main__':
    main()
