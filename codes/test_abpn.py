import argparse
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch

import options
import utils.util as util
from data import create_dataset, create_dataloader
from dataops.common import bgr2ycbcr, tensor2np, patchify_tensor, recompose_tensor
from models import create_model


def chop_forward2(lowres_img, model, scale, patch_size=200):
    batch_size, channels, img_height, img_width = lowres_img.size()
    # Patch size for patch-based testing of large images.
    # Make sure the patch size is small enough that your GPU memory is sufficient.
    # patch_size = 200 from BlindSR, 64 for ABPN
    patch_size = min(img_height, img_width, patch_size)
    overlap = patch_size // 4
    # print("lowres_img: ",lowres_img.size())

    lowres_patches = patchify_tensor(lowres_img, patch_size, overlap=overlap)
    # print("lowres_patches: ",lowres_patches.size())

    n_patches = lowres_patches.size(0)
    highres_patches = []

    with torch.no_grad():
        for p in range(n_patches):
            lowres_input = lowres_patches[p:p + 1]
            model.feed_data_batch(lowres_input, need_HR=False)
            model.test()  # test
            visuals = model.get_current_visuals_batch(need_HR=False)
            prediction = visuals['SR']
            highres_patches.append(prediction)

    highres_patches = torch.cat(highres_patches, 0)
    # print("highres_patches: ",highres_patches.size())
    highres_output = recompose_tensor(highres_patches, scale * img_height, scale * img_width, overlap=scale * overlap)

    return highres_output


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options file.')
    opt = options.parse(parser.parse_args().opt, is_train=False)
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
    chop2 = opt['chop']
    chop_patch_size = opt['chop_patch_size']
    multi_upscale = opt['multi_upscale']
    scale = opt['scale']

    util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(options.dict2str(opt))
    # Create test dataset and dataloader
    test_loaders = []
    # znorm = False
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)
        # Temporary, will turn znorm on for all the datasets. Will need to introduce a variable for each dataset and differentiate each one later in the loop.
        # if dataset_opt['znorm'] and znorm == False: 
        # znorm = True

    # Create model
    model = create_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []

        for data in test_loader:
            need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True
            img_path = data['LR_path'][0]  # because there's only 1 image per "data" dataset loader?
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            znorm = test_loader.dataset.opt['znorm']

            if chop2 == True:
                lowres_img = data['LR']  # .to('cuda')
                if multi_upscale:  # Upscale 8 times in different rotations/flips and average the results in a single image
                    LR_90 = lowres_img.transpose(2, 3).flip(2)  # PyTorch > 0.4.1
                    LR_180 = LR_90.transpose(2, 3).flip(2)  # PyTorch > 0.4.1
                    LR_270 = LR_180.transpose(2, 3).flip(2)  # PyTorch > 0.4.1
                    LR_f = lowres_img.flip(3)  # horizontal mirror (flip), dim=3 (B,C,H,W=0,1,2,3) #PyTorch > 0.4.1
                    LR_90f = LR_90.flip(3)  # horizontal mirror (flip), dim=3 (B,C,H,W=0,1,2,3) #PyTorch > 0.4.1
                    LR_180f = LR_180.flip(3)  # horizontal mirror (flip), dim=3 (B,C,H,W=0,1,2,3) #PyTorch > 0.4.1
                    LR_270f = LR_270.flip(3)  # horizontal mirror (flip), dim=3 (B,C,H,W=0,1,2,3) #PyTorch > 0.4.1

                    pred = chop_forward2(lowres_img, model, scale=scale, patch_size=chop_patch_size)
                    pred_90 = chop_forward2(LR_90, model, scale=scale, patch_size=chop_patch_size)
                    pred_180 = chop_forward2(LR_180, model, scale=scale, patch_size=chop_patch_size)
                    pred_270 = chop_forward2(LR_270, model, scale=scale, patch_size=chop_patch_size)
                    pred_f = chop_forward2(LR_f, model, scale=scale, patch_size=chop_patch_size)
                    pred_90f = chop_forward2(LR_90f, model, scale=scale, patch_size=chop_patch_size)
                    pred_180f = chop_forward2(LR_180f, model, scale=scale, patch_size=chop_patch_size)
                    pred_270f = chop_forward2(LR_270f, model, scale=scale, patch_size=chop_patch_size)

                    # convert to numpy array
                    # if znorm: #opt['datasets']['train']['znorm']: # If the image range is [-1,1] # In testing, each "dataset" can have a different name (not train, val or other)
                    #     pred = util.tensor2img(pred,min_max=(-1, 1)).clip(0, 255)  # uint8                        
                    #     pred_90 = util.tensor2img(pred_90,min_max=(-1, 1)).clip(0, 255)  # uint8                        
                    #     pred_180 = util.tensor2img(pred_180,min_max=(-1, 1)).clip(0, 255)  # uint8                        
                    #     pred_270 = util.tensor2img(pred_270,min_max=(-1, 1)).clip(0, 255)  # uint8                        
                    #     pred_f = util.tensor2img(pred_f,min_max=(-1, 1)).clip(0, 255)  # uint8                        
                    #     pred_90f = util.tensor2img(pred_90f,min_max=(-1, 1)).clip(0, 255)  # uint8                        
                    #     pred_180f = util.tensor2img(pred_180f,min_max=(-1, 1)).clip(0, 255)  # uint8                        
                    #     pred_270f = util.tensor2img(pred_270f,min_max=(-1, 1)).clip(0, 255)  # uint8                        
                    # else: # Default: Image range is [0,1]
                    #     pred = util.tensor2img(pred).clip(0, 255)  # uint8
                    #     pred_90 = util.tensor2img(pred_90).clip(0, 255)  # uint8
                    #     pred_180 = util.tensor2img(pred_180).clip(0, 255)  # uint8
                    #     pred_270 = util.tensor2img(pred_270).clip(0, 255)  # uint8
                    #     pred_f = util.tensor2img(pred_f).clip(0, 255)  # uint8
                    #     pred_90f = util.tensor2img(pred_90f).clip(0, 255)  # uint8
                    #     pred_180f = util.tensor2img(pred_180f).clip(0, 255)  # uint8
                    #     pred_270f = util.tensor2img(pred_270f).clip(0, 255)  # uint8

                    # if znorm the image range is [-1,1], Default: Image range is [0,1] # testing, each "dataset" can have a different name (not train, val or other)
                    pred = tensor2np(pred, denormalize=znorm).clip(0, 255)  # uint8
                    pred_90 = tensor2np(pred_90, denormalize=znorm).clip(0, 255)  # uint8
                    pred_180 = tensor2np(pred_180, denormalize=znorm).clip(0, 255)  # uint8
                    pred_270 = tensor2np(pred_270, denormalize=znorm).clip(0, 255)  # uint8
                    pred_f = tensor2np(pred_f, denormalize=znorm).clip(0, 255)  # uint8
                    pred_90f = tensor2np(pred_90f, denormalize=znorm).clip(0, 255)  # uint8
                    pred_180f = tensor2np(pred_180f, denormalize=znorm).clip(0, 255)  # uint8
                    pred_270f = tensor2np(pred_270f, denormalize=znorm).clip(0, 255)  # uint8

                    pred_90 = np.rot90(pred_90, 3)
                    pred_180 = np.rot90(pred_180, 2)
                    pred_270 = np.rot90(pred_270, 1)
                    pred_f = np.fliplr(pred_f)
                    pred_90f = np.rot90(np.fliplr(pred_90f), 3)
                    pred_180f = np.rot90(np.fliplr(pred_180f), 2)
                    pred_270f = np.rot90(np.fliplr(pred_270f), 1)

                    # The reason for overflow is that your NumPy arrays (im1arr im2arr) are of the uint8 type (i.e. 8-bit). This means each element of the array can only hold values up to 255, so when your sum exceeds 255, it loops back around 0:
                    # To avoid overflow, your arrays should be able to contain values beyond 255. You need to convert them to floats for instance, perform the blending operation and convert the result back to uint8:
                    # sr_img = (pred + pred_90 + pred_180 + pred_270 + pred_f + pred_90f + pred_180f + pred_270f) / 8.0
                    sr_img = (pred.astype('float') + pred_90.astype('float') + pred_180.astype(
                        'float') + pred_270.astype('float') + pred_f.astype('float') + pred_90f.astype(
                        'float') + pred_180f.astype('float') + pred_270f.astype('float')) / 8.0
                    sr_img = sr_img.astype('uint8')

                else:
                    highres_output = chop_forward2(lowres_img, model, scale=scale, patch_size=chop_patch_size)

                    # convert to numpy array
                    # highres_image = highres_output[0].permute(1, 2, 0).clamp(0.0, 1.0).cpu()
                    # if znorm: #opt['datasets']['train']['znorm']: # If the image range is [-1,1] # In testing, each "dataset" can have a different name (not train, val or other)
                    #     sr_img = util.tensor2img(highres_output,min_max=(-1, 1))  # uint8
                    # else: # Default: Image range is [0,1]
                    #     sr_img = util.tensor2img(highres_output)  # uint8

                    # if znorm the image range is [-1,1], Default: Image range is [0,1] # testing, each "dataset" can have a different name (not train, val or other)
                    sr_img = tensor2np(highres_output, denormalize=znorm)  # uint8

            else:  # will upscale each image in the batch without chopping
                model.feed_data(data, need_HR=need_HR)
                model.test()  # test
                visuals = model.get_current_visuals(need_HR=need_HR)

                # if znorm the image range is [-1,1], Default: Image range is [0,1] # testing, each "dataset" can have a different name (not train, val or other)
                sr_img = tensor2np(visuals['SR'], denormalize=znorm)  # uint8

                # if znorm: #opt['datasets']['train']['znorm']: # If the image range is [-1,1] # In testing, each "dataset" can have a different name (not train, val or other)
                #     sr_img = util.tensor2img(visuals['SR'],min_max=(-1, 1))  # uint8
                # else: # Default: Image range is [0,1]
                #     sr_img = util.tensor2img(visuals['SR'])  # uint8

            # save images
            suffix = opt['suffix']
            if suffix:
                save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
            else:
                save_img_path = os.path.join(dataset_dir, img_name + '.png')
            util.save_img(sr_img, save_img_path)

            # calculate PSNR and SSIM
            if need_HR:
                if znorm:  # opt['datasets']['train']['znorm']: # If the image range is [-1,1] # In testing, each "dataset" can have a different name (not train, val or other)
                    gt_img = util.tensor2img(visuals['HR'], min_max=(-1, 1))  # uint8
                else:  # Default: Image range is [0,1]
                    gt_img = util.tensor2img(visuals['HR'])  # uint8
                gt_img = gt_img / 255.
                sr_img = sr_img / 255.

                crop_border = test_loader.dataset.opt['scale']
                cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

                psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)

                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                    cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                    psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                    ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)
                    logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.' \
                                .format(img_name, psnr, ssim, psnr_y, ssim_y))
                else:
                    logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
            else:
                logger.info(img_name)

        if need_HR:  # metrics
            # Average PSNR/SSIM results
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            logger.info('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n' \
                        .format(test_set_name, ave_psnr, ave_ssim))
            if test_results['psnr_y'] and test_results['ssim_y']:
                ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                logger.info('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n' \
                            .format(ave_psnr_y, ave_ssim_y))


if __name__ == '__main__':
    main()
