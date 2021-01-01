import os
import sys
import logging
import time
import argparse
import numpy as np
from collections import OrderedDict

import options.options as option
import utils.util as util
from dataops.common import bgr2ycbcr, tensor2np
from data import create_dataset, create_dataloader
from models import create_model


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True,
                        help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    util.mkdirs(
        (path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
    opt = option.dict_to_nonedict(opt)

    util.setup_logger(None, opt['path']['log'],
                      'test.log', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    # Create test dataset and dataloader
    test_loaders = []
    znorm = False  # TMP
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(
            dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)
        # Temporary, will turn znorm on for all the datasets. Will need to introduce a variable for each dataset and differentiate each one later in the loop.
        if dataset_opt['znorm'] and znorm == False:
            znorm = True

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

            model.feed_data(data, need_HR=need_HR)
            img_path = data['in_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            model.test()  # test
            visuals = model.get_current_visuals(need_HR=need_HR)

            #if znorm the image range is [-1,1], Default: Image range is [0,1] # testing, each "dataset" can have a different name (not train, val or other)
            top_img = tensor2np(visuals['top_fake'])  # uint8
            bot_img = tensor2np(visuals['bottom_fake'])  # uint8

            # save images
            suffix = opt['suffix']
            if suffix:
                save_img_path = os.path.join(
                    dataset_dir, img_name + suffix)
            else:
                save_img_path = os.path.join(dataset_dir, img_name)
            util.save_img(top_img, save_img_path + '_top.png')
            util.save_img(bot_img, save_img_path + '_bot.png')


            #TODO: update to use metrics functions
            # calculate PSNR and SSIM
            if need_HR:
                #if znorm the image range is [-1,1], Default: Image range is [0,1] # testing, each "dataset" can have a different name (not train, val or other)
                gt_img = tensor2img(visuals['HR'], denormalize=znorm)  # uint8
                gt_img = gt_img / 255.
                sr_img = sr_img / 255.

                crop_border = test_loader.dataset.opt['scale']
                cropped_sr_img = sr_img[crop_border:-
                                        crop_border, crop_border:-crop_border, :]
                cropped_gt_img = gt_img[crop_border:-
                                        crop_border, crop_border:-crop_border, :]

                psnr = util.calculate_psnr(
                    cropped_sr_img * 255, cropped_gt_img * 255)
                ssim = util.calculate_ssim(
                    cropped_sr_img * 255, cropped_gt_img * 255)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)

                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    cropped_sr_img_y = sr_img_y[crop_border:-
                                                crop_border, crop_border:-crop_border]
                    cropped_gt_img_y = gt_img_y[crop_border:-
                                                crop_border, crop_border:-crop_border]
                    psnr_y = util.calculate_psnr(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                    ssim_y = util.calculate_ssim(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)
                    logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'
                                .format(img_name, psnr, ssim, psnr_y, ssim_y))
                else:
                    logger.info(
                        '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
            else:
                logger.info(img_name)

        #TODO: update to use metrics functions
        if need_HR:  # metrics
            # Average PSNR/SSIM results
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            logger.info('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'
                        .format(test_set_name, ave_psnr, ave_ssim))
            if test_results['psnr_y'] and test_results['ssim_y']:
                ave_psnr_y = sum(
                    test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(
                    test_results['ssim_y']) / len(test_results['ssim_y'])
                logger.info('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'
                            .format(ave_psnr_y, ave_ssim_y))


if __name__ == '__main__':
    main()
