import argparse
import logging
import os
import time

import torch

import options
import utils.util as util
from data import create_dataset, create_dataloader
from dataops.common import bgr2ycbcr, tensor2np
from models import create_model
from utils.metrics import calculate_psnr, calculate_ssim
from train import parse_options, dir_check, configure_loggers, get_dataloaders


def test_loop(model, opt, dataloaders, data_params):
    logger = util.get_root_logger()

    # read data_params
    znorms = data_params['znorm']

    # prepare the metric calculation classes for RGB and Y_only images
    calc_metrics = opt.get('metrics', None)
    if calc_metrics:
        test_metrics = metrics.MetricsDict(metrics = calc_metrics)
        test_metrics_y = metrics.MetricsDict(metrics = calc_metrics)

    for phase, dataloader in dataloaders.items():
        name = dataloader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(name))
        dataset_dir = os.path.join(opt['path']['results_root'], name)
        util.mkdir(dataset_dir)

        for data in dataloader:
            znorm = znorms[name]
            need_HR = False if dataloader.dataset.opt['dataroot_HR'] is None else True

            model.feed_data(data, need_HR=need_HR)  # unpack data from data loader
            img_path = data['LR_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            # test with eval mode. This only affects layers like batchnorm and dropout.
            test_mode = opt.get('test_mode', None)
            if test_mode == 'x8':
                # geometric self-ensemble
                model.test_x8()
            elif test_mode == 'chop':
                # chop images in patches/crops, to reduce VRAM usage
                model.test_chop(patch_size=opt.get('chop_patch_size', 100), 
                                step=opt.get('chop_step', 0.9))
            else:
                # normal inference
                model.test()  # run inference
            visuals = model.get_current_visuals(need_HR=need_HR)  # get image results

            sr_img = tensor2np(visuals['SR'], denormalize=znorm)  # uint8

            # save images
            suffix = opt['suffix']
            if suffix:
                save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
            else:
                save_img_path = os.path.join(dataset_dir, img_name + '.png')
            util.save_img(sr_img, save_img_path)

            # calculate metrics if HR dataset is provided and metrics are configured in options
            if need_HR and calc_metrics:
                gt_img = tensor2np(visuals['HR'], denormalize=znorm)  # uint8

                test_results = test_metrics.calculate_metrics(sr_img, gt_img, crop_size=opt['scale'])
                
                # prepare single image metrics log message
                logger_m = '{:20s} -'.format(img_name)
                for k, v in test_results:
                    formatted_res = k.upper() + ': {:.6f}, '.format(v)
                    logger_m += formatted_res

                if gt_img.shape[2] == 3:  # RGB image, calculate y_only metrics
                    test_results_y = test_metrics_y.calculate_metrics(sr_img, gt_img, crop_size=opt['scale'], only_y=True)
                    
                    # add the y only results to the single image log message
                    for k, v in test_results_y:
                        formatted_res = k.upper() + ': {:.6f}, '.format(v)
                        logger_m += formatted_res
                
                logger.info(logger_m)
            else:
                logger.info(img_name)

        # average metrics results for the dataset
        if need_HR and calc_metrics:
            
            # aggregate the metrics results (automatically resets the metric classes)
            avg_metrics = test_metrics.get_averages()
            avg_metrics_y = test_metrics_y.get_averages()

            # prepare log average metrics message
            agg_logger_m = ''
            for r in avg_metrics:
                formatted_res = r['name'].upper() + ': {:.6f}, '.format(r['average'])
                agg_logger_m += formatted_res
            logger.info('----Average metrics results for {}----\n\t'.format(name) + agg_logger_m[:-2])
            
            if len(avg_metrics_y > 0):
                # prepare log average Y channel metrics message
                agg_logger_m = ''
                for r in avg_metrics_y:
                    formatted_res = r['name'].upper() + '_Y' + ': {:.6f}, '.format(r['average'])
                    agg_logger_m += formatted_res
                logger.info('----Y channel, average metrics ----\n\t' + agg_logger_m[:-2])


def main():
    
    # parse test options
    opt = parse_options(is_train=False)

    # create the test directory
    dir_check(opt)
    
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    
    # configure loggers
    loggers = configure_loggers(opt)

    # create dataloaders
    # note: test dataloader only supports num_workers = 0, batch_size = 1 and data shuffling is disable
    dataloaders, data_params = get_dataloaders(opt)

    # create and setup model: load and print network; init
    model = create_model(opt)

    # start testing loop with configured options
    test_loop(model, opt, dataloaders, data_params)
    

if __name__ == '__main__':
    main()