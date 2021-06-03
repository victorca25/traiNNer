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
from utils.metrics import calculate_psnr, calculate_ssim, MetricsDict
from train import parse_options, dir_check, configure_loggers, get_dataloaders

from models.modules.architectures.CEM import CEMnet
from dataops.filters import GuidedFilter
from dataops.colors import rgb_to_ycbcr, ycbcr_to_rgb
from test import visuals_check, get_img_path, get_CEM



def test_loop(model, opt, dataloaders, data_params):
    logger = util.get_root_logger()

    # read data_params
    znorms = data_params['znorm']

    # prepare the metric calculation classes for RGB and Y_only images
    calc_metrics = opt.get('metrics', None)
    if calc_metrics:
        test_metrics = MetricsDict(metrics = calc_metrics)
        test_metrics_y = MetricsDict(metrics = calc_metrics)

    for phase, dataloader in dataloaders.items():
        name = dataloader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(name))
        dataset_dir = os.path.join(opt['path']['results_root'], name)
        util.mkdir(dataset_dir)

        nlls = []
        for data in dataloader:
            znorm = znorms[name]
            need_HR = False if dataloader.dataset.opt['dataroot_HR'] is None else True

            # set up per image CEM wrapper if configured
            CEM_net = get_CEM(opt, data)

            model.feed_data(data, need_HR=need_HR)  # unpack data from data loader
            # img_path = data['LR_path'][0]
            img_path = get_img_path(data)
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            # test with eval mode. This only affects layers like batchnorm and dropout.
            test_mode = opt.get('test_mode', None)
            if test_mode == 'x8':
                # geometric self-ensemble
                # model.test_x8(CEM_net=CEM_net)
                break
            elif test_mode == 'chop':
                # chop images in patches/crops, to reduce VRAM usage
                # model.test_chop(patch_size=opt.get('chop_patch_size', 100), 
                #                 step=opt.get('chop_step', 0.9),
                #                 CEM_net=CEM_net)
                break
            else:
                # normal inference
                model.test(CEM_net=CEM_net)  # run inference
            
            if hasattr(model, 'nll'):
                nll = model.nll if model.nll else 0
                nlls.append(nll)

            # get image results
            visuals = model.get_current_visuals(need_HR=need_HR)

            res_options = visuals_check(visuals.keys(), opt.get('val_comparison', None))

            # save images
            save_img_path = os.path.join(dataset_dir, img_name + opt.get('suffix', ''))
            
            # Save SR images for reference
            sr_img = None
            if hasattr(model, 'heats'):  # SRFlow
                opt['val_comparison'] = False
                for heat in model.heats:
                    for i in range(model.n_sample):
                        for save_img_name in res_options['save_imgs']:
                            imn = '_' + save_img_name if len(res_options['save_imgs']) > 1 else ''
                            imn += '_h{:03d}_s{:d}'.format(int(heat * 100), i)
                            util.save_img(tensor2np(visuals[save_img_name, heat, i], denormalize=znorm), save_img_path + imn + '.png')
            else:  # regular SR
                if not opt['val_comparison']:
                    for save_img_name in res_options['save_imgs']:
                        imn = '_' + save_img_name if len(res_options['save_imgs']) > 1 else ''
                        util.save_img(tensor2np(visuals[save_img_name], denormalize=znorm), save_img_path + imn + '.png')

            # save single images or lr / sr comparison
            if opt['val_comparison'] and len(res_options['save_imgs']) > 1:
                comp_images = [tensor2np(visuals[save_img_name], denormalize=znorm) for save_img_name in res_options['save_imgs']]
                util.save_img_comp(comp_images, save_img_path + '.png')
            # else:
                # util.save_img(sr_img, save_img_path)

            # calculate metrics if HR dataset is provided and metrics are configured in options
            if need_HR and calc_metrics and res_options['aligned_metrics']:
                metric_imgs = [tensor2np(visuals[x], denormalize=znorm) for x in res_options['compare_imgs']]
                test_results = test_metrics.calculate_metrics(metric_imgs[0], metric_imgs[1], 
                                                              crop_size=opt['scale'])
                
                # prepare single image metrics log message
                logger_m = '{:20s} -'.format(img_name)
                for k, v in test_results.items():
                    formatted_res = k.upper() + ': {:.6f}, '.format(v)
                    logger_m += formatted_res

                if metric_imgs[1].shape[2] == 3:  # RGB image, calculate y_only metrics
                    test_results_y = test_metrics_y.calculate_metrics(metric_imgs[0], metric_imgs[1], 
                                                                      crop_size=opt['scale'], only_y=True)
                    
                    # add the y only results to the single image log message
                    for k, v in test_results_y.items():
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
            agg_logger_m = ''.join(f'{met.upper()}: {avgr:.6f}, ' for met, avgr in avg_metrics.items())
            logger.info('----Average metrics results for {}----\n\t'.format(name) + agg_logger_m[:-2])
            
            if avg_metrics_y:
                # prepare log average Y channel metrics message
                agg_logger_m = ''.join(f'{met.upper()}_Y: {avgr:.6f}, ' for met, avgr in avg_metrics_y.items())
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