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

from models.modules.architectures.CEM import CEMnet
from dataops.filters import GuidedFilter
from dataops.colors import rgb_to_ycbcr, ycbcr_to_rgb


def visuals_check(visuals_keys, val_comparison=None):
    aligned_metrics = False
    save_imgs = []
    compare_imgs = []
    model_type = None
    
    if visuals_keys >= {"LR", "HR", "SR"} or visuals_keys >= {"lq", "gt", "result"}:
        # SR, 1X
        aligned_metrics = True
        if val_comparison:
            save_imgs = ["LR", "SR"] if "SR" in visuals_keys else ["lq", "results"]  # default, can filter from options file
        else: 
            save_imgs = ["SR"] if "SR" in visuals_keys else ["results"]
        compare_imgs = ["SR", "HR"] if "SR" in visuals_keys else ["results", "gt"]
        model_type = "SR"
    elif visuals_keys >= {"LR", "SR"} or visuals_keys >= {"lq", "result"}:
        # LR
        if val_comparison:
            save_imgs = ["LR", "SR"] if "SR" in visuals_keys else ["lq", "results"]
        else:
            save_imgs = ["SR"] if "SR" in visuals_keys else ["results"]
        model_type = "SR"
    elif visuals_keys >= {"real_A", "fake_B", "rec_A", "real_B", "fake_A", "rec_B"}:
        # CycleGAN
        # res_img = "fake_B" # if AtoB else "fake_A"
        save_imgs = ["real_A", "fake_B", "rec_A", "real_B", "fake_A", "rec_B"]  # default, can filter from options file
        model_type = "CiC"
    elif visuals_keys >= {"real_A", "fake_B", "real_B"}:
        # pix2pix
        # res_img = "fake_B"
        save_imgs = ["real_A", "fake_B", "real_B"]  # default, can filter from options file
        model_type = "P2P"
    elif visuals_keys >= {"interlaced", "top_fake", "bottom_fake", "top_real", "bot_real"}:
        # DVD train
        aligned_metrics = True
        save_imgs = ["top_fake", "bottom_fake"]
        compare_imgs = ["top_fake", "top_real"]  # and compare_imgs = ["bot_fake", "bot_real"]
        model_type = "DVD"
    elif visuals_keys >= {"interlaced", "top_fake", "bottom_fake"}:
        # DVD test
        save_imgs = ["top_fake", "bottom_fake"]
        model_type = "DVD"
    
    return {"aligned_metrics": aligned_metrics,
            "save_imgs": save_imgs,
            "compare_imgs": compare_imgs,
            "model_type": model_type}

def get_img_path(data):
    img_path = ''
    data_keys = data.keys()
    if data_keys >= {"LR", "HR", "LR_path", "HR_path"} or data_keys >= {"lq", "gt", "lq_path", "gt_path"}:
        # LRHR, PBR, Vid Train
        img_path = data['LR_path'][0] if 'LR_path' in data else data['lq_path'][0]
    elif data_keys >= {"LR", "LR_path"} or data_keys >= {"lq", "lq_path"}:
        # LR
        img_path = data['LR_path'][0] if 'LR_path' in data else data['lq_path'][0]
    elif data_keys >= {"A", "B", "A_path", "B_path"}:
        # pix2pix, CycleGAN
        img_path = ''  # model.image_paths
    elif data_keys >= {"in", "top", "bottom", "in_path", "top_path", "bot_path"}:
        # DVD train
        img_path = data['in_path'][0]
    elif data_keys >= {"in", "in_path"}:
        # DVD test
        img_path = data['in_path'][0]
    return img_path

def get_CEM(opt, data):
    CEM_net = None
    if opt.get('use_cem', None):
        CEM_conf = CEMnet.Get_CEM_Conf(opt['scale'])
        CEM_conf.lower_magnitude_bound = opt['cem_config'].get('cem_lmb', 0.35)
        CEM_conf.default_kernel_alg = opt['cem_config'].get('cem_alg', 'torch')
        kernel = opt['cem_config'].get('cem_kernel', 'cubic')
        if kernel == 'estimated':
            kernel = data.get('kernel', None)
        CEM_net = CEMnet.CEMnet(CEM_conf, upscale_kernel=kernel)
        CEM_net.WrapArchitecture(only_padders=True)
    return CEM_net

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
                model.test_x8(CEM_net=CEM_net)
            elif test_mode == 'chop':
                # chop images in patches/crops, to reduce VRAM usage
                model.test_chop(patch_size=opt.get('chop_patch_size', 100), 
                                step=opt.get('chop_step', 0.9),
                                CEM_net=CEM_net)
            else:
                # normal inference
                model.test(CEM_net=CEM_net)  # run inference
            
            # get image results
            visuals = model.get_current_visuals(need_HR=need_HR)

            # post-process options if using CEM
            if opt.get('use_cem', None) and opt['cem_config'].get('out_orig', False):
                # run regular inference
                if test_mode == 'x8':
                    model.test_x8()
                elif test_mode == 'chop':
                    model.test_chop(patch_size=opt.get('chop_patch_size', 100), 
                                    step=opt.get('chop_step', 0.9))
                else:
                    model.test()
                orig_visuals = model.get_current_visuals(need_HR=need_HR)

                if opt['cem_config'].get('out_filter', False):
                    GF = GuidedFilter(ks=opt['cem_config'].get('out_filter_ks', 7))
                    filt = GF(visuals['SR'].unsqueeze(0), (visuals['SR']-orig_visuals['SR']).unsqueeze(0)).squeeze(0)
                    visuals['SR'] = orig_visuals['SR']+filt

                if opt['cem_config'].get('out_keepY', False):
                    out_regY = rgb_to_ycbcr(orig_visuals['SR']).unsqueeze(0)
                    out_cemY = rgb_to_ycbcr(visuals['SR']).unsqueeze(0)
                    visuals['SR'] = ycbcr_to_rgb(torch.cat([out_regY[:, 0:1, :, :], out_cemY[:, 1:2, :, :], out_cemY[:, 2:3, :, :]], 1)).squeeze(0)

            res_options = visuals_check(visuals.keys(), opt.get('val_comparison', None))

            # save images
            save_img_path = os.path.join(dataset_dir, img_name + opt.get('suffix', ''))

            # save single images or lr / sr comparison
            if opt['val_comparison'] and len(res_options['save_imgs']) > 1:
                comp_images = [tensor2np(visuals[save_img_name], denormalize=znorm) for save_img_name in res_options['save_imgs']]
                util.save_img_comp(comp_images, save_img_path + '.png')
            else:
                for save_img_name in res_options['save_imgs']:
                    imn = '_' + save_img_name if len(res_options['save_imgs']) > 1 else ''
                    util.save_img(tensor2np(visuals[save_img_name], denormalize=znorm), save_img_path + imn + '.png')

            # calculate metrics if HR dataset is provided and metrics are configured in options
            if need_HR and calc_metrics and res_options['aligned_metrics']:
                metric_imgs = [tensor2np(visuals[x], denormalize=znorm) for x in res_options['compare_imgs']]
                test_results = test_metrics.calculate_metrics(metric_imgs[0], metric_imgs[1], 
                                                              crop_size=opt['scale'])
                
                # prepare single image metrics log message
                logger_m = '{:20s} -'.format(img_name)
                for k, v in test_results:
                    formatted_res = k.upper() + ': {:.6f}, '.format(v)
                    logger_m += formatted_res

                if gt_img.shape[2] == 3:  # RGB image, calculate y_only metrics
                    test_results_y = test_metrics_y.calculate_metrics(metric_imgs[0], metric_imgs[1], 
                                                                      crop_size=opt['scale'], only_y=True)
                    
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