import os.path
import sys
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging

import torch

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from models.modules.LPIPS import compute_dists as lpips


def cycle(dataloader):
    while True:
        for x in iter(dataloader):
            yield x

def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options file.')
    parser.add_argument('-simple', action='store_true', help='Enable simple logging.')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    use_simple_logging = parser.parse_args().simple
    
    # train from scratch OR resume training
    if opt['path']['resume_state']:
        if os.path.isdir(opt['path']['resume_state']):
            import glob
            opt['path']['resume_state'] = util.sorted_nicely(glob.glob(os.path.normpath(opt['path']['resume_state']) + '/*.state'))[-1]
        resume_state_path = opt['path']['resume_state']
        resume_state = torch.load(resume_state_path)
    else:  # training from scratch
        resume_state = None
        util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))
    
    # config loggers. Before it, the log will not work
    util.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    util.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    
    if resume_state:
        logger.info('Set [resume_state] to ' + resume_state_path)
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))
        option.check_resume(opt)  # check resume options

    logger.info(option.dict2str(opt))
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        from tensorboardX import SummaryWriter
        try:
            tb_logger = SummaryWriter(logdir='../tb_logger/' + opt['name']) #for version tensorboardX >= 1.7
        except:
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name']) #for version tensorboardX < 1.6

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            train_size = int(len(train_loader) / opt['batch_multiplier'])
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'],
                                                                      len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    train_gen = cycle(train_loader)

    # create model
    model = create_model(opt)

    # resume training
    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
        model.update_schedulers(opt['train']) # updated schedulers in case configuration has changed
    else:
        current_step = 0
        start_epoch = 0

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    epoch = start_epoch
    while current_step <= total_iters:
        for n in range(1, train_size+1):
            current_step += 1
            if current_step > total_iters:
                break
            # training
            model.optimize_parameters(train_gen, current_step)

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    if not use_simple_logging:
                        message += '{:s}:{: .4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

            # update learning rate
            model.update_learning_rate()

            # save models and training states (changed to save models before validation)
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                model.save(current_step, opt['name'])
                model.save_training_state(epoch + (n >= train_size), current_step)
                logger.info('Models and training states saved.')

            # save & overwrite backup models & training states
            if opt['logger']['backup_freq'] and current_step % opt['logger']['backup_freq'] == 0:
                model.save('backup')
                model.save_training_state(epoch + (n >= train_size), current_step, True)
                logger.info('Backup models and training states saved.')

            # validation
            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_lpips = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['HR'])  # uint8
                    
                    # print("Min. SR value:",sr_img.min()) # Debug
                    # print("Max. SR value:",sr_img.max()) # Debug
                    
                    # print("Min. GT value:",gt_img.min()) # Debug
                    # print("Max. GT value:",gt_img.max()) # Debug
                    
                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(\
                        img_name, current_step))
                    util.save_img(sr_img, save_img_path)

                    # calculate PSNR, SSIM and LPIPS distance
                    crop_size = opt['scale']

                    # For training models with only one channel ndim==2, if RGB ndim==3, etc.
                    if gt_img.ndim == 2:
                        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size]
                    else:
                        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    if sr_img.ndim == 2:
                        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size]
                    else: # Default: RGB images
                        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    
                    # LPIPS only works for RGB images
                    avg_psnr += util.calculate_psnr(cropped_sr_img, cropped_gt_img)
                    avg_ssim += util.calculate_ssim(cropped_sr_img, cropped_gt_img)
                    avg_lpips += lpips.calculate_lpips(cropped_sr_img, cropped_gt_img)
                    del cropped_gt_img, cropped_sr_img, gt_img, sr_img

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_lpips = avg_lpips / idx
                lpips.cleanup()

                # log
                # logger.info('# Validation # PSNR: {:.5g}, SSIM: {:.5g}'.format(avg_psnr, avg_ssim))
                logger.info('# Validation # PSNR: {:.5g}, SSIM: {:.5g}, LPIPS: {:.5g}'.format(avg_psnr, avg_ssim, avg_lpips))
                logger_val = logging.getLogger('val')  # validation logger
                # logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.5g}, ssim: {:.5g}'.format(
                    # epoch, current_step, avg_psnr, avg_ssim))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.5g}, ssim: {:.5g}, lpips: {:.5g}'.format(
                    epoch, current_step, avg_psnr, avg_ssim, avg_lpips))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)
                    tb_logger.add_scalar('lpips', avg_lpips, current_step)
        epoch += 1

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
