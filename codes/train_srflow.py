import argparse
import glob
import logging
import math
import os.path
import random
# import time

import torch

import options
from data import create_dataloader, create_dataset
from dataops.common import tensor2np
from models import create_model
from utils import util, metrics

from train import parse_options, dir_check, configure_loggers, get_dataloaders, get_resume_state, get_random_seed, resume_training


def fit(model, opt, dataloaders, steps_states, data_params, loggers):
    # read data_params
    batch_size = data_params['batch_size']
    virtual_batch_size = data_params['virtual_batch_size']
    total_iters = data_params['total_iters']
    total_epochs = data_params['total_epochs']

    # read steps_states
    start_epoch = steps_states["start_epoch"]
    current_step = steps_states["current_step"]
    virtual_step = steps_states["virtual_step"]

    # read loggers
    logger = util.get_root_logger()
    tb_logger = loggers["tb_logger"]
    
    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    try:
        timer = metrics.Timer()  # iteration timer
        timerData = metrics.TickTock()  # data timer
        timerEpoch = metrics.TickTock()  # epoch timer
        # outer loop for different epochs
        for epoch in range(start_epoch, (total_epochs * (virtual_batch_size // batch_size))+1):
            timerData.tick()
            timerEpoch.tick()

            # inner iteration loop within one epoch
            for n, train_data in enumerate(dataloaders['train'], start=1):
                timerData.tock()

                virtual_step += 1
                take_step = False
                if virtual_step > 0 and virtual_step * batch_size % virtual_batch_size == 0:
                    current_step += 1
                    take_step = True
                    if current_step > total_iters:
                        break

                # training
                model.feed_data(train_data)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters(virtual_step)  # calculate loss functions, get gradients, update network weights

                # log
                def eta(t_iter):
                    # calculate training ETA in hours
                    return (t_iter * (opt['train']['niter'] - current_step)) / 3600 if t_iter > 0 else 0

                if current_step % opt['logger']['print_freq'] == 0 and take_step:
                    # iteration end time
                    avg_time = timer.get_average_and_reset()
                    avg_data_time = timerData.get_average_and_reset()

                    # print training losses and save logging information to disk
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, t:{:.4f}s, td:{:.4f}s, eta:{:.4f}h> '.format(
                        epoch, current_step, model.get_current_learning_rate(current_step), avg_time, 
                        avg_data_time, eta(avg_time))
                    
                    # tensorboard training logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if current_step % opt['logger'].get('tb_sample_rate', 1) == 0: # Reduce rate of tb logs
                            # tb_logger.add_scalar('loss/nll', nll, current_step)
                            tb_logger.add_scalar('lr/base', model.get_current_learning_rate(), current_step)
                            tb_logger.add_scalar('time/iteration', timer.get_last_iteration(), current_step)
                            tb_logger.add_scalar('time/data', timerData.get_last_iteration(), current_step)

                    logs = model.get_current_log()
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        # tensorboard loss logger
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            if current_step % opt['logger'].get('tb_sample_rate', 1) == 0: # Reduce rate of tb logs
                                tb_logger.add_scalar(k, v, current_step)
                            # tb_logger.flush()
                    logger.info(message)

                    # start time for next iteration #TODO:skip the validation time from calculation
                    timer.tick()

                # update learning rate
                if model.optGstep and model.optDstep and take_step:
                    model.update_learning_rate(current_step, warmup_iter=opt['train'].get('warmup_iter', -1))
                
                # save latest models and training states every <save_checkpoint_freq> iterations
                if current_step % opt['logger']['save_checkpoint_freq'] == 0 and take_step:
                    if model.swa: 
                        model.save(current_step, opt['logger']['overwrite_chkp'], loader=dataloaders['train'])
                    else:
                        model.save(current_step, opt['logger']['overwrite_chkp'])
                    model.save_training_state(
                        epoch=epoch + (n >= len(dataloaders['train'])),
                        iter_step=current_step,
                        latest=opt['logger']['overwrite_chkp']
                    )
                    logger.info('Models and training states saved.')

                # validation
                if dataloaders.get('val', None) and current_step % opt['train']['val_freq'] == 0 and take_step:
                    val_metrics = metrics.MetricsDict(metrics=opt['train'].get('metrics', None))
                    nlls = []
                    for val_data in dataloaders['val']:
                        
                        model.feed_data(val_data)  # unpack data from data loader
                        model.test()  # run inference
                        if hasattr(model, 'nll'):
                            nll = model.nll if model.nll else 0
                            nlls.append(nll)

                        """
                        Get Visuals
                        """
                        visuals = model.get_current_visuals()  # get image results
                        img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]
                        img_dir = os.path.join(opt['path']['val_images'], img_name)
                        util.mkdir(img_dir)
                        
                        # Save SR images for reference
                        sr_img = None
                        if hasattr(model, 'heats'):  # SRFlow
                            opt['train']['val_comparison'] = False
                            for heat in model.heats:
                                for i in range(model.n_sample):
                                    sr_img = tensor2np(visuals['SR', heat, i], denormalize=opt['datasets']['train']['znorm'])
                                    if opt['train']['overwrite_val_imgs']:
                                        save_img_path = os.path.join(img_dir,
                                                                '{:s}_h{:03d}_s{:d}.png'.format(img_name, int(heat * 100), i))
                                    else:
                                        save_img_path = os.path.join(img_dir,
                                                                '{:s}_{:09d}_h{:03d}_s{:d}.png'.format(img_name,
                                                                                                        current_step,
                                                                                                        int(heat * 100), i))
                                    util.save_img(sr_img, save_img_path)
                        else:  # regular SR
                            sr_img = tensor2np(visuals['SR'], denormalize=opt['datasets']['train']['znorm'])
                            if opt['train']['overwrite_val_imgs']:
                                save_img_path = os.path.join(img_dir, '{:s}.png'.format(img_name))
                            else:
                                save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                            if not opt['train']['val_comparison']:
                                util.save_img(sr_img, save_img_path)
                        assert sr_img is not None

                        # Save GT images for reference
                        gt_img = tensor2np(visuals['HR'], denormalize=opt['datasets']['train']['znorm'])
                        if opt['train']['save_gt']:
                            save_img_path_gt = os.path.join(img_dir,
                                                            '{:s}_GT.png'.format(img_name))
                            if not os.path.isfile(save_img_path_gt):
                                util.save_img(gt_img, save_img_path_gt)

                        # Save LQ images for reference
                        if opt['train']['save_lr']:
                            save_img_path_lq = os.path.join(img_dir,
                                                            '{:s}_LQ.png'.format(img_name))
                            if not os.path.isfile(save_img_path_lq):
                                lq_img = tensor2np(visuals['LR'], denormalize=opt['datasets']['train']['znorm'])
                                util.save_img(lq_img, save_img_path_lq, scale=opt['scale'])

                        # save single images or LQ / SR comparison
                        if opt['train']['val_comparison']:
                            lr_img = tensor2np(visuals['LR'], denormalize=opt['datasets']['train']['znorm'])
                            util.save_img_comp([lr_img, sr_img], save_img_path)
                        # else:
                            # util.save_img(sr_img, save_img_path)

                        """
                        Get Metrics
                        # TODO: test using tensor based metrics (batch) instead of numpy.
                        """
                        val_metrics.calculate_metrics(sr_img, gt_img, crop_size=opt['scale'])  # , only_y=True)

                    avg_metrics = val_metrics.get_averages()
                    if nlls:
                        avg_nll = sum(nlls) / len(nlls)
                    del val_metrics

                    # log
                    logger_m = ''
                    for r in avg_metrics:
                        formatted_res = r['name'].upper() + ': {:.5g}, '.format(r['average'])
                        logger_m += formatted_res
                    if nlls:
                        logger_m += 'avg_nll: {:.4e}  '.format(avg_nll)

                    logger.info('# Validation # ' + logger_m[:-2])
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> '.format(epoch, current_step) + logger_m[:-2])
                    # memory_usage = torch.cuda.memory_allocated()/(1024.0 ** 3) # in GB
                    
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        for r in avg_metrics:
                            tb_logger.add_scalar(r['name'], r['average'], current_step)
                            if nlls:
                                tb_logger.add_scalar('average nll', avg_nll, current_step)
                            # tb_logger.flush()
                            # tb_logger_valid.add_scalar(r['name'], r['average'], current_step)
                            # tb_logger_valid.flush()
                    
                timerData.tick()
            
            timerEpoch.tock()
            logger.info('End of epoch {} / {} \t Time Taken: {:.4f} sec'.format(
                epoch, total_epochs, timerEpoch.get_last_iteration()))

        logger.info('Saving the final model.')
        if model.swa:
            model.save('latest', loader=dataloaders['train'])
        else:
            model.save('latest')
        logger.info('End of training.')

    except KeyboardInterrupt:
        # catch a KeyboardInterrupt and save the model and state to resume later
        if model.swa:
            model.save(current_step, True, loader=dataloaders['train'])
        else:
            model.save(current_step, True)
        model.save_training_state(epoch + (n >= len(dataloaders['train'])), current_step, True)
        logger.info('Training interrupted. Latest models and training states saved.')


def main():
    # parse training options
    opt = parse_options()

    # create the training directory if needed
    dir_check(opt)

    # configure loggers
    loggers = configure_loggers(opt)

    # set random seed
    opt = get_random_seed(opt)

    # resume state or create directories if needed
    resume_state = get_resume_state(opt)

    # if the model does not change and input sizes remain the same during training then there may be benefit 
    # from setting torch.backends.cudnn.benchmark = True, otherwise it may stall training
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.autograd.set_detect_anomaly(True)

    # create dataloaders
    dataloaders, data_params = get_dataloaders(opt)

    # create and setup model: load and print networks; create schedulers/optimizer; init
    model = create_model(opt, step = 0 if resume_state is None else resume_state['iter'])

    # resume training if needed
    steps_states = resume_training(opt, model, resume_state, data_params)

    # start training loop with configured options
    fit(model, opt, dataloaders, steps_states, data_params, loggers)


if __name__ == '__main__':
    main()