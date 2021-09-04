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


def parse_options(is_train=True):
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to options file.')

    args = parser.parse_args()
    opt = options.parse(args.opt, is_train=is_train)

    return opt


def dir_check(opt):
    if opt['is_train']:
        # starting from scratch, needs to create training directory
        if not opt['path']['resume_state']:
            util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                        and 'pretrain_model' not in key and 'resume' not in key))
    else:
        # create testing directory
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))


def configure_loggers(opt=None):
    tofile = opt.get('logger', {}).get('save_logfile', True)
    if opt['is_train']:
        # config loggers. Before it, the log will not work
        util.get_root_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True, tofile=tofile)
        util.get_root_logger('val', opt['path']['log'], 'val', level=logging.INFO, tofile=tofile)
    else:
        util.get_root_logger(None, opt['path']['log'], 'test', level=logging.INFO, screen=True, tofile=tofile)
    logger = util.get_root_logger()  # 'base'
    logger.info(options.dict2str(opt))

    # initialize tensorboard logger
    tb_logger = None
    if opt.get('use_tb_logger', False) and 'debug' not in opt['name']:
        version = float(torch.__version__[0:3])
        log_dir = os.path.join(opt['path']['root'], 'tb_logger', opt['name'])
        # log_dir = os.path.join(opt['path']['experiments_root'], opt['name'], 'tb')
        # logdir_valid = os.path.join(opt['path']['root'], 'tb_logger', opt['name'] + 'valid')
        # logdir_valid = os.path.join(opt['path']['experiments_root'], opt['name'], 'tb_valid')
        if version >= 1.1:  # PyTorch 1.1
            # official PyTorch tensorboard
            try:
                from torch.utils.tensorboard import SummaryWriter
            except:
                from tensorboardX import SummaryWriter    
        else:
            logger.info('You are using PyTorch {}. Using [tensorboardX].'.format(version))
            from tensorboardX import SummaryWriter
        try:
            # for versions PyTorch > 1.1 and tensorboardX < 1.6
            tb_logger = SummaryWriter(log_dir=log_dir)
            # tb_logger_valid = SummaryWriter(log_dir=logdir_valid)
        except:
            # for version tensorboardX >= 1.7
            tb_logger = SummaryWriter(logdir=log_dir)
            # tb_logger_valid = SummaryWriter(logdir=logdir_valid)
    return {"tb_logger": tb_logger}


def get_resume_state(opt):
    logger = util.get_root_logger()

    # train from scratch OR resume training
    if opt['path']['resume_state']:
        if os.path.isdir(opt['path']['resume_state']):
            resume_state_path = glob.glob(opt['path']['resume_state'] + '/*.state')
            resume_state_path = util.sorted_nicely(resume_state_path)[-1]
        else:
            resume_state_path = opt['path']['resume_state']

        if opt['gpu_ids']:
            resume_state = torch.load(resume_state_path)
        else:
            resume_state = torch.load(resume_state_path,
                            map_location=torch.device('cpu'))

        logger.info('Set [resume_state] to {}'.format(resume_state_path))
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))
        options.check_resume(opt)  # check resume options
    else:  # training from scratch
        resume_state = None
    return resume_state


def get_random_seed(opt):
    logger = util.get_root_logger()
    # set random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
        opt['train']['manual_seed'] = seed
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)
    return opt


def get_dataloaders(opt):
    logger = util.get_root_logger()

    gpu_ids = opt.get('gpu_ids', None)
    gpu_ids = gpu_ids if gpu_ids else []

    # Create datasets and dataloaders
    dataloaders = {}
    data_params = {}
    znorm = {}
    for phase, dataset_opt in opt['datasets'].items():
        if opt['is_train'] and phase not in ['train', 'val']:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

        name = dataset_opt['name']
        dataset = create_dataset(dataset_opt)

        if not dataset:
            raise Exception('Dataset "{}" for phase "{}" is empty.'.format(name, phase))

        dataloaders[phase] = create_dataloader(dataset, dataset_opt, gpu_ids)

        if opt['is_train'] and phase == 'train':
            batch_size = dataset_opt.get('batch_size', 4)
            virtual_batch_size = dataset_opt.get('virtual_batch_size', batch_size)
            virtual_batch_size = virtual_batch_size if virtual_batch_size > batch_size else batch_size
            # train_size = int(math.ceil(len(dataset) / batch_size))
            train_size = len(dataloaders[phase])
            logger.info(
                f'Number of train images: {len(dataset):,d}, '
                f'epoch iters: {train_size:,d}')
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info(
                f'Total epochs needed: {total_epochs:d} for '
                f'iters {total_iters:,d}')
            data_params = {
                "batch_size": batch_size, 
                "virtual_batch_size": virtual_batch_size, 
                "total_iters": total_iters, 
                "total_epochs": total_epochs
            }
            assert dataset is not None
        else:
            logger.info('Number of {:s} images in [{:s}]: {:,d}'.format(phase, name, len(dataset)))
            if phase != 'val':
                znorm[name] = dataset_opt.get('znorm', False)                    

    if not opt['is_train']:
        data_params['znorm'] = znorm

    if not dataloaders:
        raise Exception("No Dataloader has been created.")

    return dataloaders, data_params


def resume_training(opt, model, resume_state, data_params):
    batch_size = data_params['batch_size'] 
    virtual_batch_size = data_params['virtual_batch_size']

    # resume training
    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        virtual_step = current_step * virtual_batch_size / batch_size \
            if virtual_batch_size and virtual_batch_size > batch_size else current_step
        model.resume_training(resume_state)  # handle optimizers and schedulers
        model.update_schedulers(opt['train'])  # updated schedulers in case configuration has changed
        del resume_state
    else:
        start_epoch = 0
        current_step = 0
        virtual_step = 0
    return {"start_epoch": start_epoch, "current_step": current_step, "virtual_step": virtual_step}


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
                timerData.tock()  # timer for data loading per iteration

                virtual_step += 1
                take_step = False
                if (virtual_step > 0 and
                    virtual_step * batch_size % virtual_batch_size == 0):
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
                            # tb_logger.add_scalar('loss/nll', nll, current_step)  # srflow
                            tb_logger.add_scalar('lr/base', model.get_current_learning_rate(), current_step)
                            tb_logger.add_scalar('time/iteration', timer.get_last_iteration(), current_step)
                            tb_logger.add_scalar('time/data', timerData.get_last_iteration(), current_step)
                            # tb_logger.add_scalar('time/eta', eta(timer.get_last_iteration()), current_step)

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
                    model.update_learning_rate(
                        current_step, warmup_iter=opt['train'].get('warmup_iter', -1))

                # save latest models and training states every <save_checkpoint_freq> iterations
                if current_step % opt['logger']['save_checkpoint_freq'] == 0 and take_step:
                    if model.swa: 
                        model.save(
                            current_step, opt['logger']['overwrite_chkp'],
                            loader=dataloaders['train'])
                    else:
                        model.save(
                            current_step, opt['logger']['overwrite_chkp'])
                    model.save_training_state(
                        epoch=epoch + (n >= len(dataloaders['train'])),
                        iter_step=current_step,
                        latest=opt['logger']['overwrite_chkp']
                    )
                    logger.info('Models and training states saved.')

                # validation (for SR and other models with validations)
                if dataloaders.get('val', None) and current_step % opt['train']['val_freq'] == 0 and take_step:
                    val_metrics = metrics.MetricsDict(metrics=opt['train'].get('metrics', None))
                    nlls = []  # srflow
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
                        #     util.save_img(sr_img, save_img_path)

                        """
                        Get Metrics
                        # TODO: test using tensor based metrics (batch) instead of numpy.
                        """
                        val_metrics.calculate_metrics(sr_img, gt_img, crop_size=opt['scale'])  # , only_y=True)

                    avg_metrics = val_metrics.get_averages()
                    if nlls:  # srflow
                        avg_nll = sum(nlls) / len(nlls)
                    del val_metrics

                    # for ReduceLROnPlateau scheduler, get metric average value
                    if opt['train']['lr_scheme'] == 'ReduceLROnPlateau':
                        plateau_metric = opt['train']['plateau_metric']
                        if plateau_metric in avg_metrics:
                            model.metric = avg_metrics[plateau_metric]
                        elif nlls and plateau_metric == 'nll':
                            model.metric = avg_nll

                    # log
                    logger_m = ''.join(f'{met.upper()}: {avgr:.5g}, ' for met, avgr in avg_metrics.items())
                    if nlls:
                        logger_m += 'avg_nll: {:.4e}  '.format(avg_nll)

                    logger.info(f'# Validation # {logger_m[:-2]}')
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> '.format(epoch, current_step) + logger_m[:-2])
                    # memory_usage = torch.cuda.memory_allocated()/(1024.0 ** 3) # in GB

                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        # for r in avg_metrics:
                        for met, avgr in avg_metrics.items():
                            # tb_logger.add_scalar(r['name'], r['average'], current_step)
                            tb_logger.add_scalar(met, avgr, current_step)
                        if nlls:
                            tb_logger.add_scalar('average nll', avg_nll, current_step)
                        # tb_logger.flush()
                        # tb_logger_valid.add_scalar(r['name'], r['average'], current_step)
                        # tb_logger_valid.flush()

                # sampling training data (for image2image translation and others without validation step)
                if opt['train'].get('display_freq', None) and current_step % opt['train']['display_freq'] == 0 and take_step:
                    visuals = model.get_current_visuals()

                    img_name = ''
                    img_dir = opt['path']['disp_images']
                    util.mkdir(img_dir)
                    reala_img = tensor2np(visuals['real_A'], denormalize=opt['datasets']['train']['znorm'])
                    fakeb_img = tensor2np(visuals['fake_B'], denormalize=opt['datasets']['train']['znorm'])
                    realb_img = tensor2np(visuals['real_B'], denormalize=opt['datasets']['train']['znorm'])
                    if 'fake_A' in visuals and 'rec_A' in visuals and 'rec_B' in visuals:
                        fakea_img = tensor2np(visuals['fake_A'], denormalize=opt['datasets']['train']['znorm'])
                        reca_img = tensor2np(visuals['rec_A'], denormalize=opt['datasets']['train']['znorm'])
                        recb_img = tensor2np(visuals['rec_B'], denormalize=opt['datasets']['train']['znorm'])
                        if 'idt_B' in visuals and 'idt_A' in visuals:
                            idta_img = tensor2np(visuals['idt_A'], denormalize=opt['datasets']['train']['znorm'])
                            idtb_img = tensor2np(visuals['idt_B'], denormalize=opt['datasets']['train']['znorm'])
                    if opt['train'].get('overwrite_disp_imgs', None):
                        save_img_path = os.path.join(img_dir, '{:s}.png'.format(\
                            img_name))
                    else:
                        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(\
                            img_name, current_step))
                    if 'fake_A' in visuals and 'rec_A' in visuals and 'rec_B' in visuals:
                        if 'idt_B' in visuals and 'idt_A' in visuals:
                            util.save_img_comp([reala_img, fakeb_img, reca_img, idtb_img, realb_img, fakea_img, recb_img, idta_img], save_img_path)
                        else:
                            util.save_img_comp([reala_img, fakeb_img, reca_img, realb_img, fakea_img, recb_img], save_img_path)
                    else:
                        util.save_img_comp([reala_img, fakeb_img, realb_img], save_img_path)

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
        n = n if 'n' in locals() else 0
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