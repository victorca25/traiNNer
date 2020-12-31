import glob
import logging
import math
import os
import sys
import time

import torch

from codes.data import create_dataset, create_dataloader
from codes.dataops.common import tensor2np
from codes.models import create_model
from codes.runner import Runner
from codes.utils import metrics
from codes.utils.util import sorted_nicely, mkdir_and_rename, save_img_comp, save_img


class Trainer(Runner):

    def __init__(self, config_path: str):
        super(Trainer).__init__(config_path, trainer=True)

        # prepare state
        resume_state = None
        if self.opt['path']['resume_state']:
            # resume training
            if os.path.isdir(self.opt['path']['resume_state']):
                # get newest state
                self.opt['path']['resume_state'] = sorted_nicely(
                    glob.glob(self.opt['path']['resume_state'] + '/*.state')
                )[-1]
            resume_state = torch.load(self.opt['path']['resume_state'])
            self.logger.info('Resuming training from epoch: %d, iter: %d.', resume_state['epoch'], resume_state['iter'])
        else:
            # training from scratch
            # rotate experiments folder
            mkdir_and_rename(self.opt['path']['experiments_root'])

        # get dataloaders
        datasets = {}
        dataloaders = {}
        for phase, dataset_opt in self.opt['datasets'].items():
            if phase not in ['train', 'val']:
                raise NotImplementedError('Phase [%s] is not recognized.' % phase)
            datasets[phase] = create_dataset(dataset_opt)
            if not datasets[phase]:
                self.logger.error('Dataset [%s] for phase [%s] is empty.', dataset_opt['name'], phase)
                sys.exit(1)
            self.logger.info(
                'Number of {:s} images in [{:s}]: {:,d}'.format(phase, dataset_opt['name'], len(datasets[phase]))
            )
            dataloaders[phase] = create_dataloader(datasets[phase], dataset_opt)
        if not dataloaders:
            self.logger.error("No Dataloader has been created.")
            sys.exit(1)

        # get training data
        total_iters = int(self.opt['train']['niter'])
        batch_size = self.opt['datasets']['train'].get('batch_size', 4)
        batches = int(math.ceil(len(datasets['train']) / batch_size))
        total_epochs = int(math.ceil(total_iters / batches))
        self.logger.info(
            'Total epochs needed: {:,d} ({:,d} batches) for iters {:,d}'.format(
                total_epochs, batches, total_iters
            )
        )

        # create model
        model = create_model(self.opt)

        # resume training
        current_step = 0
        virtual_step = 0
        start_epoch = 0
        virtual_batch_size = 0
        t0 = None
        if resume_state:
            start_epoch = resume_state['epoch']
            current_step = resume_state['iter']
            model.resume_training(resume_state)  # handle optimizers and schedulers
            model.update_schedulers(self.opt['train'])  # update schedulers in case JSON configuration has changed
            del resume_state
            # compute virtual batch size
            virtual_batch_size = self.opt['datasets']['train'].get('virtual_batch_size', batch_size)
            virtual_batch_size = virtual_batch_size if virtual_batch_size > batch_size else batch_size
            if virtual_batch_size > batch_size:
                virtual_step = current_step * virtual_batch_size / batch_size
            else:
                virtual_step = current_step
            # start the iteration time when resuming
            t0 = time.time()

        # training
        self.logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
        try:
            for epoch in range(start_epoch, total_epochs * (virtual_batch_size // batch_size)):
                for n, train_data in enumerate(dataloaders['train'], start=1):

                    if virtual_step == 0:
                        # first iteration start time
                        t0 = time.time()

                    virtual_step += 1
                    take_step = False
                    if virtual_step > 0 and virtual_step * batch_size % virtual_batch_size == 0:
                        current_step += 1
                        take_step = True
                        if current_step > total_iters:
                            break

                    # training
                    model.feed_data(train_data)
                    model.optimize_parameters(virtual_step)
                    # model.update_learning_rate(virtual_step - 1)  # current_step?

                    # log
                    if current_step % self.opt['logger']['print_freq'] == 0 and take_step:
                        # iteration end time
                        t1 = time.time()

                        logs = model.get_current_log()
                        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, i_time: {:.4f} sec.> '.format(
                            epoch, current_step, model.get_current_learning_rate(current_step), (t1 - t0)
                        )
                        for k, v in logs.items():
                            message += '{:s}: {:.4e} '.format(k, v)
                            # tensorboard logger
                            if self.opt['use_tb_logger'] and 'debug' not in self.opt['name']:
                                self.tb_logger.add_scalar(k, v, current_step)
                        self.logger.info(message)

                        # # start time for next iteration
                        # t0 = time.time()

                    # update learning rate
                    if model.optGstep and model.optDstep and take_step:
                        model.update_learning_rate(current_step, warmup_iter=self.opt['train'].get('warmup_iter', -1))

                    # save models and training states (changed to save models before validation)
                    if current_step % self.opt['logger']['save_checkpoint_freq'] == 0 and take_step:
                        if model.swa:
                            model.save(current_step, self.opt['logger']['overwrite_chkp'], loader=dataloaders['train'])
                        else:
                            model.save(current_step, self.opt['logger']['overwrite_chkp'])
                        model.save_training_state(
                            epoch=epoch + (n >= len(dataloaders['train'])),
                            iter_step=current_step,
                            latest=self.opt['logger']['overwrite_chkp']
                        )
                        self.logger.info('Models and training states saved.')

                    # validation
                    if dataloaders['val'] and current_step % self.opt['train']['val_freq'] == 0 and take_step:
                        val_metrics = metrics.MetricsDict(metrics=self.opt['train'].get('metrics', None))
                        for val_data in dataloaders['val']:
                            img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]
                            img_dir = os.path.join(self.opt['path']['val_images'], img_name)
                            os.makedirs(img_dir, exist_ok=True)

                            model.feed_data(val_data)
                            model.test()

                            """
                            Get Visuals
                            """
                            visuals = model.get_current_visuals()
                            sr_img = tensor2np(visuals['SR'], denormalize=self.opt['datasets']['train']['znorm'])
                            gt_img = tensor2np(visuals['HR'], denormalize=self.opt['datasets']['train']['znorm'])

                            # Save SR images for reference
                            if self.opt['train']['overwrite_val_imgs']:
                                save_img_path = os.path.join(img_dir, '%s.png' % img_name)
                            else:
                                save_img_path = os.path.join(img_dir, '%s_%s.png' % (img_name, current_step))

                            # save single images or lr / sr comparison
                            if self.opt['train']['val_comparison']:
                                lr_img = tensor2np(visuals['LR'], denormalize=self.opt['datasets']['train']['znorm'])
                                save_img_comp([lr_img, sr_img], save_img_path)
                            else:
                                save_img(sr_img, save_img_path)

                            """
                            Get Metrics
                            """
                            # TODO: test using tensor based metrics (batch) instead of numpy.
                            crop_size = self.opt['scale']
                            val_metrics.calculate_metrics(sr_img, gt_img, crop_size=crop_size)

                        avg_metrics = val_metrics.get_averages()
                        del val_metrics

                        # log
                        logger_m = ''.join(
                            '{:s}: {:.5g}, '.format(r['name'].upper(), r['average']) for r in avg_metrics
                        )
                        self.logger.info('# Validation # %s', logger_m[:-2])
                        logger_val = logging.getLogger('val')
                        logger_val.info('<epoch:{:3d}, iter:{:8,d}> %s', epoch, current_step, logger_m[:-2])

                        # tensorboard logger
                        if self.opt['use_tb_logger'] and 'debug' not in self.opt['name']:
                            for r in avg_metrics:
                                self.tb_logger.add_scalar(r['name'], r['average'], current_step)

                    if current_step % self.opt['logger']['print_freq'] == 0 and take_step or \
                            (dataloaders['val'] and current_step % self.opt['train']['val_freq'] == 0 and take_step):
                        # reset time for next iteration to skip the validation time from calculation
                        t0 = time.time()

            self.logger.info('Saving the final model.')
            if model.swa:
                model.save('latest', loader=dataloaders['train'])
            else:
                model.save('latest')
            self.logger.info('End of training.')

        except KeyboardInterrupt:
            # catch a KeyboardInterrupt and save the model and state to resume later
            if model.swa:
                model.save(current_step, True, loader=dataloaders['train'])
            else:
                model.save(current_step, True)
            model.save_training_state(epoch + (n >= len(dataloaders['train'])), current_step, True)
            self.logger.info('Training interrupted. Latest models and training states saved.')
