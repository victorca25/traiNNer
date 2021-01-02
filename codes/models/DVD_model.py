from __future__ import absolute_import
from . import swa
from . import schedulers
from . import optimizers
from . import losses

import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn

import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')


load_amp = (hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"))
if load_amp:
    from torch.cuda.amp import autocast, GradScaler
    logger.info('AMP library available')
else:
    logger.info('AMP library not available')


class nullcast():
    #nullcontext:
    #https://github.com/python/cpython/commit/0784a2e5b174d2dbf7b144d480559e650c5cf64c
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *excinfo):
        pass


class DVDModel(BaseModel):
    def __init__(self, opt):
        super(DVDModel, self).__init__(opt)
        train_opt = opt['train']

        # specify the models you want to load/save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.is_train: # print/save/load both G and D during training
            self.model_names = ['G', 'D']
        else:  # during test time, only print/load G
            self.model_names = ['G']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netG.train()
            if train_opt['gan_weight']:
                self.netD = networks.define_D(opt).to(self.device)  # D
                self.netD.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            """
            Setup network cap
            """
            # define if the generator will have a final capping mechanism in the output
            self.outm = train_opt.get('finalcap', None)

            """
            Initialize losses
            """
            #Initialize the losses with the opt parameters
            # Generator losses:
            # for the losses that don't require high precision (can use half precision)
            self.generatorlosses = losses.GeneratorLoss(opt, self.device)
            # for losses that need high precision (use out of the AMP context)
            self.precisegeneratorlosses = losses.PreciseGeneratorLoss(
                opt, self.device)
            # TODO: show the configured losses names in logger
            # print(self.generatorlosses.loss_list)

            # Discriminator loss:
            if train_opt['gan_type'] and train_opt['gan_weight']:
                self.cri_gan = True
                diffaug = train_opt.get('diffaug', None)
                dapolicy = None
                if diffaug:  # TODO: this if should not be necessary
                    dapolicy = train_opt.get(
                        'dapolicy', 'color,translation,cutout')  # original
                self.adversarial = losses.Adversarial(
                    train_opt=train_opt, device=self.device, diffaug=diffaug, dapolicy=dapolicy)
                # D_update_ratio and D_init_iters are for WGAN
                self.D_update_ratio = train_opt.get('D_update_ratio', 1)
                self.D_init_iters = train_opt.get('D_init_iters', 0)
            else:
                self.cri_gan = False

            """
            Prepare optimizers
            """
            self.optGstep = False
            self.optDstep = False
            if self.cri_gan:
                self.optimizers, self.optimizer_G, self.optimizer_D = optimizers.get_optimizers(
                    self.cri_gan, self.netD, self.netG, train_opt, logger, self.optimizers)
            else:
                self.optimizers, self.optimizer_G = optimizers.get_optimizers(
                    None, None, self.netG, train_opt, logger, self.optimizers)
                self.optDstep = True

            """
            Prepare schedulers
            """
            self.schedulers = schedulers.get_schedulers(
                optimizers=self.optimizers, schedulers=self.schedulers, train_opt=train_opt)

            #Keep log in loss class instead?
            self.log_dict = OrderedDict()

            """
            Configure SWA
            """
            #https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
            self.swa = opt.get('use_swa', False)
            if self.swa:
                self.swa_start_iter = train_opt.get('swa_start_iter', 0)
                # self.swa_start_epoch = train_opt.get('swa_start_epoch', None)
                swa_lr = train_opt.get('swa_lr', 0.0001)
                swa_anneal_epochs = train_opt.get('swa_anneal_epochs', 10)
                swa_anneal_strategy = train_opt.get(
                    'swa_anneal_strategy', 'cos')
                #TODO: Note: This could be done in resume_training() instead, to prevent creating
                # the swa scheduler and model before they are needed
                self.swa_scheduler, self.swa_model = swa.get_swa(
                    self.optimizer_G, self.netG, swa_lr, swa_anneal_epochs, swa_anneal_strategy)
                self.load_swa()  # load swa from resume state
                logger.info('SWA enabled. Starting on iter: {}, lr: {}'.format(
                    self.swa_start_iter, swa_lr))

            """
            If using virtual batch
            """
            batch_size = opt["datasets"]["train"]["batch_size"]
            virtual_batch = opt["datasets"]["train"].get(
                'virtual_batch_size', None)
            self.virtual_batch = virtual_batch if virtual_batch \
                >= batch_size else batch_size
            self.accumulations = self.virtual_batch // batch_size
            self.optimizer_G.zero_grad()
            if self.cri_gan:
                self.optimizer_D.zero_grad()

            """
            Configure AMP
            """
            self.amp = load_amp and opt.get('use_amp', False)
            if self.amp:
                self.cast = autocast
                self.amp_scaler = GradScaler()
                logger.info('AMP enabled')
            else:
                self.cast = nullcast

            """
            Configure FreezeD
            """
            if self.cri_gan:
                loc = train_opt.get('freeze_loc', False)
                disc = opt["network_D"].get('which_model_D', False)
                if "discriminator_vgg" in disc and "fea" not in disc:
                    loc = (loc*3)-2
                elif "patchgan" in disc:
                    loc = (loc*3)-1
                #TODO: TMP, for now only tested with the vgg-like or patchgan discriminators
                if "discriminator_vgg" in disc or "patchgan" in disc:
                    self.feature_loc = loc
                    logger.info('FreezeD enabled')
                else:
                    self.feature_loc = None

        # print network
        """ 
        TODO:
        Network summary? Make optional with parameter
            could be an selector between traditional print_network() and summary()
        """
        self.print_network(verbose=False) #TODO: pass verbose flag from config file

    def feed_data(self, data, need_HR=True):
        # LR images
        self.var_I = data['in'].to(self.device)
        if need_HR:  # train or val
            # Top images
            self.var_T = data['top'].to(self.device)
            self.var_B = data['bottom'].to(self.device)
            # discriminator references
            self.var_ref_T = data['top'].to(self.device)
            self.var_ref_B = data['bottom'].to(self.device)

    def feed_data_batch(self, data, need_HR=True):
        # LR
        self.var_I = data

    def optimize_parameters(self, step):
        # G
        # freeze discriminator while generator is trained to prevent BP
        if self.cri_gan:
            self.requires_grad(self.netD, flag=False, net_type='D')

        ### Network forward, generate progressive frames
        with self.cast():
            self.fake_T, self.fake_B = self.netG(self.var_I)
        #/with self.cast():

        l_g_total = 0
        """
        Calculate and log losses
        """
        loss_results = []
        # training generator and discriminator
        # update generator (on its own if only training generator or alternatively if training GAN)
        if (self.cri_gan is not True) or (step % self.D_update_ratio == 0 and step > self.D_init_iters):
            with self.cast():  # Casts operations to mixed precision if enabled, else nullcontext
                # regular losses
                loss_results, self.log_dict = self.generatorlosses(
                    self.fake_T, self.var_T, self.log_dict)
                l_g_total += sum(loss_results)/self.accumulations

                loss_results, self.log_dict = self.generatorlosses(
                    self.fake_B, self.var_B, self.log_dict)
                l_g_total += sum(loss_results)/self.accumulations

                if self.cri_gan:
                    # adversarial loss
                    l_g_gan = self.adversarial(
                        self.fake_T, self.var_ref_T, netD=self.netD,
                        stage='generator')
                    self.log_dict['l_g_gan_t'] = l_g_gan.item()
                    l_g_total += l_g_gan/self.accumulations

                    l_g_gan = self.adversarial(
                        self.fake_B, self.var_ref_B, netD=self.netD,
                        stage='generator')
                    self.log_dict['l_g_gan_b'] = l_g_gan.item()
                    l_g_total += l_g_gan/self.accumulations

            #/with self.cast():
            # high precision generator losses (can be affected by AMP half precision)
            if self.precisegeneratorlosses.loss_list:
                precise_loss_results, self.log_dict = self.precisegeneratorlosses(
                    self.fake_T, self.var_T, self.log_dict)
                l_g_total += sum(precise_loss_results)/self.accumulations

                precise_loss_results, self.log_dict = self.precisegeneratorlosses(
                    self.fake_B, self.var_B, self.log_dict)
                l_g_total += sum(precise_loss_results)/self.accumulations

            if self.amp:
                # call backward() on scaled loss to create scaled gradients.
                self.amp_scaler.scale(l_g_total).backward()
            else:
                l_g_total.backward()

            # only step and clear gradient if virtual batch has completed
            if (step + 1) % self.accumulations == 0:
                if self.amp:
                    # unscale gradients of the optimizer's params, call
                    # optimizer.step() if no infs/NaNs in gradients, else, skipped
                    self.amp_scaler.step(self.optimizer_G)
                    # Update GradScaler scale for next iteration.
                    self.amp_scaler.update()
                    #TODO: remove. for debugging AMP
                    #print("AMP Scaler state dict: ", self.amp_scaler.state_dict())
                else:
                    self.optimizer_G.step()
                self.optimizer_G.zero_grad()
                self.optGstep = True

        if self.cri_gan:
            # update discriminator
            if isinstance(self.feature_loc, int):
                # unfreeze all D
                self.requires_grad(self.netD, flag=True)
                # then freeze up to the selected layers
                for loc in range(self.feature_loc):
                    self.requires_grad(
                        self.netD, False, target_layer=loc, net_type='D')
            else:
                # unfreeze discriminator
                self.requires_grad(self.netD, flag=True)

            l_d_total = 0

            with self.cast():  # Casts operations to mixed precision if enabled, else nullcontext
                l_d_total, gan_logs = self.adversarial(
                    self.fake_T, self.var_ref_T, netD=self.netD,
                    stage='discriminator')
                
                l_d_total, gan_logs = self.adversarial(
                    self.fake_B, self.var_ref_B, netD=self.netD,
                    stage='discriminator')

                for g_log in gan_logs:
                    self.log_dict[g_log] = gan_logs[g_log]

                l_d_total /= self.accumulations
            #/with autocast():

            if self.amp:
                # call backward() on scaled loss to create scaled gradients.
                self.amp_scaler.scale(l_d_total).backward()
            else:
                l_d_total.backward()

            # only step and clear gradient if virtual batch has completed
            if (step + 1) % self.accumulations == 0:
                if self.amp:
                    # unscale gradients of the optimizer's params, call
                    # optimizer.step() if no infs/NaNs in gradients, else, skipped
                    self.amp_scaler.step(self.optimizer_D)
                    # Update GradScaler scale for next iteration.
                    self.amp_scaler.update()
                else:
                    self.optimizer_D.step()
                self.optimizer_D.zero_grad()
                self.optDstep = True

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.is_train:
                self.fake_T, self.fake_B = self.netG(self.var_I)
            else:
                #self.fake_H = self.netG(self.var_L, isTest=True)
                self.fake_T, self.fake_B = self.netG(self.var_I)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['interlaced'] = self.var_I.detach()[0].float().cpu()
        out_dict['top_fake'] = self.fake_T.detach()[0].float().cpu()
        out_dict['bottom_fake'] = self.fake_B.detach()[0].float().cpu()
        if need_HR:
            out_dict['top_real'] = self.var_T.detach()[0].float().cpu()
            out_dict['bot_real'] = self.var_B.detach()[0].float().cpu()
        return out_dict

    def get_current_visuals_batch(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['interlaced'] = self.var_I.detach().float().cpu()
        out_dict['top_fake'] = self.fake_T.detach().float().cpu()
        out_dict['bottom_fake'] = self.fake_B.detach().float().cpu()
        if need_HR:
            out_dict['top_real'] = self.var_T.detach().float().cpu()
            out_dict['bot_real'] = self.var_B.detach().float().cpu()
        return out_dict
