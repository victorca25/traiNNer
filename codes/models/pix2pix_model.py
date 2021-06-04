from __future__ import absolute_import

import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn

import models.networks as networks
from .base_model import BaseModel, nullcast

logger = logging.getLogger('base')

from . import losses
from . import optimizers
from . import schedulers
from . import swa

from dataops.batchaug import BatchAug
from dataops.filters import FilterHigh, FilterLow #, FilterX

load_amp = (hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"))
if load_amp:
    from torch.cuda.amp import autocast, GradScaler
    logger.info('AMP library available')
else:
    logger.info('AMP library not available')


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model for learning a mapping 
    from input images to output images given paired data.
    The model training uses by default:
        netG: unet256 U-Net generator (unet_net, num_downs: 8)
        netD: basic discriminator (PatchGAN)
        gan_type: vanilla GAN loss (the cross-entropy objective used
            in the orignal GAN paper)
        norm: batch (batchnorm)
        dataset_mode: aligned
        L1 weight for pixel loss: 100.0 (lambda_L1)
        pool_size: 0 (image buffer not used)
        lr: 0.0002
        lr_policy: linear

    The original training objective is:
        GAN Loss + lambda_L1 * ||G(A)-B||_1

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option dictionary): stores all the experiment flags
        """
        super(Pix2PixModel, self).__init__(opt)
        train_opt = opt['train']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        
        # specify the models you want to load/save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # For training and testing, a generator 'G' is needed
        self.model_names = ['G']

        # define networks (both generator and discriminator) and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        
        if self.is_train:
            self.netG.train()
            if train_opt['gan_weight']:
                self.model_names.append('D') # add discriminator to the network list
                # define a discriminator; conditional GANs need to take both input and output images;
                # Therefore, input channels for D must be input_nc + output_nc
                self.netD = networks.define_D(opt).to(self.device)  # D
                self.netD.train()
        self.load()  # load G and D if needed

        if self.is_train:
            """
            Setup batch augmentations
            #TODO: test
            """
            self.mixup = train_opt.get('mixup', None)
            if self.mixup:
                self.mixopts = train_opt.get('mixopts', ["blend", "rgb", "mixup", "cutmix", "cutmixup"]) #, "cutout", "cutblur"]
                self.mixprob = train_opt.get('mixprob', [1.0, 1.0, 1.0, 1.0, 1.0]) #, 1.0, 1.0]
                self.mixalpha = train_opt.get('mixalpha', [0.6, 1.0, 1.2, 0.7, 0.7]) #, 0.001, 0.7]
                self.aux_mixprob = train_opt.get('aux_mixprob', 1.0)
                self.aux_mixalpha = train_opt.get('aux_mixalpha', 1.2)
                self.mix_p = train_opt.get('mix_p', None)

            """
            Setup frequency separation
            #TODO: test
            """
            self.fs = train_opt.get('fs', None)
            self.f_low = None
            self.f_high = None
            if self.fs:
                lpf_type = train_opt.get('lpf_type', "average")
                hpf_type = train_opt.get('hpf_type', "average")
                self.f_low = FilterLow(filter_type=lpf_type).to(self.device)
                self.f_high = FilterHigh(filter_type=hpf_type).to(self.device)

            """
            Initialize losses
            """
            #Initialize the losses with the opt parameters
            # Generator losses:
            # for the losses that don't require high precision (can use half precision)
            self.generatorlosses = losses.GeneratorLoss(opt, self.device)
            # for losses that need high precision (use out of the AMP context)
            self.precisegeneratorlosses = losses.PreciseGeneratorLoss(opt, self.device)
            # TODO: show the configured losses names in logger
            # print(self.generatorlosses.loss_list)

            # Discriminator loss:
            if train_opt['gan_type'] and train_opt['gan_weight']:
                self.cri_gan = True
                diffaug = train_opt.get('diffaug', None)
                dapolicy = None
                if diffaug: #TODO: this if should not be necessary
                    dapolicy = train_opt.get('dapolicy', 'color,translation,cutout') #original
                self.adversarial = losses.Adversarial(train_opt=train_opt, device=self.device,
                                                      diffaug=diffaug, dapolicy=dapolicy,
                                                      conditional=True)
                #TODO:
                # D_update_ratio and D_init_iters are for WGAN
                # self.D_update_ratio = train_opt.get('D_update_ratio', 1)
                # self.D_init_iters = train_opt.get('D_init_iters', 0)
            else:
                self.cri_gan = False

            """
            Initialize optimizers
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

            """
            Configure SWA
            #TODO: test
            """
            self.swa = opt.get('use_swa', False)
            if self.swa:
                self.swa_start_iter = train_opt.get('swa_start_iter', 0)
                # self.swa_start_epoch = train_opt.get('swa_start_epoch', None)
                swa_lr = train_opt.get('swa_lr', 0.0001)
                swa_anneal_epochs = train_opt.get('swa_anneal_epochs', 10)
                swa_anneal_strategy = train_opt.get('swa_anneal_strategy', 'cos')
                #TODO: Note: This could be done in resume_training() instead, to prevent creating
                # the swa scheduler and model before they are needed
                self.swa_scheduler, self.swa_model = swa.get_swa(
                        self.optimizer_G, self.netG, swa_lr, swa_anneal_epochs, swa_anneal_strategy)
                self.load_swa() #load swa from resume state
                logger.info('SWA enabled. Starting on iter: {}, lr: {}'.format(self.swa_start_iter, swa_lr))

            """
            If using virtual batch
            """
            batch_size = opt["datasets"]["train"]["batch_size"]
            virtual_batch = opt["datasets"]["train"].get('virtual_batch_size', None)
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
                self.feature_loc = None
                loc = train_opt.get('freeze_loc', False)
                if loc:
                    disc = opt["network_D"].get('which_model_D', False)
                    if "discriminator_vgg" in disc and "fea" not in disc:
                        loc = (loc*3)-2
                    elif "patchgan" in disc:
                        loc = (loc*3)-1
                    #TODO: TMP, for now only tested with the vgg-like or patchgan discriminators
                    if "discriminator_vgg" in disc or "patchgan" in disc:
                        self.feature_loc = loc
                        logger.info('FreezeD enabled')
            
            self.log_dict = OrderedDict()

        self.print_network(verbose=False) #TODO: pass verbose flag from config file

    def feed_data(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            data (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        #TODO: images currently being flipped with BtoA during read, check logic
        # AtoB = self.opt.get('direction') == 'AtoB'
        # self.real_A = data['A' if AtoB else 'B'].to(self.device)
        # self.real_B = data['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = data['A_path' if AtoB else 'B_path']
        self.real_A = data['A'].to(self.device)
        self.real_B = data['B'].to(self.device)
        self.image_paths = data['A_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        
        l_d_total = 0
        with self.cast():
            l_d_total, gan_logs = self.adversarial(
                self.fake_B, self.real_A, self.real_B, netD=self.netD,
                stage='discriminator', fsfilter=self.f_high)

            for g_log in gan_logs:
                self.log_dict[g_log] = gan_logs[g_log]

            l_d_total /= self.accumulations

        # calculate gradients
        if self.amp:
            # call backward() on scaled loss to create scaled gradients.
            self.amp_scaler.scale(l_d_total).backward()
        else:
            l_d_total.backward()

    def backward_G(self):
        """Calculate GAN and reconstruction losses for the generator"""
        
        l_g_total = 0
        with self.cast():
            if self.cri_gan:
                # First, G(A) should fake the discriminator
                # adversarial loss
                l_g_gan = self.adversarial(
                    self.fake_B, self.real_A, netD=self.netD,
                    stage='generator', fsfilter=self.f_high)
                self.log_dict['l_g_gan'] = l_g_gan.item()
                l_g_total += l_g_gan / self.accumulations

            # Second, G(A) = B, calculate losses
            loss_results = []
            loss_results, self.log_dict = self.generatorlosses(
                self.fake_B, self.real_B, self.log_dict, self.f_low)
            l_g_total += sum(loss_results) / self.accumulations

        # high precision generator losses (can be affected by AMP half precision)
        if self.precisegeneratorlosses.loss_list:
            precise_loss_results, self.log_dict = self.precisegeneratorlosses(
                    self.fake_B, self.real_B, self.log_dict, self.f_low)
            l_g_total += sum(precise_loss_results) / self.accumulations

        # calculate gradients
        if self.amp:
            # call backward() on scaled loss to create scaled gradients.
            self.amp_scaler.scale(l_g_total).backward()
        else:
            l_g_total.backward()

    def optimize_parameters(self, step):
        # batch (mixup) augmentations
        aug = None
        if self.mixup:
            self.real_B, self.real_A, mask, aug = BatchAug(
                self.real_B, self.real_A,
                self.mixopts, self.mixprob, self.mixalpha,
                self.aux_mixprob, self.aux_mixalpha, self.mix_p
                )

        # run G(A)
        with self.cast(): # Casts operations to mixed precision if enabled, else nullcontext
            self.forward()  # compute fake images: G(A)

        # batch (mixup) augmentations
        # cutout-ed pixels are discarded when calculating loss by masking removed pixels
        if aug == "cutout":
            self.fake_B, self.real_B = self.fake_B*mask, self.real_B*mask

        if self.cri_gan:
            # update D
            self.requires_grad(self.netD, True)  # enable backprop for D
            if isinstance(self.feature_loc, int):
                # freeze up to the selected layers
                for loc in range(self.feature_loc):
                    self.requires_grad(self.netD, False, target_layer=loc, net_type='D')
            
            self.backward_D()  # calculate gradients for D
            # only step and clear gradient if virtual batch has completed
            if (step + 1) % self.accumulations == 0:
                if self.amp:
                    self.amp_scaler.step(self.optimizer_D)
                    self.amp_scaler.update()
                else:
                    self.optimizer_D.step()  # update D's weights
                self.optimizer_D.zero_grad()  # set D's gradients to zero
                self.optDstep = True

        # update G
        if self.cri_gan:
            # D requires no gradients when optimizing G
            self.requires_grad(self.netD, flag=False, net_type='D')
            
        self.backward_G()  # calculate graidents for G
        # only step and clear gradient if virtual batch has completed
        if (step + 1) % self.accumulations == 0:
            if self.amp:
                self.amp_scaler.step(self.optimizer_G)
                self.amp_scaler.update()
            else:
                self.optimizer_G.step()  # udpdate G's weights
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.optGstep = True

    def get_current_log(self):
        """Return traning losses / errors. train.py will print out these on the
            console, and save them to a file"""
        return self.log_dict

    def get_current_visuals(self):
        """Return visualization images. train.py will display and/or save these images"""
        out_dict = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                out_dict[name] = getattr(self, name).detach()[0].float().cpu()
        return out_dict

