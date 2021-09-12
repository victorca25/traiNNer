from __future__ import absolute_import

import os
import logging
from collections import OrderedDict
import torch
import torch.nn as nn

import models.networks as networks
from .base_model import BaseModel
from . import losses
from dataops.filters import FilterHigh, FilterLow

logger = logging.getLogger('base')


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

        # specify the images you want to save/display.
        # The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        # specify the models you want to load/save to the disk.
        # The training/test scripts will call <BaseModel.save_networks>
        # and <BaseModel.load_networks>
        # for training and testing, a generator 'G' is needed
        self.model_names = ['G']

        # define networks (both generator and discriminator) and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netG.train()
            opt_G_nets = [self.netG]
            opt_D_nets = []
            if train_opt['gan_weight']:
                self.model_names.append('D')  # add discriminator to the network list
                # define a discriminator; conditional GANs need to take both input and output images;
                # Therefore, input channels for D must be input_nc + output_nc
                self.netD = networks.define_D(opt).to(self.device)  # D
                self.netD.train()
                opt_D_nets.append(self.netD)

            # configure AdaTarget
            self.setup_atg()
            if self.atg:
                opt_G_nets.append(self.netLoc)
        self.load()  # load G, D and other networks if needed

        # define losses, optimizer, scheduler and other components
        if self.is_train:
            # setup batch augmentations
            self.setup_batchaug()

            # setup frequency separation
            self.setup_fs()

            # initialize losses
            # generator losses:
            self.generatorlosses = losses.GeneratorLoss(opt, self.device)

            # TODO: show the configured losses names in logger
            # print(self.generatorlosses.loss_list)

            # discriminator loss:
            self.setup_gan(conditional=True)

            # configure FreezeD
            if self.cri_gan:
                self.setup_freezeD()

            # prepare optimizers
            self.setup_optimizers(opt_G_nets, opt_D_nets, init_setup=True)

            # prepare schedulers
            self.setup_schedulers()

            # set gradients to zero
            self.optimizer_G.zero_grad()
            if self.cri_gan:
                self.optimizer_D.zero_grad()

            # init loss log
            self.log_dict = OrderedDict()

            # configure SWA
            self.setup_swa()

            # configure virtual batch
            self.setup_virtual_batch()

            # configure AMP
            self.setup_amp()

        # print network
        # TODO: pass verbose flag from config file
        self.print_network(verbose=False)

    def feed_data(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            data (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # TODO: images currently being flipped with BtoA during read, check logic
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
        """Calculate GAN loss for the discriminator."""
        self.log_dict = self.backward_D_Basic(
            self.netD, self.real_B, self.fake_B, self.log_dict,
            self.real_A)

    def backward_G(self):
        """Calculate GAN and reconstruction losses for the generator."""
        l_g_total = 0
        with self.cast():
            if self.cri_gan:
                # First, G(A) should fake the discriminator
                # adversarial loss
                l_g_gan = self.adversarial(
                    self.fake_B, condition=self.real_A, netD=self.netD,
                    stage='generator', fsfilter=self.f_high)
                self.log_dict['l_g_gan'] = l_g_gan.item()
                l_g_total += l_g_gan / self.accumulations

            # Second, G(A) = B, calculate losses
            loss_results = []
            loss_results, self.log_dict = self.generatorlosses(
                self.fake_B, self.real_B, self.log_dict, self.f_low)
            l_g_total += sum(loss_results) / self.accumulations

        # high precision generator losses (can be affected by AMP half precision)
        if self.generatorlosses.precise_loss_list:
            loss_results, self.log_dict = self.generatorlosses(
                self.fake_B, self.real_B, self.log_dict, self.f_low,
                precise=True)
            l_g_total += sum(loss_results) / self.accumulations

        # calculate G gradients
        self.calc_gradients(l_g_total)

    def optimize_parameters(self, step):
        """Calculate losses, gradients, and update network weights;
        called in every training iteration."""
        eff_step = step/self.accumulations

        # switch ATG to train
        if self.atg:
            if eff_step > self.atg_start_iter:
                self.switch_atg(True)
            else:
                self.switch_atg(False)

        # batch (mixup) augmentations
        if self.mixup:
            self.real_B, self.real_A = self.batchaugment(self.real_B, self.real_A)

        # run G(A)
        with self.cast():  # Casts operations to mixed precision if enabled, else nullcontext
            self.forward()  # compute fake images: G(A)

        # apply mask if batchaug == "cutout"
        if self.mixup:
            self.fake_B, self.real_B = self.batchaugment.apply_mask(self.fake_B, self.real_B)

        # adatarget
        if self.atg:
            self.fake_H = self.ada_out(
                output=self.fake_H, target=self.real_H,
                loc_model=self.netLoc)

        if self.cri_gan:
            # update D
            self.requires_grad(self.netD, flag=True)  # enable backprop for D
            if isinstance(self.feature_loc, int):
                # freeze up to the selected layers
                for loc in range(self.feature_loc):
                    self.requires_grad(
                        self.netD, False, target_layer=loc, net_type='D')

            # calculate D backward step and gradients
            self.backward_D()

            # step D optimizer
            self.optimizer_step(step, self.optimizer_D, "D")

        if (self.cri_gan is not True) or (eff_step % self.D_update_ratio == 0
            and eff_step > self.D_init_iters):
            # update G
            if self.cri_gan:
                # D requires no gradients when optimizing G
                self.requires_grad(self.netD, flag=False, net_type='D')

            # calculate G backward step and gradients
            self.backward_G()

            # step G optimizer
            self.optimizer_step(step, self.optimizer_G, "G")

    def get_current_log(self):
        """Return traning losses / errors. train.py will print out
        these on the console, and save them to a file."""
        return self.log_dict

    def get_current_visuals(self):
        """Return visualization images. train.py will display and/or save these images."""
        out_dict = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                out_dict[name] = getattr(self, name).detach()[0].float().cpu()
        return out_dict
