from __future__ import absolute_import

import os
import logging
from collections import OrderedDict
import torch
import torch.nn as nn

import models.networks as networks
from .base_model import BaseModel
from . import losses

logger = logging.getLogger('base')


class PBRModel(BaseModel):
    def __init__(self, opt):
        super(PBRModel, self).__init__(opt)
        train_opt = opt['train']

        # specify the models you want to load/save to the disk.
        # The training/test scripts will call <BaseModel.save_networks>
        # and <BaseModel.load_networks>
        # for training and testing, a generator 'G' is needed
        self.model_names = ['G']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt, step=step).to(self.device)  # G
        if self.is_train:
            self.netG.train()
            opt_G_nets = [self.netG]
            opt_D_nets = []
            if train_opt['gan_weight']:
                self.model_names.append('D')  # add discriminator to the network list
                self.netD = networks.define_D(opt).to(self.device)  # D
                self.netD.train()
                opt_D_nets.append(self.netD)
        self.load()  # load G, D and other networks if needed

        # define losses, optimizer, scheduler and other components
        if self.is_train:
            # setup network cap
            # define if the generator will have a final
            # capping mechanism in the output
            self.outm = train_opt.get('finalcap', None)

            # setup batch augmentations
            self.setup_batchaug()

            # setup frequency separation
            self.setup_fs()

            # initialize losses
            # generator losses:
            # Generator losses for 3 channel maps: diffuse, albedo and normal:
            # for the losses that don't require high precision (can use half precision)
            self.generatorlosses = losses.GeneratorLoss(opt, self.device)
            # for losses that need high precision (use out of the AMP context)
            self.precisegeneratorlosses = losses.PreciseGeneratorLoss(opt, self.device)
            # TODO: show the configured losses names in logger
            # print(self.generatorlosses.loss_list)

            # Generator losses for 1 channel maps (does not support feature networks like VGG):
            # using new option in the loss builder: allow_featnets = False
            # TODO: does it make sense to make fake 3ch images with the 1ch maps?
            # for the losses that don't require high precision (can use half precision)
            self.generatorlosses1ch = losses.GeneratorLoss(opt, self.device, False)
            # for losses that need high precision (use out of the AMP context)
            self.precisegeneratorlosses1ch = losses.PreciseGeneratorLoss(opt, self.device, False)

            # discriminator loss:
            self.setup_gan()
 
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

    def feed_data(self, data, need_HR=True):
        # LR images
        self.var_L = data['LR'].to(self.device)
        if need_HR:  # train or val
            # HR images
            self.real_H = data['HR'].to(self.device)
            # discriminator references
            input_ref = data.get('ref', data['HR'])
            self.var_ref = input_ref.to(self.device)

        if isinstance(data.get('NO', None), torch.Tensor):
            self.real_NO = data['NO'].to(self.device)
        else:
            self.real_NO = None

        if isinstance(data.get('AL', None), torch.Tensor):
            self.real_AL = data['AL'].to(self.device)
        else:
            self.real_AL = None

        if isinstance(data.get('AO', None), torch.Tensor):
            self.real_AO = data['AO'].to(self.device)
        else:
            self.real_AO = None

        if isinstance(data.get('HE', None), torch.Tensor):
            self.real_HE = data['HE'].to(self.device)
        else:
            self.real_HE = None

        if isinstance(data.get('ME', None), torch.Tensor):
            self.real_ME = data['ME'].to(self.device)
        else:
            self.real_ME = None

        if isinstance(data.get('RE', None), torch.Tensor):
            self.real_RE = data['RE'].to(self.device)
        else:
            self.real_RE = None

        if isinstance(data.get('RO', None), torch.Tensor):
            self.real_RO = data['RO'].to(self.device)
        else:
            self.real_RO = None

    def feed_data_batch(self, data, need_HR=True):
        # LR
        self.var_L = data

    def optimize_parameters(self, step):
        """Calculate losses, gradients, and update network weights;
        called in every training iteration."""
        eff_step = step/self.accumulations

        # G
        # freeze discriminator while generator is trained to prevent BP
        if self.cri_gan:
            self.requires_grad(self.netD, flag=False, net_type='D')

        # # batch (mixup) augmentations
        # if self.mixup:
        #     self.real_H, self.var_L = self.batchaugment(self.real_H, self.var_L)

        ### Network forward, generate SR
        with self.cast():
            if self.outm: #if the model has the final activation option
                self.fake_H = self.netG(self.var_L, outm=self.outm)
            else: #regular models without the final activation option
                self.fake_H = self.netG(self.var_L)
        #/with self.cast():

        fake_SR = self.fake_H[:, 0:3, :, :]
        fake_NO = self.fake_H[:, 3:6, :, :]
        fake_AL = self.fake_H[:, 6:9, :, :]
        fake_AO = self.fake_H[:, 9:10, :, :]
        fake_HE = self.fake_H[:, 10:11, :, :]
        fake_ME = self.fake_H[:, 11:12, :, :]
        fake_RE = self.fake_H[:, 12:13, :, :]
        fake_RO = self.fake_H[:, 13:14, :, :]

        # # apply mask if batchaug == "cutout"
        # if self.mixup:
        #     self.fake_H, self.real_H = self.batchaugment.apply_mask(self.fake_H, self.real_H)

        # calculate and log losses
        loss_results = []
        l_g_total = 0
        # training generator and discriminator
        # update generator (on its own if only training generator or alternatively if training GAN)
        if (self.cri_gan is not True) or (step % self.D_update_ratio == 0 and step > self.D_init_iters):
            with self.cast(): # Casts operations to mixed precision if enabled, else nullcontext
                # regular losses
                log_dict_diffuse = {}
                loss_results, log_dict_diffuse = self.generatorlosses(
                    fake_SR, self.real_H, log_dict_diffuse, self.f_low)
                l_g_total += sum(loss_results)/self.accumulations

                #TODO: for now only showing the logs for the diffuse losses,
                # need to append the other logs
                self.log_dict = log_dict_diffuse

                if isinstance(self.real_NO, torch.Tensor):
                    NO_loss_results = []
                    log_dict_normal = {}
                    NO_loss_results, log_dict_normal = self.generatorlosses(
                        fake_NO, self.real_NO, log_dict_normal, self.f_low)
                    l_g_total += sum(NO_loss_results)/self.accumulations

                if isinstance(self.real_AL, torch.Tensor):
                    AL_loss_results = []
                    log_dict_albedo = {}
                    AL_loss_results, log_dict_albedo = self.generatorlosses(
                        fake_AL, self.real_AL, log_dict_albedo, self.f_low)
                    l_g_total += sum(AL_loss_results)/self.accumulations

                if isinstance(self.real_AO, torch.Tensor):
                    AO_loss_results = []
                    log_dict_ao = {}
                    AO_loss_results, log_dict_ao = self.generatorlosses1ch(
                        fake_AO, self.real_AO, log_dict_ao, self.f_low)
                    l_g_total += sum(AO_loss_results)/self.accumulations

                if isinstance(self.real_HE, torch.Tensor):
                    HE_loss_results = []
                    log_dict_height = {}
                    HE_loss_results, log_dict_height = self.generatorlosses1ch(
                        fake_HE, self.real_HE, log_dict_height, self.f_low)
                    l_g_total += sum(HE_loss_results)/self.accumulations

                if isinstance(self.real_ME, torch.Tensor):
                    ME_loss_results = []
                    log_dict_metalness = {}
                    ME_loss_results, log_dict_metalness = self.generatorlosses1ch(
                        fake_ME, self.real_ME, log_dict_metalness, self.f_low)
                    l_g_total += sum(ME_loss_results)/self.accumulations

                if isinstance(self.real_RE, torch.Tensor):
                    RE_loss_results = []
                    log_dict_reflection = {}
                    RE_loss_results, log_dict_reflection = self.generatorlosses1ch(
                        fake_RE, self.real_RE, log_dict_reflection, self.f_low)
                    l_g_total += sum(RE_loss_results)/self.accumulations

                if isinstance(self.real_RO, torch.Tensor):
                    RO_loss_results = []
                    log_dict_roughness = {}
                    RO_loss_results, log_dict_roughness = self.generatorlosses1ch(
                        fake_RO, self.real_RO, log_dict_roughness, self.f_low)
                    l_g_total += sum(RO_loss_results)/self.accumulations

                #TODO: for now only one GAN for the diffuse image, can have one for each map
                if self.cri_gan:
                    # adversarial loss
                    l_g_gan = self.adversarial(
                        fake_SR, self.var_ref, netD=self.netD, 
                        stage='generator', fsfilter = self.f_high)  # (sr, hr)
                    self.log_dict['l_g_gan'] = l_g_gan.item()
                    l_g_total += l_g_gan/self.accumulations

            #/with self.cast():

            # high precision generator losses (can be affected by AMP half precision)
            #TODO: for now only showing the logs for the diffuse losses, need to append the other logs
            if self.precisegeneratorlosses.loss_list and self.precisegeneratorlosses1ch.loss_list:
                precise_loss_results, self.log_dict = self.precisegeneratorlosses(
                    fake_SR, self.real_H, self.log_dict, self.f_low)
                l_g_total += sum(precise_loss_results)/self.accumulations

                if isinstance(self.real_NO, torch.Tensor):
                    NO_loss_results = []
                    NO_loss_results, log_dict_normal = self.precisegeneratorlosses(
                        fake_NO, self.real_NO, log_dict_normal, self.f_low)
                    l_g_total += sum(NO_loss_results)/self.accumulations

                if isinstance(self.real_AL, torch.Tensor):
                    AL_loss_results = []
                    AL_loss_results, log_dict_albedo = self.precisegeneratorlosses(
                        fake_AL, self.real_AL, log_dict_albedo, self.f_low)
                    l_g_total += sum(AL_loss_results)/self.accumulations

                if isinstance(self.real_AO, torch.Tensor):
                    AO_loss_results = []
                    AO_loss_results, log_dict_ao = self.precisegeneratorlosses1ch(
                        fake_AO, self.real_AO, log_dict_ao, self.f_low)
                    l_g_total += sum(AO_loss_results)/self.accumulations

                if isinstance(self.real_HE, torch.Tensor):
                    HE_loss_results = []
                    HE_loss_results, log_dict_height = self.precisegeneratorlosses1ch(
                        fake_HE, self.real_HE, log_dict_height, self.f_low)
                    l_g_total += sum(HE_loss_results)/self.accumulations

                if isinstance(self.real_ME, torch.Tensor):
                    ME_loss_results = []
                    ME_loss_results, log_dict_metalness = self.precisegeneratorlosses1ch(
                        fake_ME, self.real_ME, log_dict_metalness, self.f_low)
                    l_g_total += sum(ME_loss_results)/self.accumulations

                if isinstance(self.real_RE, torch.Tensor):
                    RE_loss_results = []
                    RE_loss_results, log_dict_reflection = self.precisegeneratorlosses1ch(
                        fake_RE, self.real_RE, log_dict_reflection, self.f_low)
                    l_g_total += sum(RE_loss_results)/self.accumulations

                if isinstance(self.real_RO, torch.Tensor):
                    RO_loss_results = []
                    RO_loss_results, log_dict_roughness = self.precisegeneratorlosses1ch(
                        fake_RO, self.real_RO, log_dict_roughness, self.f_low)
                    l_g_total += sum(RO_loss_results)/self.accumulations

            # calculate G gradients
            self.calc_gradients(l_g_total)

            # step G optimizer
            self.optimizer_step(step, self.optimizer_G, "G")

        #TODO: for now only one GAN for the diffuse image, can have one for each map
        if self.cri_gan:
            # update discriminator
            # unfreeze discriminator
            for p in self.netD.parameters():
                p.requires_grad = True
            l_d_total = 0

            with self.cast(): # Casts operations to mixed precision if enabled, else nullcontext
                l_d_total, gan_logs = self.adversarial(
                    fake_SR, self.var_ref, netD=self.netD,
                    stage='discriminator', fsfilter = self.f_high) # (sr, hr)

                for g_log in gan_logs:
                    self.log_dict[g_log] = gan_logs[g_log]

                l_d_total /= self.accumulations
            #/with autocast():

            # calculate D gradients
            self.calc_gradients(l_d_total)

            # step D optimizer
            self.optimizer_step(step, self.optimizer_D, "D")

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.is_train:
                self.fake_H = self.netG(self.var_L)
            else:
                #self.fake_H = self.netG(self.var_L, isTest=True)
                self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def get_current_visuals_batch(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach().float().cpu()
        out_dict['SR'] = self.fake_H.detach().float().cpu()
        if need_HR:
            out_dict['HR'] = self.real_H.detach().float().cpu()
        return out_dict
