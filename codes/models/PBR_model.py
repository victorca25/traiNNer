from __future__ import absolute_import

import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn

import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')

from . import losses
from . import optimizers
from . import schedulers

from dataops.batchaug import BatchAug
from dataops.filters import FilterHigh, FilterLow #, FilterX

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


class PBRModel(BaseModel):
    def __init__(self, opt):
        super(PBRModel, self).__init__(opt)
        train_opt = opt['train']

        # set if data should be normalized (-1,1) or not (0,1)
        if self.is_train:
            z_norm = opt['datasets']['train'].get('znorm', False)
        
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
            Setup batch augmentations
            """
            self.mixup = train_opt.get('mixup', None)
            if self.mixup: 
                #TODO: cutblur and cutout need model to be modified so LR and HR have the same dimensions (1x)
                self.mixopts = train_opt.get('mixopts', ["blend", "rgb", "mixup", "cutmix", "cutmixup"]) #, "cutout", "cutblur"]
                self.mixprob = train_opt.get('mixprob', [1.0, 1.0, 1.0, 1.0, 1.0]) #, 1.0, 1.0]
                self.mixalpha = train_opt.get('mixalpha', [0.6, 1.0, 1.2, 0.7, 0.7]) #, 0.001, 0.7]
                self.aux_mixprob = train_opt.get('aux_mixprob', 1.0)
                self.aux_mixalpha = train_opt.get('aux_mixalpha', 1.2)
                self.mix_p = train_opt.get('mix_p', None)
            
            """
            Setup frequency separation
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
            # Generator losses for 3 channel maps: diffuse, albedo and normal:
            self.generatorlosses = losses.GeneratorLoss(opt, self.device)
            # TODO: show the configured losses names in logger
            # print(self.generatorlosses.loss_list)

            # Generator losses for 1 channel maps (does not support feature networks like VGG):
            # using new option in the loss builder: allow_featnets = False
            # TODO: does it make sense to make fake 3ch images with the 1ch maps?
            self.generatorlosses1ch = losses.GeneratorLoss(opt, self.device, False)

            # Discriminator loss:
            if train_opt['gan_type'] and train_opt['gan_weight']:
                self.cri_gan = True
                diffaug = train_opt.get('diffaug', None)
                dapolicy = None
                if diffaug: #TODO: this if should not be necessary
                    dapolicy = train_opt.get('dapolicy', 'color,translation,cutout') #original
                self.adversarial = losses.Adversarial(train_opt=train_opt, device=self.device, diffaug = diffaug, dapolicy = dapolicy)
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
            If using virtual batch
            """
            batch_size = opt["datasets"]["train"]["batch_size"]
            virtual_batch = opt["datasets"]["train"].get('virtual_batch_size', None)
            self.virtual_batch = virtual_batch if virtual_batch \
                >= batch_size else batch_size
            self.accumulations = self.virtual_batch // batch_size
            self.optimizer_G.zero_grad()
            if self. cri_gan:
                self.optimizer_D.zero_grad()
            
            """
            Configure AMP
            """
            self.amp = load_amp and opt.get('use_amp', False)
            if self.amp:
                self.cast = autocast
                self.amp_scaler =  GradScaler()
                logger.info('AMP enabled')
            else:
                self.cast = nullcast

        # print network
        """ 
        TODO:
        Network summary? Make optional with parameter
            could be an selector between traditional print_network() and summary()
        """
        #self.print_network() #TODO

    def feed_data(self, data, need_HR=True):
        # LR images
        self.var_L = data['LR'].to(self.device)
        if need_HR:  # train or val
            # HR images
            self.var_H = data['HR'].to(self.device)
            # discriminator references
            input_ref = data.get('ref', data['HR'])
            self.var_ref = input_ref.to(self.device)
        
        if isinstance(data.get('NO', None), torch.Tensor):
            self.var_NO = data['NO'].to(self.device)
        else:
            self.var_NO = None
        
        if isinstance(data.get('AL', None), torch.Tensor):
            self.var_AL = data['AL'].to(self.device)
        else:
            self.var_AL = None

        if isinstance(data.get('AO', None), torch.Tensor):
            self.var_AO = data['AO'].to(self.device)
        else:
            self.var_AO = None
        
        if isinstance(data.get('HE', None), torch.Tensor):
            self.var_HE = data['HE'].to(self.device)
        else:
            self.var_HE = None
        
        if isinstance(data.get('ME', None), torch.Tensor):
            self.var_ME = data['ME'].to(self.device)
        else:
            self.var_ME = None
        
        if isinstance(data.get('RE', None), torch.Tensor):
            self.var_RE = data['RE'].to(self.device)
        else:
            self.var_RE = None
        
        if isinstance(data.get('RO', None), torch.Tensor):
            self.var_RO = data['RO'].to(self.device)
        else:
            self.var_RO = None
        

    def feed_data_batch(self, data, need_HR=True):
        # LR
        self.var_L = data
        
    def optimize_parameters(self, step):       
        # G
        # freeze discriminator while generator is trained to prevent BP
        if self.cri_gan:
            for p in self.netD.parameters():
                p.requires_grad = False

        # batch (mixup) augmentations
        # aug = None
        # if self.mixup:
        #     self.var_H, self.var_L, mask, aug = BatchAug(
        #         self.var_H, self.var_L,
        #         self.mixopts, self.mixprob, self.mixalpha,
        #         self.aux_mixprob, self.aux_mixalpha, self.mix_p
        #         )
        
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

        # batch (mixup) augmentations
        # cutout-ed pixels are discarded when calculating loss by masking removed pixels
        # if aug == "cutout":
        #     self.fake_H, self.var_H = self.fake_H*mask, self.var_H*mask
        
        l_g_total = 0
        """
        Calculate and log losses
        """
        loss_results = []
        # training generator and discriminator        
        # update generator (on its own if only training generator or alternatively if training GAN)
        if (self.cri_gan is not True) or (step % self.D_update_ratio == 0 and step > self.D_init_iters):
            with self.cast(): # Casts operations to mixed precision if enabled, else nullcontext
                # regular losses
                log_dict_diffuse = {}
                loss_results, log_dict_diffuse = self.generatorlosses(fake_SR, self.var_H, log_dict_diffuse, self.f_low)
                l_g_total += sum(loss_results)/self.accumulations

                #TODO: for now only showing the logs for the diffuse losses, need to append the other logs
                self.log_dict = log_dict_diffuse
                
                if isinstance(self.var_NO, torch.Tensor):
                    NO_loss_results = []
                    log_dict_normal = {}
                    NO_loss_results, log_dict_normal = self.generatorlosses(fake_NO, self.var_NO, log_dict_normal, self.f_low)
                    l_g_total += sum(NO_loss_results)/self.accumulations

                if isinstance(self.var_AL, torch.Tensor):
                    AL_loss_results = []
                    log_dict_albedo = {}
                    AL_loss_results, log_dict_albedo = self.generatorlosses(fake_AL, self.var_AL, log_dict_albedo, self.f_low)
                    l_g_total += sum(AL_loss_results)/self.accumulations
                
                if isinstance(self.var_AO, torch.Tensor):
                    AO_loss_results = []
                    log_dict_ao = {}
                    AO_loss_results, log_dict_ao = self.generatorlosses1ch(fake_AO, self.var_AO, log_dict_ao, self.f_low)
                    l_g_total += sum(AO_loss_results)/self.accumulations
                
                if isinstance(self.var_HE, torch.Tensor):
                    HE_loss_results = []
                    log_dict_height = {}
                    HE_loss_results, log_dict_height = self.generatorlosses1ch(fake_HE, self.var_HE, log_dict_height, self.f_low)
                    l_g_total += sum(HE_loss_results)/self.accumulations
                
                if isinstance(self.var_ME, torch.Tensor):
                    ME_loss_results = []
                    log_dict_metalness = {}
                    ME_loss_results, log_dict_metalness = self.generatorlosses1ch(fake_ME, self.var_ME, log_dict_metalness, self.f_low)
                    l_g_total += sum(ME_loss_results)/self.accumulations
                
                if isinstance(self.var_RE, torch.Tensor):
                    RE_loss_results = []
                    log_dict_reflection = {}
                    RE_loss_results, log_dict_reflection = self.generatorlosses1ch(fake_RE, self.var_RE, log_dict_reflection, self.f_low)
                    l_g_total += sum(RE_loss_results)/self.accumulations
                
                if isinstance(self.var_RO, torch.Tensor):
                    RO_loss_results = []
                    log_dict_roughness = {}
                    RO_loss_results, log_dict_roughness = self.generatorlosses1ch(fake_RO, self.var_RO, log_dict_roughness, self.f_low)
                    l_g_total += sum(RO_loss_results)/self.accumulations

                #TODO: for now only one GAN for the diffuse image, can have one for each map
                if self.cri_gan:
                    # adversarial loss
                    l_g_gan = self.adversarial(
                        fake_SR, self.var_ref, netD=self.netD, 
                        stage='generator', fsfilter = self.f_high) # (sr, hr)
                    self.log_dict['l_g_gan'] = l_g_gan.item()
                    l_g_total += l_g_gan/self.accumulations

            #/with self.cast():
            
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
            # unfreeze discriminator
            for p in self.netD.parameters():
                p.requires_grad = True
            l_d_total = 0
            
            with self.cast(): # Casts operations to mixed precision if enabled, else nullcontext
                l_d_total, gan_logs = self.adversarial(
                    self.fake_H, self.var_ref, netD=self.netD, 
                    stage='discriminator', fsfilter = self.f_high) # (sr, hr)

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
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        #TODO for PPON ?
        #if get stages 1 and 2
            #out_dict['SR_content'] = ...
            #out_dict['SR_structure'] = ...
        return out_dict

    def get_current_visuals_batch(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach().float().cpu()
        out_dict['SR'] = self.fake_H.detach().float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach().float().cpu()
        #TODO for PPON ?
        #if get stages 1 and 2
            #out_dict['SR_content'] = ...
            #out_dict['SR_structure'] = ...
        return out_dict
        
    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:
            # Discriminator
            if self.cri_gan:
                s, n = self.get_network_description(self.netD)
                if isinstance(self.netD, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                    self.netD.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netD.__class__.__name__)

                logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

            #TODO: feature network is not being trained, is it necessary to visualize? Maybe just name?
            # maybe show the generatorlosses instead?
            '''
            if self.generatorlosses.cri_fea:  # F, Perceptual Network
                #s, n = self.get_network_description(self.netF)
                s, n = self.get_network_description(self.generatorlosses.netF) #TODO
                #s, n = self.get_network_description(self.generatorlosses.loss_list.netF) #TODO
                if isinstance(self.generatorlosses.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.generatorlosses.netF.__class__.__name__,
                                                    self.generatorlosses.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.generatorlosses.netF.__class__.__name__)

                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)
            '''

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            strict = self.opt['path'].get('strict', None)
            self.load_network(load_path_G, self.netG, strict)
        if self.opt['is_train'] and self.opt['train']['gan_weight']:
            load_path_D = self.opt['path']['pretrain_model_D']
            if self.opt['is_train'] and load_path_D is not None:
                logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
                self.load_network(load_path_D, self.netD)

    def save(self, iter_step, latest=None):
        self.save_network(self.netG, 'G', iter_step, latest)
        if self.cri_gan:
            self.save_network(self.netD, 'D', iter_step, latest)
