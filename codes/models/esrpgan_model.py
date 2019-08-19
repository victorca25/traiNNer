import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss, HFENL1Loss, HFENL2Loss, TVLoss, CharbonnierLoss, ElasticLoss, SRPFeaLoss, SRPGANLoss, SRPGANDiscriminatorLoss
from models.modules.ssim2 import SSIM, MS_SSIM
logger = logging.getLogger('base')

import models.lr_schedulerR as lr_schedulerR

class ESRPGANModel(BaseModel):
    def __init__(self, opt):
        super(ESRPGANModel, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netG.train()
            if train_opt['gan_weight'] > 0:
                self.netD = networks.define_D(opt).to(self.device)  # D
                self.netD.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif l_pix_type == 'cb':
                    self.cri_pix = CharbonnierLoss(eps=1e-8).to(self.device)
                elif l_pix_type == 'elastic':
                    self.cri_pix = ElasticLoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                #Later will allow to choose loss type, the paper uses Charbonnier loss
                """
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                elif l_fea_type == 'cb':
                    self.cri_fea = CharbonnierLoss().to(self.device)
                elif l_fea_type == 'elastic':
                    self.cri_fea = ElasticLoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                """
                self.cri_fea = SRPFeaLoss(eps=1e-8).to(self.device)
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            #VGG network is not needed
            #if self.cri_fea:  # load VGG perceptual loss
                #self.netF = networks.define_F(opt, use_bn=False).to(self.device)
            
            #HFEN loss
            if train_opt['hfen_weight'] > 0:
                l_hfen_type = train_opt['hfen_criterion']
                if l_hfen_type == 'l1':
                    self.cri_hfen =  HFENL1Loss().to(self.device) #RelativeHFENL1Loss().to(self.device)
                elif l_hfen_type == 'l2':
                    self.cri_hfen = HFENL2Loss().to(self.device)
                elif l_hfen_type == 'rel_l1':
                    self.cri_hfen = RelativeHFENL1Loss().to(self.device)
                elif l_hfen_type == 'rel_l2':
                    self.cri_hfen = RelativeHFENL2Loss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_hfen_type))
                self.l_hfen_w = train_opt['hfen_weight']
            else:
                logger.info('Remove HFEN loss.')
                self.cri_hfen = None
                
            #TV loss
            if train_opt['tv_weight'] > 0:
                self.l_tv_w = train_opt['tv_weight']
                l_tv_type = train_opt['tv_type']
                if l_tv_type == 'normal':
                    self.cri_tv = TVLoss(self.l_tv_w).to(self.device) 
                elif l_tv_type == '4D':
                    self.cri_tv = TVLoss4D(self.l_tv_w).to(self.device) #Total Variation regularization in 4 directions
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_tv_type))
            else:
                logger.info('Remove TV loss.')
                self.cri_tv = None
                
            #SSIM loss
            if train_opt['ssim_weight'] > 0:
                self.l_ssim_w = train_opt['ssim_weight']
                l_ssim_type = train_opt['ssim_type']
                if l_ssim_type == 'ssim':
                    self.cri_ssim = SSIM(win_size=11, win_sigma=1.5, size_average=True, data_range=1., channel=3).to(self.device)
                elif l_ssim_type == 'ms-ssim':
                    self.cri_ssim = MS_SSIM(win_size=11, win_sigma=1.5, size_average=True, data_range=1., channel=3).to(self.device)
            else:
                logger.info('Remove SSIM loss.')
                self.cri_ssim = None

            # SRPGD discriminator loss
            if train_opt['gan_weight'] > 0:
                self.cri_gan = SRPGANLoss().to(self.device)  #*0.001
                self.cri_gan_d = SRPGANDiscriminatorLoss().to(self.device)
                self.l_gan_w = train_opt['gan_weight']
            else:
                logger.info('Remove GAN loss.')
                self.cri_gan = None
                self.cri_gan_d = None

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            
            # D
            if self.cri_gan:
                wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], \
                    weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
                self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            elif train_opt['lr_scheme'] == 'MultiStepLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_schedulerR.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'StepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.StepLR(optimizer, \
                        train_opt['lr_step_size'], train_opt['lr_gamma']))
            elif train_opt['lr_scheme'] == 'StepLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_schedulerR.StepLR_Restart(optimizer, step_sizes=train_opt['lr_step_sizes'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_schedulerR.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('Learning rate scheme ("lr_scheme") not defined or not recognized.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data['LR'].to(self.device)

        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)

            #input_ref = data['ref'] if 'ref' in data else data['HR']
            #self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        if self.cri_gan:
            self.fake_H = self.netG(self.var_L)
            
            d_hr_out, d_hr_feat_maps = self.netD(self.var_H)  # Sigmoid output #pred_d_real
            d_sr_out, d_sr_feat_maps = self.netD(self.fake_H.detach())  # detach to avoid BP to G  # Sigmoid output #pred_d_fake
            
            # D
            #for p in self.netD.parameters():
            #    p.requires_grad = True

            self.optimizer_D.zero_grad()
            l_d_total = 0
            
            l_d_total = self.cri_gan_d(d_hr_out, d_sr_out)

            l_d_total.backward(retain_graph=True)
            self.optimizer_D.step()
            
            # lg = la(G,D)+λ1lp+λ2ly
            
            # G
            #for p in self.netD.parameters():
            #    p.requires_grad = False

            self.optimizer_G.zero_grad()

            l_g_total = 0
        
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_ssim: # structural loss
                l_g_ssim = 1.-(self.l_ssim_w *self.cri_ssim(self.fake_H, self.var_H)) 
                if torch.isnan(l_g_ssim).any(): #at random, l_g_ssim is returning NaN for ms-ssim, which breaks the model. Temporary hack, until I find out what's going on.
                    l_g_total = l_g_total
                else:
                    l_g_total += l_g_ssim
            if self.cri_hfen:  # HFEN loss 
                l_g_HFEN = self.l_hfen_w * self.cri_hfen(self.fake_H, self.var_H)
                l_g_total += l_g_HFEN
            if self.cri_tv: #TV loss
                l_g_tv = self.cri_tv(self.fake_H) #note: the weight is already multiplied inside the function, doesn't need to be here
                l_g_total += l_g_tv
            if self.cri_fea:  # G feature loss
                l_g_fea = self.l_fea_w * self.cri_fea(d_hr_feat_maps, d_sr_feat_maps)
                l_g_total += l_g_fea
            
            # G gan + cls loss
            l_g_gan = self.cri_gan(d_sr_out)
            l_g_total += l_g_gan
            
            l_g_total.backward()
            self.optimizer_G.step()
            
            # set log
            # G
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            if self.cri_hfen:
                self.log_dict['l_g_HFEN'] = l_g_HFEN.item()
            if self.cri_tv:
                self.log_dict['l_g_tv'] = l_g_tv.item()
            if self.cri_ssim:
                self.log_dict['l_g_ssim'] = l_g_ssim.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()

            # D outputs
            self.log_dict['l_d_total'] = l_d_total.item()
            
        else:
            # G
            for p in self.netD.parameters():
                p.requires_grad = False

            self.optimizer_G.zero_grad()
            self.fake_H = self.netG(self.var_L)

            l_g_total = 0
            
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_ssim: # structural loss
                l_g_ssim = 1.-(self.l_ssim_w *self.cri_ssim(self.fake_H, self.var_H)) 
                if torch.isnan(l_g_ssim).any(): #at random, l_g_ssim is returning NaN for ms-ssim, which breaks the model. Temporary hack, until I find out what's going on.
                    l_g_total = l_g_total
                else:
                    l_g_total += l_g_ssim
            if self.cri_hfen:  # HFEN loss 
                l_g_HFEN = self.l_hfen_w * self.cri_hfen(self.fake_H, self.var_H)
                l_g_total += l_g_HFEN
            if self.cri_tv: #TV loss
                l_g_tv = self.cri_tv(self.fake_H) #note: the weight is already multiplied inside the function, doesn't need to be here
                l_g_total += l_g_tv
                
            l_g_total.backward()
            self.optimizer_G.step()

            # set log
            # G
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            if self.cri_hfen:
                self.log_dict['l_g_HFEN'] = l_g_HFEN.item()
            if self.cri_tv:
                self.log_dict['l_g_tv'] = l_g_tv.item()
            if self.cri_ssim:
                self.log_dict['l_g_ssim'] = l_g_ssim.item()


    def test(self):
        self.netG.eval()
        with torch.no_grad():
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


    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        if self.opt['is_train'] and self.opt['train']['gan_weight'] > 0:        
            load_path_D = self.opt['path']['pretrain_model_D']
            if self.opt['is_train'] and load_path_D is not None:
                logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
                self.load_network(load_path_D, self.netD)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        if self.cri_gan:
            self.save_network(self.netD, 'D', iter_step)
