#this is the pseudo model after outsourcing the components to other classes, it should end up very short
#note that the functionality between this model file and train.py is becomming blurry, similar to what would
#happen if using PytorchLightning. Can try to follow PL best practices so later it is easier to migrate to

# another reference: https://github.com/NVIDIA/pix2pixHD/blob/master/models/pix2pixHD_model.py
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/generative_adversarial_net.py


from __future__ import absolute_import

import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')

import models.lr_schedulerR as lr_schedulerR

from . import losses

#TMP
from dataops.mixup import mixaug

""" #debug
def save_images(image, num_rep, sufix):
    from utils import util
    import uuid, cv2
    hex = uuid.uuid4().hex
    img_ds = util.tensor2img(image)  # uint8
    cv2.imwrite("D:/tmp_test/fake_"+sufix+"_"+str(num_rep)+hex+".png",img_ds*255) #random name to save + had to multiply by 255, else getting all black image
#"""


class SRRaGANModel(BaseModel):
    def __init__(self, opt):
        super(SRRaGANModel, self).__init__(opt)
        train_opt = opt['train']

        # set if data should be normalized (-1,1) or not (0,1)
        if self.is_train:
            if opt['datasets']['train']['znorm']:
                z_norm = opt['datasets']['train']['znorm']
            else:
                z_norm = False
        
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
            # define if the generator will have a final capping mechanism in the output
            self.outm = train_opt['finalcap'] if train_opt['finalcap'] else None
            # define if mixup or cutmix will be used (for now only possible if training without discriminator)
            self.mixup = train_opt['mixup'] if train_opt['mixup'] else None
            # setup mixup augmentations
            if self.mixup: 
                #TODO: cutblur and cutout need model to be modified so LR and HR have the same dimensions (1x)
                self.mixopts = train_opt['mixopts'] if train_opt['mixopts'] else ["blend", "rgb", "mixup", "cutmix", "cutmixup"] #, "cutout", "cutblur"]
                self.mixprob = train_opt['mixprob'] if train_opt['mixprob'] else [1.0, 1.0, 1.0, 1.0, 1.0] #, 1.0, 1.0]
                self.mixalpha = train_opt['mixalpha'] if train_opt['mixalpha'] else [0.6, 1.0, 1.2, 0.7, 0.7] #, 0.001, 0.7]
                self.aux_mixprob = train_opt['aux_mixprob'] if train_opt['aux_mixprob'] else 1.0
                self.aux_mixalpha = train_opt['aux_mixalpha'] if train_opt['aux_mixalpha'] else 1.2
                self.mix_p = train_opt['mix_p'] if train_opt['mix_p'] else None
                
            """
            Initialize losses
            """
            #Initialize the losses with the opt parameters
            # Generator losses:
            self.generatorlosses = losses.GeneratorLoss(opt, self.device)
            print(self.generatorlosses.loss_list)

            # Discriminator loss:
            if train_opt['gan_type'] and train_opt['gan_weight']:
                self.cri_gan = True
                diffaug = train_opt['diffaug'] if train_opt['diffaug'] else None
                if diffaug:
                    dapolicy = train_opt['dapolicy'] if train_opt['dapolicy'] else 'color,translation,cutout' #original
                self.adversarial = losses.Adversarial(train_opt=train_opt, device=self.device, diffaug = diffaug, dapolicy = dapolicy)
                # D_update_ratio and D_init_iters are for WGAN
                self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
                self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
            else:
                self.cri_gan = False

            """
            # TODO CHANGE
            """
            
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
            elif train_opt['lr_scheme'] == 'ReduceLROnPlateau':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        #lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
                        lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode=train_opt['plateau_mode'], factor=train_opt['plateau_factor'], 
                            threshold=train_opt['plateau_threshold'], patience=train_opt['plateau_patience']))
            else:
                raise NotImplementedError('Learning rate scheme ("lr_scheme") not defined or not recognized.')

            """
            # TODO CHANGE
            """

            #Keep log in loss class instead?
            self.log_dict = OrderedDict()
        # print network
        """
        Network summary? Make optional with parameter
            could be an selector between traditional print_network() and summary()
        """
        self.print_network() 

    def feed_data(self, data, need_HR=True):
        # LR images
        self.var_L = data['LR'].to(self.device)
        if need_HR:  # train or val
            # HR images
            self.var_H = data['HR'].to(self.device)
            # discriminator references
            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)

    def feed_data_batch(self, data, need_HR=True):
        # LR
        self.var_L = data
        
    def optimize_parameters(self, step):
        # G
        # freeze discriminator while generator is trained to prevent BP
        if self.cri_gan:
            for p in self.netD.parameters():
                p.requires_grad = False
        self.optimizer_G.zero_grad()

        ### Network forward, generate SR
        # mixup augmentations
        if self.mixup:
            self.var_H, self.var_L, mask, aug = mixaug(
                self.var_H, self.var_L,
                self.mixopts, self.mixprob, self.mixalpha,
                self.aux_mixprob, self.aux_mixalpha, self.mix_p
                )
                
        if self.outm: #if the model has the final activation option
            self.fake_H = self.netG(self.var_L, outm=self.outm)
        else: #regular models without the final activation option
            self.fake_H = self.netG(self.var_L)

        # mixup augmentations
        if aug == "cutout":
            self.fake_H, self.var_H = self.fake_H*mask, self.var_H*mask

        l_g_total = 0

        """ # Debug
        print ("SR min. val: ", torch.min(self.fake_H))
        print ("SR max. val: ", torch.max(self.fake_H))
        
        print ("LR min. val: ", torch.min(self.var_L))
        print ("LR max. val: ", torch.max(self.var_L))
        
        print ("HR min. val: ", torch.min(self.var_H))
        print ("HR max. val: ", torch.max(self.var_H))
        #"""
        
        """ #debug
        #####################################################################
        #test_save_img = False
        # test_save_img = None
        # test_save_img = True
        if test_save_img:
            save_images(self.var_H, 0, "self.var_H")
            save_images(self.fake_H.detach(), 0, "self.fake_H")
        #####################################################################
        #"""


        """
        Calculate and log losses
        """
        loss_results = []
        # training generator and discriminator
        if self.cri_gan:
            # update generator alternatively
            if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                loss_results, self.log_dict = self.generatorlosses(self.fake_H, self.var_H, self.log_dict)
                l_g_total = sum(loss_results)

                l_g_gan = self.adversarial(self.fake_H, self.var_ref, netD=self.netD, stage='generator') # (sr, hr)
                self.log_dict['l_g_gan'] = l_g_gan.item()
                l_g_total += l_g_gan

                l_g_total.backward()
                self.optimizer_G.step()

            # update discriminator
            # unfreeze discriminator
            for p in self.netD.parameters():
                p.requires_grad = True
            self.optimizer_D.zero_grad()
            l_d_total = 0
            l_d_total, gan_logs = self.adversarial(self.fake_H, self.var_ref, netD=self.netD, stage='discriminator') # (sr, hr)

            for g_log in gan_logs:
                self.log_dict[g_log] = gan_logs[g_log]

            l_d_total.backward()
            self.optimizer_D.step()
        
        # only training generator
        else:
            loss_results, self.log_dict = self.generatorlosses(self.fake_H, self.var_H, self.log_dict)
            l_g_total = sum(loss_results)

            l_g_total.backward()
            self.optimizer_G.step()
        
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

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        if self.opt['is_train'] and self.opt['train']['gan_weight']:
            load_path_D = self.opt['path']['pretrain_model_D']
            if self.opt['is_train'] and load_path_D is not None:
                logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
                self.load_network(load_path_D, self.netD)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        if self.cri_gan:
            self.save_network(self.netD, 'D', iter_step)
