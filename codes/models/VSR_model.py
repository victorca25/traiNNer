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
from dataops.colors import ycbcr_to_rgb

# TODO: TMP
import torch.nn.functional as F
from dataops.debug import tmp_vis, tmp_vis_flow, describe_numpy, describe_tensor


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


class VSRModel(BaseModel):
    def __init__(self, opt):
        super(VSRModel, self).__init__(opt)
        train_opt = opt['train']
        self.scale = opt.get('scale', 4)
        self.tensor_shape = opt.get('tensor_shape', 'TCHW')

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
            #TODO: will need testing. Also consider batch augmentations may be ill defined with the temporal data
            '''
            self.mixup = train_opt.get('mixup', None)
            if self.mixup: 
                #TODO: cutblur and cutout need model to be modified so LR and HR have the same dimensions (1x)
                self.mixopts = train_opt.get('mixopts', ["blend", "rgb", "mixup", "cutmix", "cutmixup"]) #, "cutout", "cutblur"]
                self.mixprob = train_opt.get('mixprob', [1.0, 1.0, 1.0, 1.0, 1.0]) #, 1.0, 1.0]
                self.mixalpha = train_opt.get('mixalpha', [0.6, 1.0, 1.2, 0.7, 0.7]) #, 0.001, 0.7]
                self.aux_mixprob = train_opt.get('aux_mixprob', 1.0)
                self.aux_mixalpha = train_opt.get('aux_mixalpha', 1.2)
                self.mix_p = train_opt.get('mix_p', None)
            '''
            
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
                self.adversarial = losses.Adversarial(train_opt=train_opt, device=self.device, diffaug = diffaug, dapolicy = dapolicy)
                # D_update_ratio and D_init_iters are for WGAN
                self.D_update_ratio = train_opt.get('D_update_ratio', 1)
                self.D_init_iters = train_opt.get('D_init_iters', 0)
            else:
                self.cri_gan = False

            # Optical Flow Reconstruction loss:
            ofr_type = train_opt.get('ofr_type', None)
            ofr_weight = train_opt.get('ofr_weight', [0.1, 0.2, 0.1, 0.01])
            if ofr_type and ofr_weight:
                self.ofr_weight = ofr_weight[3] #lambda 4
                self.ofr_wl1 = ofr_weight[0] #lambda 1
                self.ofr_wl2 = ofr_weight[1] #lambda 2
                ofr_wl3 = ofr_weight[2] #lambda 3
                if ofr_type == 'ofr':
                    from models.modules.loss import OFR_loss
                    #TODO: make the regularization weight an option. lambda3 = 0.1
                    self.cri_ofr = OFR_loss(reg_weight=ofr_wl3).to(self.device)
            else:
                self.cri_ofr = False
 
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
        # data
        if len(data['LR'].size()) == 4:
            b, n_frames, h_lr, w_lr = data['LR'].size()
            LR = data['LR'].view(b, -1, 1, h_lr, w_lr) # b, t, c, h, w
        elif len(data['LR'].size()) == 5: #for networks that work with 3 channel images
            if self.tensor_shape == 'CTHW':
                _, _, n_frames, _, _ = data['LR'].size() # b, c, t, h, w
            else: # TCHW
                _, n_frames, _, _, _ = data['LR'].size() # b, t, c, h, w
            LR = data['LR']

        self.idx_center = (n_frames - 1) // 2
        self.n_frames = n_frames

        # LR images (LR_y_cube)
        self.var_L = LR.to(self.device)

        # bicubic upscaled LR and RGB center HR
        if isinstance(data['HR_center'], torch.Tensor):
            self.var_H_center = data['HR_center'].to(self.device)
        else:
            self.var_H_center = None
        if isinstance(data['LR_bicubic'], torch.Tensor): 
            self.var_LR_bic = data['LR_bicubic'].to(self.device)
        else:
            self.var_LR_bic = None

        if need_HR:  # train or val
            # HR images
            if len(data['HR'].size()) == 4:
                HR = data['HR'].view(b, -1, 1, h_lr * self.scale, w_lr * self.scale) # b, t, c, h, w
            elif len(data['HR'].size()) == 5: #for networks that work with 3 channel images
                HR = data['HR'] # b, t, c, h, w 
            self.var_H = HR.to(self.device)

            # discriminator references
            input_ref = data.get('ref', data['HR'])
            if len(input_ref.size()) == 4:
                input_ref = input_ref.view(b, -1, 1, h_lr * self.scale, w_lr * self.scale) # b, t, c, h, w
                self.var_ref = input_ref.to(self.device)
            elif len(input_ref.size()) == 5: #for networks that work with 3 channel images    
                self.var_ref = input_ref.to(self.device)

    def feed_data_batch(self, data, need_HR=True):
        #TODO
        # LR
        self.var_L = data
        
    def optimize_parameters(self, step):       
        # G
        # freeze discriminator while generator is trained to prevent BP
        if self.cri_gan:
            for p in self.netD.parameters():
                p.requires_grad = False

        # batch (mixup) augmentations
        aug = None
        '''
        if self.mixup:
            self.var_H, self.var_L, mask, aug = BatchAug(
                self.var_H, self.var_L,
                self.mixopts, self.mixprob, self.mixalpha,
                self.aux_mixprob, self.aux_mixalpha, self.mix_p
                )
        '''
        
        ### Network forward, generate SR
        with self.cast():
            # inference
            self.fake_H = self.netG(self.var_L)
            if not isinstance(self.fake_H, torch.Tensor) and len(self.fake_H) == 4:
                flow_L1, flow_L2, flow_L3, self.fake_H = self.fake_H
        #/with self.cast():

        # batch (mixup) augmentations
        # cutout-ed pixels are discarded when calculating loss by masking removed pixels
        '''
        if aug == "cutout":
            self.fake_H, self.var_H = self.fake_H*mask, self.var_H*mask
        '''

        #TODO: TMP test to view samples of the optical flows
        # tmp_vis(self.var_H[:, self.idx_center, :, :, :], True)
        # print(flow_L1[0].shape)
        # tmp_vis(flow_L1[0][:, 0:1, :, :], to_np=True, rgb2bgr=False)
        # tmp_vis(flow_L2[0][:, 0:1, :, :], to_np=True, rgb2bgr=False)
        # tmp_vis(flow_L3[0][:, 0:1, :, :], to_np=True, rgb2bgr=False)
        # tmp_vis_flow(flow_L1[0])
        # tmp_vis_flow(flow_L2[0])
        # tmp_vis_flow(flow_L3[0])
        
        l_g_total = 0
        """
        Calculate and log losses
        """
        loss_results = []
        # training generator and discriminator        
        # update generator (on its own if only training generator or alternatively if training GAN)
        if (self.cri_gan is not True) or (step % self.D_update_ratio == 0 and step > self.D_init_iters):
            with self.cast(): # Casts operations to mixed precision if enabled, else nullcontext
                # get the central frame for SR losses
                if isinstance(self.var_LR_bic, torch.Tensor) and isinstance(self.var_H_center, torch.Tensor):
                    # tmp_vis(ycbcr_to_rgb(self.var_LR_bic), True)
                    # print("fake_H:", self.fake_H.shape)
                    fake_H_cb = self.var_LR_bic[:, 1, :, :].to(self.device)
                    # print("fake_H_cb: ", fake_H_cb.shape)
                    fake_H_cr = self.var_LR_bic[:, 2, :, :].to(self.device)
                    # print("fake_H_cr: ", fake_H_cr.shape)
                    centralSR = ycbcr_to_rgb(torch.stack((self.fake_H.squeeze(1), fake_H_cb, fake_H_cr), -3))
                    # print("central rgb", centralSR.shape)
                    # tmp_vis(centralSR, True)
                    # centralHR = ycbcr_to_rgb(self.var_H_center) #Not needed, can send the rgb HR from dataloader
                    centralHR = self.var_H_center
                    # print(centralHR.shape)
                    # tmp_vis(centralHR)
                else: #if self.var_L.shape[2] == 1:
                    centralSR = self.fake_H
                    centralHR = self.var_H[:, :, self.idx_center, :, :] if self.tensor_shape == 'CTHW' else self.var_H[:, self.idx_center, :, :, :]
                
                # tmp_vis(torch.cat((centralSR, centralHR), -1))

                # regular losses
                # loss_SR = criterion(self.fake_H, self.var_H[:, idx_center, :, :, :]) #torch.nn.MSELoss()
                loss_results, self.log_dict = self.generatorlosses(centralSR, 
                                    centralHR, self.log_dict, self.f_low)
                l_g_total += sum(loss_results)/self.accumulations

                # optical flow reconstruction loss
                #TODO: see if can be moved into loss file
                #TODO 2: test if AMP could affect the loss due to loss of precision
                if self.cri_ofr: #OFR_loss()
                    l_g_ofr = 0
                    for i in range(self.n_frames):
                        if i != self.idx_center:
                            loss_L1 = self.cri_ofr(F.avg_pool2d(self.var_L[:, i, :, :, :], kernel_size=2),
                                            F.avg_pool2d(self.var_L[:, self.idx_center, :, :, :], kernel_size=2),
                                            flow_L1[i])
                            loss_L2 = self.cri_ofr(self.var_L[:, i, :, :, :], self.var_L[:, self.idx_center, :, :, :], flow_L2[i])
                            loss_L3 = self.cri_ofr(self.var_H[:, i, :, :, :], self.var_H[:, self.idx_center, :, :, :], flow_L3[i])
                            # ofr weights option. lambda2 = 0.2, lambda1 = 0.1 in the paper
                            l_g_ofr += loss_L3 + self.ofr_wl2 * loss_L2 + self.ofr_wl1 * loss_L1

                    # ofr weight option. lambda4 = 0.01 in the paper
                    l_g_ofr = self.ofr_weight * l_g_ofr / (self.n_frames - 1)
                    self.log_dict['ofr'] = l_g_ofr.item()
                    l_g_total += l_g_ofr/self.accumulations

                if self.cri_gan:
                    # adversarial loss
                    l_g_gan = self.adversarial(
                        centralSR, centralHR, netD=self.netD, 
                        stage='generator', fsfilter = self.f_high) # (sr, hr)
                    self.log_dict['l_g_gan'] = l_g_gan.item()
                    l_g_total += l_g_gan/self.accumulations

            #/with self.cast():
            
            # high precision generator losses (can be affected by AMP half precision)
            if self.precisegeneratorlosses.loss_list:
                precise_loss_results, self.log_dict = self.precisegeneratorlosses(
                        self.fake_H, self.var_H, self.log_dict, self.f_low)
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
            # unfreeze discriminator
            for p in self.netD.parameters():
                p.requires_grad = True
            l_d_total = 0
            
            with self.cast(): # Casts operations to mixed precision if enabled, else nullcontext
                l_d_total, gan_logs = self.adversarial(
                    centralSR, centralHR, netD=self.netD, 
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
        #TODO: test/val code
        self.netG.eval()
        with torch.no_grad():
            if self.is_train:
                self.fake_H = self.netG(self.var_L)
                if len(self.fake_H) == 4:
                    _, _, _, self.fake_H = self.fake_H
            else:
                #self.fake_H = self.netG(self.var_L, isTest=True)
                self.fake_H = self.netG(self.var_L)
                if len(self.fake_H) == 4:
                    _, _, _, self.fake_H = self.fake_H
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        #TODO: temporal considerations
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
        #TODO: temporal considerations
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
