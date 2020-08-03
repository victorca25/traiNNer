import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from collections import OrderedDict

from models.modules.loss import GANLoss, GradientPenaltyLoss, HFENLoss, TVLoss, CharbonnierLoss, ElasticLoss, RelativeL1, L1CosineSim, ContextualLoss, ClipL1
from models.modules.losses import spl_loss as spl
from models.modules.losses.ssim2 import SSIM, MS_SSIM #implementation for use with any PyTorch
from models.modules.LPIPS import perceptual_loss as ps
import models.networks as networks

from dataops.diffaug import DiffAugment

#TODO TMP: 
channels = 3


#loss builder
def get_loss_fn(loss_type=None, weight=0, recurrent=False, reduction='mean', network = None, device = 'cuda:0', opt = None):
    if loss_type == 'skip':
        loss_function = None
    # pixel / content losses
    if loss_type == 'MSE' or loss_type == 'l2':
        loss_function = nn.MSELoss(reduction=reduction)
        loss_type = 'pix-'+loss_type
    elif loss_type == 'L1' or loss_type == 'l1':
        loss_function = nn.L1Loss(reduction=reduction)
        loss_type = 'pix-'+loss_type
    elif loss_type == 'cb':
        loss_function = CharbonnierLoss()
        loss_type = 'pix-'+loss_type
    elif loss_type == 'elastic':
        loss_function = ElasticLoss(reduction=reduction)
        loss_type = 'pix-'+loss_type
    elif loss_type == 'relativel1':
        loss_function = RelativeL1(reduction=reduction)
        loss_type = 'pix-'+loss_type
    #TODO
    #elif loss_type == 'relativel2':
        #loss_function = RelativeL2(reduction=reduction)
        #loss_type = 'pix-'+loss_type
    elif loss_type == 'l1cosinesim' or loss_type == 'L1CosineSim':
        loss_function = L1CosineSim(reduction=reduction)
        loss_type = 'pix-'+loss_type
    elif loss_type == 'clipl1':
        loss_function = ClipL1()
        loss_type = 'pix-'+loss_type
    # SSIM losses
    #TODO: pass SSIM options, maybe from opt_train. Specialy channel
    elif loss_type == 'ssim' or loss_type == 'SSIM': #l_ssim_type
        loss_function =  SSIM(win_size=11, win_sigma=1.5, size_average=True, data_range=1., channel=channels) # TODO channels
    elif loss_type == 'ms-ssim' or loss_type == 'MSSSIM': #l_ssim_type
        loss_function =  MS_SSIM(win_size=11, win_sigma=1.5, size_average=True, data_range=1., channel=channels) # TODO channels
    # HFEN loss
    elif loss_type.find('hfen') >= 0:
        hfen_loss_f = get_loss_fn(loss_type.split('-')[1], recurrent=True, reduction='sum')
        #print(hfen_loss_f)
        #TODO device will not be needed
        loss_function = HFENLoss(loss_f=hfen_loss_f, device='cpu')
    # SPL losses
    elif loss_type == 'gpl':
        # Gradient Profile Loss
        #TODO fix parameters to training variables (spl_norm, yuv_norm)
        loss_function = spl.GPLoss(spl_norm=False)
    elif loss_type == 'cpl':
        # Color Profile Loss
        #TODO fix parameters to training variables (spl_norm, yuv_norm)
        loss_function = spl.CPLoss(rgb=True,yuv=True,yuvgrad=True,spl_norm=False,yuv_norm=False)
    # TV regularization
    elif loss_type.find('tv') >= 0:
        tv_type = loss_type.split('-')[0]
        tv_norm = loss_type.split('-')[1]
        if tv_norm == 'l1':
            tv_norm = 1
        elif tv_norm == 'l2':
            tv_norm = 2
        if tv_type == 'tv':
            loss_function = TVLoss(0.5, p=tv_norm)
        #4D diagonal tv, incorporate from the diagonal gradient calculation
        elif tv_type == 'dtv': 
            #loss_function = DTVLoss(0.5, p=tv_norm)
            loss_function = TVLoss(0.5, p=tv_norm) #TMP, while the DTV is added
    #Feature loss
    #fea-vgg19-l1, fea-vgg16-l2, fea-lpips-... ("vgg" | "alex" | "squeeze" / net-lin | net )
    elif loss_type.find('fea') >= 0:
        if loss_type.split('-')[1] == 'lpips':
            #TODO: make lpips behave more like regular feature networks
            loss_function = PerceptualLoss(criterion='lpips', network=network, rotations=False, flips=False)
        else: #if loss_type.split('-')[1][:3] == 'vgg': #if vgg16, vgg19, resnet, etc
            fea_loss_f = get_loss_fn(loss_type.split('-')[2], recurrent=True, reduction='mean') #sum? also worked on accidental test, but larger loss magnitudes
            #TODO
            #network = networks.define_F(opt, use_bn=False).to(self.device)
            #TMP #loss_type.split('-')[1] # = vgg16, vgg19, resnet, ...
            #network = define_F(use_bn=False, feat_network = loss_type.split('-')[1])
            loss_function = PerceptualLoss(criterion=fea_loss_f, network=network, rotations=False, flips=False)
        #TODO: pass rotation and flips options, maybe from opt_train
    elif loss_type == 'contextual':
        loss_function = ContextualLoss(band_width = 0.1, loss_type = 'cosine', use_vgg = True, vgg_layer = 'relu3_4')
    else:
        loss_function = None
        #raise NotImplementedError('Loss type [{:s}] not recognized.'.format(loss_type))

    if loss_function:
        if recurrent:
            return loss_function.to(device)
        else:
            loss = {
                    'name': loss_type,
                    'weight': float(weight),
                    'function': loss_function.to(device)}
            return loss
                    


def check_loss_names(pixel_criterion=None, feature_criterion=None, feature_network=None, hfen_criterion=None, tv_type=None, tv_norm=None, lpips_criterion=None, lpips_network = None):
    """
        standardize the losses names, for backward compatibility
        (should be temporary and eventually deprecated for the new names)
    """

    if pixel_criterion:
        pixel_criterion = 'pix-'+pixel_criterion
        return pixel_criterion

    #, "feature_criterion": "l1" //"l1" | "l2" | "cb" | "elastic" //feature loss (VGG feature network)
    if feature_criterion:
        if feature_criterion == 'l1' or feature_criterion == 'L1':
            feature_criterion = 'fea-'+feature_network+'-l1'
        elif feature_criterion == 'l2' or feature_criterion == 'L2':
            feature_criterion = 'fea-'+feature_network+'-l2'
        elif feature_criterion == 'cb':
            feature_criterion = 'fea-'+feature_network+'-cb'
        elif feature_criterion == 'elastic':
            feature_criterion = 'fea-'+feature_network+'-elastic'
        elif feature_criterion == 'relativel1' or feature_criterion == 'relativel2': #TODO
            feature_criterion = 'fea-'+feature_network+'-relativel1'
        #elif feature_criterion == 'relativel2':
            #feature_criterion = 'fea-'+feature_network+'-relativel2'
        elif feature_criterion == 'l1cosinesim':
            feature_criterion = 'fea-'+feature_network+'-l1cosinesim'
        return feature_criterion

    #//, "dis_feature_criterion": "l1" //"l1" | "l2" | "cb" | "elastic" //discriminator feature loss (only for asrragan)

    #//, "hfen_criterion": "l1" //hfen: "l1" | "l2" | "rel_l1" | "rel_l2" //helps in deblurring and finding edges, lines
    if hfen_criterion:
        if hfen_criterion == 'l1' or hfen_criterion == 'L1':
            hfen_criterion = 'hfen-l1'
        elif hfen_criterion == 'l2' or hfen_criterion == 'L2':
            hfen_criterion = 'hfen-l2'
        elif hfen_criterion == 'rel_l1' or hfen_criterion == 'rel_l2': #TODO, rel_l2 not available, easy to do
            hfen_criterion = 'hfen-relativel1'
        #elif hfen_criterion == 'rel_l2':
            #hfen_criterion = 'hfen-relativel2'
        elif hfen_criterion == 'cb': 
            hfen_criterion = 'hfen-cb'
        elif hfen_criterion == 'elastic':
            hfen_criterion = 'hfen-elastic'
        elif hfen_criterion == 'l1cosinesim': 
            hfen_criterion = 'hfen-l1cosinesim'
        return hfen_criterion

    #//, "tv_type": "normal" //helps in denoising, reducing upscale artefacts
    if tv_type and tv_norm:
        #get norm
        if tv_norm == 1 or tv_norm == 'L1':
            tv_norm = 'l1'
        elif tv_norm == 2 or tv_norm == 'L2':
            tv_norm = 'l2'
        #get type
        if tv_type == 'normal':
            tv_type = 'tv'
        elif tv_type == '4D':
            tv_type = 'dtv'
        #final combined type
        tv_type = tv_type+'-'+tv_norm
        return tv_type
    
    if lpips_criterion and lpips_network:
        if lpips_criterion.split('-')[0] != 'fea':
            return 'fea-lpips-'+lpips_criterion+'-'+lpips_network

    return None


class PerceptualLoss(nn.Module):
    def __init__(self, criterion=None, network=None, rotations=False, flips=False):
        super(PerceptualLoss, self).__init__()
        self.criterion = criterion
        #self.loss = PerceptualLossLPIPS()
        self.network = network
        self.rotations = rotations
        self.flips = flips

    def forward(self, x, y):
        if self.rotations:
            k_rot = random.choice([-1, 0, 1])
            x = torch.rot90(x, k_rot, [2, 3])
            y = torch.rot90(y, k_rot, [2, 3])
        if self.flips:
            if random.choice([True, False]):
                x = torch.flip(x, (2,))
                y = torch.flip(y, (2,))
            if random.choice([True, False]):
                x = torch.flip(x, (3,))
                y = torch.flip(y, (3,))
        if self.criterion == 'lpips':
            return self.network(x, y, normalize=True).mean()
        else:
            fea_x = self.network(x)
            fea_y = self.network(y).detach()
            return self.criterion(fea_x, fea_y)



class Adversarial(nn.Module):
    def __init__(self, train_opt=None, device = 'cpu', diffaug = False, dapolicy = ''):
        super(Adversarial, self).__init__()

        self.device = device
        self.diffaug = diffaug
        self.dapolicy = dapolicy
        self.gan_type = train_opt['gan_type']
        
        self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
        self.l_gan_w = train_opt['gan_weight']

        if self.gan_type == 'wgan-gp': 
            self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
            # gradient penalty loss
            self.cri_gp = GradientPenaltyLoss(device=self.device).to(self.device)
            self.l_gp_w = train_opt['gp_weigth']


    def forward(self, fake, real, netD=None, stage='discriminator'): # (fake_H,self.var_ref)

        if self.diffaug:
            real = DiffAugment(real, policy=self.dapolicy)
            fake = DiffAugment(fake, policy=self.dapolicy)
        
        if stage == 'generator':
            # updating generator
            #For the Generator only if GAN is enabled, everything else can happen any time
            # G gan + cls loss
            pred_g_real = netD(real).detach() # detach to avoid backpropagation to D
            pred_g_fake = netD(fake)
            l_g_gan = self.l_gan_w * (self.cri_gan(pred_g_real - torch.mean(pred_g_fake), False) +
                                        self.cri_gan(pred_g_fake - torch.mean(pred_g_real), True)) / 2
            
            #l_g_gan: generator gan loss
            #l_g_total += l_g_gan
            #l_g_total.backward()
            return l_g_gan

        else: # elif stage == 'discriminator':
            # updating discriminator
            pred_d_real = netD(real)
            pred_d_fake = netD(fake.detach())  # detach to avoid backpropagation to G
            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)

            l_d_total = (l_d_real + l_d_fake) / 2

            #logs for losses and D outputs
            gan_logs = {
                    'l_d_real': l_d_real.item(),
                    'l_d_fake': l_d_fake.item(),
                    'D_real': torch.mean(pred_d_real.detach()).item(),
                    'D_fake': torch.mean(pred_d_fake.detach()).item()
                    }

            #if self.opt['train']['gan_type'] == 'wgan-gp': #TODO
            if self.gan_type == 'wgan-gp':
                batch_size = real.size(0)
                if self.random_pt.size(0) != batch_size:
                    self.random_pt.resize_(batch_size, 1, 1, 1)
                self.random_pt.uniform_()  # Draw random interpolation points
                interp = self.random_pt * fake.detach() + (1 - self.random_pt) * real
                interp.requires_grad = True
                interp_crit = netD(interp)
                l_d_gp = self.l_gp_w * self.cri_gp(interp, interp_crit)
                l_d_total += l_d_gp
                # attach gradient penalty loss to log
                gan_logs['l_d_gp'] = l_d_gp.item()
            
            #l_d_gan: discriminator gan loss
            #l_d_total.backward()
            #self.optimizers.optimizer_D.step()
            return l_d_total, gan_logs


class GeneratorLoss(nn.Module):
    def __init__(self, opt=None, device = 'cpu'):
        super(GeneratorLoss, self).__init__()

        train_opt = opt['train']

        pixel_weight = train_opt['pixel_weight'] if train_opt['pixel_weight'] else 0
        pixel_criterion = train_opt['pixel_criterion'] if train_opt['pixel_criterion'] else None # 'skip'

        feature_weight = train_opt['feature_weight'] if train_opt['feature_weight'] else 0
        feature_network = train_opt['feature_network'] if train_opt['feature_network'] else 'vgg19'  # TODO 
        #feature_criterion = train_opt['feature_criterion'] if train_opt['feature_criterion'] else None # 'skip'
        feature_criterion = check_loss_names(feature_criterion=train_opt['feature_criterion'], feature_network=feature_network)
        
        hfen_weight = train_opt['hfen_weight'] if train_opt['hfen_weight'] else 0
        hfen_criterion = check_loss_names(hfen_criterion=train_opt['hfen_criterion'])

        tv_weight = train_opt['tv_weight'] if train_opt['tv_weight'] else 0
        tv_type = check_loss_names(tv_type=train_opt['tv_type'], tv_norm=train_opt['tv_norm'])

        ssim_weight = train_opt['ssim_weight'] if train_opt['ssim_weight'] else 0
        ssim_type = train_opt['ssim_type'] if train_opt['ssim_type'] else None
        
        lpips_weight = train_opt['lpips_weight'] if train_opt['lpips_weight'] else 0
        lpips_network = train_opt['lpips_net'] if train_opt['lpips_net'] else 'vgg'
        lpips_type = train_opt['lpips_type'] if train_opt['lpips_type'] else 'net-lin'
        lpips_criterion = check_loss_names(lpips_criterion=train_opt['lpips_type'], lpips_network=lpips_network)

        spl_weight = train_opt['spl_weight'] if train_opt['spl_weight'] else 0
        spl_type = train_opt['spl_type'] if train_opt['spl_type'] else None
        if spl_type == 'spl':
            cpl_type = 'cpl'
            cpl_weight = spl_weight
            gpl_type = 'gpl'
            gpl_weight = spl_weight
        elif spl_type == 'cpl':
            cpl_type = 'cpl'
            cpl_weight = spl_weight
        elif spl_type == 'gpl':
            gpl_type = 'gpl'
            gpl_weight = spl_weight

        cx_weight = train_opt['cx_weight'] if train_opt['cx_weight'] else 0
        cx_type = train_opt['cx_type'] if train_opt['cx_type'] else None

        '''
        #old
        //, "dis_feature_criterion": "l1" //"l1" | "l2" | "cb" | "elastic" //discriminator feature loss (only for asrragan)
        //, "dis_feature_weight": 1 //(only for asrragan)

        #new
        #TODO
        '''  

        #fixed
        # TODO      
        

        self.loss_list = []

        if pixel_weight > 0 and pixel_criterion:
            cri_pix = get_loss_fn(pixel_criterion, pixel_weight) 
            self.loss_list.append(cri_pix)

        if hfen_weight > 0 and hfen_criterion:
            cri_hfen = get_loss_fn(hfen_criterion, hfen_weight)
            self.loss_list.append(cri_hfen)

        if ssim_weight > 0 and ssim_type:
            cri_ssim = get_loss_fn(ssim_type, ssim_weight)
            self.loss_list.append(cri_ssim)
        
        if tv_weight > 0 and tv_type:
            cri_tv = get_loss_fn(tv_type, tv_weight)
            self.loss_list.append(cri_tv)

        if cx_weight > 0 and cx_type:
            cri_cx = get_loss_fn(cx_type, cx_weight, device = device)
            self.loss_list.append(cri_cx)

        if feature_weight > 0 and feature_criterion:
            self.netF = networks.define_F(opt, use_bn=False).to(device)
            cri_fea = get_loss_fn(feature_criterion, feature_weight, network=self.netF)
            self.loss_list.append(cri_fea)
            self.cri_fea = True
        else: 
            self.cri_fea = None

        if lpips_weight > 0 and lpips_criterion:
            lpips_spatial = True #False # Return a spatial map of perceptual distance. Needs to use .mean() for the backprop if True, the mean distance is approximately the same as the non-spatial distance
            #self.netF = networks.define_F(opt, use_bn=False).to(device)
            # TODO: fix use_gpu 
            lpips_network  = ps.PerceptualLoss(model=lpips_type, net=lpips_network, use_gpu=torch.cuda.is_available(), model_path=None, spatial=lpips_spatial) #.to(self.device) 
            cri_lpips = get_loss_fn(lpips_criterion, lpips_weight, network=lpips_network)
            self.loss_list.append(cri_lpips)
            #self.cri_lpips = True #self.cri_fea
        #else: 
            #self.cri_fea = None #self.cri_fea
            
        '''
            if train_opt['lpips_weight']:
                if z_norm == True: # if images are in [-1,1] range
                    self.lpips_norm = False # images are already in the [-1,1] range
                else:
                    self.lpips_norm = True # normalize images from [0,1] range to [-1,1]
                    
                self.l_lpips_w = train_opt['lpips_weight']
                # Can use original off-the-shelf uncalibrated networks 'net' or Linearly calibrated models (LPIPS) 'net-lin'
                if train_opt['lpips_type']:
                    lpips_type = train_opt['lpips_type']
                else: # Default use linearly calibrated models, better results
                    lpips_type = 'net-lin'
                # Can set net = 'alex', 'squeeze' or 'vgg' or Low-level metrics 'L2' or 'ssim'
                if train_opt['lpips_net']:
                    lpips_net = train_opt['lpips_net']
                else: # Default use VGG for feature extraction
                    lpips_net = 'vgg'
                self.cri_lpips = 
        '''

        #self.log_dict = OrderedDict()


    def forward(self, sr, hr, log_dict): 
        loss_results = []
        for i, l in enumerate(self.loss_list):
            if l['function']:
                if l['name'].find('tv') >= 0:
                    effective_loss = l['weight']*l['function'](sr) # fake_H
                elif l['name'].find('ssim') >= 0:
                    effective_loss = 1 - l['weight']*l['function'](sr, hr) # (fake_H, var_H)
                else:    
                    effective_loss = l['weight']*l['function'](sr, hr) # (fake_H, var_H)
                #print(l['name'],effective_loss)
                loss_results.append(effective_loss)
                log_dict[l['name']] = effective_loss.item()
        return loss_results, log_dict