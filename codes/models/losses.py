import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from collections import OrderedDict

from models.modules.loss import *
from models.modules.ssim import SSIM, MS_SSIM
from models.modules.LPIPS import perceptual_loss as ps
import models.networks as networks
from dataops.diffaug import DiffAugment

from dataops.debug import *



# loss builder
def get_loss_fn(loss_type=None,
                weight=0,
                recurrent=False,
                reduction='mean',
                network = None,
                device = 'cuda',
                opt = None,
                allow_featnets = True):
    if loss_type == 'skip':
        loss_function = None
    # pixel / content losses
    if loss_type in ('MSE', 'l2'):
        loss_function = nn.MSELoss(reduction=reduction)
        loss_type = 'pix-{}'.format(loss_type)
    elif loss_type in ('L1', 'l1'):
        loss_function = nn.L1Loss(reduction=reduction)
        loss_type = 'pix-{}'.format(loss_type)
    elif loss_type == 'cb':
        loss_function = CharbonnierLoss()
        loss_type = 'pix-{}'.format(loss_type)
    elif loss_type == 'elastic':
        loss_function = ElasticLoss(reduction=reduction)
        loss_type = 'pix-{}'.format(loss_type)
    elif loss_type == 'relativel1':
        loss_function = RelativeL1(reduction=reduction)
        loss_type = 'pix-{}'.format(loss_type)
    # TODO
    # elif loss_type == 'relativel2':
        # loss_function = RelativeL2(reduction=reduction)
        # loss_type = 'pix-{}'.format(loss_type)
    elif loss_type in ('l1cosinesim', 'L1CosineSim'):
        loss_function = L1CosineSim(reduction=reduction)
        loss_type = 'pix-{}'.format(loss_type)
    elif loss_type == 'clipl1':
        loss_function = ClipL1()
        loss_type = 'pix-{}'.format(loss_type)
    # Multiscale content/pixel loss
    # elif loss_type == 'multiscale-l1':
    elif loss_type.find('multiscale') >= 0:
        ms_loss_f = get_loss_fn(loss_type.split('-')[1], recurrent=True)
        loss_function = MultiscalePixelLoss(loss_f=ms_loss_f)
        loss_type = 'pix-{}'.format(loss_type)
    # SSIM losses
    # TODO: pass SSIM options from opt_train
    elif loss_type in ('ssim', 'SSIM'):  # l_ssim_type
        if not allow_featnets:
            image_channels = 1
        else:
            image_channels = opt['image_channels'] if opt['image_channels'] else 3
        loss_function = SSIM(window_size=11, window_sigma=1.5, size_average=True, data_range=1., channels=image_channels)
    elif loss_type in ('ms-ssim', 'MSSSIM'):  # l_ssim_type
        if not allow_featnets:
            image_channels = 1
        else:
            image_channels = opt['image_channels'] if opt['image_channels'] else 3
        loss_function = MS_SSIM(window_size=11, window_sigma=1.5, size_average=True, data_range=1., channels=image_channels, normalize='relu')
    # HFEN loss
    elif loss_type.find('hfen') >= 0:
        hfen_loss_f = get_loss_fn(loss_type.split('-')[1], recurrent=True, reduction='sum')
        # print(hfen_loss_f)
        # TODO: can pass function options from opt_train
        loss_function = HFENLoss(loss_f=hfen_loss_f)
    # Gradient loss
    elif loss_type.find('grad') >= 0:
        gradientdir = loss_type.split('-')[1]
        grad_loss_f = get_loss_fn(loss_type.split('-')[2], recurrent=True)
        # TODO: can pass function options from opt_train
        loss_function = GradientLoss(loss_f=grad_loss_f, gradientdir=gradientdir)
    # SPL losses
    # Gradient Profile Loss
    elif loss_type == 'gpl':
        # TODO fix parameters to training variables (spl_denorm, yuv_denorm)
        # currently won't work in range [-1,1]
        loss_function = GPLoss(spl_denorm=False)
    # Color Profile Loss
    elif loss_type == 'cpl':
        # TODO fix parameters to training variables (spl_denorm, yuv_denorm)
        # currently won't work in range [-1,1]
        loss_function = CPLoss(rgb=True,yuv=True,yuvgrad=True,spl_denorm=False,yuv_denorm=False)
    # TV regularization
    elif loss_type.find('tv') >= 0:
        tv_type = loss_type.split('-')[0]
        tv_norm = loss_type.split('-')[1]
        if tv_norm == 'l1':
            tv_norm = 1
        elif tv_norm == 'l2':
            tv_norm = 2
        if tv_type == 'tv':
            loss_function = TVLoss(tv_type = 'tv', p = tv_norm)
        elif tv_type == 'dtv': 
            loss_function = TVLoss(tv_type = 'dtv', p = tv_norm)  # DTV
    # Feature loss
    # fea-vgg19-l1, fea-vgg16-l2, fea-lpips-... ("vgg" | "alex" | "squeeze" / net-lin | net )
    # TODO: pass rotation, flips and other options, from opt_train, else defaults
    elif loss_type.find('fea') >= 0:
        if loss_type.split('-')[1] == 'lpips':
            # TODO: make lpips behave more like regular feature networks
            # lpips needs normalize = true if images are in [0,1] range
            norm = opt['datasets']['train'].get('znorm', False)
            loss_function = PerceptualLoss(criterion='lpips', network=network, normalize=(not norm), rotations=False, flips=False)
        else: #if loss_type.split('-')[1][:3] == 'vgg': #if vgg16, vgg19, resnet, etc
            fea_loss_f = get_loss_fn(loss_type.split('-')[2], recurrent=True, reduction='mean')  # sum? also worked on accidental test, but larger loss magnitudes
            # TODO
            # network = networks.define_F(opt, use_bn=False).to(self.device)
            # TMP #loss_type.split('-')[1] # = vgg16, vgg19, resnet, ...
            # network = define_F(use_bn=False, feat_network = loss_type.split('-')[1])
            loss_function = PerceptualLoss(criterion=fea_loss_f, network=network, rotations=False, flips=False)
    # Contextual Loss
    elif loss_type == 'contextual':
        layers = opt.get('cx_vgg_layers', {"conv_3_2": 1.0, "conv_4_2": 1.0})
        loss_function = Contextual_Loss(layers, max_1d_size=64, distance_type = 'cosine', calc_type = 'regular')
        # loss_function = Contextual_Loss(layers, max_1d_size=32, distance_type = 0, crop_quarter=True) # for L1, L2
    elif loss_type == 'fft':
        loss_function = FFTloss()
    elif loss_type == 'overflow':
        loss_function = OFLoss()
    # Range limiting loss
    elif loss_type == 'range':
        legit_range = [-1,1] if opt['datasets']['train'].get('znorm', False) else [0,1]
        loss_function = RangeLoss(legit_range=legit_range)
    elif loss_type.find('color') >= 0:
        color_loss_f = get_loss_fn(loss_type.split('-')[1], recurrent=True)
        ds_f = torch.nn.AvgPool2d(kernel_size=opt['scale'])
        loss_function = ColorLoss(loss_f=color_loss_f, ds_f=ds_f) 
    elif loss_type.find('avg') >= 0:
        avg_loss_f = get_loss_fn(loss_type.split('-')[1], recurrent=True)
        ds_f = torch.nn.AvgPool2d(kernel_size=opt['scale'])
        loss_function = AverageLoss(loss_f=avg_loss_f, ds_f=ds_f)
    elif loss_type == 'fdpl':
        diff_means = opt.get('diff_means', "./models/modules/FDPL/diff_means.pt")
        loss_function = FDPLLoss(dataset_diff_means_file=diff_means, device=device)
    else:
        loss_function = None
        # raise NotImplementedError('Loss type [{:s}] not recognized.'.format(loss_type))

    if loss_function:
        if recurrent:
            return loss_function.to(device)
        else:
            loss = {
                    'name': loss_type,
                    'weight': float(weight), # TODO: check if float is needed
                    'function': loss_function.to(device)}
            return loss
                    

def check_loss_names(pixel_criterion=None, feature_criterion=None, feature_network=None, hfen_criterion=None, tv_type=None, tv_norm=None, lpips_criterion=None, lpips_network = None):
    """
        standardize the losses names, for backward compatibility
        (should be temporary and eventually deprecated for the new names)
    """

    if pixel_criterion:
        return 'pix-{}'.format(pixel_criterion.lower())

    # , "feature_criterion": "l1" //"l1" | "l2" | "cb" | "elastic" //feature loss (VGG feature network)
    if feature_criterion:
        return 'fea-{}-{}'.format(feature_network.lower(), feature_criterion.lower())

    # //, "dis_feature_criterion": "l1" //"l1" | "l2" | "cb" | "elastic" //discriminator feature loss (only for asrragan)
    # TODO

    # //, "hfen_criterion": "l1" //hfen: "l1" | "l2" | "rel_l1" | "rel_l2" //helps in deblurring and finding edges, lines
    if hfen_criterion:
        if hfen_criterion in ('rel_l1', 'rel_l2'): # TODO, rel_l2 not available, easy to do
            hfen_criterion = 'hfen-relativel1'
        # elif hfen_criterion == 'rel_l2':
            # hfen_criterion = 'hfen-relativel2'
            return hfen_criterion
        else:
            return 'hfen-{}'.format(hfen_criterion.lower())

    # //, "tv_type": "normal" //helps in denoising, reducing upscale artefacts
    if tv_type and tv_norm:
        # get norm
        if tv_norm in (1, 'L1'):
            tv_norm = 'l1'
        elif tv_norm in (2, 'L2'):
            tv_norm = 'l2'
        # get type
        if tv_type == 'normal':
            tv_type = 'tv'
        elif tv_type == '4D':
            tv_type = 'dtv'
        # final combined type
        tv_type = '{}-{}'.format(tv_type, tv_norm)
        return tv_type
    
    if lpips_criterion and lpips_network:
        if lpips_criterion.split('-')[0] != 'fea':
            return 'fea-lpips-{}-{}'.format(lpips_criterion,lpips_network)

    return None


class PerceptualLoss(nn.Module):
    def __init__(self, criterion=None, network=None, normalize=True, rotations=False, flips=False):
        super(PerceptualLoss, self).__init__()
        self.criterion = criterion
        self.network = network
        self.rotations = rotations
        self.flips = flips
        self.normalize = normalize

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
            return self.network(x, y, normalize=self.normalize).mean()
        else:
            fea_x = self.network(x)
            fea_y = self.network(y).detach()
            return self.criterion(fea_x, fea_y)



class Adversarial(nn.Module):
    def __init__(self, train_opt=None, device = 'cpu', diffaug = False, dapolicy = '',
                 conditional=False):
        super(Adversarial, self).__init__()

        self.device = device
        self.diffaug = diffaug
        self.dapolicy = dapolicy
        self.conditional = conditional
        self.gan_type = train_opt['gan_type']
        self.use_featmaps  = train_opt.get('gan_featmaps', None)
        if self.use_featmaps:
            dis_feature_criterion  = train_opt.get('dis_feature_criterion', 'l1')
            dis_feature_weight  = train_opt.get('dis_feature_weight', 0.0001)
            self.cri_disfea = get_loss_fn(dis_feature_criterion, dis_feature_weight)

        self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
        self.l_gan_w = train_opt['gan_weight']
        # self.simple_dis = True  # TODO: non-relativistic GAN TMP check

        if self.gan_type == 'wgan-gp':
            self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
            # gradient penalty loss
            self.cri_gp = GradientPenaltyLoss(device=self.device).to(self.device)
            self.l_gp_w = train_opt['gp_weigth']


    def forward(self, fake, real, realB=None, netD=None, stage='discriminator',
                fsfilter=None):  # (fake_H,self.var_ref)
        # Note: "fake" is fakeB, "real" is realA

        if self.conditional:
            # using conditional GANs, we need to feed both input and output to the discriminator
            if stage == 'generator':
                # updating generator
                # G(A) should fake the discriminator
                fake_AB = torch.cat((real, fake), 1)  # Fake G(A)

                # TODO: test
                # if fsfilter: # filter
                #     fake_AB = fsfilter(fake_AB)
                # if self.diffaug:
                #     fake_AB = DiffAugment(fake_AB, policy=self.dapolicy)

                pred_g_fake = netD(fake_AB)
                l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
                return l_g_gan
            else:  # elif stage == 'discriminator':
                # updating discriminator
                fake_AB = torch.cat((real, fake), 1)  # Fake G(A)
                real_AB = torch.cat((real, realB), 1)  # Real B

                # TODO: test
                # if fsfilter: # filter
                #     fake_AB = fsfilter(fake_AB)
                #     real_AB = fsfilter(real_AB)
                # if self.diffaug:
                #     fake_AB = DiffAugment(fake_AB, policy=self.dapolicy)
                #     real_AB = DiffAugment(real_AB, policy=self.dapolicy)

                # stop backprop to the generator by detaching fake_AB
                pred_d_fake = netD(fake_AB.detach())
                l_d_fake = self.cri_gan(pred_d_fake, False)

                pred_d_real = netD(real_AB)
                l_d_real = self.cri_gan(pred_d_real, True)

                # combine loss and log
                l_d_total = (l_d_fake + l_d_real) * 0.5

                gan_logs = {
                    'l_d_real': l_d_real.item(),
                    'l_d_fake': l_d_fake.item(),
                    'D_real': torch.mean(pred_d_real.detach()).item(),
                    'D_fake': torch.mean(pred_d_fake.detach()).item()
                    }

                return l_d_total, gan_logs

        else:
            # use regular discriminator for real and fake samples

            # apply frequency separation
            if fsfilter:  # filter
                real = fsfilter(real)
                fake = fsfilter(fake)

            # apply differential augmentations
            if self.diffaug:
                real = DiffAugment(real, policy=self.dapolicy)
                fake = DiffAugment(fake, policy=self.dapolicy)

            # tmp_vis(real) # to visualize the batch for debugging
            # tmp_vis(fake) # to visualize the batch for debugging

            if stage == 'generator':
                # updating generator
                # for the Generator only if GAN is enabled, everything else can happen any time
                # G gan + cls loss
                if self.use_featmaps:  # extrating feature maps from discriminator
                    pred_g_real = netD(real, return_maps=self.use_featmaps)
                    # TODO: test. if is a list, its [pred_g_real, feats_d_real], else its just pred_g_real
                    if isinstance(pred_g_real, list):
                        feats_d_real = pred_g_real[1]
                        pred_g_real = pred_g_real[0].detach()  # detach to avoid backpropagation to D

                    pred_g_fake = netD(fake, return_maps=self.use_featmaps)
                    # TODO: test. if is a list, its [pred_g_fake, feats_d_fake], else its just pred_g_fake
                    if isinstance(pred_g_fake, list) and self.use_featmaps:
                        feats_d_fake = pred_g_fake[1]
                        pred_g_fake = pred_g_fake[0]
                else:  # normal gan
                    pred_g_real = netD(real)  # .detach() # detach to avoid backpropagation to D
                    pred_g_fake = netD(fake)

                if isinstance(pred_g_real, list) and isinstance(pred_g_fake, list):
                    # for multiscale discriminator
                    l_g_gan = 0
                    for preds in zip(pred_g_real, pred_g_fake):
                        l_g_gan += self.l_gan_w * (self.cri_gan(preds[0][0].detach() - torch.mean(preds[1][0]), False) +
                                            self.cri_gan(preds[1][0] - torch.mean(preds[0][0].detach()), True)) / 2
                # TODO: TMP check
                # elif self.simple_dis:
                #     # pred_g_real not needed if non-relativistic
                #     l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
                else:  # regular single scale discriminators
                    pred_g_real = pred_g_real.detach()  # detach to avoid backpropagation to D
                    l_g_gan = self.l_gan_w * (self.cri_gan(pred_g_real - torch.mean(pred_g_fake), False) +
                                                self.cri_gan(pred_g_fake - torch.mean(pred_g_real), True)) / 2

                # SRPGAN-like Features Perceptual loss, extracted from the discriminator
                if self.use_featmaps:
                    l_g_disfea = 0
                    for hr_feat_map, sr_feat_map in zip(feats_d_fake, feats_d_real):
                        l_g_disfea += self.cri_disfea['function'](sr_feat_map, hr_feat_map)
                    l_g_disfea = (self.cri_disfea['weight']*l_g_disfea)/len(feats_d_real)
                    l_g_gan += l_g_disfea

                return l_g_gan

            else:  # elif stage == 'discriminator':
                # updating discriminator
                pred_d_real = netD(real)  # Real
                pred_d_fake = netD(fake.detach())  # detach Fake to avoid backpropagation to G

                if isinstance(pred_d_real, list) and isinstance(pred_d_fake, list):
                    # for multiscale discriminator
                    l_d_real = 0
                    l_d_fake = 0
                    for preds in zip(pred_d_real, pred_d_fake):
                        l_d_real += self.cri_gan(preds[0][0] - torch.mean(preds[1][0]), True)
                        l_d_fake += self.cri_gan(preds[1][0] - torch.mean(preds[0][0]), False)
                    pred_d_real = pred_d_real[0][0]  # leave only the largest D for the logs
                    pred_d_fake = pred_d_fake[0][0]  # leave only the largest D for the logs
                # TODO: TMP check
                # elif self.simple_dis:
                #     l_d_real = self.criterionGAN(pred_d_real, True)
                #     l_d_fake = self.criterionGAN(pred_d_fake, False)
                else:  # regular single scale discriminators
                    l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                    l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)

                # combined loss
                l_d_total = (l_d_real + l_d_fake) / 2

                # logs for losses and D outputs
                gan_logs = {
                        'l_d_real': l_d_real.item(),
                        'l_d_fake': l_d_fake.item(),
                        'D_real': torch.mean(pred_d_real.detach()).item(),
                        'D_fake': torch.mean(pred_d_fake.detach()).item()
                        }

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

                return l_d_total, gan_logs


class GeneratorLoss(nn.Module):
    def __init__(self, opt=None, device = 'cpu', allow_featnets=True):
        super(GeneratorLoss, self).__init__()

        train_opt = opt['train']

        #TODO: these checks can be moved to options.py when everything is stable
        # parsing the losses options
        pixel_weight  = train_opt.get('pixel_weight', 0)
        pixel_criterion  = train_opt.get('pixel_criterion', None) # 'skip'

        if allow_featnets:
            feature_weight = train_opt.get('feature_weight', 0)
            feature_network = train_opt.get('feature_network', 'vgg19') # TODO 
            feature_criterion = check_loss_names(feature_criterion=train_opt['feature_criterion'], feature_network=feature_network)
        else:
            feature_weight = 0
        
        hfen_weight  = train_opt.get('hfen_weight', 0)
        hfen_criterion = check_loss_names(hfen_criterion=train_opt['hfen_criterion'])

        # grad_weight  = train_opt.get('grad_weight', 0)
        # grad_type  = train_opt.get('grad_type', None) 

        tv_weight  = train_opt.get('tv_weight', 0)
        tv_type = check_loss_names(tv_type=train_opt['tv_type'], tv_norm=train_opt['tv_norm'])

        # ssim_weight  = train_opt.get('ssim_weight', 0)
        # ssim_type  = train_opt.get('ssim_type', None)

        if allow_featnets:
            lpips_weight  = train_opt.get('lpips_weight', 0)
            lpips_network  = train_opt.get('lpips_net', 'vgg')
            lpips_type  = train_opt.get('lpips_type', 'net-lin')
            lpips_criterion = check_loss_names(lpips_criterion=train_opt['lpips_type'], lpips_network=lpips_network)
        else:
            lpips_weight = 0

        color_weight  = train_opt.get('color_weight', 0)
        color_criterion  = train_opt.get('color_criterion', None)

        avg_weight  = train_opt.get('avg_weight', 0)
        avg_criterion  = train_opt.get('avg_criterion', None)

        ms_weight  = train_opt.get('ms_weight', 0)
        ms_criterion  = train_opt.get('ms_criterion', None)

        spl_weight  = train_opt.get('spl_weight', 0)
        spl_type  = train_opt.get('spl_type', None)

        gpl_type = None
        gpl_weight = -1
        cpl_type = None
        cpl_weight = -1
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

        if allow_featnets:
            cx_weight  = train_opt.get('cx_weight', 0)
            cx_type  = train_opt.get('cx_type', None)
        else:
            cx_weight = 0

        # fft_weight  = train_opt.get('fft_weight', 0)
        # fft_type  = train_opt.get('fft_type', None)

        of_weight  = train_opt.get('of_weight', 0)
        of_type  = train_opt.get('of_type', None)

        # building the loss
        self.loss_list = []

        if pixel_weight > 0 and pixel_criterion:
            cri_pix = get_loss_fn(pixel_criterion, pixel_weight) 
            self.loss_list.append(cri_pix)

        if hfen_weight > 0 and hfen_criterion:
            cri_hfen = get_loss_fn(hfen_criterion, hfen_weight)
            self.loss_list.append(cri_hfen)
        
        # if grad_weight > 0 and grad_type:
        #     cri_grad = get_loss_fn(grad_type, grad_weight, device = device)
        #     self.loss_list.append(cri_grad)

        # if ssim_weight > 0 and ssim_type:
        #     cri_ssim = get_loss_fn(ssim_type, ssim_weight, opt = train_opt, allow_featnets = allow_featnets)
        #     self.loss_list.append(cri_ssim)
        
        if tv_weight > 0 and tv_type:
            cri_tv = get_loss_fn(tv_type, tv_weight)
            self.loss_list.append(cri_tv)

        if cx_weight > 0 and cx_type:
            cri_cx = get_loss_fn(cx_type, cx_weight, device = device, opt = train_opt)
            self.loss_list.append(cri_cx)

        if feature_weight > 0 and feature_criterion:
            #TODO: can move the self.netF to the loss class instead, like lpips, change where the network is printed from
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
            cri_lpips = get_loss_fn(lpips_criterion, lpips_weight, network=lpips_network, opt = opt)
            self.loss_list.append(cri_lpips)

        if  cpl_weight > 0 and cpl_type:
            cri_cpl = get_loss_fn(cpl_type, cpl_weight) 
            self.loss_list.append(cri_cpl)

        if  gpl_weight > 0 and gpl_type:
            cri_gpl = get_loss_fn(gpl_type, gpl_weight) 
            self.loss_list.append(cri_gpl)

        # if fft_weight > 0 and fft_type:
        #     cri_fft = get_loss_fn(fft_type, fft_weight, device = device)
        #     self.loss_list.append(cri_fft)

        if of_weight > 0 and of_type:
            cri_of = get_loss_fn(of_type, of_weight, device = device)
            self.loss_list.append(cri_of)

        if color_weight > 0 and color_criterion:
            cri_color = get_loss_fn(color_criterion, color_weight, opt = opt) 
            self.loss_list.append(cri_color)

        if avg_weight > 0 and avg_criterion:
            cri_avg = get_loss_fn(avg_criterion, avg_weight, opt = opt) 
            self.loss_list.append(cri_avg)
        
        if ms_weight > 0 and ms_criterion:
            cri_avg = get_loss_fn(ms_criterion, ms_weight, opt = opt) 
            self.loss_list.append(cri_avg)


    def forward(self, sr, hr, log_dict, fsfilter=None, selector=None):
        if fsfilter: #low-pass filter
            hr_f = fsfilter(hr)
            sr_f = fsfilter(sr)
        
        # to visualize the batch for debugging
        #tmp_vis(sr)
        #tmp_vis(hr)
        #tmp_vis(sr_f)
        #tmp_vis(hr_f)

        if selector and isinstance(selector, list):
            loss_list = []
            for selected_loss in selector:
                for i, l in enumerate(self.loss_list):
                    if l['function']:
                        if l['name'].find(selected_loss) >= 0:
                            if selected_loss == 'pix' and l['name'].find('pix-multiscale') >= 0:
                                continue
                            loss_list.append(l)
        else:
            loss_list = self.loss_list

        loss_results = []
        for i, l in enumerate(loss_list):
            if l['function']:
                if fsfilter: # branch selecting the ones that should use LPF
                    if l['name'].find('tv') >= 0 or l['name'].find('overflow') >= 0 \
                            or l['name'].find('range') >= 0:
                        #Note: Using sr_f here means it will only preserve total variation / denoise on 
                        # the LF, which may or may not be a valid proposition, test!
                        effective_loss = l['weight']*l['function'](sr_f) # fake_H
                    elif l['name'].find('pix') >= 0 or l['name'].find('hfen') >= 0 \
                            or l['name'].find('cpl') >= 0 or l['name'].find('gpl') >= 0 \
                            or l['name'].find('gradient') >= 0 or l['name'].find('fdpl') >= 0:
                        effective_loss = l['weight']*l['function'](sr_f, hr_f) # (fake_H, var_H)
                    elif l['name'].find('ssim') >= 0:
                        effective_loss = l['weight']*(1 - l['function'](sr_f, hr_f)) # (fake_H, var_H)
                    else:
                        effective_loss = l['weight']*l['function'](sr, hr) # (fake_H, var_H)
                else:
                    if l['name'].find('tv') >= 0 or l['name'].find('overflow') >= 0 \
                            or l['name'].find('range') >= 0:
                        effective_loss = l['weight']*l['function'](sr) # fake_H
                    elif l['name'].find('ssim') >= 0:
                        effective_loss = l['weight']*(1 - l['function'](sr, hr)) # (fake_H, var_H)
                    else:
                        effective_loss = l['weight']*l['function'](sr, hr) # (fake_H, var_H)
                #print(l['name'],effective_loss)
                loss_results.append(effective_loss)
                log_dict[l['name']] = effective_loss.item()
        return loss_results, log_dict

class PreciseGeneratorLoss(nn.Module):
    """
    To separately instantiate losses that require high precision calculations and 
        cannot be inside the AMP context. 
        In principle, it's the same as GeneratorLoss, but each one initialize 
        different losses if configured in the options.
        If not using AMP, both classes will be equivalent.
    """
    def __init__(self, opt=None, device = 'cpu', allow_featnets=True):
        super(PreciseGeneratorLoss, self).__init__()

        train_opt = opt['train']

        #TODO: these checks can be moved to options.py when everything is stable
        # parsing the losses options
        # pixel_weight  = train_opt.get('pixel_weight', 0)
        # pixel_criterion  = train_opt.get('pixel_criterion', None) # 'skip'

        # if allow_featnets:
        #     feature_weight = train_opt.get('feature_weight', 0)
        #     feature_network = train_opt.get('feature_network', 'vgg19') # TODO 
        #     feature_criterion = check_loss_names(feature_criterion=train_opt['feature_criterion'], feature_network=feature_network)
        # else:
        #     feature_weight = 0
        
        # hfen_weight  = train_opt.get('hfen_weight', 0)
        # hfen_criterion = check_loss_names(hfen_criterion=train_opt['hfen_criterion'])

        grad_weight  = train_opt.get('grad_weight', 0)
        grad_type  = train_opt.get('grad_type', None) 

        # tv_weight  = train_opt.get('tv_weight', 0)
        # tv_type = check_loss_names(tv_type=train_opt['tv_type'], tv_norm=train_opt['tv_norm'])

        ssim_weight  = train_opt.get('ssim_weight', 0)
        ssim_type  = train_opt.get('ssim_type', None)

        # if allow_featnets:
        #     lpips_weight  = train_opt.get('lpips_weight', 0)
        #     lpips_network  = train_opt.get('lpips_net', 'vgg')
        #     lpips_type  = train_opt.get('lpips_type', 'net-lin')
        #     lpips_criterion = check_loss_names(lpips_criterion=train_opt['lpips_type'], lpips_network=lpips_network)
        # else:
        #     lpips_weight = 0

        # if allow_featnets:
        #     cx_weight  = train_opt.get('cx_weight', 0)
        #     cx_type  = train_opt.get('cx_type', None)
        # else:
        #     cx_weight = 0

        fft_weight  = train_opt.get('fft_weight', 0)
        fft_type  = train_opt.get('fft_type', None)

        fdpl_weight  = train_opt.get('fdpl_weight', 0)
        fdpl_type  = train_opt.get('fdpl_type', None)

        range_weight = train_opt.get('range_weight', 0) 
        range_type = 'range'


        # building the loss
        self.loss_list = []

        # if hfen_weight > 0 and hfen_criterion:
        #     cri_hfen = get_loss_fn(hfen_criterion, hfen_weight)
        #     self.loss_list.append(cri_hfen)
        
        if grad_weight > 0 and grad_type:
            cri_grad = get_loss_fn(grad_type, grad_weight, device = device)
            self.loss_list.append(cri_grad)

        if ssim_weight > 0 and ssim_type:
            cri_ssim = get_loss_fn(ssim_type, ssim_weight, opt = train_opt, allow_featnets = allow_featnets)
            self.loss_list.append(cri_ssim)
        
        # if tv_weight > 0 and tv_type:
        #     cri_tv = get_loss_fn(tv_type, tv_weight)
        #     self.loss_list.append(cri_tv)

        # if cx_weight > 0 and cx_type:
        #     cri_cx = get_loss_fn(cx_type, cx_weight, device = device, opt = train_opt)
        #     self.loss_list.append(cri_cx)

        # if feature_weight > 0 and feature_criterion:
        #     #TODO: can move the self.netF to the loss class instead, like lpips, change where the network is printed from
        #     self.netF = networks.define_F(opt, use_bn=False).to(device)
        #     cri_fea = get_loss_fn(feature_criterion, feature_weight, network=self.netF)
        #     self.loss_list.append(cri_fea)
        #     self.cri_fea = True
        # else: 
        #     self.cri_fea = None

        # if lpips_weight > 0 and lpips_criterion:
        #     lpips_spatial = True #False # Return a spatial map of perceptual distance. Needs to use .mean() for the backprop if True, the mean distance is approximately the same as the non-spatial distance
        #     #self.netF = networks.define_F(opt, use_bn=False).to(device)
        #     # TODO: fix use_gpu 
        #     lpips_network  = ps.PerceptualLoss(model=lpips_type, net=lpips_network, use_gpu=torch.cuda.is_available(), model_path=None, spatial=lpips_spatial) #.to(self.device) 
        #     cri_lpips = get_loss_fn(lpips_criterion, lpips_weight, network=lpips_network, opt = opt)
        #     self.loss_list.append(cri_lpips)

        if fft_weight > 0 and fft_type:
            cri_fft = get_loss_fn(fft_type, fft_weight, device = device)
            self.loss_list.append(cri_fft)

        if fdpl_weight > 0 and fdpl_type:
            cri_fdpl = get_loss_fn(fdpl_type, fdpl_weight, device = device, opt = train_opt)
            self.loss_list.append(cri_fdpl)

        if range_weight > 0 and range_type:
            cri_range = get_loss_fn(range_type, range_weight, device = device, opt = opt)
            self.loss_list.append(cri_range)

    def forward(self, sr, hr, log_dict, fsfilter=None, selector=None):
        # make sure both sr and hr are not in float16 or int, change to float32 (torch.float32/torch.float)
        # in some cases could even need torch.float64/torch.double, may have to add the case here
        if sr.dtype in (torch.float16, torch.int8, torch.int32):
            sr = sr.float()
        if hr.dtype in (torch.float16, torch.int8, torch.int32):
            hr = hr.float()
        
        assert sr.dtype == hr.dtype, \
            'Error: SR and HR have different precision in precise losses: {} and {}'.format(sr.dtype, hr.dtype)
        assert sr.type() == hr.type(), \
            'Error: SR and HR are on different devices in precise losses: {} and {}'.format(sr.type(), hr.type())
        
        if fsfilter: #low-pass filter
            hr_f = fsfilter(hr)
            sr_f = fsfilter(sr)
        
        # to visualize the batch for debugging
        #tmp_vis(sr)
        #tmp_vis(hr)
        #tmp_vis(sr_f)
        #tmp_vis(hr_f)

        if selector and isinstance(selector, list):
            loss_list = []
            for selected_loss in selector:
                for i, l in enumerate(self.loss_list):
                    if l['function']:
                        if l['name'].find(selected_loss) >= 0:
                            if selected_loss == 'pix' and l['name'].find('pix-multiscale') >= 0:
                                continue
                            loss_list.append(l)
        else:
            loss_list = self.loss_list

        loss_results = []
        for i, l in enumerate(loss_list):
            if l['function']:
                if fsfilter: # branch selecting the ones that should use LPF
                    if l['name'].find('tv') >= 0 or l['name'].find('overflow') >= 0 \
                            or l['name'].find('range') >= 0:
                        #Note: Using sr_f here means it will only preserve total variation / denoise on 
                        # the LF, which may or may not be a valid proposition, test!
                        effective_loss = l['weight']*l['function'](sr_f) # fake_H
                    elif l['name'].find('pix') >= 0 or l['name'].find('hfen') >= 0 \
                            or l['name'].find('cpl') >= 0 or l['name'].find('gpl') >= 0 \
                            or l['name'].find('gradient') >= 0 or l['name'].find('fdpl') >= 0:
                        effective_loss = l['weight']*l['function'](sr_f, hr_f) # (fake_H, var_H)
                    elif l['name'].find('ssim') >= 0:
                        effective_loss = l['weight']*(1 - l['function'](sr_f, hr_f)) # (fake_H, var_H)
                    else:
                        effective_loss = l['weight']*l['function'](sr, hr) # (fake_H, var_H)
                else:
                    if l['name'].find('tv') >= 0 or l['name'].find('overflow') >= 0 \
                            or l['name'].find('range') >= 0:
                        effective_loss = l['weight']*l['function'](sr) # fake_H
                    elif l['name'].find('ssim') >= 0:
                        effective_loss = l['weight']*(1 - l['function'](sr, hr)) # (fake_H, var_H)
                    else:
                        effective_loss = l['weight']*l['function'](sr, hr) # (fake_H, var_H)
                #print(l['name'],effective_loss)
                loss_results.append(effective_loss)
                log_dict[l['name']] = effective_loss.item()
        return loss_results, log_dict