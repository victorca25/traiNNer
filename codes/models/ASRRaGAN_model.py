from __future__ import absolute_import

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from . import networks
from .base_model import BaseModel
from .modules.LPIPS.perceptual_loss import PerceptualLoss
from .modules.loss import GPLoss, CPLoss, CharbonnierLoss, ElasticLoss, RelativeL1, L1CosineSim, TVLoss, GANLoss, \
    GradientPenaltyLoss
from .modules.ssim import SSIM, MS_SSIM
from .schedulers import MultiStepLR_Restart, StepLR_Restart, CosineAnnealingLR_Restart

logger = logging.getLogger('base')


class ASRRaGANModel(BaseModel):
    def __init__(self, opt):
        super(ASRRaGANModel, self).__init__(opt)
        train_opt = opt['train']

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
            # Define if the generator will have a final capping mechanism in the output
            self.outm = None
            if train_opt['finalcap']:
                self.outm = train_opt['finalcap']

            # G pixel loss
            # """
            if train_opt['pixel_weight']:
                if train_opt['pixel_criterion']:
                    l_pix_type = train_opt['pixel_criterion']
                else:  # default to cb
                    l_fea_type = 'cb'

                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif l_pix_type == 'cb':
                    self.cri_pix = CharbonnierLoss().to(self.device)
                elif l_pix_type == 'elastic':
                    self.cri_pix = ElasticLoss().to(self.device)
                elif l_pix_type == 'relativel1':
                    self.cri_pix = RelativeL1().to(self.device)
                elif l_pix_type == 'l1cosinesim':
                    self.cri_pix = L1CosineSim().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None
            # """

            # G feature loss
            # """
            if train_opt['feature_weight']:
                if train_opt['feature_criterion']:
                    l_fea_type = train_opt['feature_criterion']
                else:  # default to l1
                    l_fea_type = 'l1'

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
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
            # """

            # self.cri_disfea is the feature loss extracted from feature maps in the discriminator
            # will be enabled if the feature maps are enabled in the discriminator
            if train_opt['dis_feature_weight']:
                if train_opt['dis_feature_criterion']:
                    l_disfea_type = train_opt['dis_feature_criterion']
                else:  # default to l1
                    l_disfea_type = 'l1'

                if l_disfea_type == 'l1':
                    self.cri_disfea = nn.L1Loss().to(self.device)
                elif l_disfea_type == 'l2':
                    self.cri_disfea = nn.MSELoss().to(self.device)
                elif l_disfea_type == 'cb':
                    self.cri_disfea = CharbonnierLoss().to(self.device)
                elif l_disfea_type == 'elastic':
                    self.cri_disfea = ElasticLoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                # self.cri_disfea = None
                # self.cri_disfea = self.cri_fea
                self.l_disfea_w = train_opt['dis_feature_weight']
            else:
                self.cri_disfea = None

            # HFEN loss
            # """
            if train_opt['hfen_weight']:
                l_hfen_type = train_opt['hfen_criterion']
                if train_opt['hfen_presmooth']:
                    pre_smooth = train_opt['hfen_presmooth']
                else:
                    pre_smooth = False  # train_opt['hfen_presmooth']
                if l_hfen_type:
                    relative = l_hfen_type in ['rel_l1', 'rel_l2']
                if l_hfen_type:
                    raise NotImplementedError('HFENLoss for ASRRaGAN needs to be fixed!')
                    # TODO: HFENLoss has changed it's wanted arguments seemingly quite heavily
                    #       possibly quite a while back, so I do not know what should be applicable
                    #       for it now.
                    # self.cri_hfen = HFENLoss(
                    #     loss_f=l_hfen_type,
                    #     device=self.device,
                    #     pre_smooth=pre_smooth,
                    #     relative=relative
                    # ).to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_hfen_type))
                self.l_hfen_w = train_opt['hfen_weight']
            else:
                logger.info('Remove HFEN loss.')
                self.cri_hfen = None
            # """

            # TV loss
            # """
            if train_opt['tv_weight']:
                self.l_tv_w = train_opt['tv_weight']
                l_tv_type = train_opt['tv_type']
                if train_opt['tv_norm']:
                    tv_norm = train_opt['tv_norm']
                else:
                    tv_norm = 1

                if l_tv_type == 'normal':
                    self.cri_tv = TVLoss(self.l_tv_w, p=tv_norm).to(self.device)
                elif l_tv_type == '4D':
                    # Total Variation regularization in 4 directions
                    raise NotImplementedError('TVLoss4D for ASRRaGAN needs to be re-implemented!')
                    # TODO: TVLoss4D function seems to have been MIA since like 1-2 years, I remember this
                    #       being missing back in 2019 or so when I forked the code back then as well.
                    # self.cri_tv = TVLoss4D(self.l_tv_w).to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_tv_type))
            else:
                logger.info('Remove TV loss.')
                self.cri_tv = None
            # """

            # SSIM loss
            # """
            if train_opt['ssim_weight']:
                self.l_ssim_w = train_opt['ssim_weight']
                self.cri_ssim = {
                    'ssim': SSIM,
                    'ms-ssim': MS_SSIM
                }.get(train_opt.get('ssim_type', 'ms-ssim'), None)(
                    window_size=11,
                    window_sigma=1.5,
                    size_average=True,
                    data_range=1.,
                    channels=3
                ).to(self.device)
            else:
                logger.info('Remove SSIM loss.')
                self.cri_ssim = None
            # """

            # LPIPS loss
            """
            lpips_spatial = False
            if train_opt['lpips_spatial']:
                #lpips_spatial = True if train_opt['lpips_spatial'] == True else False
                lpips_spatial = True if train_opt['lpips_spatial'] else False
            lpips_GPU = False
            if train_opt['lpips_GPU']:
                #lpips_GPU = True if train_opt['lpips_GPU'] == True else False
                lpips_GPU = True if train_opt['lpips_GPU'] else False
            #"""
            # """
            lpips_spatial = True  # False # Return a spatial map of perceptual distance. Meeds to use .mean() for the backprop if True, the mean distance is approximately the same as the non-spatial distance
            lpips_GPU = True  # Whether to use GPU for LPIPS calculations
            if train_opt['lpips_weight']:
                if z_norm == True:  # if images are in [-1,1] range
                    self.lpips_norm = False  # images are already in the [-1,1] range
                else:
                    self.lpips_norm = True  # normalize images from [0,1] range to [-1,1]

                self.l_lpips_w = train_opt['lpips_weight']
                # Can use original off-the-shelf uncalibrated networks 'net' or Linearly calibrated models (LPIPS) 'net-lin'
                if train_opt['lpips_type']:
                    lpips_type = train_opt['lpips_type']
                else:  # Default use linearly calibrated models, better results
                    lpips_type = 'net-lin'
                # Can set net = 'alex', 'squeeze' or 'vgg' or Low-level metrics 'L2' or 'ssim'
                if train_opt['lpips_net']:
                    lpips_net = train_opt['lpips_net']
                else:  # Default use VGG for feature extraction
                    lpips_net = 'vgg'
                self.cri_lpips = PerceptualLoss(
                    model=lpips_type,
                    net=lpips_net,
                    use_gpu=lpips_GPU,
                    model_path=None,
                    spatial=lpips_spatial
                )  # .to(self.device)
                # Linearly calibrated models (LPIPS)
                # self.cri_lpips = PerceptualLoss(model='net-lin', net='alex', use_gpu=lpips_GPU, model_path=None, spatial=lpips_spatial) #.to(self.device)
                # self.cri_lpips = PerceptualLoss(model='net-lin', net='vgg', use_gpu=lpips_GPU, model_path=None, spatial=lpips_spatial) #.to(self.device)
                # Off-the-shelf uncalibrated networks
                # Can set net = 'alex', 'squeeze' or 'vgg'
                # self.cri_lpips = PerceptualLoss(model='net', net='alex', use_gpu=lpips_GPU, model_path=None, spatial=lpips_spatial)
                # Low-level metrics
                # self.cri_lpips = PerceptualLoss(model='L2', colorspace='Lab', use_gpu=lpips_GPU)
                # self.cri_lpips = PerceptualLoss(model='ssim', colorspace='RGB', use_gpu=lpips_GPU)
            else:
                logger.info('Remove LPIPS loss.')
                self.cri_lpips = None
            # """

            # SPL loss
            # """
            if train_opt['spl_weight']:
                self.l_spl_w = train_opt['spl_weight']
                l_spl_type = train_opt['spl_type']
                # SPL Normalization (from [-1,1] images to [0,1] range, if needed)
                if z_norm == True:  # if images are in [-1,1] range
                    self.spl_norm = True  # normalize images to [0, 1]
                else:
                    self.spl_norm = False  # images are already in [0, 1] range
                # YUV Normalization (from [-1,1] images to [0,1] range, if needed, but mandatory)
                if z_norm == True:  # if images are in [-1,1] range
                    self.yuv_norm = True  # normalize images to [0, 1] for yuv calculations
                else:
                    self.yuv_norm = False  # images are already in [0, 1] range
                if l_spl_type in ['spl', 'gpl']:
                    # Gradient Profile Loss
                    self.cri_gpl = GPLoss(spl_denorm=self.spl_norm)
                if l_spl_type in ['spl', 'cpl']:
                    # Color Profile Loss
                    self.cri_cpl = CPLoss(
                        rgb=True,
                        yuv=True,
                        yuvgrad=True,
                        spl_denorm=self.spl_norm,
                        yuv_denorm=self.yuv_norm
                    )
            else:
                logger.info('Remove SPL loss.')
                self.cri_gpl = None
                self.cri_cpl = None
            # """

            # GD gan loss
            # """
            if train_opt['gan_weight']:
                if train_opt['gan_type'] == 'basic':
                    self.cri_gan = nn.BCELoss()  # SRPGAN
                    # self.cri_gan = nn.BCEWithLogitsLoss() # SASRGAN: https://github.com/mitulrm/SRGAN/blob/master/SR_GAN.ipynb
                    self.l_gan_w = train_opt['gan_weight']
                    self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
                    self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
                else:
                    self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
                    self.l_gan_w = train_opt['gan_weight']
                    # D_update_ratio and D_init_iters are for WGAN
                    self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
                    self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

                    if train_opt['gan_type'] == 'wgan-gp':
                        self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                        # gradient penalty loss
                        self.cri_gp = GradientPenaltyLoss(device=self.device).to(self.device)
                        self.l_gp_w = train_opt['gp_weigth']
            else:
                logger.info('Remove GAN loss.')
                self.cri_gan = None
            # """

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(
                optim_params,
                lr=train_opt['lr_G'],
                weight_decay=wd_G,
                betas=(train_opt['beta1_G'], 0.999)
            )
            self.optimizers.append(self.optimizer_G)

            # D
            if self.cri_gan:
                wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
                self.optimizer_D = torch.optim.Adam(
                    self.netD.parameters(),
                    lr=train_opt['lr_D'],
                    weight_decay=wd_D,
                    betas=(train_opt['beta1_D'], 0.999)
                )
                self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(
                        optimizer,
                        train_opt['lr_steps'],
                        train_opt['lr_gamma']
                    ))
            elif train_opt['lr_scheme'] == 'MultiStepLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        MultiStepLR_Restart(
                            optimizer,
                            train_opt['lr_steps'],
                            restarts=train_opt['restarts'],
                            weights=train_opt['restart_weights'],
                            gamma=train_opt['lr_gamma'],
                            clear_state=train_opt['clear_state']
                        ))
            elif train_opt['lr_scheme'] == 'StepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.StepLR(
                        optimizer,
                        train_opt['lr_step_size'],
                        train_opt['lr_gamma']
                    ))
            elif train_opt['lr_scheme'] == 'StepLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        StepLR_Restart(
                            optimizer,
                            step_sizes=train_opt['lr_step_sizes'],
                            restarts=train_opt['restarts'],
                            weights=train_opt['restart_weights'],
                            gamma=train_opt['lr_gamma'],
                            clear_state=train_opt['clear_state']
                        ))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt['T_period'],
                            eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'],
                            weights=train_opt['restart_weights']
                        ))
            elif train_opt['lr_scheme'] == 'ReduceLROnPlateau':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        # lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
                        lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            mode=train_opt['plateau_mode'],
                            factor=train_opt['plateau_factor'],
                            threshold=train_opt['plateau_threshold'],
                            patience=train_opt['plateau_patience']
                        ))
            else:
                raise NotImplementedError('Learning rate scheme ("lr_scheme") not defined or not recognized.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data, need_hr=True):
        # LR
        self.var_L = data['LR'].to(self.device)
        if need_hr:  # train or val
            self.var_H = data['HR'].to(self.device)
            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)

    def feed_data_batch(self, data):
        # LR
        self.var_L = data

    def optimize_parameters(self, step):

        # print("lpips norm: ", self.lpips_norm)
        # print("spl norm: ", self.spl_norm)
        # print("yuv norm: ", self.yuv_norm)

        # G
        if self.cri_gan:
            for p in self.netD.parameters():
                p.requires_grad = False
        self.optimizer_G.zero_grad()
        if self.outm:  # if the model has the final activation option
            self.fake_H = self.netG(self.var_L, outm=self.outm)
        else:  # regular models without the final activation option
            self.fake_H = self.netG(self.var_L)
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
        test_save_img = True
        if test_save_img:
            save_images(self.var_L, 0, "self.var_L")
            save_images(self.var_H, 0, "self.var_H")
            save_images(self.fake_H.detach(), 0, "self.fake_H")
        #####################################################################
        #"""

        if self.cri_gan:
            # D
            for p in self.netD.parameters():
                p.requires_grad = True

            self.optimizer_D.zero_grad()
            l_d_total = 0
            # pred_d_real = self.netD(self.var_ref) # original
            pred_d_real, feats_d_real = self.netD(self.var_ref)  # original with Feature maps
            # pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G # original
            pred_d_fake, feats_d_fake = self.netD(
                self.fake_H.detach())  # detach to avoid BP to G # original with Feature maps
            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)  # Original
            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)  # Original
            l_d_total = (l_d_real + l_d_fake) / 2  # Original

            # If calculating gradient penalty (https://github.com/lycutter/SRGAN-SpectralNorm/blob/master/train.py)
            if self.opt['train']['gan_type'] == 'wgan-gp':
                batch_size = self.var_ref.size(0)
                if self.random_pt.size(0) != batch_size:
                    self.random_pt.resize_(batch_size, 1, 1, 1)
                self.random_pt.uniform_()  # Draw random interpolation points
                interp = self.random_pt * self.fake_H.detach() + (1 - self.random_pt) * self.var_ref
                interp.requires_grad = True
                interp_crit = self.netD(interp)
                l_d_gp = self.l_gp_w * self.cri_gp(interp, interp_crit)
                l_d_total += l_d_gp

            # l_d_total.backward() # Original and SRGAN, Discriminator after Generator
            l_d_total.backward(retain_graph=True)  # SRPGAN
            self.optimizer_D.step()

            if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                if self.cri_pix:  # pixel loss
                    l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                    l_g_total += l_g_pix
                if self.cri_ssim:  # structural loss
                    l_g_ssim = 1. - (self.l_ssim_w * self.cri_ssim(self.fake_H, self.var_H))  # using ssim2.py
                    if torch.isnan(l_g_ssim).any():
                        l_g_total = l_g_total
                    else:
                        l_g_total += l_g_ssim
                if self.cri_fea:  # feature loss
                    real_fea = self.netF(self.var_H).detach()
                    fake_fea = self.netF(self.fake_H)
                    # VGG Features Perceptual Loss 
                    l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                    # print(l_g_fea)
                    l_g_total += l_g_fea
                if self.cri_disfea:  # SRPGAN-like Features Perceptual loss, extracted from the discriminator
                    l_g_disfea = 0
                    for hr_feat_map, sr_feat_map in zip(feats_d_fake, feats_d_real):
                        # Note: VGG features order of magnitude is around 1 and the sum of the feature_maps losses
                        # is around order of magnitude of 6. The 0.000001 is to make them roughly equivalent.
                        # l_g_disfea += 0.00001 * self.l_disfea_w * self.cri_disfea(sr_feat_map, hr_feat_map)
                        l_g_disfea += self.l_disfea_w * self.cri_disfea(sr_feat_map, hr_feat_map)
                    l_g_disfea /= len(sr_feat_map)
                    # print(l_g_disfea)
                    l_g_total += l_g_disfea
                if self.cri_hfen:  # HFEN loss 
                    l_g_HFEN = self.l_hfen_w * self.cri_hfen(self.fake_H, self.var_H)
                    l_g_total += l_g_HFEN
                if self.cri_tv:  # TV loss
                    l_g_tv = self.cri_tv(
                        self.fake_H)  # note: the weight is already multiplied inside the function, doesn't need to be here
                    l_g_total += l_g_tv
                if self.cri_lpips:  # LPIPS loss
                    # If "spatial = False" .forward() returns a scalar value, if "spatial = True", returns a map (5 layers for vgg and alex or 7 for squeeze)
                    # NOTE: .mean() is only to make the resulting loss into a scalar if "spatial = True", the mean distance is approximately the same as the non-spatial distance: https://github.com/richzhang/PerceptualSimilarity/blob/master/test_network.py
                    # l_g_lpips = self.cri_lpips.forward(self.fake_H,self.var_H).mean() # -> If normalize is False (default), assumes the images are already between [-1,+1]
                    l_g_lpips = self.cri_lpips.forward(self.fake_H, self.var_H,
                                                       normalize=self.lpips_norm).mean()  # -> # If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
                    # l_g_lpips = self.cri_lpips.forward(self.fake_H, self.var_H, normalize=True) # If "spatial = False" should return a scalar value
                    # print(l_g_lpips)
                    l_g_total += l_g_lpips
                if self.cri_gpl:  # GPL Loss (SPL)
                    l_g_gpl = self.l_spl_w * self.cri_gpl(self.fake_H, self.var_H)
                    # l_spl = l_g_gpl + l_g_cpl
                    l_g_total += l_g_gpl
                if self.cri_cpl:  # CPL Loss (SPL)
                    l_g_cpl = self.l_spl_w * self.cri_cpl(self.fake_H, self.var_H)
                    l_g_total += l_g_cpl

                # G gan + cls loss
                # pred_g_fake = self.netD(self.fake_H) # Original
                pred_g_fake, _ = self.netD(self.fake_H)  # Original with Feature maps
                # pred_d_real = self.netD(self.var_ref).detach() # Original
                pred_d_real, _ = self.netD(self.var_ref)  # Original with Feature maps
                pred_d_real = pred_d_real.detach()  # Original with Feature maps
                l_g_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                          self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2  # Original
                l_g_total += l_g_gan  # Original

                l_g_total.backward()
                self.optimizer_G.step()

            # set log
            if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                # G
                if self.cri_pix:
                    self.log_dict['l_g_pix'] = l_g_pix.item()
                if self.cri_fea:
                    self.log_dict['l_g_fea'] = l_g_fea.item()
                if self.cri_disfea:
                    self.log_dict['l_g_disfea'] = l_g_disfea.item()
                if self.cri_hfen:
                    self.log_dict['l_g_HFEN'] = l_g_HFEN.item()
                if self.cri_tv:
                    self.log_dict['l_g_tv'] = l_g_tv.item()
                if self.cri_ssim:
                    self.log_dict['l_g_ssim'] = l_g_ssim.item()
                if self.cri_lpips:
                    self.log_dict['l_g_lpips'] = l_g_lpips.item()
                if self.cri_gpl:
                    self.log_dict['l_g_gpl'] = l_g_gpl.item()
                if self.cri_cpl:
                    self.log_dict['l_g_cpl'] = l_g_cpl.item()
                self.log_dict['l_g_gan'] = l_g_gan.item()
            # D
            self.log_dict['l_d_real'] = l_d_real.item()
            self.log_dict['l_d_fake'] = l_d_fake.item()

            if self.opt['train']['gan_type'] == 'wgan-gp':
                self.log_dict['l_d_gp'] = l_d_gp.item()
            # D outputs
            self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
            self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

        else:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_ssim:  # structural loss (Structural Dissimilarity)
                l_g_ssim = (1. - (self.l_ssim_w * self.cri_ssim(self.fake_H, self.var_H))) / 2  # using ssim2.py
                if torch.isnan(
                        l_g_ssim).any():  # at random, l_g_ssim is returning NaN for ms-ssim, which breaks the model. Temporary hack, until I find out what's going on.
                    l_g_total = l_g_total
                else:
                    l_g_total += l_g_ssim
            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
            if self.cri_hfen:  # HFEN loss 
                l_g_HFEN = self.l_hfen_w * self.cri_hfen(self.fake_H, self.var_H)
                l_g_total += l_g_HFEN
            if self.cri_tv:  # TV loss
                l_g_tv = self.cri_tv(
                    self.fake_H)  # note: the weight is already multiplied inside the function, doesn't need to be here
                l_g_total += l_g_tv
            if self.cri_lpips:  # LPIPS loss
                # If "spatial = False" .forward() returns a scalar value, if "spatial = True", returns a map (5 layers for vgg and alex or 7 for squeeze)
                # NOTE: .mean() is only to make the resulting loss into a scalar if "spatial = True", the mean distance is approximately the same as the non-spatial distance: https://github.com/richzhang/PerceptualSimilarity/blob/master/test_network.py
                # l_g_lpips = self.cri_lpips.forward(self.fake_H,self.var_H).mean() # -> If normalize is False (default), assumes the images are already between [-1,+1]
                l_g_lpips = self.cri_lpips.forward(self.fake_H, self.var_H,
                                                   normalize=self.lpips_norm).mean()  # -> # If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
                # l_g_lpips = self.cri_lpips.forward(self.fake_H, self.var_H, normalize=True) # If "spatial = False" returns a scalar value
                # print(l_g_lpips)
                l_g_total += l_g_lpips
            # SPL note: we can compute the GP loss between output and input and the colour loss 
            # between output and target to train a generator in an auto-encoder fashion for 
            # style-transfer applications.
            if self.cri_gpl:  # GPL Loss (SPL)
                l_g_gpl = self.l_spl_w * self.cri_gpl(self.fake_H, self.var_H)
                l_g_total += l_g_gpl
            if self.cri_cpl:  # CPL Loss (SPL)
                l_g_cpl = self.l_spl_w * self.cri_cpl(self.fake_H, self.var_H)
                l_g_total += l_g_cpl

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
            if self.cri_lpips:
                self.log_dict['l_g_lpips'] = l_g_lpips.item()
            if self.cri_gpl:
                self.log_dict['l_g_gpl'] = l_g_gpl.item()
            if self.cri_cpl:
                self.log_dict['l_g_cpl'] = l_g_cpl.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.is_train:
                self.fake_H = self.netG(self.var_L)
            else:
                self.fake_H = self.netG(self.var_L, isTest=True)
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

    def get_current_visuals_batch(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach().float().cpu()
        out_dict['SR'] = self.fake_H.detach().float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach().float().cpu()
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

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)

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
