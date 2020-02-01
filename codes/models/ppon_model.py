from __future__ import absolute_import

import logging
from collections import OrderedDict

import codes.models.networks as networks
import torch
import torch.nn as nn
from codes.models.modules.LPIPS import (
    perceptual_loss as models,
)  # import models.modules.LPIPS as models
from codes.models.modules.loss import (
    GANLoss,
    GradientPenaltyLoss,
    HFENLoss,
    TVLoss,
    CharbonnierLoss,
    ElasticLoss,
    RelativeL1,
    L1CosineSim,
)
from codes.models.modules.losses import spl_loss as spl
from codes.models.modules.losses.ssim2 import (
    SSIM,
    MS_SSIM,
)  # implementation for use with any PyTorch
from torch.optim import lr_scheduler

from .base_model import BaseModel

# from models.modules.losses.ssim3 import SSIM, MS_SSIM #for use of the PyTorch 1.1.1+ optimized implementation
logger = logging.getLogger("base")

import codes.models.lr_schedulerR as lr_schedulerR

"""
import numpy as np
def convert_shape(img):
    img = np.transpose((img * 255.0).round(), (1, 2, 0))
    img = np.uint8(np.clip(img, 0, 255))
    return img
"""

""" #debug
def save_images(image, num_rep, sufix):
    from utils import util
    import uuid, cv2
    hex = uuid.uuid4().hex
    img_ds = util.tensor2img(image)  # uint8
    cv2.imwrite("D:/tmp_test/fake_"+sufix+"_"+str(num_rep)+hex+".png",img_ds*255) #random name to save + had to multiply by 255, else getting all black image
#"""


class PPONModel(BaseModel):
    def __init__(self, opt):
        super(PPONModel, self).__init__(opt)
        train_opt = opt["train"]

        if self.is_train:
            if opt["datasets"]["train"]["znorm"]:
                z_norm = opt["datasets"]["train"]["znorm"]
            else:
                z_norm = False

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netG.train()
            if train_opt["gan_weight"]:
                self.netD = networks.define_D(opt).to(self.device)  # D
                self.netD.train()
            # PPON
            """
            self.phase1_s = train_opt['phase1_s']
            if self.phase1_s is None:
                self.phase1_s = 138000
            self.phase2_s = train_opt['phase2_s']
            if self.phase2_s is None:
                self.phase2_s = 138000+34500
            self.phase3_s = train_opt['phase3_s']
            if self.phase3_s is None:
                self.phase3_s = 138000+34500+34500
            """
            self.phase1_s = train_opt["phase1_s"] if train_opt["phase1_s"] else 138000
            self.phase2_s = (
                train_opt["phase2_s"] if train_opt["phase2_s"] else (138000 + 34500)
            )
            self.phase3_s = (
                train_opt["phase3_s"]
                if train_opt["phase3_s"]
                else (138000 + 34500 + 34500)
            )
            self.train_phase = (
                train_opt["train_phase"] - 1 if train_opt["train_phase"] else 0
            )  # change to start from 0 (Phase 1: from 0 to 1, Phase 1: from 1 to 2, etc)
            self.restarts = train_opt["restarts"] if train_opt["restarts"] else [0]

        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # Define if the generator will have a final capping mechanism in the output
            self.outm = None
            if train_opt["finalcap"]:
                self.outm = train_opt["finalcap"]

            # G pixel loss
            # """
            if train_opt["pixel_weight"]:
                if train_opt["pixel_criterion"]:
                    l_pix_type = train_opt["pixel_criterion"]
                else:  # default to cb
                    l_fea_type = "cb"

                if l_pix_type == "l1":
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == "l2":
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif l_pix_type == "cb":
                    self.cri_pix = CharbonnierLoss().to(self.device)
                elif l_pix_type == "elastic":
                    self.cri_pix = ElasticLoss().to(self.device)
                elif l_pix_type == "relativel1":
                    self.cri_pix = RelativeL1().to(self.device)
                elif l_pix_type == "l1cosinesim":
                    self.cri_pix = L1CosineSim().to(self.device)
                else:
                    raise NotImplementedError(
                        "Loss type [{:s}] not recognized.".format(l_pix_type)
                    )
                self.l_pix_w = train_opt["pixel_weight"]
            else:
                logger.info("Remove pixel loss.")
                self.cri_pix = None
            # """

            # G feature loss
            # """
            if train_opt["feature_weight"]:
                if train_opt["feature_criterion"]:
                    l_fea_type = train_opt["feature_criterion"]
                else:  # default to l1
                    l_fea_type = "l1"

                if l_fea_type == "l1":
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == "l2":
                    self.cri_fea = nn.MSELoss().to(self.device)
                elif l_fea_type == "cb":
                    self.cri_fea = CharbonnierLoss().to(self.device)
                elif l_fea_type == "elastic":
                    self.cri_fea = ElasticLoss().to(self.device)
                else:
                    raise NotImplementedError(
                        "Loss type [{:s}] not recognized.".format(l_fea_type)
                    )
                self.l_fea_w = train_opt["feature_weight"]
            else:
                logger.info("Remove feature loss.")
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
            # """

            # HFEN loss
            # """
            if train_opt["hfen_weight"]:
                l_hfen_type = train_opt["hfen_criterion"]
                if train_opt["hfen_presmooth"]:
                    pre_smooth = train_opt["hfen_presmooth"]
                else:
                    pre_smooth = False  # train_opt['hfen_presmooth']
                if l_hfen_type:
                    if l_hfen_type == "rel_l1" or l_hfen_type == "rel_l2":
                        relative = True
                    else:
                        relative = False  # True #train_opt['hfen_relative']
                if l_hfen_type:
                    self.cri_hfen = HFENLoss(
                        loss_f=l_hfen_type,
                        device=self.device,
                        pre_smooth=pre_smooth,
                        relative=relative,
                    ).to(self.device)
                else:
                    raise NotImplementedError(
                        "Loss type [{:s}] not recognized.".format(l_hfen_type)
                    )
                self.l_hfen_w = train_opt["hfen_weight"]
            else:
                logger.info("Remove HFEN loss.")
                self.cri_hfen = None
            # """

            # TV loss
            # """
            if train_opt["tv_weight"]:
                self.l_tv_w = train_opt["tv_weight"]
                l_tv_type = train_opt["tv_type"]
                if train_opt["tv_norm"]:
                    tv_norm = train_opt["tv_norm"]
                else:
                    tv_norm = 1

                if l_tv_type == "normal":
                    self.cri_tv = TVLoss(self.l_tv_w, p=tv_norm).to(self.device)
                elif l_tv_type == "4D":
                    self.cri_tv = TVLoss4D(self.l_tv_w).to(
                        self.device
                    )  # Total Variation regularization in 4 directions
                else:
                    raise NotImplementedError(
                        "Loss type [{:s}] not recognized.".format(l_tv_type)
                    )
            else:
                logger.info("Remove TV loss.")
                self.cri_tv = None
            # """

            # SSIM loss
            # """
            if train_opt["ssim_weight"]:
                self.l_ssim_w = train_opt["ssim_weight"]

                if train_opt["ssim_type"]:
                    l_ssim_type = train_opt["ssim_type"]
                else:  # default to ms-ssim
                    l_ssim_type = "ms-ssim"

                if l_ssim_type == "ssim":
                    self.cri_ssim = SSIM(
                        win_size=11,
                        win_sigma=1.5,
                        size_average=True,
                        data_range=1.0,
                        channel=3,
                    ).to(self.device)
                elif l_ssim_type == "ms-ssim":
                    self.cri_ssim = MS_SSIM(
                        win_size=11,
                        win_sigma=1.5,
                        size_average=True,
                        data_range=1.0,
                        channel=3,
                    ).to(self.device)
            else:
                logger.info("Remove SSIM loss.")
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
            if train_opt["lpips_weight"]:
                if z_norm == True:  # if images are in [-1,1] range
                    self.lpips_norm = False  # images are already in the [-1,1] range
                else:
                    self.lpips_norm = (
                        True  # normalize images from [0,1] range to [-1,1]
                    )

                self.l_lpips_w = train_opt["lpips_weight"]
                # Can use original off-the-shelf uncalibrated networks 'net' or Linearly calibrated models (LPIPS) 'net-lin'
                if train_opt["lpips_type"]:
                    lpips_type = train_opt["lpips_type"]
                else:  # Default use linearly calibrated models, better results
                    lpips_type = "net-lin"
                # Can set net = 'alex', 'squeeze' or 'vgg' or Low-level metrics 'L2' or 'ssim'
                if train_opt["lpips_net"]:
                    lpips_net = train_opt["lpips_net"]
                else:  # Default use VGG for feature extraction
                    lpips_net = "vgg"
                self.cri_lpips = models.PerceptualLoss(
                    model=lpips_type,
                    net=lpips_net,
                    use_gpu=lpips_GPU,
                    model_path=None,
                    spatial=lpips_spatial,
                )  # .to(self.device)
                # Linearly calibrated models (LPIPS)
                # self.cri_lpips = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=lpips_GPU, model_path=None, spatial=lpips_spatial) #.to(self.device)
                # self.cri_lpips = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=lpips_GPU, model_path=None, spatial=lpips_spatial) #.to(self.device)
                # Off-the-shelf uncalibrated networks
                # Can set net = 'alex', 'squeeze' or 'vgg'
                # self.cri_lpips = models.PerceptualLoss(model='net', net='alex', use_gpu=lpips_GPU, model_path=None, spatial=lpips_spatial)
                # Low-level metrics
                # self.cri_lpips = models.PerceptualLoss(model='L2', colorspace='Lab', use_gpu=lpips_GPU)
                # self.cri_lpips = models.PerceptualLoss(model='ssim', colorspace='RGB', use_gpu=lpips_GPU)
            else:
                logger.info("Remove LPIPS loss.")
                self.cri_lpips = None
            # """

            # SPL loss
            # """
            if train_opt["spl_weight"]:
                self.l_spl_w = train_opt["spl_weight"]
                l_spl_type = train_opt["spl_type"]
                # SPL Normalization (from [-1,1] images to [0,1] range, if needed)
                if z_norm == True:  # if images are in [-1,1] range
                    self.spl_norm = True  # normalize images to [0, 1]
                else:
                    self.spl_norm = False  # images are already in [0, 1] range
                # YUV Normalization (from [-1,1] images to [0,1] range, if needed, but mandatory)
                if z_norm == True:  # if images are in [-1,1] range
                    self.yuv_norm = (
                        True  # normalize images to [0, 1] for yuv calculations
                    )
                else:
                    self.yuv_norm = False  # images are already in [0, 1] range
                if l_spl_type == "spl":  # Both GPL and CPL
                    # Gradient Profile Loss
                    self.cri_gpl = spl.GPLoss(spl_norm=self.spl_norm)
                    # Color Profile Loss
                    # You can define the desired color spaces in the initialization
                    # default is True for all
                    self.cri_cpl = spl.CPLoss(
                        rgb=True,
                        yuv=True,
                        yuvgrad=True,
                        spl_norm=self.spl_norm,
                        yuv_norm=self.yuv_norm,
                    )
                elif l_spl_type == "gpl":  # Only GPL
                    # Gradient Profile Loss
                    self.cri_gpl = spl.GPLoss(spl_norm=self.spl_norm)
                    self.cri_cpl = None
                elif l_spl_type == "cpl":  # Only CPL
                    # Color Profile Loss
                    # You can define the desired color spaces in the initialization
                    # default is True for all
                    self.cri_cpl = spl.CPLoss(
                        rgb=True,
                        yuv=True,
                        yuvgrad=True,
                        spl_norm=self.spl_norm,
                        yuv_norm=self.yuv_norm,
                    )
                    self.cri_gpl = None
            else:
                logger.info("Remove SPL loss.")
                self.cri_gpl = None
                self.cri_cpl = None
            # """

            # GD gan loss
            # """
            if train_opt["gan_weight"]:
                self.cri_gan = GANLoss(train_opt["gan_type"], 1.0, 0.0).to(self.device)
                self.l_gan_w = train_opt["gan_weight"]
                # D_update_ratio and D_init_iters are for WGAN
                self.D_update_ratio = (
                    train_opt["D_update_ratio"] if train_opt["D_update_ratio"] else 1
                )
                self.D_init_iters = (
                    train_opt["D_init_iters"] if train_opt["D_init_iters"] else 0
                )

                if train_opt["gan_type"] == "wgan-gp":
                    self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                    # gradient penalty loss
                    self.cri_gp = GradientPenaltyLoss(device=self.device).to(
                        self.device
                    )
                    self.l_gp_w = train_opt["gp_weigth"]
            else:
                logger.info("Remove GAN loss.")
                self.cri_gan = None
            # """

            # optimizers
            # G
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0

            optim_params = []
            for (
                    k,
                    v,
            ) in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning("Params [{:s}] will not optimize.".format(k))
            self.optimizer_G = torch.optim.Adam(
                optim_params,
                lr=train_opt["lr_G"],
                weight_decay=wd_G,
                betas=(train_opt["beta1_G"], 0.999),
            )
            self.optimizers.append(self.optimizer_G)

            # D
            if self.cri_gan:
                wd_D = train_opt["weight_decay_D"] if train_opt["weight_decay_D"] else 0
                self.optimizer_D = torch.optim.Adam(
                    self.netD.parameters(),
                    lr=train_opt["lr_D"],
                    weight_decay=wd_D,
                    betas=(train_opt["beta1_D"], 0.999),
                )
                self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR(
                            optimizer, train_opt["lr_steps"], train_opt["lr_gamma"]
                        )
                    )
            elif train_opt["lr_scheme"] == "MultiStepLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_schedulerR.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "StepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.StepLR(
                            optimizer, train_opt["lr_step_size"], train_opt["lr_gamma"]
                        )
                    )
            elif train_opt["lr_scheme"] == "StepLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_schedulerR.StepLR_Restart(
                            optimizer,
                            step_sizes=train_opt["lr_step_sizes"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_schedulerR.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt["T_period"],
                            eta_min=train_opt["eta_min"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                        )
                    )
            elif train_opt["lr_scheme"] == "ReduceLROnPlateau":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        # lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
                        lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            mode=train_opt["plateau_mode"],
                            factor=train_opt["plateau_factor"],
                            threshold=train_opt["plateau_threshold"],
                            patience=train_opt["plateau_patience"],
                        )
                    )
            else:
                raise NotImplementedError(
                    'Learning rate scheme ("lr_scheme") not defined or not recognized.'
                )

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data["LR"].to(self.device)
        if need_HR:  # train or val
            self.var_H = data["HR"].to(self.device)
            input_ref = data["ref"] if "ref" in data else data["HR"]
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        # G
        # Freeze Discriminator during the Generator training
        if self.cri_gan:
            for p in self.netD.parameters():
                p.requires_grad = False

        self.optimizer_G.zero_grad()

        # Prevent the new lr after a restart from being used by the previous phase between phase changes
        # Originally this was in the end of the function, but led to one step during phase change where there were no parameters with requires_grad for l_g_total.backward() to use, which stops with an error
        if step in self.restarts:
            # Freeze all Generator layers to prevent training leaks during phase change
            for p in self.netG.parameters():
                p.requires_grad = False

        ### PPON freeze and unfreeze the components at each phase (Content, Structure, Perceptual)
        # Phase 1
        if step > 0 and step < self.phase1_s and self.phase1_s > 0:
            # Freeze/Unfreeze layers
            if self.train_phase == 0 or (step in self.restarts):
                print("Starting phase 1")
                self.train_phase = 1
                self.log_dict = OrderedDict()  # Clear the loss logs
            # Freeze all layers
            for p in self.netG.parameters():
                # print(p)
                p.requires_grad = False
            """
            for param_name in self.netG.state_dict():
                self.netG.state_dict()[param_name].requires_grad = False
            #"""
            # Unfreeze the Content Layers, CFEM and CRM
            # CFEM_param = self.netG.module.CFEM.parameters()
            for p in self.netG.module.CFEM.parameters():
                p.requires_grad = True
            """
            for param_name in self.netG.module.CFEM.state_dict():
                print('Name: ' + str(param_name))
                self.netG.module.CFEM.state_dict()[param_name].requires_grad = True
                print('\tRequires Grad: ' + str(self.netG.module.CFEM.state_dict()[param_name].requires_grad))
            """
            # CRM_param = self.netG.module.CRM.parameters()
            for p in self.netG.module.CRM.parameters():
                p.requires_grad = True
            """
            for param_name in self.netG.module.CRM.state_dict():
                self.netG.module.CRM.state_dict()[param_name].requires_grad = True
            """
            self.optimizer_G.zero_grad()

            self.fake_Hc, self.fake_Hs, self.fake_Hp = self.netG(self.var_L)
            self.fake_H = self.fake_Hc

            # Calculate losses
            l_g_total = 0
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            # if self.cri_ssim: # structural loss
            # l_g_ssim = 1.-(self.l_ssim_w *self.cri_ssim(self.fake_H, self.var_H)) #using ssim2.py
            # if torch.isnan(l_g_ssim).any(): #at random, l_g_ssim is returning NaN for ms-ssim, which breaks the model. Temporary hack, until I find out what's going on.
            # l_g_total = l_g_total
            # else:
            # l_g_total += l_g_ssim
            if self.cri_tv:  # TV loss
                l_g_tv = self.cri_tv(
                    self.fake_H
                )  # note: the weight is already multiplied inside the function, doesn't need to be here
                l_g_total += l_g_tv
            if self.cri_lpips:  # LPIPS loss
                # If "spatial = False" .forward() returns a scalar value, if "spatial = True", returns a map (5 layers for vgg and alex or 7 for squeeze)
                # NOTE: .mean() is only to make the resulting loss into a scalar if "spatial = True", the mean distance is approximately the same as the non-spatial distance: https://github.com/richzhang/PerceptualSimilarity/blob/master/test_network.py
                # l_g_lpips = self.cri_lpips.forward(self.fake_H,self.var_H).mean() # -> If normalize is False (default), assumes the images are already between [-1,+1]
                l_g_lpips = self.cri_lpips.forward(
                    self.fake_H, self.var_H, normalize=self.lpips_norm
                ).mean()  # -> # If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
                # l_g_lpips = self.cri_lpips.forward(self.fake_H, self.var_H, normalize=True) # If "spatial = False" returns a scalar value
                # print(l_g_lpips)
                l_g_total += l_g_lpips
            if self.cri_gpl:  # GPL Loss (SPL)
                l_g_gpl = self.l_spl_w * self.cri_gpl(self.fake_H, self.var_H)
                l_g_total += l_g_gpl
            if self.cri_cpl:  # CPL Loss (SPL)
                l_g_cpl = self.l_spl_w * self.cri_cpl(self.fake_H, self.var_H)
                l_g_total += l_g_cpl

            try:  # Prevent error if there are no parameter for autograd during phase change
                l_g_total.backward()
                self.optimizer_G.step()
            except:
                print("skipping iteration", step)
                print("error in the backward pass")

            # set log
            # G
            if self.cri_pix:
                self.log_dict["l_g_pix"] = l_g_pix.item()
            # if self.cri_fea:
            #    self.log_dict['l_g_fea'] = l_g_fea.item()
            # if self.cri_hfen:
            #    self.log_dict['l_g_HFEN'] = l_g_HFEN.item()
            if self.cri_tv:
                self.log_dict["l_g_tv"] = l_g_tv.item()
            # if self.cri_ssim:
            #    self.log_dict['l_g_ssim'] = l_g_ssim.item()
            if self.cri_lpips:
                self.log_dict["l_g_lpips"] = l_g_lpips.item()
            if self.cri_gpl:
                self.log_dict["l_g_gpl"] = l_g_gpl.item()
            if self.cri_cpl:
                self.log_dict["l_g_cpl"] = l_g_cpl.item()

        # Phase 2
        elif step > self.phase1_s and step < self.phase2_s and self.phase2_s > 0:
            # Freeze/Unfreeze layers
            if self.train_phase == 1 or (step in self.restarts):
                print("Starting phase 2")
                self.train_phase = 2
                self.log_dict = OrderedDict()  # Clear the loss logs
            # Freeze all layers
            for p in self.netG.parameters():
                p.requires_grad = False
            # Unfreeze the Structure Layers, SFEM and SRM
            # SFEM_param = self.netG.module.SFEM.parameters()
            for p in self.netG.module.SFEM.parameters():
                p.requires_grad = True
            # SRM_param = self.netG.module.SRM.parameters()
            for p in self.netG.module.SRM.parameters():
                p.requires_grad = True
            self.optimizer_G.zero_grad()

            self.fake_Hc, self.fake_Hs, self.fake_Hp = self.netG(self.var_L)
            self.fake_H = self.fake_Hs

            # Calculate losses
            l_g_total = 0
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_ssim:  # structural loss
                l_g_ssim = 1.0 - (
                        self.l_ssim_w * self.cri_ssim(self.fake_H, self.var_H)
                )
                if torch.isnan(
                        l_g_ssim
                ).any():  # at random, l_g_ssim is returning NaN for ms-ssim, which breaks the model. Temporary hack, until I find out what's going on.
                    l_g_total = l_g_total
                else:
                    l_g_total += l_g_ssim
            if self.cri_hfen:  # HFEN loss
                l_g_HFEN = self.l_hfen_w * self.cri_hfen(self.fake_H, self.var_H)
                l_g_total += l_g_HFEN
            if self.cri_lpips:  # LPIPS loss
                # If "spatial = False" .forward() returns a scalar value, if "spatial = True", returns a map (5 layers for vgg and alex or 7 for squeeze)
                # NOTE: .mean() is only to make the resulting loss into a scalar if "spatial = True", the mean distance is approximately the same as the non-spatial distance: https://github.com/richzhang/PerceptualSimilarity/blob/master/test_network.py
                # l_g_lpips = self.cri_lpips.forward(self.fake_H,self.var_H).mean() # -> If normalize is False (default), assumes the images are already between [-1,+1]
                l_g_lpips = self.cri_lpips.forward(
                    self.fake_H, self.var_H, normalize=self.lpips_norm
                ).mean()  # -> # If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
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

            try:  # Prevent error if there are no parameter for autograd during phase change
                l_g_total.backward()
                self.optimizer_G.step()
            except:
                print("skipping iteration", step)
                print("error in the backward pass")

            # set log
            # G
            if self.cri_pix:
                self.log_dict["l_g_pix"] = l_g_pix.item()
            # if self.cri_fea:
            #    self.log_dict['l_g_fea'] = l_g_fea.item()
            if self.cri_hfen:
                self.log_dict["l_g_HFEN"] = l_g_HFEN.item()
            # if self.cri_tv:
            #    self.log_dict['l_g_tv'] = l_g_tv.item()
            if self.cri_ssim:
                self.log_dict["l_g_ssim"] = l_g_ssim.item()
            if self.cri_lpips:
                self.log_dict["l_g_lpips"] = l_g_lpips.item()
            if self.cri_gpl:
                self.log_dict["l_g_gpl"] = l_g_gpl.item()
            if self.cri_cpl:
                self.log_dict["l_g_cpl"] = l_g_cpl.item()

        # Phase 3
        elif step > self.phase2_s and step < self.phase3_s and self.phase3_s > 0:
            # Freeze/Unfreeze layers
            if self.train_phase == 2 or (step in self.restarts):
                print("Starting phase 3")
                self.train_phase = 3
                self.log_dict = OrderedDict()  # Clear the loss logs
            # Freeze all layers
            for p in self.netG.parameters():
                p.requires_grad = False
            # Unfreeze the Perceptual Layers, PFEM and PRM
            # PFEM_param = self.netG.module.PFEM.parameters()
            for p in self.netG.module.PFEM.parameters():
                p.requires_grad = True
            # PRM_param = self.netG.module.PRM.parameters()
            for p in self.netG.module.PRM.parameters():
                p.requires_grad = True
            self.optimizer_G.zero_grad()

            self.fake_Hc, self.fake_Hs, self.fake_Hp = self.netG(self.var_L)
            self.fake_H = self.fake_Hp

            # test_save_img = False
            # test_save_img = None
            """ #debug
            #####################################################################
            if test_save_img:
                save_images(self.var_H, 0, "self.var_H")
                save_images(self.fake_H.detach(), 0, "self.fake_H")
            #####################################################################
            #"""

            # Calculate losses
            l_g_total = 0
            if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                # if self.cri_pix:  # pixel loss
                # l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                # l_g_total += l_g_pix
                # if self.cri_ssim: # structural loss
                # l_g_ssim = 1.-(self.l_ssim_w *self.cri_ssim(self.fake_H, self.var_H)) #using ssim2.py
                # if torch.isnan(l_g_ssim).any():
                # l_g_total = l_g_total
                # else:
                # l_g_total += l_g_ssim
                if self.cri_fea:  # feature loss
                    real_fea = self.netF(self.var_H).detach()
                    fake_fea = self.netF(self.fake_H)
                    l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                    l_g_total += l_g_fea
                # if self.cri_hfen:  # HFEN loss
                # l_g_HFEN = self.l_hfen_w * self.cri_hfen(self.fake_H, self.var_H)
                # l_g_total += l_g_HFEN
                # if self.cri_tv: #TV loss
                # l_g_tv = self.cri_tv(self.fake_H) #note: the weight is already multiplied inside the function, doesn't need to be here
                # l_g_total += l_g_tv
                if self.cri_lpips:  # LPIPS loss
                    # If "spatial = False" .forward() returns a scalar value, if "spatial = True", returns a map (5 layers for vgg and alex or 7 for squeeze)
                    # NOTE: .mean() is only to make the resulting loss into a scalar if "spatial = True", the mean distance is approximately the same as the non-spatial distance: https://github.com/richzhang/PerceptualSimilarity/blob/master/test_network.py
                    # l_g_lpips = self.cri_lpips.forward(self.fake_H,self.var_H).mean() # -> If normalize is False (default), assumes the images are already between [-1,+1]
                    l_g_lpips = self.cri_lpips.forward(
                        self.fake_H, self.var_H, normalize=self.lpips_norm
                    ).mean()  # -> # If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
                    # l_g_lpips = self.cri_lpips.forward(self.fake_H, self.var_H, normalize=True) # If "spatial = False" should return a scalar value
                    # print(l_g_lpips)
                    l_g_total += l_g_lpips
                if self.cri_gpl:  # GPL Loss (SPL)
                    l_g_gpl = self.l_spl_w * self.cri_gpl(self.fake_H, self.var_H)
                    l_g_total += l_g_gpl
                if self.cri_cpl:  # CPL Loss (SPL)
                    l_g_cpl = self.l_spl_w * self.cri_cpl(self.fake_H, self.var_H)
                    l_g_total += l_g_cpl
                if self.cri_gan:  # GAN loss
                    # G gan + cls loss
                    pred_g_fake = self.netD(self.fake_H)
                    pred_d_real = self.netD(self.var_ref).detach()
                    l_g_gan = (
                            self.l_gan_w
                            * (
                                    self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False)
                                    + self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)
                            )
                            / 2
                    )
                    l_g_total += l_g_gan

            try:  # Prevent error if there are no parameter for autograd during phase change
                l_g_total.backward()
                self.optimizer_G.step()
            except:
                print("skipping iteration", step)
                print("error in the backward pass")

            # D
            # Unfreeze the Discriminator for training
            if self.cri_gan:
                for p in self.netD.parameters():
                    p.requires_grad = True

                self.optimizer_D.zero_grad()
                l_d_total = 0
                pred_d_real = self.netD(self.var_ref)
                pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
                l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)

                l_d_total = (l_d_real + l_d_fake) / 2

                if self.opt["train"]["gan_type"] == "wgan-gp":
                    batch_size = self.var_ref.size(0)
                    if self.random_pt.size(0) != batch_size:
                        self.random_pt.resize_(batch_size, 1, 1, 1)
                    self.random_pt.uniform_()  # Draw random interpolation points
                    interp = (
                            self.random_pt * self.fake_H.detach()
                            + (1 - self.random_pt) * self.var_ref
                    )
                    interp.requires_grad = True
                    interp_crit = self.netD(interp)
                    l_d_gp = self.l_gp_w * self.cri_gp(interp, interp_crit)
                    l_d_total += l_d_gp

                try:  # Prevent error if there are no parameter for autograd during phase change
                    l_d_total.backward()
                    self.optimizer_D.step()
                except:
                    print("skipping iteration", step)
                    print("error in the backward pass")

                # set log
                if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                    # G
                    # if self.cri_pix:
                    # self.log_dict['l_g_pix'] = l_g_pix.item()
                    if self.cri_fea:
                        self.log_dict["l_g_fea"] = l_g_fea.item()
                    # if self.cri_hfen:
                    #    self.log_dict['l_g_HFEN'] = l_g_HFEN.item()
                    # if self.cri_tv:
                    #    self.log_dict['l_g_tv'] = l_g_tv.item()
                    # if self.cri_ssim:
                    #    self.log_dict['l_g_ssim'] = l_g_ssim.item()
                    if self.cri_lpips:
                        self.log_dict["l_g_lpips"] = l_g_lpips.item()
                    if self.cri_gpl:
                        self.log_dict["l_g_gpl"] = l_g_gpl.item()
                    if self.cri_cpl:
                        self.log_dict["l_g_cpl"] = l_g_cpl.item()
                    if self.cri_gan:
                        self.log_dict["l_g_gan"] = l_g_gan.item()
                    # D
                    self.log_dict["l_d_real"] = l_d_real.item()
                    self.log_dict["l_d_fake"] = l_d_fake.item()

                    if self.opt["train"]["gan_type"] == "wgan-gp":
                        self.log_dict["l_d_gp"] = l_d_gp.item()
                    # D outputs
                    self.log_dict["D_real"] = torch.mean(pred_d_real.detach())
                    self.log_dict["D_fake"] = torch.mean(pred_d_fake.detach())

        # Potential additional phase, can be disabled
        '''
        #Phase 4
        # Test recurrent training of model CFEM
        elif step > self.phase3_s and step < self.niter and self.phase4_s > 0:
            #Freeze/Unfreeze layers
            if self.train_phase == 3 or (step in self.restarts):
                print('Starting phase 4')
                self.train_phase = 4
            #Freeze all layers
            for p in self.netG.parameters():
                p.requires_grad = False
            #Unfreeze the Content Layers, CFEM and CRM
            #CFEM_param = self.netG.module.CFEM.parameters()
            for p in self.netG.module.CFEM.parameters():
                p.requires_grad = True
            #CRM_param = self.netG.module.CRM.parameters()
            for p in self.netG.module.CRM.parameters():
                p.requires_grad = True
            
            if self.bm_count == 0:
                #self.optimizer.step()
                self.optimizer_G.zero_grad()
                self.bm_count = self.batch_multiplier
                print("reset count")
            
            test_save_img = False
            # test_save_img = None
            
            self.fake_Hc, self.fake_Hs, self.fake_Hp = self.netG(self.var_L)
            self.fake_H = self.fake_Hc
            
            #"""
            #####################################################################
            if test_save_img:
                save_images(self.var_H, 0, "self.var_H")
                save_images(self.fake_H.detach(), 0, "self.fake_H")
            #####################################################################
            #"""
            
            #Calculate losses
            l_g_total = 0
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)/self.batch_multiplier
                l_g_total += l_g_pix
            if self.cri_tv: #TV loss
                l_g_tv = self.cri_tv(self.fake_H)/self.batch_multiplier #note: the weight is already multiplied inside the function, doesn't need to be here
                l_g_total += l_g_tv

            try: # Prevent error if there are no parameter for autograd during phase change
                l_g_total.backward()
                #self.optimizer_G.step()
            except:
                #print(sys.exc_info()[0])
                print("skipping iteration", step)
                #pass
            
            self.bm_count -= 1
            print(self.bm_count)
            print(l_g_total)
            
            if self.bm_count == 0:
                print("step")
                self.optimizer_G.step()
            
            # set log
            # G
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            #if self.cri_fea:
            #    self.log_dict['l_g_fea'] = l_g_fea.item()
            #if self.cri_hfen:
            #    self.log_dict['l_g_HFEN'] = l_g_HFEN.item()
            if self.cri_tv:
                self.log_dict['l_g_tv'] = l_g_tv.item()
            #if self.cri_ssim:
            #    self.log_dict['l_g_ssim'] = l_g_ssim.item()
        #'''

        """
        print('Network Parameters:')
        model_dict = self.netG.state_dict()
        for param_name in model_dict:
            param = model_dict[param_name]
            if param.requires_grad == True:
                print('Name: ' + str(param_name))
                print('\tRequires Grad: ' + str(param.requires_grad))
        #"""

        # Prevent the new lr after a restart from being used by the previous phase between phase changes
        if step in self.restarts:
            # Freeze all layers
            for p in self.netG.parameters():
                p.requires_grad = False

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.out_c, self.out_s, self.out_p = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict["LR"] = self.var_L.detach()[0].float().cpu()
        out_dict["img_c"], out_dict["img_s"], out_dict["img_p"] = (
            self.out_c.detach()[0].float().cpu(),
            self.out_s.detach()[0].float().cpu(),
            self.out_p.detach()[0].float().cpu(),
        )

        if need_HR:
            out_dict["HR"] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = "{} - {}".format(
                self.netG.__class__.__name__, self.netG.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.netG.__class__.__name__)

        logger.info(
            "Network G structure: {}, with parameters: {:,d}".format(net_struc_str, n)
        )
        logger.info(s)
        if self.is_train:
            # Discriminator
            if self.cri_gan:
                s, n = self.get_network_description(self.netD)
                if isinstance(self.netD, nn.DataParallel):
                    net_struc_str = "{} - {}".format(
                        self.netD.__class__.__name__,
                        self.netD.module.__class__.__name__,
                    )
                else:
                    net_struc_str = "{}".format(self.netD.__class__.__name__)

                logger.info(
                    "Network D structure: {}, with parameters: {:,d}".format(
                        net_struc_str, n
                    )
                )
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = "{} - {}".format(
                        self.netF.__class__.__name__,
                        self.netF.module.__class__.__name__,
                    )
                else:
                    net_struc_str = "{}".format(self.netF.__class__.__name__)

                logger.info(
                    "Network F structure: {}, with parameters: {:,d}".format(
                        net_struc_str, n
                    )
                )
                logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading pretrained model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.netG)
        if self.opt["is_train"] and self.opt["train"]["gan_weight"]:
            load_path_D = self.opt["path"]["pretrain_model_D"]
            if self.opt["is_train"] and load_path_D is not None:
                logger.info(
                    "Loading pretrained model for D [{:s}] ...".format(load_path_D)
                )
                self.load_network(load_path_D, self.netD)

    def save(self, iter_step):
        self.save_network(self.netG, "G", iter_step)
        if self.cri_gan and self.train_phase >= 3:
            self.save_network(self.netD, "D", iter_step)
