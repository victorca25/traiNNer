import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from models.modules.loss import GANLoss, GradientPenaltyLoss
from .__model__ import Model

logger = logging.getLogger("base")


class SFTGAN_ACD_Model(Model):
    def __init__(self, opt):
        super(SFTGAN_ACD_Model, self).__init__(opt)
        train_opt = opt["train"]

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)  # D
            self.netG.train()
            self.netD.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt["pixel_weight"] > 0:
                l_pix_type = train_opt["pixel_criterion"]
                if l_pix_type == "l1":
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == "l2":
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError(
                        "Loss type [{:s}] not recognized.".format(l_pix_type)
                    )
                self.l_pix_w = train_opt["pixel_weight"]
            else:
                logging.info("Remove pixel loss.")
                self.cri_pix = None

            # G feature loss
            if train_opt["feature_weight"] > 0:
                l_fea_type = train_opt["feature_criterion"]
                if l_fea_type == "l1":
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == "l2":
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError(
                        "Loss type [{:s}] not recognized.".format(l_fea_type)
                    )
                self.l_fea_w = train_opt["feature_weight"]
            else:
                logging.info("Remove feature loss.")
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # GD gan loss
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
                self.cri_gp = GradientPenaltyLoss(device=self.device).to(self.device)
                self.l_gp_w = train_opt["gp_weigth"]

            # D cls loss
            self.cri_ce = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
            # ignore background, since bg images may conflict with other classes

            # optimizers
            # G
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params_SFT = []
            optim_params_other = []
            for (
                    k,
                    v,
            ) in self.netG.named_parameters():  # can optimize for a part of the model
                if "SFT" in k or "Cond" in k:
                    optim_params_SFT.append(v)
                else:
                    optim_params_other.append(v)
            self.optimizer_G_SFT = torch.optim.Adam(
                optim_params_SFT,
                lr=train_opt["lr_G"] * 5,
                weight_decay=wd_G,
                betas=(train_opt["beta1_G"], 0.999),
            )
            self.optimizer_G_other = torch.optim.Adam(
                optim_params_other,
                lr=train_opt["lr_G"],
                weight_decay=wd_G,
                betas=(train_opt["beta1_G"], 0.999),
            )
            self.optimizers.append(self.optimizer_G_SFT)
            self.optimizers.append(self.optimizer_G_other)
            # D
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
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data["LR"].to(self.device)
        # seg
        self.var_seg = data["seg"].to(self.device)
        # category
        self.var_cat = data["category"].long().to(self.device)

        if need_HR:  # train or val
            self.var_H = data["HR"].to(self.device)

    def optimize_parameters(self, step):
        # G
        self.optimizer_G_SFT.zero_grad()
        self.optimizer_G_other.zero_grad()
        self.fake_H = self.netG((self.var_L, self.var_seg))

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
            # G gan + cls loss
            pred_g_fake, cls_g_fake = self.netD(self.fake_H)
            l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
            l_g_cls = self.l_gan_w * self.cri_ce(cls_g_fake, self.var_cat)
            l_g_total += l_g_gan
            l_g_total += l_g_cls

            l_g_total.backward()
            self.optimizer_G_SFT.step()
        if step > 20000:
            self.optimizer_G_other.step()

        # D
        self.optimizer_D.zero_grad()
        l_d_total = 0
        # real data
        pred_d_real, cls_d_real = self.netD(self.var_H)
        l_d_real = self.cri_gan(pred_d_real, True)
        l_d_cls_real = self.cri_ce(cls_d_real, self.var_cat)
        # fake data
        pred_d_fake, cls_d_fake = self.netD(
            self.fake_H.detach()
        )  # detach to avoid BP to G
        l_d_fake = self.cri_gan(pred_d_fake, False)
        l_d_cls_fake = self.cri_ce(cls_d_fake, self.var_cat)

        l_d_total = l_d_real + l_d_cls_real + l_d_fake + l_d_cls_fake

        if self.opt["train"]["gan_type"] == "wgan-gp":
            batch_size = self.var_H.size(0)
            if self.random_pt.size(0) != batch_size:
                self.random_pt.resize_(batch_size, 1, 1, 1)
            self.random_pt.uniform_()  # Draw random interpolation points
            interp = (
                    self.random_pt * self.fake_H.detach()
                    + (1 - self.random_pt) * self.var_H
            )
            interp.requires_grad = True
            interp_crit, _ = self.netD(interp)
            l_d_gp = self.l_gp_w * self.cri_gp(
                interp, interp_crit
            )  # maybe wrong in cls?
            l_d_total += l_d_gp

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # G
            if self.cri_pix:
                self.log_dict["l_g_pix"] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict["l_g_fea"] = l_g_fea.item()
            self.log_dict["l_g_gan"] = l_g_gan.item()
        # D
        self.log_dict["l_d_real"] = l_d_real.item()
        self.log_dict["l_d_fake"] = l_d_fake.item()
        self.log_dict["l_d_cls_real"] = l_d_cls_real.item()
        self.log_dict["l_d_cls_fake"] = l_d_cls_fake.item()
        if self.opt["train"]["gan_type"] == "wgan-gp":
            self.log_dict["l_d_gp"] = l_d_gp.item()
        # D outputs
        self.log_dict["D_real"] = torch.mean(pred_d_real.detach())
        self.log_dict["D_fake"] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG((self.var_L, self.var_seg))
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict["LR"] = self.var_L.detach()[0].float().cpu()
        out_dict["SR"] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict["HR"] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # G
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
            # D
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel):
                net_struc_str = "{} - {}".format(
                    self.netD.__class__.__name__, self.netD.module.__class__.__name__
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
        load_path_D = self.opt["path"]["pretrain_model_D"]
        if self.opt["is_train"] and load_path_D is not None:
            logger.info("Loading pretrained model for D [{:s}] ...".format(load_path_D))
            self.load_network(load_path_D, self.netD)

    def save(self, iter_step):
        self.save_network(self.netG, "G", iter_step)
        self.save_network(self.netD, "D", iter_step)
