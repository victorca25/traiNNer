from __future__ import absolute_import

import os
import logging
from collections import OrderedDict
import torch
import torch.nn as nn

import models.networks as networks
from .base_model import BaseModel
from . import losses
from dataops.batchaug import BatchAug
from dataops.filters import FilterHigh, FilterLow
from utils.image_pool import ImagePool

logger = logging.getLogger('base')


class CycleGANModel(BaseModel):
    """ This class implements the CycleGAN model, for learning
    image-to-image translation from A (source domain) to B (target
    domain) without paired data.
    The model training uses by default:
        netG: resnet_9blocks ResNet generator
        netD: basic discriminator (PatchGAN)
        gan_type: lsgan (least-square GANs objective)
        norm: instance (instancenorm)
        dataset_mode: unaligned
        pool_size: 50
        lambda_A: 10.0 weight for cycle loss (A -> B -> A)
        lambda_B: 10.0 weight for cycle loss (B -> A -> B)
        lambda_identity: 0.5 to use identity mapping
        lr: 0.0002
        lr_policy: linear

    For CycleGAN, in addition to GAN losses, it introduces lambda_A,
        lambda_B, and lambda_identity as losses in as follow:
        Generators: G_A (A -> B); G_B (B -> A)
        Discriminators: D_A (G_A(A) vs. B); D_B (G_B(B) vs. A)
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional):
            Setting lambda_identity other than 0 has an effect of scaling the weight
            of the identity mapping loss in respect to the cycle generator loss.
            For example, if the weight of the identity loss should be 10 times
            smaller than the weight of the reconstruction loss, set
            lambda_identity = 0.1. The full equation is:
            lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A)
            (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option dictionary): stores all the experiment flags.
        """
        super(CycleGANModel, self).__init__(opt)
        train_opt = opt['train']

        # fetch lambda_idt if provided for identity loss
        self.lambda_idt = train_opt['lambda_identity']

        # specify the images you want to save/display.
        # The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.is_train and self.lambda_idt and self.lambda_idt > 0.0:
            # if identity loss is used, also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        # combine visualizations for A and B
        self.visual_names = visual_names_A + visual_names_B

        # specify the models you want to load/save to the disk.
        # The training/test scripts will call <BaseModel.save_networks>
        # and <BaseModel.load_networks>
        # for training and testing, a generator 'G' is needed
        self.model_names = ['G_A']

        # define networks (both generator and discriminator) and load pretrained models
        # *The naming is different from those used in the paper.
        #   Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt).to(self.device)  # G_A
        if self.is_train:
            # for training 2 generators are needed, add to list and define
            self.model_names.append('G_B')
            self.netG_B = networks.define_G(opt).to(self.device)  # G_B
            self.netG_A.train()
            self.netG_B.train()
            opt_G_nets = [self.netG_A, self.netG_B]
            opt_D_nets = []
            if train_opt['gan_weight']:
                # add discriminators to the network list
                self.model_names.append('D_A')
                self.model_names.append('D_B')
                self.netD_A = networks.define_D(opt).to(self.device)  # D_A
                self.netD_B = networks.define_D(opt).to(self.device)  # D_B
                self.netD_A.train()
                self.netD_B.train()
                opt_D_nets.extend([self.netD_A, self.netD_B])

            # configure AdaTarget
            self.setup_atg()
            if self.atg:
                opt_G_nets.append(self.netLoc)
        self.load()  # load 'G_A', 'G_B', 'D_A' and 'D_B' if needed

        if self.is_train:
            if self.lambda_idt and self.lambda_idt > 0.0:
                # only works when input and output images have the same
                # number of channels
                assert opt['input_nc'] == opt['output_nc']

            # create image buffers to store previously generated images
            self.fake_A_pool = ImagePool(opt['pool_size'])
            self.fake_B_pool = ImagePool(opt['pool_size'])

            # setup batch augmentations
            self.setup_batchaug()

            # setup frequency separation
            self.setup_fs()

            # initialize losses
            # generator losses:
            # for the losses that don't require high precision (can use half precision)
            self.cyclelosses = losses.GeneratorLoss(opt, self.device)
            # for losses that need high precision (use out of the AMP context)
            self.precisecyclelosses = losses.PreciseGeneratorLoss(opt, self.device)
            # TODO: show the configured losses names in logger
            # print(self.cyclelosses.loss_list)

            # add identity loss if configured
            if self.is_train and self.lambda_idt and self.lambda_idt > 0.0:
                # TODO: using the same losses as cycle/generator, could be
                #   different. Use filters like in WBC.
                # self.idtlosses = losses.GeneratorLoss(opt, self.device)
                self.idtlosses = self.cyclelosses
                # self.preciseidtlosses = losses.PreciseGeneratorLoss(opt, self.device)
                self.preciseidtlosses = self.precisecyclelosses

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
            self.log_dict_A = OrderedDict()
            self.log_dict_B = OrderedDict()

            # configure SWA
            # TODO: configure SWA for two Generators
            # self.setup_swa()

            # configure virtual batch
            self.setup_virtual_batch()

            # configure AMP
            self.setup_amp()

        # print network
        # TODO: pass verbose flag from config file
        self.print_network(verbose=False)

    def feed_data(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            data (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # TODO: images currently being flipped with BtoA during read, check logic
        # AtoB = self.opt.get('direction') == 'AtoB'
        # self.real_A = data['A' if AtoB else 'B'].to(self.device)
        # self.real_B = data['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = data['A_path' if AtoB else 'B_path']
        self.real_A = data['A'].to(self.device)
        self.real_B = data['B'].to(self.device)
        self.image_paths = data['A_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A."""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.log_dict_A = self.backward_D_Basic(
            self.netD_A, self.real_B, fake_B, self.log_dict_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B."""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.log_dict_B = self.backward_D_Basic(
            self.netD_B, self.real_A, fake_A, self.log_dict_B)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B."""
        l_g_total = 0
        with self.cast():
            if self.lambda_idt and self.lambda_idt > 0:
                self.idt_A = self.netG_A(self.real_B)
                log_idt_dict_A = OrderedDict()
                log_idt_dict_B = OrderedDict()

            # Identity loss (fp16)
            if self.lambda_idt and self.lambda_idt > 0 and self.idtlosses.loss_list:
                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                loss_idt_A, log_idt_dict_A = self.idtlosses(
                    self.idt_A, self.real_B, log_idt_dict_A, self.f_low)
                l_g_total += sum(loss_idt_A) * self.lambda_idt / self.accumulations
                for kidt_A, vidt_A in log_idt_dict_A.items():
                    self.log_dict_A[f'{kidt_A}_idt'] = vidt_A

                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B = self.netG_B(self.real_A)
                loss_idt_B, log_idt_dict_B = self.idtlosses(
                    self.idt_B, self.real_A, log_idt_dict_B, self.f_low)
                l_g_total += sum(loss_idt_B) * self.lambda_idt / self.accumulations
                for kidt_B, vidt_B in log_idt_dict_B.items():
                    self.log_dict_B[f'{kidt_B}_idt'] = vidt_B

            if self.cri_gan:
                # adversarial lossses
                # GAN loss D_A(G_A(A)) (if non-relativistic)
                l_g_gan_A = self.adversarial(
                    self.fake_B, self.real_A, netD=self.netD_A,
                    stage='generator', fsfilter=self.f_high)  # (fake_B, real_A)
                self.log_dict_A['l_g_gan'] = l_g_gan_A.item()
                l_g_total += l_g_gan_A / self.accumulations

                # GAN loss D_B(G_B(B)) (if non-relativistic)
                l_g_gan_B = self.adversarial(
                    self.fake_A, self.real_B, netD=self.netD_B,
                    stage='generator', fsfilter=self.f_high)  # (fake_A, real_B)
                self.log_dict_B['l_g_gan'] = l_g_gan_B.item()
                l_g_total += l_g_gan_B / self.accumulations

            loss_results = []
            # Forward cycle loss || G_B(G_A(A)) - A||
            loss_results, self.log_dict_A = self.cyclelosses(
                self.rec_A, self.real_A, self.log_dict_A, self.f_low)
            l_g_total += sum(loss_results) / self.accumulations

            # Backward cycle loss || G_A(G_B(B)) - B||
            loss_results, self.log_dict_B = self.cyclelosses(
                self.rec_B, self.real_B, self.log_dict_B, self.f_low)
            l_g_total += sum(loss_results) / self.accumulations

        if self.lambda_idt and self.lambda_idt > 0 and self.preciseidtlosses.loss_list:
            # Identity loss (precise losses)
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            # self.idt_A = self.netG_A(self.real_B)
            precise_loss_idt_A, log_idt_dict_A = self.preciseidtlosses(
                self.idt_A, self.real_B, log_idt_dict_A, self.f_low)
            l_g_total += sum(precise_loss_idt_A) * self.lambda_idt / self.accumulations
            for kidt_A, vidt_A in log_idt_dict_A.items():
                self.log_dict_A[f'{kidt_A}_idt'] = vidt_A

            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            # self.idt_B = self.netG_B(self.real_A)
            precise_loss_idt_B, log_idt_dict_B = self.preciseidtlosses(
                self.idt_B, self.real_A, log_idt_dict_B, self.f_low)
            l_g_total += sum(precise_loss_idt_B) * self.lambda_idt / self.accumulations
            for kidt_B, vidt_B in log_idt_dict_B.items():
                self.log_dict_B[f'{kidt_B}_idt'] = vidt_B

        # high precision generator losses (can be affected by AMP half precision)
        if self.precisecyclelosses.loss_list:
            precise_loss_results = []
            # Forward cycle loss || G_B(G_A(A)) - A||
            precise_loss_results, self.log_dict_A = self.precisecyclelosses(
                    self.rec_A, self.real_A, self.log_dict_A, self.f_low)
            l_g_total += sum(precise_loss_results) / self.accumulations

            # Backward cycle loss || G_A(G_B(B)) - B||
            precise_loss_results, self.log_dict_B = self.precisecyclelosses(
                    self.rec_B, self.real_B, self.log_dict_B, self.f_low)
            l_g_total += sum(precise_loss_results) / self.accumulations

        # calculate G gradients
        self.calc_gradients(l_g_total)

        # aggregate both cycles logs to global logger
        for kls_A, vls_A in self.log_dict_A.items():
            self.log_dict[f'{kls_A}_A'] = vls_A
        for kls_B, vls_B in self.log_dict_B.items():
            self.log_dict[f'{kls_B}_B'] = vls_B

    def optimize_parameters(self, step):
        """Calculate losses, gradients, and update network weights;
        called in every training iteration."""
        eff_step = step/self.accumulations

        # switch ATG to train
        if self.atg:
            if eff_step > self.atg_start_iter:
                self.switch_atg(True)
            else:
                self.switch_atg(False)

        # batch (mixup) augmentations
        aug = None
        if self.mixup:
            self.real_B, self.real_A, mask, aug = BatchAug(
                self.real_B, self.real_A,
                self.mixopts, self.mixprob, self.mixalpha,
                self.aux_mixprob, self.aux_mixalpha, self.mix_p
                )

        # run G_A(A), G_B(G_A(A)), G_B(B), G_A(G_B(B))
        with self.cast():  # Casts operations to mixed precision if enabled, else nullcontext
            self.forward()  # compute fake images and reconstruction images.

        # batch (mixup) augmentations
        # cutout-ed pixels are discarded when calculating loss by masking removed pixels
        if aug == "cutout":
            self.fake_B, self.real_B = self.fake_B * mask, self.real_B * mask
            self.fake_A, self.real_A = self.fake_A * mask, self.real_A * mask

        # adatarget
        if self.atg:
            self.fake_H = self.ada_out(
                output=self.fake_H, target=self.real_H,
                loc_model=self.netLoc)

        # update G_A and G_B
        if self.cri_gan:
            # Ds require no gradients when optimizing Gs
            self.requires_grad(self.netD_A, flag=False, net_type='D')
            self.requires_grad(self.netD_B, flag=False, net_type='D')

        if (self.cri_gan is not True) or (eff_step % self.D_update_ratio == 0
            and eff_step > self.D_init_iters):
            # calculate gradients for G_A and G_B
            self.backward_G()

            # step G_A and G_B optimizer and update weights
            self.optimizer_step(step, self.optimizer_G, "G")

        if self.cri_gan:
            # update D_A and D_B
            self.requires_grad(self.netD_A, True)  # enable backprop for D_A
            self.requires_grad(self.netD_B, True)  # enable backprop for D_B
            if isinstance(self.feature_loc, int):
                # freeze up to the selected layers
                for loc in range(self.feature_loc):
                    self.requires_grad(
                        self.netD_A, False, target_layer=loc, net_type='D')
                    self.requires_grad(
                        self.netD_B, False, target_layer=loc, net_type='D')

            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate gradients for D_B

            # step D_A and D_B optimizer and update weights
            self.optimizer_step(step, self.optimizer_D, "D")

    def get_current_log(self, direction=None):
        """Return traning losses / errors. train.py will print out these on the
        console, and save them to a file"""
        if direction == 'A':
            return self.log_dict_A
        elif direction == 'B':
            return self.log_dict_B
        return self.log_dict

    def get_current_visuals(self):
        """Return visualization images. train.py will display and/or save these images"""
        out_dict = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                out_dict[name] = getattr(self, name).detach()[0].float().cpu()
        return out_dict
