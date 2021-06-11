from __future__ import absolute_import

import os
import logging
from collections import OrderedDict
import itertools

import torch
import torch.nn as nn

import models.networks as networks
from .base_model import BaseModel, nullcast

from . import losses
from . import optimizers
from . import schedulers
from . import swa

from dataops.batchaug import BatchAug
from dataops.filters import FilterHigh, FilterLow  # , FilterX
from utils.image_pool import ImagePool

logger = logging.getLogger('base')

load_amp = (hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"))
if load_amp:
    from torch.cuda.amp import autocast, GradScaler
    logger.info('AMP library available')
else:
    logger.info('AMP library not available')


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
            opt (Option dictionary): stores all the experiment flags
        """
        super(CycleGANModel, self).__init__(opt)
        train_opt = opt['train']

        # fetch lambda_idt if provided for identity loss
        self.lambda_idt = train_opt['lambda_identity']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.is_train and self.lambda_idt and self.lambda_idt > 0.0:
            # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        # combine visualizations for A and B
        self.visual_names = visual_names_A + visual_names_B

        # specify the models you want to load/save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
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
            if train_opt['gan_weight']:
                # add discriminators to the network list
                self.model_names.append('D_A')
                self.model_names.append('D_B')
                self.netD_A = networks.define_D(opt).to(self.device)  # D_A
                self.netD_B = networks.define_D(opt).to(self.device)  # D_B
                self.netD_A.train()
                self.netD_B.train()
        self.load()  # load 'G_A', 'G_B', 'D_A' and 'D_B' if needed

        if self.is_train:
            if self.lambda_idt and self.lambda_idt > 0.0:
                # only works when input and output images have the same number of channels
                assert opt['input_nc'] == opt['output_nc']

            # create image buffers to store previously generated images
            self.fake_A_pool = ImagePool(opt['pool_size'])
            self.fake_B_pool = ImagePool(opt['pool_size'])

            """
            Setup batch augmentations
            #TODO: test
            """
            self.mixup = train_opt.get('mixup', None)
            if self.mixup:
                self.mixopts = train_opt.get('mixopts', ["blend", "rgb", "mixup", "cutmix", "cutmixup"])  # , "cutout", "cutblur"]
                self.mixprob = train_opt.get('mixprob', [1.0, 1.0, 1.0, 1.0, 1.0])  # , 1.0, 1.0]
                self.mixalpha = train_opt.get('mixalpha', [0.6, 1.0, 1.2, 0.7, 0.7])  # , 0.001, 0.7]
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
            # Initialize the losses with the opt parameters
            # Generator losses:
            # for the losses that don't require high precision (can use half precision)
            self.cyclelosses = losses.GeneratorLoss(opt, self.device)
            # for losses that need high precision (use out of the AMP context)
            self.precisecyclelosses = losses.PreciseGeneratorLoss(opt, self.device)
            # TODO: show the configured losses names in logger
            # print(self.cyclelosses.loss_list)

            # add identity loss if configured
            if self.is_train and self.lambda_idt and self.lambda_idt > 0.0:
                # TODO: using the same losses as cycle/generator, could be different
                # self.idtlosses = losses.GeneratorLoss(opt, self.device)
                self.idtlosses = self.cyclelosses
                # self.preciseidtlosses = losses.PreciseGeneratorLoss(opt, self.device)
                self.preciseidtlosses = self.precisecyclelosses

            # Discriminator loss:
            if train_opt['gan_type'] and train_opt['gan_weight']:
                # TODO:
                # self.criterionGAN = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
                self.cri_gan = True
                diffaug = train_opt.get('diffaug', None)
                dapolicy = None
                if diffaug:  # TODO: this if should not be necessary
                    dapolicy = train_opt.get('dapolicy', 'color,translation,cutout')  # original
                self.adversarial = losses.Adversarial(train_opt=train_opt, device=self.device,
                                                      diffaug=diffaug, dapolicy=dapolicy,
                                                      conditional=False)
                # TODO:
                # D_update_ratio and D_init_iters are for WGAN
                # self.D_update_ratio = train_opt.get('D_update_ratio', 1)
                # self.D_init_iters = train_opt.get('D_init_iters', 0)
            else:
                self.cri_gan = False

            """
            Initialize optimizers
            """
            self.optGstep = False
            self.optDstep = False
            if self.cri_gan:
                self.optimizers, self.optimizer_G, self.optimizer_D = optimizers.get_optimizers(
                    cri_gan=self.cri_gan,
                    optim_paramsD=itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                    optim_paramsG=itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                    train_opt=train_opt, logger=logger, optimizers=self.optimizers)
            else:
                self.optimizers, self.optimizer_G = optimizers.get_optimizers(
                    None, None,
                    optim_paramsG=itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                    train_opt=train_opt, logger=logger, optimizers=self.optimizers)
                self.optDstep = True

            """
            Prepare schedulers
            """
            self.schedulers = schedulers.get_schedulers(
                optimizers=self.optimizers, schedulers=self.schedulers, train_opt=train_opt)

            """
            Configure SWA
            """
            # TODO: configure SWA for two Generators
            # self.swa = opt.get('use_swa', False)
            # if self.swa:
            #     self.swa_start_iter = train_opt.get('swa_start_iter', 0)
            #     # self.swa_start_epoch = train_opt.get('swa_start_epoch', None)
            #     swa_lr = train_opt.get('swa_lr', 0.0001)
            #     swa_anneal_epochs = train_opt.get('swa_anneal_epochs', 10)
            #     swa_anneal_strategy = train_opt.get('swa_anneal_strategy', 'cos')
            #     #TODO: Note: This could be done in resume_training() instead, to prevent creating
            #     # the swa scheduler and model before they are needed
            #     self.swa_scheduler, self.swa_model_A = swa.get_swa(
            #             self.optimizer_G, itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            #             swa_lr, swa_anneal_epochs, swa_anneal_strategy)
            #     self.load_swa() #load swa from resume state
            #     logger.info('SWA enabled. Starting on iter: {}, lr: {}'.format(self.swa_start_iter, swa_lr))

            """
            If using virtual batch
            """
            batch_size = opt["datasets"]["train"]["batch_size"]
            virtual_batch = opt["datasets"]["train"].get('virtual_batch_size', None)
            self.virtual_batch = virtual_batch if virtual_batch \
                >= batch_size else batch_size
            self.accumulations = self.virtual_batch // batch_size
            self.optimizer_G.zero_grad()
            if self.cri_gan:
                self.optimizer_D.zero_grad()

            """
            Configure AMP
            """
            self.amp = load_amp and opt.get('use_amp', False)
            if self.amp:
                self.cast = autocast
                self.amp_scaler = GradScaler()
                logger.info('AMP enabled')
            else:
                self.cast = nullcast

            """
            Configure FreezeD
            """
            if self.cri_gan:
                self.feature_loc = None
                loc = train_opt.get('freeze_loc', False)
                if loc:
                    disc = opt["network_D"].get('which_model_D', False)
                    if "discriminator_vgg" in disc and "fea" not in disc:
                        loc = (loc*3)-2
                    elif "patchgan" in disc:
                        loc = (loc*3)-1
                    # TODO: TMP, for now only tested with the vgg-like or patchgan discriminators
                    if "discriminator_vgg" in disc or "patchgan" in disc:
                        self.feature_loc = loc
                        logger.info('FreezeD enabled')

            self.log_dict = OrderedDict()
            self.log_dict_A = OrderedDict()
            self.log_dict_B = OrderedDict()

        self.print_network(verbose=False)  # TODO: pass verbose flag from config file

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

    def backward_D_basic(self, netD, real, fake, log_dict):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network): the discriminator D
            real (tensor array): real images
            fake (tensor array): images generated by a generator
        Return the discriminator loss.
        Also calls l_d_total.backward() to calculate the gradients.
        """

        l_d_total = 0
        with self.cast():
            l_d_total, gan_logs = self.adversarial(
                fake, real, netD=netD,
                stage='discriminator', fsfilter=self.f_high)

            for g_log in gan_logs:
                log_dict[g_log] = gan_logs[g_log]

            l_d_total /= self.accumulations

        # calculate gradients
        if self.amp:
            # call backward() on scaled loss to create scaled gradients.
            self.amp_scaler.scale(l_d_total).backward()
        else:
            l_d_total.backward()
        # return l_d_total
        return log_dict

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.log_dict_A = self.backward_D_basic(
            self.netD_A, self.real_B, fake_B, self.log_dict_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.log_dict_B = self.backward_D_basic(
            self.netD_B, self.real_A, fake_A, self.log_dict_B)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""

        l_g_total = 0
        with self.cast():
            # Identity loss (fp16)
            if self.lambda_idt and self.lambda_idt > 0 and self.idtlosses.loss_list:
                log_idt_dict_A = OrderedDict()
                log_idt_dict_B = OrderedDict()

                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                self.idt_A = self.netG_A(self.real_B)
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

        # calculate gradients
        if self.amp:
            # call backward() on scaled loss to create scaled gradients.
            self.amp_scaler.scale(l_g_total).backward()
        else:
            l_g_total.backward()

        # aggregate both cycles logs to global logger
        for kls_A, vls_A in self.log_dict_A.items():
            self.log_dict[f'{kls_A}_A'] = vls_A
        for kls_B, vls_B in self.log_dict_B.items():
            self.log_dict[f'{kls_B}_B'] = vls_B

    def optimize_parameters(self, step):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
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
            self.fake_B, self.real_B = self.fake_B*mask, self.real_B*mask
            self.fake_A, self.real_A = self.fake_A*mask, self.real_A*mask

        # update G_A and G_B
        if self.cri_gan:
            # Ds require no gradients when optimizing Gs
            self.requires_grad(self.netD_A, flag=False, net_type='D')
            self.requires_grad(self.netD_B, flag=False, net_type='D')

        self.backward_G()  # calculate gradients for G_A and G_B
        # only step and clear gradient if virtual batch has completed
        if (step + 1) % self.accumulations == 0:
            if self.amp:
                self.amp_scaler.step(self.optimizer_G)
                self.amp_scaler.update()
            else:
                self.optimizer_G.step()  # update G_A and G_B's weights
            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.optGstep = True

        if self.cri_gan:
            # update D_A and D_B
            self.requires_grad(self.netD_A, True)  # enable backprop for D_A
            self.requires_grad(self.netD_B, True)  # enable backprop for D_B
            if isinstance(self.feature_loc, int):
                # freeze up to the selected layers
                for loc in range(self.feature_loc):
                    self.requires_grad(self.netD_A, False, target_layer=loc, net_type='D')
                    self.requires_grad(self.netD_B, False, target_layer=loc, net_type='D')

            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate graidents for D_B
            # only step and clear gradient if virtual batch has completed
            if (step + 1) % self.accumulations == 0:
                if self.amp:
                    self.amp_scaler.step(self.optimizer_D)
                    self.amp_scaler.update()
                else:
                    self.optimizer_D.step()  # update D_A and D_B's weights
                self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
                self.optDstep = True

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
