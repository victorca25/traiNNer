from __future__ import absolute_import

import os
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from joblib import Parallel, delayed, parallel_backend

import models.networks as networks
from .base_model import BaseModel, nullcast
from . import losses
from dataops.filters import FilterHigh, FilterLow, GuidedFilter
from dataops.colors import ColorShift
from dataops.augmennt.augmennt import transforms
from dataops.common import tensor2np, np2tensor
from utils.image_pool import ImagePool

logger = logging.getLogger('base')


# TODO: can move to some common module and reuse for other fns
def batch_superpixel(batch_image: torch.Tensor,
    superpixel_fn: callable, num_job:int=None) -> torch.Tensor:
    """ Convert a batch of images to superpixel in parallel.
    Args:
        batch_image: the batch of images. Shape must be [b,c,h,w].
        superpixel_fn: the callable function to apply in parallel.
        num_job: the number of threads to parallelize on. Default: will
            use as many threads as the batch size 'b'.
    Returns:
        superpixel tensor, shape = [b,c,h,w]
    """
    if not num_job:
        num_job = batch_image.shape[0]

    with parallel_backend('threading', n_jobs=num_job):
        batch_out = Parallel()(delayed(superpixel_fn)
                        (image) for image in batch_image)

    return torch.stack(batch_out, dim=0)


def get_sp_transform(train_opt:dict, znorm:bool=True):
    n_segments = train_opt.get('sp_n_segments', 200)  # 500
    max_size = train_opt.get('sp_max_size', None)  # crop_size
    # 'selective' 'cluster' 'rag' None
    reduction = train_opt.get('sp_reduction', 'selective')
    # 'seeds', 'slic', 'slico', 'mslic', 'sk_slic', 'sk_felzenszwalb'
    algo = train_opt.get('sp_algo', 'sk_felzenszwalb')
    gamma_range = train_opt.get('sp_gamma_range', (100, 120))

    superpixel_fn = transforms.Compose([
        transforms.Lambda(lambda img: tensor2np(img, rgb2bgr=True,
                            denormalize=znorm, remove_batch=False)),
        transforms.Superpixels(
            p_replace=1, n_segments=n_segments, algo=algo,
            reduction=reduction, max_size=max_size, p=1),
        transforms.RandomGamma(gamma_range=gamma_range, gain=1, p=1),
        transforms.Lambda(lambda img: np2tensor(img, bgr2rgb=True,
                            normalize=znorm, add_batch=False))
    ])
    return superpixel_fn


class WBCModel(BaseModel):
    """ This class implements the white-box cartoonization (WBC) model,
    for learning image-to-image translation from A (source domain) to B
    (target domain) without paired data.

    WBC paper:
    https://systemerrorwang.github.io/White-box-Cartoonization/paper/06791.pdf
    """

    def __init__(self, opt):
        """Initialize the WBC model class.
        Parameters:
            opt (Option dictionary): stores all the experiment flags.
        """
        super(WBCModel, self).__init__(opt)
        train_opt = opt['train']

        # fetch lambda_idt if provided for identity loss
        self.lambda_idt = train_opt['lambda_identity']

        # specify the images you want to save/display. The training/test
        # scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        if self.is_train and self.lambda_idt and self.lambda_idt > 0.0:
            # if identity loss is used, we also visualize idt_B=G(B)
            self.visual_names.append('idt_B')

        # specify the models you want to load/save to the disk.
        # The training/test scripts will call <BaseModel.save_networks>
        # and <BaseModel.load_networks>
        # for training and testing, a generator 'G' is needed
        self.model_names = ['G']

        # define networks (both generator and discriminator) and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netG.train()
            opt_G_nets = [self.netG]
            opt_D_nets = []
            if train_opt['gan_weight']:
                # add discriminators to the network list
                self.model_names.append('D_S')  # surface
                self.model_names.append('D_T')  # texture
                self.netD_S = networks.define_D(opt).to(self.device)
                self.netD_S.train()
                t_opt = opt.copy()  # TODO: tmp to reuse same config.
                t_opt['network_D']['input_nc'] = 1
                self.netD_T = networks.define_D(t_opt).to(self.device)
                self.netD_T.train()
                opt_D_nets.extend([self.netD_S, self.netD_T])

            # configure AdaTarget
            self.setup_atg()
            if self.atg:
                opt_G_nets.append(self.netLoc)
        self.load()  # load 'G', 'D_T' and 'D_S' if needed

        # additional WBC component, initial guided filter
        # TODO: parameters for GFs can be in options file
        self.guided_filter = GuidedFilter(r=1, eps=1e-2)

        if self.is_train:
            if self.lambda_idt and self.lambda_idt > 0.0:
                # only works when input and output images have the same
                # number of channels
                assert opt['input_nc'] == opt['output_nc']

            # create image buffers to store previously generated images
            self.fake_S_pool = ImagePool(opt['pool_size'])
            self.fake_T_pool = ImagePool(opt['pool_size'])

            # setup batch augmentations
            self.setup_batchaug()

            # setup frequency separation
            self.setup_fs()

            # initialize losses
            # generator losses:
            # for the losses that don't require high precision (can use half precision)
            self.generatorlosses = losses.GeneratorLoss(opt, self.device)
            # for losses that need high precision (use out of the AMP context)
            self.precisegeneratorlosses = losses.PreciseGeneratorLoss(opt, self.device)
            # TODO: show the configured losses names in logger
            # print(self.generatorlosses.loss_list)

            # set filters losses for each representation
            self.surf_losses = opt['train'].get('surf_losses', [])
            self.text_losses = opt['train'].get('text_losses', [])
            self.struct_losses = opt['train'].get('struct_losses', ['fea'])
            self.cont_losses = opt['train'].get('cont_losses', ['fea'])
            self.reg_losses = opt['train'].get('reg_losses', ['tv'])

            # add identity loss if configured
            self.idt_losses = []
            if self.is_train and self.lambda_idt and self.lambda_idt > 0.0:
                self.idt_losses = opt['train'].get('idt_losses', ['pix'])

            # custom representations scales
            self.stru_w = opt['train'].get('struct_scale', 1)
            self.cont_w = opt['train'].get('content_scale', 1)
            self.text_w = opt['train'].get('texture_scale', 1)
            self.surf_w = opt['train'].get('surface_scale', 0.1)
            self.reg_w = opt['train'].get('reg_scale', 1)

            # additional WBC components
            self.colorshift = ColorShift()
            self.guided_filter_surf = GuidedFilter(r=5, eps=2e-1)
            self.sp_transform = get_sp_transform(
                train_opt, opt['datasets']['train']['znorm'])

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

            # create logs dictionaries
            self.log_dict = OrderedDict()
            self.log_dict_T = OrderedDict()
            self.log_dict_S = OrderedDict()

            # configure SWA
            self.setup_swa()

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
        fake_B = self.netG(self.real_A)  # G(A)
        self.fake_B = self.guided_filter(self.real_A, fake_B)

        if self.is_train:
            # generate representations images
            # surface: fake_blur
            self.fake_blur = self.guided_filter_surf(
                self.fake_B, self.fake_B)
            # surface: real_blur (cartoon)
            self.real_blur = self.guided_filter_surf(
                self.real_B, self.real_B)
            # texture: fake_gray, real_gray (cartoon)
            self.fake_gray, self.real_gray = self.colorshift(
                self.fake_B, self.real_B)
            # structure: get superpixels (sp_real)
            self.sp_real = (
                batch_superpixel(
                    self.fake_B.detach(),  # self.real_A, #
                    self.sp_transform)
                ).to(self.device)

    def backward_D_T(self):
        """Calculate GAN loss for texture discriminator D_T."""
        fake_gray = self.fake_T_pool.query(self.fake_gray)
        self.log_dict_T = self.backward_D_Basic(
            self.netD_T, self.real_gray, fake_gray, self.log_dict_T)
        # aggregate logs to global logger
        for kls_T, vls_T in self.log_dict_T.items():
            self.log_dict[f'{kls_T}_T'] = vls_T  # * self.text_w

    def backward_D_S(self):
        """Calculate GAN loss for surface discriminator D_S."""
        fake_blur = self.fake_S_pool.query(self.fake_blur)
        self.log_dict_S = self.backward_D_Basic(
            self.netD_S, self.real_blur, fake_blur, self.log_dict_S)
        # aggregate logs to global logger
        for kls_S, vls_S in self.log_dict_S.items():
            self.log_dict[f'{kls_S}_S'] = vls_S  # * self.surf_w

    def backward_G(self):
        """Calculate the loss for generator G."""
        # prepare losses and image pairs
        rep_names = ['surf', 'text', 'struct', 'cont', 'reg']
        selectors = [self.surf_losses, self.text_losses,
            self.struct_losses, self.cont_losses, self.reg_losses]
        sel_fakes = [self.fake_blur, self.fake_gray,
            self.fake_B, self.fake_B, self.fake_B]
        sel_reals = [self.real_blur, self.real_gray,
            self.sp_real, self.real_A, self.real_B]
        rep_ws = [self.surf_w, self.text_w,
            self.stru_w, self.cont_w, self.reg_w]

        l_g_total = 0
        # l_g_total = torch.zeros(1)  # 0
        with self.cast():
            if self.lambda_idt and self.lambda_idt > 0:
                self.idt_B = self.netG(self.real_B)
                log_idt_dict = OrderedDict()

            # Identity loss (fp16)
            if self.lambda_idt and self.lambda_idt > 0 and self.idt_losses:
                # G should be identity if real_B is fed: ||G(B) - B|| = 0
                loss_idt_B, log_idt_dict = self.generatorlosses(
                    self.idt_B, self.real_B, log_idt_dict,
                    self.f_low, selector=self.idt_losses)
                l_g_total += sum(loss_idt_B) * self.lambda_idt / self.accumulations
                for kidt_B, vidt_B in log_idt_dict.items():
                    self.log_dict[f'{kidt_B}_idt'] = vidt_B

            if self.cri_gan:
                # texture adversarial loss
                l_g_gan_T = self.adversarial(
                    self.fake_gray, self.real_gray, netD=self.netD_T,
                    stage='generator', fsfilter=self.f_high)
                self.log_dict_T['l_g_gan'] = l_g_gan_T.item()
                l_g_total += self.text_w * l_g_gan_T / self.accumulations

                # surface adversarial loss
                l_g_gan_S = self.adversarial(
                    self.fake_blur, self.real_blur, netD=self.netD_S,
                    stage='generator', fsfilter=self.f_high)
                self.log_dict_S['l_g_gan'] = l_g_gan_S.item()
                l_g_total += self.surf_w * l_g_gan_S / self.accumulations

            # calculate remaining losses
            for sn, fake, real, sel, w in zip(
                rep_names, sel_fakes, sel_reals, selectors, rep_ws):
                if not sel:
                    continue
                loss_results, log_dict = self.generatorlosses(
                    fake, real, {}, self.f_low, selector=sel)
                l_g_total += w * sum(loss_results) / self.accumulations
                for ksel, vsel in log_dict.items():
                    self.log_dict[f'{ksel}_{sn}'] = vsel  # * w

        # high precision generator losses (can be affected by AMP half precision)
        if self.precisegeneratorlosses.loss_list:
            if self.lambda_idt and self.lambda_idt > 0 and self.idt_losses:
                # Identity loss (precise losses)
                # G should be identity if real_B is fed: ||G(B) - B|| = 0
                precise_loss_idt_B, log_idt_dict = self.precisegeneratorlosses(
                    self.idt_B, self.real_B, log_idt_dict,
                    self.f_low, selector=self.idt_losses)
                l_g_total += sum(precise_loss_idt_B) * self.lambda_idt / self.accumulations
                for kidt_B, vidt_B in log_idt_dict.items():
                    self.log_dict[f'{kidt_B}_idt'] = vidt_B

            for sn, fake, real, sel, w in zip(
                rep_names, sel_fakes, sel_reals, selectors, rep_ws):
                if not sel:
                    continue
                precise_loss_results, log_dict = self.precisegeneratorlosses(
                    fake, real, {}, self.f_low, selector=sel)
                l_g_total += w * sum(precise_loss_results) / self.accumulations
                for ksel, vsel in log_dict.items():
                    self.log_dict[f'{ksel}_{sn}'] = vsel  # * w

        # calculate G gradients
        self.calc_gradients(l_g_total)

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
        if self.mixup:
            self.real_B, self.real_A = self.batchaugment(self.real_B, self.real_A)

        # run G(A)
        with self.cast():  # casts operations to mixed precision if enabled, else nullcontext
            self.forward()  # compute fake images: G(A)

        # apply mask if batchaug == "cutout"
        if self.mixup:
            self.fake_B, self.real_B = self.batchaugment.apply_mask(self.fake_B, self.real_B)

        # adatarget
        if self.atg:
            self.fake_H = self.ada_out(
                output=self.fake_H, target=self.real_H,
                loc_model=self.netLoc)

        if self.cri_gan:
            # update D_T and D_S
            self.requires_grad(self.netD_T, True)  # enable backprop for D_T
            self.requires_grad(self.netD_S, True)  # enable backprop for D_S
            if isinstance(self.feature_loc, int):
                # freeze up to the selected layers
                for loc in range(self.feature_loc):
                    self.requires_grad(
                        self.netD_T, False, target_layer=loc, net_type='D')
                    self.requires_grad(
                        self.netD_S, False, target_layer=loc, net_type='D')

            self.backward_D_T()  # calculate gradients for D_T
            self.backward_D_S()  # calculate gradidents for D_S

            # step D_T and D_S optimizer and update weights
            self.optimizer_step(step, self.optimizer_D, "D")

        # update G
        if self.cri_gan:
            # Ds require no gradients when optimizing G
            self.requires_grad(self.netD_T, flag=False, net_type='D')
            self.requires_grad(self.netD_S, flag=False, net_type='D')

        if (self.cri_gan is not True) or (eff_step % self.D_update_ratio == 0
            and eff_step > self.D_init_iters):
            # calculate gradients for G_A and G_B
            self.backward_G()

            # step G optimizer and update weights
            self.optimizer_step(step, self.optimizer_G, "G")

    def get_current_log(self):
        """Return traning losses / errors. train.py will print out these on the
            console, and save them to a file."""
        return self.log_dict

    def get_current_visuals(self):
        """Return visualization images. train.py will display and/or save these images."""
        out_dict = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                out_dict[name] = getattr(self, name).detach()[0].float().cpu()
        return out_dict
