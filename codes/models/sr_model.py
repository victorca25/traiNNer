from __future__ import absolute_import

import os
import logging
from collections import OrderedDict
import torch
import torch.nn as nn

import models.networks as networks
from .base_model import BaseModel
from . import losses
from dataops.common import extract_patches_2d, recompose_tensor

logger = logging.getLogger('base')


class SRModel(BaseModel):
    """ This class implements a super-resolution or restoration 
    model given paired data, using one generator 'G' and, if
    configured, one discriminator 'D'.
    """
    def __init__(self, opt, step=0):
        super(SRModel, self).__init__(opt)
        train_opt = opt['train']

        # specify the models you want to load/save to the disk.
        # The training/test scripts will call <BaseModel.save_networks>
        # and <BaseModel.load_networks>
        # for training and testing, a generator 'G' is needed
        self.model_names = ['G']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt, step=step).to(self.device)  # G
        if self.is_train:
            self.netG.train()
            opt_G_nets = [self.netG]
            opt_D_nets = []
            if train_opt['gan_weight']:
                self.model_names.append('D')  # add discriminator to the network list
                self.netD = networks.define_D(opt).to(self.device)  # D
                self.netD.train()
                opt_D_nets.append(self.netD)

            # configure AdaTarget
            self.setup_atg()
            if self.atg:
                opt_G_nets.append(self.netLoc)
        self.load()  # load G, D and other networks if needed

        self.outm = None

        # define losses, optimizer, scheduler and other components
        if self.is_train:
            # setup network cap
            # define if the generator will have a final
            # capping mechanism in the output
            self.outm = train_opt.get('finalcap', None)

            # setup batch augmentations
            self.setup_batchaug()

            # setup frequency separation
            self.setup_fs()

            # initialize losses
            # generator losses:
            self.generatorlosses = losses.GeneratorLoss(opt, self.device)

            # TODO: show the configured losses names in logger
            # print(self.generatorlosses.loss_list)

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

            # configure SWA
            self.setup_swa()

            # configure virtual batch
            self.setup_virtual_batch()

            # configure AMP
            self.setup_amp()

            # initialize CEM and wrap training generator
            self.setup_cem()

            # setup unshuffle wrapper
            self.setup_unshuffle()

        # print network
        # TODO: pass verbose flag from config file
        self.print_network(verbose=False)

    def feed_data(self, data, need_HR=True):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            data (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # LR images
        self.var_L = data['LR'].to(self.device)  # LQ
        if need_HR:  # train or val
            # HR images
            self.real_H = data['HR'].to(self.device)  # GT
            # discriminator references
            input_ref = data.get('ref', data['HR'])
            self.var_ref = input_ref.to(self.device)

    def feed_data_batch(self, data, need_HR=True):
        # LR
        self.var_L = data

    def forward(self, data=None, CEM_net=None):
        """
        Run forward pass; called by <optimize_parameters> and <test> functions.
        Can be used either with 'data' passed directly or loaded 'self.var_L'.
        CEM_net can be used during inference to pass different CEM wrappers.
        """
        if isinstance(data, torch.Tensor):
            if CEM_net is not None:
                wrapped_netG = CEM_net.WrapArchitecture(self.netG)
                return wrapped_netG(data)
            elif self.unshuffle is not None:
                return self.netG(self.unshuffle(data))
            else:
                return self.netG(data)

        if CEM_net is not None:
            wrapped_netG = CEM_net.WrapArchitecture(self.netG)
            self.fake_H = wrapped_netG(self.var_L)  # G(LR)
        elif self.unshuffle is not None:
            self.fake_H = self.netG(self.unshuffle(self.var_L))
        else:
            if self.outm:
                # if the model has the final activation option
                self.fake_H = self.netG(self.var_L, outm=self.outm)
            else:
                # regular models without the final activation option
                self.fake_H = self.netG(self.var_L)  # G(LR)

    def backward_G(self):
        """Calculate GAN and reconstruction losses for the generator."""
        l_g_total = 0
        with self.cast():
        # casts operations to mixed precision if enabled, else nullcontext
            # calculate regular losses
            loss_results, self.log_dict = self.generatorlosses(
                self.fake_H, self.real_H, self.log_dict, self.f_low)
            l_g_total += sum(loss_results) / self.accumulations

            if self.cri_gan:
                # adversarial loss
                l_g_gan = self.adversarial(
                    self.fake_H, self.var_ref, netD=self.netD,  # (sr, hr)
                    stage='generator', fsfilter=self.f_high)
                self.log_dict['l_g_gan'] = l_g_gan.item()
                l_g_total += l_g_gan / self.accumulations

        # high precision generator losses (can be affected by AMP half precision)
        if self.generatorlosses.precise_loss_list:
            loss_results, self.log_dict = self.generatorlosses(
                self.fake_H, self.real_H, self.log_dict, self.f_low,
                precise=True)
            l_g_total += sum(loss_results) / self.accumulations

        # calculate G gradients
        self.calc_gradients(l_g_total)

    def backward_D(self):
        """Calculate GAN loss for the discriminator."""
        self.log_dict = self.backward_D_Basic(
            self.netD, self.var_ref, self.fake_H, self.log_dict)

    def optimize_parameters(self, step):
        """Calculate losses, gradients, and update network weights;
        called in every training iteration."""
        eff_step = step/self.accumulations

        # G
        # freeze discriminator while generator is trained to prevent BP
        if self.cri_gan:
            self.requires_grad(self.netD, flag=False, net_type='D')

        # switch ATG to train
        if self.atg:
            if eff_step > self.atg_start_iter:
                self.switch_atg(True)
            else:
                self.switch_atg(False)

        # match HR resolution for batchaugment = cutblur
        if self.upsample:
            # TODO: assumes model and process scale == 4x
            self.var_L = nn.functional.interpolate(
                self.var_L, scale_factor=self.upsample, mode="nearest")

        # batch (mixup) augmentations
        if self.mixup:
            self.real_H, self.var_L = self.batchaugment(self.real_H, self.var_L)

        # network forward, generate SR
        with self.cast():
            self.forward()

        # apply mask if batchaug == "cutout"
        if self.mixup:
            self.fake_H, self.real_H = self.batchaugment.apply_mask(self.fake_H, self.real_H)

        # unpad images if using CEM
        if self.CEM:
            self.fake_H = self.CEM_net.HR_unpadder(self.fake_H)
            self.real_H = self.CEM_net.HR_unpadder(self.real_H)
            self.var_ref = self.CEM_net.HR_unpadder(self.var_ref)

        # adatarget
        if self.atg:
            self.fake_H = self.ada_out(
                output=self.fake_H, target=self.real_H,
                loc_model=self.netLoc)

        # calculate and log losses
        # training generator and discriminator
        # update generator (on its own if only training generator
        # or alternatively if training GAN)
        if (self.cri_gan is not True) or (eff_step % self.D_update_ratio == 0
            and eff_step > self.D_init_iters):
            # calculate G backward step and gradients
            self.backward_G()

            # step G optimizer
            self.optimizer_step(step, self.optimizer_G, "G")

        if self.cri_gan:
            # update discriminator
            self.requires_grad(self.netD, flag=True)  # unfreeze all D
            if isinstance(self.feature_loc, int):
                # then freeze up to the selected layers
                for loc in range(self.feature_loc):
                    self.requires_grad(
                        self.netD, False, target_layer=loc, net_type='D')

            # calculate D backward step and gradients
            self.backward_D()

            # step D optimizer
            self.optimizer_step(step, self.optimizer_D, "D")

    def test(self, CEM_net=None):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so
        intermediate steps for backprop are not saved.
        """
        self.netG.eval()
        with torch.no_grad():
            self.forward(CEM_net=CEM_net)
        self.netG.train()

    def test_x8(self, CEM_net=None):
        """Geometric self-ensemble forward function used in test time.
        Will upscale each image 8 times in different rotations/flips
        and average the results into a single image.
        """
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.forward(data=aug, CEM_net=CEM_net) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)
        self.netG.train()

    def test_chop(self, patch_size=200, step=1.0, CEM_net=None):
        """Chop forward function used in test time.
        Converts large images into patches of size (patch_size, patch_size).
        Make sure the patch size is small enough that your GPU memory is
        sufficient. Examples: patch_size = 200 for BlindSR, 64 for ABPN
        """
        batch_size, channels, img_height, img_width = self.var_L.size()
        # if (patch_size * (1.0 - step)) % 1 < 0.5:
        #     patch_size += 1
        patch_size = min(img_height, img_width, patch_size)
        scale = self.opt['scale']

        img_patches = extract_patches_2d(img=self.var_L,
            patch_shape=(patch_size, patch_size), step=[step, step],
            batch_first=True).squeeze(0)

        n_patches = img_patches.size(0)
        highres_patches = []

        self.netG.eval()
        with torch.no_grad():
            for p in range(n_patches):
                lowres_input = img_patches[p:p + 1]
                prediction = self.forward(
                    data=lowres_input, CEM_net=CEM_net)
                highres_patches.append(prediction)

        highres_patches = torch.cat(highres_patches, 0)

        self.fake_H = recompose_tensor(highres_patches, img_height,
            img_width, step=step, scale=scale)
        self.netG.train()

    def get_current_log(self):
        """Return traning losses / errors. train.py will print out
        these on the console, and save them to a file"""
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        """Return visualization images."""
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def get_current_visuals_batch(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach().float().cpu()
        out_dict['SR'] = self.fake_H.detach().float().cpu()
        if need_HR:
            out_dict['HR'] = self.real_H.detach().float().cpu()
        return out_dict
