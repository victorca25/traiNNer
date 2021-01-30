from __future__ import absolute_import

import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn

import models.networks as networks
from .base_model import BaseModel, nullcast

logger = logging.getLogger('base')

from . import losses
from . import optimizers
from . import schedulers
from . import swa

from dataops.batchaug import BatchAug
from dataops.filters import FilterHigh, FilterLow #, FilterX
from dataops.common import extract_patches_2d, recompose_tensor
from models.modules.architectures.CEM import CEMnet

load_amp = (hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"))
if load_amp:
    from torch.cuda.amp import autocast, GradScaler
    logger.info('AMP library available')
else:
    logger.info('AMP library not available')


class SRRaGANModel(BaseModel):
    def __init__(self, opt):
        super(SRRaGANModel, self).__init__(opt)
        train_opt = opt['train']

        # set if data should be normalized (-1,1) or not (0,1)
        if self.is_train:
            z_norm = opt['datasets']['train'].get('znorm', False)
        
        # specify the models you want to load/save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # for training and testing, a generator 'G' is needed 
        self.model_names = ['G']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netG.train()
            if train_opt['gan_weight']:
                self.model_names.append('D') # add discriminator to the network list
                self.netD = networks.define_D(opt).to(self.device)  # D
                self.netD.train()
        self.load()  # load G and D if needed

        self.outm = None

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
            self.mixup = train_opt.get('mixup', None)
            if self.mixup: 
                #TODO: cutblur and cutout need model to be modified so LR and HR have the same dimensions (1x)
                self.mixopts = train_opt.get('mixopts', ["blend", "rgb", "mixup", "cutmix", "cutmixup"]) #, "cutout", "cutblur"]
                self.mixprob = train_opt.get('mixprob', [1.0, 1.0, 1.0, 1.0, 1.0]) #, 1.0, 1.0]
                self.mixalpha = train_opt.get('mixalpha', [0.6, 1.0, 1.2, 0.7, 0.7]) #, 0.001, 0.7]
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
            Configure SWA
            """
            #https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
            self.swa = opt.get('use_swa', False)
            if self.swa:
                self.swa_start_iter = train_opt.get('swa_start_iter', 0)
                # self.swa_start_epoch = train_opt.get('swa_start_epoch', None)
                swa_lr = train_opt.get('swa_lr', 0.0001)
                swa_anneal_epochs = train_opt.get('swa_anneal_epochs', 10)
                swa_anneal_strategy = train_opt.get('swa_anneal_strategy', 'cos')
                #TODO: Note: This could be done in resume_training() instead, to prevent creating
                # the swa scheduler and model before they are needed
                self.swa_scheduler, self.swa_model = swa.get_swa(
                        self.optimizer_G, self.netG, swa_lr, swa_anneal_epochs, swa_anneal_strategy)
                self.load_swa() #load swa from resume state
                logger.info('SWA enabled. Starting on iter: {}, lr: {}'.format(self.swa_start_iter, swa_lr))

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
                self.amp_scaler =  GradScaler()
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
                    #TODO: TMP, for now only tested with the vgg-like or patchgan discriminators
                    if "discriminator_vgg" in disc or "patchgan" in disc:
                        self.feature_loc = loc
                        logger.info('FreezeD enabled')

            """
            Initialize CEM and wrap training generator 
            """
            self.CEM = opt.get('use_cem', None)
            if self.CEM:
                CEM_conf = CEMnet.Get_CEM_Conf(opt['scale'])
                CEM_conf.sigmoid_range_limit = bool(opt['network_G'].get('sigmoid_range_limit', 0))
                if CEM_conf.sigmoid_range_limit:
                    CEM_conf.input_range = [-1,1] if z_norm else [0,1]
                kernel = None  # note: could pass a kernel here, but None will use default cubic kernel
                self.CEM_net = CEMnet.CEMnet(CEM_conf, upscale_kernel=kernel)
                self.CEM_net.WrapArchitecture(only_padders=True)
                self.netG = self.CEM_net.WrapArchitecture(self.netG, training_patch_size=opt['datasets']['train']['HR_size'])
                logger.info('CEM enabled')

        # print network
        """ 
        TODO:
        Network summary? Make optional with parameter
            could be an selector between traditional print_network() and summary()
        """
        self.print_network(verbose=False) #TODO: pass verbose flag from config file

    def feed_data(self, data, need_HR=True):
        # LR images
        self.var_L = data['LR'].to(self.device)
        if need_HR:  # train or val
            # HR images
            self.var_H = data['HR'].to(self.device)
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
            else:
                return self.netG(data)
        
        if CEM_net is not None:
            wrapped_netG = CEM_net.WrapArchitecture(self.netG)
            self.fake_H = wrapped_netG(self.var_L)  # G(LR)
        else:
            if self.outm: #if the model has the final activation option
                self.fake_H = self.netG(self.var_L, outm=self.outm)
            else: #regular models without the final activation option
                self.fake_H = self.netG(self.var_L)  # G(LR)

    def optimize_parameters(self, step):       
        # G
        # freeze discriminator while generator is trained to prevent BP
        if self.cri_gan:
            self.requires_grad(self.netD, flag=False, net_type='D')

        # batch (mixup) augmentations
        aug = None
        if self.mixup:
            self.var_H, self.var_L, mask, aug = BatchAug(
                self.var_H, self.var_L,
                self.mixopts, self.mixprob, self.mixalpha,
                self.aux_mixprob, self.aux_mixalpha, self.mix_p
                )
        
        ### Network forward, generate SR
        with self.cast():
            self.forward()
        #/with self.cast():

        # batch (mixup) augmentations
        # cutout-ed pixels are discarded when calculating loss by masking removed pixels
        if aug == "cutout":
            self.fake_H, self.var_H = self.fake_H*mask, self.var_H*mask
        
        # unpad images if using CEM
        if self.CEM:
            self.fake_H = self.CEM_net.HR_unpadder(self.fake_H)
            self.var_H = self.CEM_net.HR_unpadder(self.var_H)
            self.var_ref = self.CEM_net.HR_unpadder(self.var_ref)

        l_g_total = 0
        """
        Calculate and log losses
        """
        loss_results = []
        # training generator and discriminator
        # update generator (on its own if only training generator or alternatively if training GAN)
        if (self.cri_gan is not True) or (step % self.D_update_ratio == 0 and step > self.D_init_iters):
            with self.cast(): # Casts operations to mixed precision if enabled, else nullcontext
                # regular losses
                loss_results, self.log_dict = self.generatorlosses(self.fake_H, self.var_H, self.log_dict, self.f_low)
                l_g_total += sum(loss_results) / self.accumulations

                if self.cri_gan:
                    # adversarial loss
                    l_g_gan = self.adversarial(
                        self.fake_H, self.var_ref, netD=self.netD, 
                        stage='generator', fsfilter = self.f_high) # (sr, hr)
                    self.log_dict['l_g_gan'] = l_g_gan.item()
                    l_g_total += l_g_gan / self.accumulations

            #/with self.cast():
            # high precision generator losses (can be affected by AMP half precision)
            if self.precisegeneratorlosses.loss_list:
                precise_loss_results, self.log_dict = self.precisegeneratorlosses(
                        self.fake_H, self.var_H, self.log_dict, self.f_low)
                l_g_total += sum(precise_loss_results) / self.accumulations
            
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
            if isinstance(self.feature_loc, int):
                # unfreeze all D
                self.requires_grad(self.netD, flag=True)
                # then freeze up to the selected layers
                for loc in range(self.feature_loc):
                    self.requires_grad(self.netD, False, target_layer=loc, net_type='D')
            else:
                # unfreeze discriminator
                self.requires_grad(self.netD, flag=True)
            
            l_d_total = 0
            
            with self.cast(): # Casts operations to mixed precision if enabled, else nullcontext
                l_d_total, gan_logs = self.adversarial(
                    self.fake_H, self.var_ref, netD=self.netD, 
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

    def test(self, CEM_net=None):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so intermediate steps 
        for backprop are not saved.
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
        Make sure the patch size is small enough that your GPU memory is sufficient.
        Examples: patch_size = 200 for BlindSR, 64 for ABPN
        """
        batch_size, channels, img_height, img_width = self.var_L.size()
        # if (patch_size * (1.0 - step)) % 1 < 0.5:
        #     patch_size += 1
        patch_size = min(img_height, img_width, patch_size)
        scale = self.opt['scale']

        img_patches = extract_patches_2d(img=self.var_L, 
                                        patch_shape=(patch_size, patch_size), 
                                        step=[step, step], 
                                        batch_first=True).squeeze(0)
        
        n_patches = img_patches.size(0)
        highres_patches = []

        self.netG.eval()
        with torch.no_grad():
            for p in range(n_patches):
                lowres_input = img_patches[p:p + 1]
                prediction = self.forward(data=lowres_input, CEM_net=CEM_net)
                highres_patches.append(prediction)

        highres_patches = torch.cat(highres_patches, 0)

        self.fake_H = recompose_tensor(highres_patches, img_height, 
                                        img_width, step=step, scale=scale)
        self.netG.train()
    
    def get_current_log(self):
        """Return traning losses / errors. train.py will print out these on the 
        console, and save them to a file"""
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        """Return visualization images."""
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
