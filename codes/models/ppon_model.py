import logging

import torch

from .SRRaGAN_model import SRRaGANModel
from .base_model import nullcast
logger = logging.getLogger('base')


class PPONModel(SRRaGANModel):
    def __init__(self, opt):
        super(PPONModel, self).__init__(opt)

        if self.is_train:
            # Generator losses:
            # Note: self.generatorlosses and self.precisegeneratorlosses already 
            # defined in SRRaGANModel, only need to select which losses will be
            # used in each phase. Discriminator by default only on phase 3. 
            # Content
            self.p1_losses = opt['train'].get('p1_losses', ['pix'])

            # Structure
            self.p2_losses = opt['train'].get('p2_losses', ['pix-multiscale', 'ms-ssim'])

            # Perceptual (as well as self.adversarial)
            self.p3_losses = opt['train'].get('p3_losses', ['contextual'])

            # PPON stages milestones
            self.stages_m = opt['train'].get('ppon_stages', [50000, 75000])

        # Set default phase (for inference)
        self.phase = opt.get('ppon_phase', 3)

    def update_stage(self, current_step: int = None):
        if not current_step:
            current_step = 0
        phase = 1
        
        for i, s in enumerate(self.stages_m):
            if current_step >= s:
                phase = i + 2
                self.log_dict = {}  # Clear the loss logs
        
        self.phase = 'p{}'.format(phase)

    def set_optim_params(self):
        """ Freeze layers according to the current phase.
            p1: Only training the Content Layers: CFEM and CRM
            p2: Only training the Structure Layers: SFEM and SRM
            p3: Only training the the Perceptual Layers: PFEM and PRM
        """
        # phase 2
        if self.phase == 'p2': # structure
            for param in self.netG.module.CFEM.parameters():
                param.requires_grad = False
            for param in self.netG.module.CRM.parameters():
                param.requires_grad = False

        # phase 3
        if self.phase == 'p3': # perceptual
            for param in self.netG.module.CFEM.parameters():
                param.requires_grad = False
            for param in self.netG.module.CRM.parameters():
                param.requires_grad = False
            for param in self.netG.module.SFEM.parameters():
                param.requires_grad = False
            for param in self.netG.module.SRM.parameters():
                param.requires_grad = False
        
        # optim_params = []
        # for k, v in self.netG.named_parameters():
        #     if v.requires_grad:
        #         optim_params.append(v)
        #     else:
        #         print('Warning: params [{:s}] will not be optimized.'.format(k))

    # override
    def forward(self, data=None):
        """Run forward pass; called by <optimize_parameters> and <test> functions."""
        if isinstance(data, torch.Tensor):
            PPON_out = self.netG(data)  # G(LR)
            if self.phase == 'p1':
                return PPON_out[0]
            elif self.phase == 'p2':
                return PPON_out[1]
            else:  # 'p3'
                return PPON_out[2]
        
        PPON_out = self.netG(self.var_L)  # G(LR)
        if self.phase == 'p1':
            self.fake_H = PPON_out[0]
        elif self.phase == 'p2':
            self.fake_H = PPON_out[1]
        else:  # 'p3'
            self.fake_H = PPON_out[2]

    # override
    def optimize_parameters(self, step):
        self.update_stage(step)
        if self.phase == 'p3':
            losses_selector = self.p3_losses
        elif self.phase == 'p2':
            losses_selector = self.p2_losses
        else: # 'p1'
            losses_selector = self.p1_losses

        # G
        # freeze discriminator while generator is trained to prevent BP
        if self.cri_gan:
            self.requires_grad(self.netD, flag=False, net_type='D')
        self.set_optim_params()

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
                loss_results, self.log_dict = self.generatorlosses(self.fake_H, self.var_H, self.log_dict,
                                                                   self.f_low, selector=losses_selector)
                l_g_total += sum(loss_results) / self.accumulations

                if self.cri_gan and self.phase == 'p3':
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
                        self.fake_H, self.var_H, self.log_dict, self.f_low, selector=losses_selector)
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

        if self.cri_gan and self.phase == 'p3':
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
