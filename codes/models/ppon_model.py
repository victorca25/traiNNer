import logging
import torch

from .sr_model import SRModel

logger = logging.getLogger('base')


class PPONModel(SRModel):
    def __init__(self, opt):
        super(PPONModel, self).__init__(opt)

        if self.is_train:
            # Generator losses:
            # Note: self.generatorlosses and self.precisegeneratorlosses already
            # defined in SRModel, only need to select which losses will be
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

        current_phase = self.phase

        for i, s in enumerate(self.stages_m):
            if current_step >= s:
                phase = i + 2
                self.log_dict = {}  # Clear the loss logs

        phase = f"p{phase}"
        if phase != current_phase:
            self.phase = phase
            self.set_optim_params()
            logger.info(
                f"Switching to phase: {phase}, step: {current_step}")

    def set_optim_params(self):
        """ Freeze layers according to the current phase.
            p1: Only training the Content Layers: CFEM and CRM
            p2: Only training the Structure Layers: SFEM and SRM
            p3: Only training the the Perceptual Layers: PFEM and PRM
        """
        # phase 1
        if self.phase == 'p1':  # content
            for param in self.netG.module.CFEM.parameters():
                param.requires_grad = True
            for param in self.netG.module.CRM.parameters():
                param.requires_grad = True
            for param in self.netG.module.SFEM.parameters():
                param.requires_grad = False
            for param in self.netG.module.SRM.parameters():
                param.requires_grad = False
            for param in self.netG.module.PFEM.parameters():
                param.requires_grad = False
            for param in self.netG.module.PRM.parameters():
                param.requires_grad = False

        # phase 2
        if self.phase == 'p2':  # structure
            for param in self.netG.module.CFEM.parameters():
                param.requires_grad = False
            for param in self.netG.module.CRM.parameters():
                param.requires_grad = False
            for param in self.netG.module.SFEM.parameters():
                param.requires_grad = True
            for param in self.netG.module.SRM.parameters():
                param.requires_grad = True
            for param in self.netG.module.PFEM.parameters():
                param.requires_grad = False
            for param in self.netG.module.PRM.parameters():
                param.requires_grad = False

        # phase 3
        if self.phase == 'p3':  # perceptual
            for param in self.netG.module.CFEM.parameters():
                param.requires_grad = False
            for param in self.netG.module.CRM.parameters():
                param.requires_grad = False
            for param in self.netG.module.SFEM.parameters():
                param.requires_grad = False
            for param in self.netG.module.SRM.parameters():
                param.requires_grad = False
            for param in self.netG.module.PFEM.parameters():
                param.requires_grad = True
            for param in self.netG.module.PRM.parameters():
                param.requires_grad = True

        # optim_params = []
        # for k, v in self.netG.named_parameters():
        #     if v.requires_grad:
        #         optim_params.append(v)
        #     else:
        #         print('Warning: params [{:s}] will not be optimized.'.format(k))

    # override
    def forward(self, data=None, CEM_net=None):
        """
        Run forward pass; called by <optimize_parameters> and <test> functions.
        Can be used either with 'data' passed directly or loaded 'self.var_L'.
        """
        if isinstance(data, torch.Tensor):
            if self.unshuffle is not None:
                PPON_out = self.netG(self.unshuffle(data))
            else:
                PPON_out = self.netG(data)  # G(LR)
            if self.phase == 'p1':
                return PPON_out[0]
            elif self.phase == 'p2':
                return PPON_out[1]
            else:  # 'p3'
                return PPON_out[2]

        if self.unshuffle is not None:
            PPON_out = self.netG(self.unshuffle(self.var_L))
        else:
            PPON_out = self.netG(self.var_L)  # G(LR)
        if self.phase == 'p1':
            self.fake_H = PPON_out[0]
        elif self.phase == 'p2':
            self.fake_H = PPON_out[1]
        else:  # 'p3'
            self.fake_H = PPON_out[2]

    # override
    def optimize_parameters(self, step):
        eff_step = step/self.accumulations

        self.update_stage(eff_step)
        if self.phase == 'p3':
            losses_selector = self.p3_losses
        elif self.phase == 'p2':
            losses_selector = self.p2_losses
        else:  # 'p1'
            losses_selector = self.p1_losses

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
            self.var_L = torch.nn.functional.interpolate(
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

        # adatarget
        if self.atg:
            self.fake_H = self.ada_out(
                output=self.fake_H, target=self.real_H,
                loc_model=self.netLoc)

        # calculate and log losses
        loss_results = []
        # training generator and discriminator
        # update generator (on its own if only training generator
        # or alternatively if training GAN)
        l_g_total = 0
        if (self.cri_gan is not True) or (eff_step % self.D_update_ratio == 0
            and eff_step > self.D_init_iters):
            with self.cast():
                # casts operations to mixed precision if enabled, else nullcontext
                # regular losses
                loss_results, self.log_dict = self.generatorlosses(
                    self.fake_H, self.real_H, self.log_dict, self.f_low,
                    selector=losses_selector)
                l_g_total += sum(loss_results) / self.accumulations

                if self.cri_gan and self.phase == 'p3':
                    # adversarial loss
                    l_g_gan = self.adversarial(
                        self.fake_H, self.var_ref, netD=self.netD,
                        stage='generator', fsfilter=self.f_high)  # (sr, hr)
                    self.log_dict['l_g_gan'] = l_g_gan.item()
                    l_g_total += l_g_gan / self.accumulations

            # high precision generator losses (can be affected by AMP half precision)
            if self.precisegeneratorlosses.loss_list:
                precise_loss_results, self.log_dict = self.precisegeneratorlosses(
                    self.fake_H, self.real_H, self.log_dict, self.f_low,
                    selector=losses_selector)
                l_g_total += sum(precise_loss_results) / self.accumulations

            # calculate G gradients
            self.calc_gradients(l_g_total)

            # step G optimizer
            self.optimizer_step(step, self.optimizer_G, "G")

        if self.cri_gan and self.phase == 'p3':
            # update discriminator
            self.requires_grad(self.netD, flag=True)  # unfreeze all D
            if isinstance(self.feature_loc, int):
                # then freeze up to the selected layers
                for loc in range(self.feature_loc):
                    self.requires_grad(
                        self.netD, False, target_layer=loc, net_type='D')

            # calculate D backward step and gradients
            self.log_dict = self.backward_D_Basic(
                self.netD, self.var_ref, self.fake_H, self.log_dict)

            # step D optimizer
            self.optimizer_step(step, self.optimizer_D, "D")
