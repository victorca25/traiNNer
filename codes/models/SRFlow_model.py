import logging
from collections import OrderedDict

import torch

from .sr_model import SRModel
from .base_model import nullcast
logger = logging.getLogger('base')

from . import optimizers
from . import schedulers

from dataops.batchaug import BatchAug
from options.options import opt_get


class SRFlowModel(SRModel):
    def __init__(self, opt, step):
        super(SRFlowModel, self).__init__(opt, step)
        train_opt = opt['train']

        self.heats = opt_get(opt, ['val', 'heats'], 0.0)
        self.n_sample = opt_get(opt, ['val', 'n_sample'], 1)
        hr_size = opt_get(opt, ['datasets', 'train', 'HR_size'], 160)
        self.lr_size = hr_size // opt['scale']
        self.nll = None

        if self.is_train:
            """
            Initialize losses
            """
            # nll loss
            self.fl_weight = opt_get(self.opt, ['train', 'fl_weight'], 1)

            """
            Prepare optimizer
            """
            self.optDstep = True # no Discriminator being used
            self.optimizers, self.optimizer_G = optimizers.get_optimizers_filter(
                    None, None, self.netG, train_opt, logger, self.optimizers, param_filter='RRDB')

            """
            Prepare schedulers
            """
            self.schedulers = schedulers.get_schedulers(
                optimizers=self.optimizers, schedulers=self.schedulers, train_opt=train_opt)
            
            """
            Set RRDB training state
            """
            train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
            if train_RRDB_delay is not None and step < int(train_RRDB_delay * self.opt['train']['niter']) \
                and self.netG.module.RRDB_training:
                if self.netG.module.set_rrdb_training(False):
                    logger.info('RRDB module frozen, will unfreeze at iter: {}'.format(
                        int(train_RRDB_delay * self.opt['train']['niter'])))


    # TODO: CEM is WIP
    # def forward(self, gt=None, lr=None, z=None, eps_std=None, reverse=False, 
    #             epses=None, reverse_with_grad=False, lr_enc=None, add_gt_noise=False, 
    #             step=None, y_label=None, CEM_net=None)
    #     """
    #     Run forward pass G(LR); called by <optimize_parameters> and <test> functions.
    #     Can be used either with 'data' passed directly or loaded 'self.var_L'. 
    #     CEM_net can be used during inference to pass different CEM wrappers.
    #     """
    #     if isinstance(lr, torch.Tensor):
    #         gt=gt, lr=lr
    #     else:
    #         gt=self.real_H, lr=self.var_L

    #     if CEM_net is not None:
    #         wrapped_netG = CEM_net.WrapArchitecture(self.netG)
    #         net_out = wrapped_netG(gt=gt, lr=lr, z=z, eps_std=eps_std, reverse=reverse, 
    #                             epses=epses, reverse_with_grad=reverse_with_grad, 
    #                             lr_enc=lr_enc, add_gt_noise=add_gt_noise, step=step, 
    #                             y_label=y_label)
    #     else:
    #         net_out = self.netG(gt=gt, lr=lr, z=z, eps_std=eps_std, reverse=reverse, 
    #                         epses=epses, reverse_with_grad=reverse_with_grad, 
    #                         lr_enc=lr_enc, add_gt_noise=add_gt_noise, step=step, 
    #                         y_label=y_label)

    #     if reverse:
    #         sr, logdet = net_out
    #         return sr, logdet
    #     else:
    #         z, nll, y_logits = net_out
    #         return z, nll, y_logits

    def add_optimizer_and_scheduler_RRDB(self, train_opt):
        #Note: this function from the original SRFLow code seems partially broken.
        #Since the RRDB optimizer is being created on init, this is not being used
        # optimizers
        assert len(self.optimizers) == 1, self.optimizers
        assert len(self.optimizer_G.param_groups[1]['params']) == 0, self.optimizer_G.param_groups[1]
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                if '.RRDB.' in k:
                    self.optimizer_G.param_groups[1]['params'].append(v)
        assert len(self.optimizer_G.param_groups[1]['params']) > 0

    def optimize_parameters(self, step):
        # unfreeze RRDB module if train_RRDB_delay is set
        train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
        if train_RRDB_delay is not None and \
                int(step/self.accumulations) > int(train_RRDB_delay * self.opt['train']['niter']) \
                and not self.netG.module.RRDB_training:
            if self.netG.module.set_rrdb_training(True): 
                logger.info('Unfreezing RRDB module.')
                if len(self.optimizers) == 1:
                    # add the RRDB optimizer only if missing
                    self.add_optimizer_and_scheduler_RRDB(self.opt['train'])

        # self.print_rrdb_state()

        self.netG.train()

        """
        Calculate and log losses
        """
        l_g_total = 0
        if self.fl_weight > 0:
            # compute the negative log-likelihood of the output z assuming a unit-norm Gaussian prior
            # with self.cast():  # needs testing, reduced precision could affect results
            z, nll, y_logits = self.netG(gt=self.real_H, lr=self.var_L, reverse=False)
            nll_loss = torch.mean(nll)
            l_g_nll = self.fl_weight * nll_loss
            # # /with self.cast():
            self.log_dict['nll_loss'] = l_g_nll.item()
            l_g_total += l_g_nll / self.accumulations

        if self.generatorlosses.loss_list or self.precisegeneratorlosses.loss_list:
            # batch (mixup) augmentations
            aug = None
            if self.mixup:
                self.real_H, self.var_L, mask, aug = BatchAug(
                    self.real_H, self.var_L,
                    self.mixopts, self.mixprob, self.mixalpha,
                    self.aux_mixprob, self.aux_mixalpha, self.mix_p
                    )

            with self.cast():
                z = self.get_z(heat=0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
                self.fake_H, logdet = self.netG(lr=self.var_L, z=z, eps_std=0, reverse=True, reverse_with_grad=True)
            
            # batch (mixup) augmentations
            # cutout-ed pixels are discarded when calculating loss by masking removed pixels
            if aug == "cutout":
                self.fake_H, self.real_H = self.fake_H*mask, self.real_H*mask
            
            # TODO: CEM is WIP
            # unpad images if using CEM
            # if self.CEM:
            #     self.fake_H = self.CEM_net.HR_unpadder(self.fake_H)
            #     self.real_H = self.CEM_net.HR_unpadder(self.real_H)
            #     self.var_ref = self.CEM_net.HR_unpadder(self.var_ref)
            
            if self.generatorlosses.loss_list:
                with self.cast():
                    # regular losses
                    loss_results, self.log_dict = self.generatorlosses(self.fake_H, self.real_H, self.log_dict, self.f_low)
                    l_g_total += sum(loss_results) / self.accumulations

            if self.precisegeneratorlosses.loss_list:
                # high precision generator losses (can be affected by AMP half precision)
                precise_loss_results, self.log_dict = self.precisegeneratorlosses(
                        self.fake_H, self.real_H, self.log_dict, self.f_low)
                l_g_total += sum(precise_loss_results) / self.accumulations

        if self.amp:
            self.amp_scaler.scale(l_g_total).backward()
        else:
            l_g_total.backward()
        
        # only step and clear gradient if virtual batch has completed
        if (step + 1) % self.accumulations == 0:
            if self.amp:
                self.amp_scaler.step(self.optimizer_G)
                self.amp_scaler.update()
            else:
                self.optimizer_G.step()
            self.optimizer_G.zero_grad()
            self.optGstep = True

    def print_rrdb_state(self):
        for name, param in self.netG.module.named_parameters():
            if "RRDB.conv_first.weight" in name:
                print(name, param.requires_grad, param.data.abs().sum())
        print('params', [len(p['params']) for p in self.optimizer_G.param_groups])

    def test(self, CEM_net=None):
        self.netG.eval()
        self.fake_H = {}
        for heat in self.heats:
            for i in range(self.n_sample):
                z = self.get_z(heat, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
                with torch.no_grad():
                    self.fake_H[(heat, i)], logdet = self.netG(lr=self.var_L, z=z, eps_std=heat, reverse=True)
        with torch.no_grad():
            _, nll, _ = self.netG(gt=self.real_H, lr=self.var_L, reverse=False)
        self.netG.train()
        self.nll = nll.mean().item()

    # TODO
    def get_encode_nll(self, lq, gt):
        self.netG.eval()
        with torch.no_grad():
            _, nll, _ = self.netG(gt=gt, lr=lq, reverse=False)
        self.netG.train()
        return nll.mean().item()

    # TODO: only used for testing code
    def get_sr(self, lq, heat=None, seed=None, z=None, epses=None):
        return self.get_sr_with_z(lq, heat, seed, z, epses)[0]

    # TODO
    def get_encode_z(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, _, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise)
        self.netG.train()
        return z

    # TODO
    def get_encode_z_and_nll(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, nll, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise)
        self.netG.train()
        return z, nll

    # TODO: used by get_sr
    def get_sr_with_z(self, lq, heat=None, seed=None, z=None, epses=None):
        self.netG.eval()
        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) if z is None and epses is None else z

        with torch.no_grad():
            sr, logdet = self.netG(lr=lq, z=z, eps_std=heat, reverse=True, epses=epses)
        self.netG.train()
        return sr, z

    # TODO: used in optimize_parameters and test
    def get_z(self, heat, seed=None, batch_size=1, lr_shape=None):
        if seed: torch.manual_seed(seed)
        if opt_get(self.opt, ['network_G', 'flow', 'split', 'enable']):
            C = self.netG.module.flowUpsamplerNet.C
            H = int(self.opt['scale'] * lr_shape[2] // self.netG.module.flowUpsamplerNet.scaleH)
            W = int(self.opt['scale'] * lr_shape[3] // self.netG.module.flowUpsamplerNet.scaleW)
            size = (batch_size, C, H, W)
            z = torch.normal(mean=0, std=heat, size=size) if heat > 0 else torch.zeros(
                size)
        else:
            L = opt_get(self.opt, ['network_G', 'flow', 'L']) or 3
            fac = 2 ** (L - 3)
            z_size = int(self.lr_size // (2 ** (L - 3)))
            z = torch.normal(mean=0, std=heat, size=(batch_size, 3 * 8 * 8 * fac * fac, z_size, z_size))
        return z

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = True
        for heat in self.heats:
            for i in range(self.n_sample):
                out_dict[('SR', heat, i)] = self.fake_H[(heat, i)].detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.real_H.detach()[0].float().cpu()
        return out_dict
