import os
from collections import Counter
from copy import deepcopy
from shutil import copyfile

import torch
import torch.nn as nn

from models.networks import model_val, cem2normal

import logging
logger = logging.getLogger('base')


class nullcast():
    #nullcontext:
    #https://github.com/python/cpython/commit/0784a2e5b174d2dbf7b144d480559e650c5cf64c
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, *excinfo):
        pass


class BaseModel:
    """
    This class is a base class for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call super(NewModel, self).__init__(opt)
        -- <feed_data>:                     unpack data from dataset and apply any preprocessing.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <get_current_visuals>:           -
        -- <get_current_losses>:            -
        -- <print_network>:                 -
        -- <save>:                          -
        -- <load>:                          -
    """

    def __init__(self, opt: dict):
        """
        Initialize the BaseModel class.

        When creating your custom class, you need to implement your own initialization.
        Then, you need to define:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define
                                                    one optimizer for each network. If two networks are
                                                    updated at the same time, you can use itertools.chain
                                                    to group them.
            -- self.schedulers (schedulers list):   a scheduler for each optimizer
            -- other model options

        :param opt: stores all the experiment flags
        """
        self.opt = opt
        if opt['gpu_ids']:
            torch.cuda.current_device()
            torch.cuda.empty_cache()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'

        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        self.swa = None
        self.swa_start_iter = None
        self.metric = 0  # used for learning rate policy 'plateau'

    def feed_data(self, data: dict):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.

        :param data: includes the data itself and any metadata information.
        """
        pass

    def optimize_parameters(self, step):
        """Calculate losses, gradients, and update network weights; called in every training iteration."""
        pass

    def get_current_visuals(self):
        """Return visualization images images for validation, visualization and logging."""
        pass

    def get_current_losses(self):
        """Return training losses. train.py will print out these errors on console, and save them to a file."""
        pass
        
    def print_network(self, verbose: bool = False):
        """Print the total number of parameters in the network and (if verbose) network architecture.
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                s, n = self.get_network_description(net)
                if isinstance(net, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                    net_struc_str = '{} - {}'.format(net.__class__.__name__,
                                                     net.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(net.__class__.__name__)
            
            logger.info(f'Network {name} structure: {net_struc_str}, with parameters: {n:,d}')
            if verbose:
                logger.info(s)

        #TODO: feature network is not being trained, is it necessary to visualize? Maybe just name?
        # maybe show the generatorlosses instead?
        '''
        if self.generatorlosses.cri_fea:  # F, Perceptual Network
            #s, n = self.get_network_description(self.netF)
            s, n = self.get_network_description(self.generatorlosses.netF) #TODO
            #s, n = self.get_network_description(self.generatorlosses.loss_list.netF) #TODO
            if isinstance(self.generatorlosses.netF, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.generatorlosses.netF.__class__.__name__,
                                                self.generatorlosses.netF.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.generatorlosses.netF.__class__.__name__)

            logger.info('Network F structure: {net_struc_str}, with parameters: {n:,d}')
            logger.info(s)
        '''

    def save(self, iter_step, latest=None, loader=None):
        """
        Save all the networks to disk.

        :param iter_step: current iteration; used in the file name '%s_net_%s.pth' % (iter_step, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                self.save_network(net, name, iter_step, latest)

        # self.save_network(self.netG, 'G', iter_step, latest)
        # if self.cri_gan:
        #     self.save_network(self.netD, 'D', iter_step, latest)
        if self.swa:
            # when training with networks that use BN
            # # Update bn statistics for the swa_model only at the end of training
            # if not isinstance(iter_step, int): #TODO: not sure if it should be done only at the end
            self.swa_model = self.swa_model.cpu()
            torch.optim.swa_utils.update_bn(loader, self.swa_model)
            self.swa_model = self.swa_model.cuda()
            # Check swa BN statistics
            # for module in self.swa_model.modules():
            #     if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            #         print(module.running_mean)
            #         print(module.running_var)
            #         print(module.momentum)
            #         break
            self.save_network(self.swa_model, 'swaG', iter_step, latest)

    def load(self):
        """Load all the networks from disk."""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                load_path_opt = f'pretrain_model_{name}'
                load_path = self.opt['path'][load_path_opt]
                load_submodule = self.opt['path'].get('load_submodule', None)  # 'RRDB' -> pretrained RRDB for SRFlow
                if load_path is not None:
                    logger.info(f'Loading pretrained model for {name} [{load_path:s}]')
                    model_type = 'D' if 'D' in name else 'G'
                    strict = True
                    if f'network_{name}' in self.opt:
                        strict = self.opt[f'network_{name}'].get('strict', None)
                    elif f'network_{model_type}' in self.opt:
                        strict = self.opt[f'network_{model_type}'].get('strict', None)
                    self.load_network(load_path, net, strict, model_type=name, submodule=load_submodule)

    def load_swa(self):
        if self.opt['is_train'] and self.opt['use_swa']:
            load_path_swaG = self.opt['path']['pretrain_model_swaG']
            if self.opt['is_train'] and load_path_swaG is not None:
                logger.info(f'Loading pretrained model for SWA G [{load_path_swaG:s}]')
                self.load_network(load_path_swaG, self.swa_model)

        # if self.opt['is_train'] and self.opt['use_swa']:
        #     if self.opt['model'] == 'cyclegan':
        #         #TODO: SWA for cyclegan complete and test
        #         model_keys = ['_A', '_B']
        #     else:
        #         model_keys = ['']

        #     for mkey in model_keys:
        #         swanet = getattr(self, 'swa_model' + mkey)
        #         load_path_swaG = self.opt['path'][f"pretrain_model_swaG{mkey}"]
        #         if self.opt['is_train'] and load_path_swaG is not None:
        #             logger.info(f'Loading pretrained model for SWA G{mkey} [{load_path_swaG:s}]')
        #             self.load_network(load_path_swaG, swanet)

    def _set_lr(self, lr_groups_l: list):
        """
        Set learning rate for warmup.
        :param lr_groups_l: List for lr_groups, one for each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler (for warmup)."""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_step: int = None, warmup_iter: int = -1):
        """
        Update learning rate of all networks.
        :param current_step: Current iteration.
        :param warmup_iter: Warmup iter numbers. -1 for no warmup.
        """
        # SWA scheduler only steps if current_step > swa_start_iter
        if self.swa and current_step and isinstance(self.swa_start_iter, int) and current_step > self.swa_start_iter:
            self.swa_model.update_parameters(self.netG)
            self.swa_scheduler.step()

            # TODO: uncertain, how to deal with the discriminator schedule when the generator enters SWA regime
            # alt 1): D continues with its normal scheduler (current option)
            # alt 2): D also trained using SWA scheduler
            # alt 3): D lr is not modified any longer (Should D be frozen?)
            sched_count = 0
            for scheduler in self.schedulers:
                # first scheduler is G, skip
                if sched_count > 0:
                    if self.opt['train']['lr_scheme'] == 'ReduceLROnPlateau':
                        scheduler.step(self.metric)
                    else:
                        scheduler.step()
                sched_count += 1
        # regular schedulers
        else:
            # print(self.schedulers)
            # print(str(scheduler.__class__) + ": " + str(scheduler.__dict__))
            for scheduler in self.schedulers:
                if self.opt['train']['lr_scheme'] == 'ReduceLROnPlateau':
                    scheduler.step(self.metric)
                else:
                    scheduler.step()
            #### if configured, set up warm up learning rate
            if current_step < warmup_iter:
                # get initial lr for each group
                init_lr_g_l = self._get_init_lr()
                # modify warming-up learning rates
                warm_up_lr_l = []
                for init_lr_g in init_lr_g_l:
                    warm_up_lr_l.append([v / warmup_iter * current_step for v in init_lr_g])
                # set learning rate
                self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self, current_step=None):
        if torch.__version__ >= '1.4.0':
            # Note: SWA only works for torch.__version__ >= '1.6.0'
            if self.swa and current_step and isinstance(self.swa_start_iter,
                                                        int) and current_step > self.swa_start_iter:
                # SWA scheduler lr
                return self.swa_scheduler.get_last_lr()[0]
            # Regular G scheduler lr
            if self.opt['train']['lr_scheme'] == 'ReduceLROnPlateau':
                #TODO: to deal with PyTorch bug: https://github.com/pytorch/pytorch/issues/50715
                return self.schedulers[0]._last_lr[0]
            return self.schedulers[0].get_last_lr()[0]
        else:
            # return self.schedulers[0].get_lr()[0]
            return self.optimizers[0].param_groups[0]['lr']

    def get_network_description(self, network: nn.Module):
        """Get the string and total parameters of the network"""
        if isinstance(network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def requires_grad(self, model, flag: bool = True, target_layer: int = None, net_type: str = None):
        """
        Set requires_grad for all the networks. Use flag=False to avoid unnecessary computations.
        :param model: the network to be updated
        :param flag: whether the networks require gradients or not
        :param target_layer: (optional) for supported networks, can set a specific layer up to which the defined
        flag will be set, for example, to freeze the network up to layer "target_layer"
        :param net_type: (optional) used with target_layer to identify what type of supported network it is.
        """
        # for p in model.parameters():
        #     p.requires_grad = flag
        for name, param in model.named_parameters():
            if target_layer is None:  # every layer
                param.requires_grad = flag
            else:  # elif target_layer in name:  # target layer
                if net_type == 'D':
                    if 'features.' in name:  # vgg-d
                        layer=f'features.{target_layer}.'
                    elif 'conv' in name:  # vgg-fea-d
                        layer=f'conv{target_layer}.'
                    elif 'model.' in name:  # patch-d
                        layer=f'model.{target_layer}.'
                
                if layer in name:
                    # print(name, layer)
                    param.requires_grad = flag

    def save_network(self, network: nn.Module, network_label: str, iter_step: int, latest=False):
        if latest:
            save_filename = f'latest_{network_label}.pth'
        else:
            save_filename = f'{iter_step}_{network_label}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if os.path.exists(save_path):
            prev_path = os.path.join(self.opt['path']['models'], f'previous_{network_label}.pth')
            copyfile(save_path, prev_path)
        if isinstance(network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        
        # unwrap a CEM model if necessary, keep only original parameters
        if str(list(state_dict.keys())[0]).startswith('generated_image_model'):
            state_dict = cem2normal(state_dict)

        try:  # save model in the pre-1.4.0 non-zipped format
            torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)
        except:  # pre 1.4.0, normal torch.save
            torch.save(state_dict, save_path)

    def load_network(self, load_path:str, network:nn.Module,
            strict:bool=True, submodule:str=None,
            model_type:str=None, param_key:str=None):
        """
        Load pretrained model into instantiated network.
        :param load_path: The path of model to be loaded into the network.
        :param network: the network.
        :param strict: Whether if the model will be strictly loaded.
        :param submodule: Specify a submodule of the network to load the model into.
        :param model_type: To do additional validations if needed (either 'G' or 'D').
        :param param_key: The parameter key of loaded model. If set to None, will use the root 'path'.
        """

        # Get bare model, especially under wrapping with DistributedDataParallel or DataParallel.
        if isinstance(network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module
        # network.load_state_dict(torch.load(load_path), strict=strict)

        # load into a specific submodule of the network
        if not (submodule is None or submodule.lower() == 'none'.lower()):
            network = network.__getattr__(submodule)

        # load_net = torch.load(load_path)
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)

        # to allow loading state_dicts
        if 'state_dict' in load_net:
            load_net = load_net['state_dict']

        # load specific keys of the model
        if param_key is not None:
            load_net = load_net[param_key]

        # remove unnecessary 'module.' if needed
        # for k, v in deepcopy(load_net).items():
        #     if k.startswith('module.'):
        #         load_net[k[7:]] = v
        #         load_net.pop(k)

        # validate model type to be loaded in the network can do
        # any additional conversion or modification steps here
        # (requires 'model_type', either 'G' or 'D')
        if model_type:
            load_net = model_val(
                opt_net=self.opt,
                state_dict=load_net,
                model_type=model_type
            )

        # to remove running_mean and running_var from models using
        # InstanceNorm2d trained with PyTorch before 0.4.0:
        # for k in list(load_net.keys()):
        #     if (k.find('running_mean') > 0) or (k.find('running_var') > 0):
        #         del load_net[k]

        network.load_state_dict(load_net, strict=strict)

        # If loading a network with more parameters into a model with less parameters:
        # model = ABPN_v5(input_dim=3, dim=32)
        # model = model.to(device)
        # pretrained_dict = torch.load(model_name, map_location=lambda storage, loc: storage)
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

    def save_training_state(self, epoch: int, iter_step: int, latest=False):
        """Saves training state during training, which will be used for resuming"""
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        if self.opt['is_train'] and self.opt['use_swa']:
            # only save swa_scheduler if needed
            state['swa_scheduler'] = []
            if self.swa and isinstance(self.swa_start_iter, int) and iter_step > self.swa_start_iter:
                state['swa_scheduler'].append(self.swa_scheduler.state_dict())
        if self.opt['is_train'] and self.opt['use_amp'] and state.get('amp_scaler'):
                state['amp_scaler'] = self.amp_scaler.state_dict()

        if latest:
            save_filename = 'latest.state'
        else:
            save_filename = f'{iter_step}.state'
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        if os.path.exists(save_path):
            prev_path = os.path.join(self.opt['path']['training_state'], 'previous.state')
            copyfile(save_path, prev_path)
        torch.save(state, save_path)

    def resume_training(self, resume_state: dict):
        """Resume the optimizers and schedulers for training"""
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong length of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong length of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            if hasattr(self.schedulers[i], 'milestones'):  # for schedulers without milestones attribute
                if isinstance(self.schedulers[i].milestones, Counter) and isinstance(s['milestones'], list):
                    s['milestones'] = Counter(s['milestones'])
            self.schedulers[i].load_state_dict(s)
        if self.opt['is_train'] and self.opt['use_swa']:
            # Only load the swa_scheduler if it exists in the state
            if resume_state.get('swa_scheduler', None):
                resume_swa_scheduler = resume_state['swa_scheduler']
                for i, s in enumerate(resume_swa_scheduler):
                    self.swa_scheduler.load_state_dict(s)
        if self.opt['is_train'] and self.opt['use_amp']:
            if resume_state.get('amp_scaler', None):
                self.amp_scaler.load_state_dict(resume_state['amp_scaler'])

    # TODO: check all these updates
    def update_schedulers(self, train_opt: dict):
        """Update scheduler parameters if they are changed in the configuration"""
        if train_opt['lr_scheme'] == 'StepLR':
            for i, s in enumerate(self.schedulers):
                if self.schedulers[i].step_size != train_opt['lr_step_size'] and train_opt['lr_step_size'] is not None:
                    print("Updating step_size from {} to {}".format(self.schedulers[i].step_size,
                                                                    train_opt['lr_step_size']))
                    self.schedulers[i].step_size = train_opt['lr_step_size']
                # common
                if self.schedulers[i].gamma != train_opt['lr_gamma'] and train_opt['lr_gamma'] is not None:
                    print("Updating lr_gamma from {} to {}".format(self.schedulers[i].gamma, train_opt['lr_gamma']))
                    self.schedulers[i].gamma = train_opt['lr_gamma']
        if train_opt['lr_scheme'] == 'StepLR_Restart':
            for i, s in enumerate(self.schedulers):
                if self.schedulers[i].step_sizes != train_opt['lr_step_sizes'] and train_opt[
                    'lr_step_sizes'] is not None:
                    print("Updating step_sizes from {} to {}".format(self.schedulers[i].step_sizes,
                                                                     train_opt['lr_step_sizes']))
                    self.schedulers[i].step_sizes = train_opt['lr_step_sizes']
                if self.schedulers[i].restarts != train_opt['restarts'] and train_opt['restarts'] is not None:
                    print("Updating restarts from {} to {}".format(self.schedulers[i].restarts, train_opt['restarts']))
                    self.schedulers[i].restarts = train_opt['restarts']
                if self.schedulers[i].restart_weights != train_opt['restart_weights'] and train_opt[
                    'restart_weights'] is not None:
                    print("Updating restart_weights from {} to {}".format(self.schedulers[i].restart_weights,
                                                                          train_opt['restart_weights']))
                    self.schedulers[i].restart_weights = train_opt['restart_weights']
                if self.schedulers[i].clear_state != train_opt['clear_state'] and train_opt['clear_state'] is not None:
                    print("Updating clear_state from {} to {}".format(self.schedulers[i].clear_state,
                                                                      train_opt['clear_state']))
                    self.schedulers[i].clear_state = train_opt['clear_state']
                # common
                if self.schedulers[i].gamma != train_opt['lr_gamma'] and train_opt['lr_gamma'] is not None:
                    print("Updating lr_gamma from {} to {}".format(self.schedulers[i].gamma, train_opt['lr_gamma']))
                    self.schedulers[i].gamma = train_opt['lr_gamma']
        if train_opt['lr_scheme'] == 'MultiStepLR':
            for i, s in enumerate(self.schedulers):
                if list(self.schedulers[i].milestones) != train_opt['lr_steps'] and train_opt['lr_steps'] is not None:
                    if not list(train_opt['lr_steps']) == sorted(train_opt['lr_steps']):
                        raise ValueError('lr_steps should be a list of'
                                         ' increasing integers. Got {}', train_opt['lr_steps'])
                    print("Updating lr_steps from {} to {}".format(list(self.schedulers[i].milestones),
                                                                   train_opt['lr_steps']))
                    if isinstance(self.schedulers[i].milestones, Counter):
                        self.schedulers[i].milestones = Counter(train_opt['lr_steps'])
                    else:
                        self.schedulers[i].milestones = train_opt['lr_steps']
                # common
                if self.schedulers[i].gamma != train_opt['lr_gamma'] and train_opt['lr_gamma'] is not None:
                    print("Updating lr_gamma from {} to {}".format(self.schedulers[i].gamma, train_opt['lr_gamma']))
                    self.schedulers[i].gamma = train_opt['lr_gamma']
        if train_opt['lr_scheme'] == 'MultiStepLR_Restart':
            for i, s in enumerate(self.schedulers):
                if list(self.schedulers[i].milestones) != train_opt['lr_steps'] and train_opt['lr_steps'] is not None:
                    if not list(train_opt['lr_steps']) == sorted(train_opt['lr_steps']):
                        raise ValueError('lr_steps should be a list of'
                                         ' increasing integers. Got {}', train_opt['lr_steps'])
                    print("Updating lr_steps from {} to {}".format(list(self.schedulers[i].milestones),
                                                                   train_opt['lr_steps']))
                    if isinstance(self.schedulers[i].milestones, Counter):
                        self.schedulers[i].milestones = Counter(train_opt['lr_steps'])
                    else:
                        self.schedulers[i].milestones = train_opt['lr_steps']
                if self.schedulers[i].restarts != train_opt['restarts'] and train_opt['restarts'] is not None:
                    print("Updating restarts from {} to {}".format(self.schedulers[i].restarts, train_opt['restarts']))
                    self.schedulers[i].restarts = train_opt['restarts']
                if self.schedulers[i].restart_weights != train_opt['restart_weights'] and train_opt[
                    'restart_weights'] is not None:
                    print("Updating restart_weights from {} to {}".format(self.schedulers[i].restart_weights,
                                                                          train_opt['restart_weights']))
                    self.schedulers[i].restart_weights = train_opt['restart_weights']
                if self.schedulers[i].clear_state != train_opt['clear_state'] and train_opt['clear_state'] is not None:
                    print("Updating clear_state from {} to {}".format(self.schedulers[i].clear_state,
                                                                      train_opt['clear_state']))
                    self.schedulers[i].clear_state = train_opt['clear_state']
                # common
                if self.schedulers[i].gamma != train_opt['lr_gamma'] and train_opt['lr_gamma'] is not None:
                    print("Updating lr_gamma from {} to {}".format(self.schedulers[i].gamma, train_opt['lr_gamma']))
                    self.schedulers[i].gamma = train_opt['lr_gamma']
