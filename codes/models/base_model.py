import os
import torch
import torch.nn as nn


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def update_learning_rate(self, epoch=None):
        for scheduler in self.schedulers:
            if epoch:
                scheduler.step(epoch)
            else:
                scheduler.step()

    def get_current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_step):
        save_filename = '{}_{}.pth'.format(iter_step, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def update_schedulers(self, train_opt):
        '''Update scheduler parameters if they are changed in the JSON configuration'''
        if train_opt['lr_scheme'] == 'StepLR':
            for i, s in enumerate(self.schedulers):
                if self.schedulers[i].step_size != train_opt['lr_step_size'] and train_opt['lr_step_size'] is not None:
                    print("Updating step_size from ",self.schedulers[i].step_size ," to", train_opt['lr_step_size'])
                    self.schedulers[i].step_size = train_opt['lr_step_size']
                #common
                if self.schedulers[i].gamma !=train_opt['lr_gamma'] and train_opt['lr_gamma'] is not None:
                    print("Updating lr_gamma from ",self.schedulers[i].gamma," to", train_opt['lr_gamma'])
                    self.schedulers[i].gamma =train_opt['lr_gamma']
        if train_opt['lr_scheme'] == 'StepLR_Restart':
            for i, s in enumerate(self.schedulers):
                if self.schedulers[i].step_sizes != train_opt['lr_step_sizes'] and train_opt['lr_step_sizes'] is not None:
                    print("Updating step_sizes from ",self.schedulers[i].step_sizes," to", train_opt['lr_step_sizes'])
                    self.schedulers[i].step_sizes = train_opt['lr_step_sizes']
                if self.schedulers[i].restarts != train_opt['restarts'] and train_opt['restarts'] is not None:
                    print("Updating restarts from ",self.schedulers[i].restarts," to", train_opt['restarts'])
                    self.schedulers[i].restarts = train_opt['restarts']
                if self.schedulers[i].restart_weights != train_opt['restart_weights'] and train_opt['restart_weights'] is not None:
                    print("Updating restart_weights from ",self.schedulers[i].restart_weights," to", train_opt['restart_weights'])
                    self.schedulers[i].restart_weights = train_opt['restart_weights']
                if self.schedulers[i].clear_state != train_opt['clear_state'] and train_opt['clear_state'] is not None:
                    print("Updating clear_state from ",self.schedulers[i].clear_state," to", train_opt['clear_state'])
                    self.schedulers[i].clear_state = train_opt['clear_state']
                #common
                if self.schedulers[i].gamma !=train_opt['lr_gamma'] and train_opt['lr_gamma'] is not None:
                    print("Updating lr_gamma from ",self.schedulers[i].gamma," to", train_opt['lr_gamma'])
                    self.schedulers[i].gamma =train_opt['lr_gamma']
        if train_opt['lr_scheme'] == 'MultiStepLR':
            for i, s in enumerate(self.schedulers):
                if self.schedulers[i].milestones != train_opt['lr_steps'] and train_opt['lr_steps'] is not None:
                    if not list(train_opt['lr_steps']) == sorted(train_opt['lr_steps']):
                        raise ValueError('lr_steps should be a list of'
                             ' increasing integers. Got {}', train_opt['lr_steps'])
                    print("Updating lr_steps from ",self.schedulers[i].milestones ," to", train_opt['lr_steps'])
                    self.schedulers[i].milestones = train_opt['lr_steps']
                #common
                if self.schedulers[i].gamma !=train_opt['lr_gamma'] and train_opt['lr_gamma'] is not None:
                    print("Updating lr_gamma from ",self.schedulers[i].gamma," to", train_opt['lr_gamma'])
                    self.schedulers[i].gamma =train_opt['lr_gamma']
        if train_opt['lr_scheme'] == 'MultiStepLR_Restart':
            for i, s in enumerate(self.schedulers):
                if self.schedulers[i].milestones != train_opt['lr_steps'] and train_opt['lr_steps'] is not None:
                    if not list(train_opt['lr_steps']) == sorted(train_opt['lr_steps']):
                        raise ValueError('lr_steps should be a list of'
                             ' increasing integers. Got {}', train_opt['lr_steps'])
                if self.schedulers[i].restarts != train_opt['restarts'] and train_opt['restarts'] is not None:
                    print("Updating restarts from ",self.schedulers[i].restarts," to", train_opt['restarts'])
                    self.schedulers[i].restarts = train_opt['restarts']
                if self.schedulers[i].restart_weights != train_opt['restart_weights'] and train_opt['restart_weights'] is not None:
                    print("Updating restart_weights from ",self.schedulers[i].restart_weights," to", train_opt['restart_weights'])
                    self.schedulers[i].restart_weights = train_opt['restart_weights']
                if self.schedulers[i].clear_state != train_opt['clear_state'] and train_opt['clear_state'] is not None:
                    print("Updating clear_state from ",self.schedulers[i].clear_state," to", train_opt['clear_state'])
                    self.schedulers[i].clear_state = train_opt['clear_state']
                #common
                if self.schedulers[i].gamma !=train_opt['lr_gamma'] and train_opt['lr_gamma'] is not None:
                    print("Updating lr_gamma from ",self.schedulers[i].gamma," to", train_opt['lr_gamma'])
                    self.schedulers[i].gamma =train_opt['lr_gamma']
        
