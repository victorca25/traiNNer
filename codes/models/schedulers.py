import math
from collections import Counter
from collections import defaultdict
import torch
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler


def get_schedulers(optimizers=None, schedulers=None, train_opt=None):
    ''' Returns a learning rate scheduler for each optimizer 
    provided, according to the configuration in train_opt
    
    Parameters:
        optimizers (list)  -- the list of optimizers, one for each of the networks
        schedulers (list)  -- the list where the schedulers will be appended to, one for each of the networks
        train_opt          -- stores all the experiment flags;
                              train_opt['lr_scheme'] is the name of learning rate policy
    For 'linear', we keep the same learning rate for the first <niter> epochs
        and linearly decay the rate to zero over the next <niter_decay> epochs.
    For other schedulers (step, multistep, plateau, cosine, etc), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    '''
    for optimizer in optimizers:
        if train_opt['lr_scheme'] == 'Linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 - train_opt['fixed_niter']) / float(train_opt['niter_decay'] + 1)
                return max(0, lr_l) # make sure lr is always >= 0

            sched = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        elif train_opt['lr_scheme'] == 'MultiStepLR':
            sched = lr_scheduler.MultiStepLR(optimizer, train_opt['lr_steps'], 
                        train_opt['lr_gamma'])
    
        elif train_opt['lr_scheme'] == 'MultiStepLR_Restart':
            sched = MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                    restarts=train_opt['restarts'],
                                    weights=train_opt['restart_weights'],
                                    gamma=train_opt['lr_gamma'],
                                    clear_state=train_opt['clear_state'])
    
        elif train_opt['lr_scheme'] == 'StepLR':
            sched = lr_scheduler.StepLR(optimizer, step_size=train_opt['lr_step_size'], 
                        gamma=train_opt['lr_gamma'])
    
        elif train_opt['lr_scheme'] == 'StepLR_Restart':
            sched = StepLR_Restart(optimizer, step_sizes=train_opt['lr_step_sizes'],
                                    restarts=train_opt['restarts'],
                                    weights=train_opt['restart_weights'],
                                    gamma=train_opt['lr_gamma'],
                                    clear_state=train_opt['clear_state'])
    
        elif train_opt['lr_scheme'] == 'ProgressiveMultiStepLR':
            sched = ProgressiveMultiStepLR(optimizer, milestones=train_opt['gen_lr_steps'],
                        group_starts=train_opt['progressive_starts'],
                        gamma=train_opt['lr_gamma'])

        elif train_opt['lr_scheme'] == 'CosineAnnealingLR':
            sched = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_opt['T_max'], 
                        eta_min=train_opt['eta_min'])
                    
        elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            sched = CosineAnnealingLR_Restart(optimizer, T_period=train_opt['T_period'], 
                        eta_min=train_opt['eta_min'], restarts=train_opt['restarts'], 
                        weights=train_opt['restart_weights'])
        
        elif train_opt['lr_scheme'] == 'ReduceLROnPlateau':
            #lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
            sched = lr_scheduler.ReduceLROnPlateau(optimizer, mode=train_opt['plateau_mode'], 
                        factor=train_opt['plateau_factor'], threshold=train_opt['plateau_threshold'], 
                        patience=train_opt['plateau_patience'])
        else:
            raise NotImplementedError('Learning rate scheme [{}] not defined or not recognized.'.format(
                        train_opt['lr_scheme']))
        schedulers.append(sched)
    return schedulers







class MultiStepLR_Restart(_LRScheduler):
    def __init__(self, optimizer, milestones, restarts=None, weights=None, gamma=0.1,
                 clear_state=False, last_epoch=-1, force_lr=False):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.clear_state = clear_state
        restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in restarts]
        self.restart_weights = weights if weights else [1]
        self.force_lr = force_lr
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.force_lr:
            return [group['initial_lr'] for group in self.optimizer.param_groups]
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

    def load_state_dict(self, s):
        # Allow this scheduler to use newly appointed milestones partially through a training run.
        milestones_cache = self.milestones
        super(MultiStepLR_Restart, self).load_state_dict(s)
        self.milestones = milestones_cache

class StepLR_Restart(_LRScheduler):
    def __init__(self, optimizer, step_sizes, restarts=None, weights=None, gamma=0.1,
                 clear_state=False, last_epoch=-1):
        self.step_sizes = step_sizes
        self.step_size = self.step_sizes[0]
        self.gamma = gamma
        self.clear_state = clear_state
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        self.weight = 1.0
        self.epoch_offset = 0        
        super(StepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            self.weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            self.step_size = self.step_sizes[self.restarts.index(self.last_epoch) + 1]
            self.epoch_offset = self.last_epoch
            return [base_lr * self.weight for base_lr in self.base_lrs]
        return [base_lr * self.weight * self.gamma ** ((self.last_epoch - self.epoch_offset) // self.step_size)
                for base_lr in self.base_lrs]

class ProgressiveMultiStepLR(_LRScheduler):
    '''
    This scheduler is specifically designed to modulate the learning rate 
    of several different param groups configured by a generator or 
    discriminator that slowly adds new stages one at a time, e.g. like 
    progressive growing of GANs.
    '''
    def __init__(self, optimizer, milestones, group_starts, gamma=0.1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.group_starts = group_starts
        super(ProgressiveMultiStepLR, self).__init__(optimizer)

    def get_lr(self):
        group_lrs = []
        assert len(self.optimizer.param_groups) == len(self.group_starts)
        for group, group_start in zip(self.optimizer.param_groups, self.group_starts):
            if self.last_epoch - group_start not in self.milestones:
                group_lrs.append(group['lr'])
            else:
                group_lrs.append(group['lr'] * self.gamma)
        return group_lrs

class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)


    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]





if __name__ == "__main__":
    optimizer = torch.optim.Adam([torch.zeros(3, 64, 3, 3)], lr=2e-4, weight_decay=0,
                                 betas=(0.9, 0.99))

    ##############################
    # MultiStepLR_Restart
    ##############################

    ## Original
    lr_steps = [200000, 400000, 600000, 800000]
    restarts = None
    restart_weights = None

    ## two
    lr_steps = [100000, 200000, 300000, 400000, 490000, 600000, 700000, 800000, 900000, 990000]
    restarts = [500000]

    restart_weights = [1]

    ## four
    lr_steps = [
        50000, 100000, 150000, 200000, 240000, 300000, 350000, 400000, 450000, 490000, 550000,
        600000, 650000, 700000, 740000, 800000, 850000, 900000, 950000, 990000
    ]

    restarts = [250000, 500000, 750000]
    restart_weights = [1, 1, 1]

    scheduler = MultiStepLR_Restart(optimizer, lr_steps, restarts, restart_weights, gamma=0.5,
                                    clear_state=False)

    ##############################
    # Cosine Annealing Restart
    ##############################

    ## two
    T_period = [500000, 500000]
    restarts = [500000]
    restart_weights = [1]

    ## four
    T_period = [250000, 250000, 250000, 250000]
    restarts = [250000, 500000, 750000]
    restart_weights = [1, 1, 1]

    scheduler = CosineAnnealingLR_Restart(optimizer, T_period, eta_min=1e-7, restarts=restarts,
                                          weights=restart_weights)


    ##############################
    # Draw figure
    ##############################

    N_iter = 1000000
    lr_l = list(range(N_iter))
    for i in range(N_iter):
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr_l[i] = current_lr

    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick
    mpl.style.use('default')
    import seaborn
    seaborn.set(style='whitegrid')
    seaborn.set_context('paper')

    plt.figure(1)
    plt.subplot(111)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title('Title', fontsize=16, color='k')
    plt.plot(list(range(N_iter)), lr_l, linewidth=1.5, label='learning rate scheme')
    legend = plt.legend(loc='upper right', shadow=False)
    ax = plt.gca()
    labels = ax.get_xticks().tolist()
    for k, v in enumerate(labels):
        labels[k] = str(int(v / 1000)) + 'K'
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    ax.set_ylabel('Learning rate')
    ax.set_xlabel('Iteration')
    fig = plt.gcf()
    plt.show()
