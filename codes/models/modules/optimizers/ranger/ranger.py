import torch
from torch.optim.optimizer import Optimizer, required
import math


def centralized_gradient(x, use_gc=True, gc_conv_only=False):
    '''credit - https://github.com/Yonghongwei/Gradient-Centralization '''
    if use_gc:
        if gc_conv_only:
            if len(list(x.size())) > 3:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
        else:
            if len(list(x.size())) > 1:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
    return x


class Ranger(Optimizer):
    r"""Implements Ranger deep learning optimizer proposed in
        https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer, a synergistic
        optimizer combining RAdam (Rectified Adam) LookAhead, and GC (gradient
        centralization) into one optimizer. Based on version 2020.9.4 of the code.
    Refs:
        Gradient Centralization: https://arxiv.org/abs/2004.01461v2
        RAdam: https://github.com/LiyuanLucasLiu/RAdam
        Lookahead: https://arxiv.org/abs/1907.08610
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. In comparison to regular
            ADAM, beta1 (momentum) of .95 seems to work better than .90, but
            but can require test on case by case basis (default: (0.95, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-5)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        alpha (float): look ahead param alpha (Default: 0.5)
        k (int): look ahead param k (Default: 6)
        N_sma_threshhold (int): N_sma_threshold of 5 seems better in testing than 4,
            but can require test on case by case basis (Default: 5)
        use_gc (bool): Gradient centralization on or off (default: True)
        gc_conv_only (bool): Wheter gradient centralization is applied
            only to conv layers or conv + fc layers (default: False)
        gc_loc (bool): (default: True)
    """

    def __init__(self, params, lr=1e-3, betas=(.95, 0.999), eps=1e-5,
                 weight_decay=0, alpha=0.5, k=6, N_sma_threshhold=5,
                 use_gc=True, gc_conv_only=False, gc_loc=True):
        if not 0.0 < lr:
            raise ValueError(f'Invalid learning Rate: {lr}')
        if not 0.0 < eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                    alpha=alpha, k=k, N_sma_threshhold=N_sma_threshhold, step_counter=0)
        super(Ranger, self).__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params
        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.gc_loc = gc_loc
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only
        # level of gradient centralization
        #self.gc_gradient_threshold = 3 if gc_conv_only else 1

        # print(
        #     f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        # if (self.use_gc and self.gc_conv_only == False):
        #     print(f"GC applied to both conv and fc layers")
        # elif (self.use_gc and self.gc_conv_only == True):
        #     print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        # print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                # State initialization
                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    #print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                # begin computations. Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GC operation for Conv layers and FC layers
                # if grad.dim() > self.gc_gradient_threshold:
                #    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
                if self.gc_loc:
                    grad = centralized_gradient(grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # compute mean moving avg
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    bias_correction1 = 1 - beta1 ** state['step']
                    beta2_t = beta2 ** state['step']
                    buffered[0] = state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * \
                        state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                            N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / bias_correction1
                    buffered[2] = step_size

                # if group['weight_decay'] != 0:
                #    p_data_fp32.add_(-group['weight_decay']
                #                     * group['lr'], p_data_fp32)

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    G_grad = exp_avg / denom
                else:
                    G_grad = exp_avg

                if group['weight_decay'] != 0:
                    G_grad.add_(p_data_fp32, alpha=group['weight_decay'])

                # GC operation
                if self.gc_loc == False:
                    G_grad = centralized_gradient(G_grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)

                p_data_fp32.add_(G_grad, alpha=-step_size * group['lr'])
                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    # get access to slow param tensor
                    slow_p = state['slow_buffer']
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)

        return loss
