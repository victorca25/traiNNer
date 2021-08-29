# different optimizers can be used here, can be changed in the options

import torch
import logging
import itertools
from models.modules.optimizers.adamp import AdamP, SGDP
from models.modules.optimizers.ranger import Ranger
from models.modules.optimizers.madgrad import MADGRAD

logger = logging.getLogger('base')



def get_optim_params(networks, only_requires_grad:bool=True,
    param_filter=None):
    """Get all parameters that will be added to an optimizer.
    Can return all parameters in one or multiple networks or
    filter parameters on two conditions: if requires_grad==True
    or based on a parameter name filter.
    Args:
        networks: a single network or a list of networks
        (torch.nn.Module)
        only_requires_grad: only return the parameters with
            requires_grad==True.
        param_filter: the str filter to select the parameters.
            Will return two lists, one for the filtered results
            and another for the remaining parameters.
    Returns:
        Two lists of networks parameters.
    """

    if isinstance(networks, torch.nn.Module):
        networks = [networks]

    optim_params = []
    optim_params_filter = []

    if only_requires_grad and param_filter:
        # optimize for a part of the model based on parameter name
        for net in networks:
            for k, v in net.named_parameters():
                if v.requires_grad:
                    if param_filter in k:
                        optim_params_filter.append(v)
                    else:
                        optim_params.append(v)
                else:
                    if logger:
                        logger.warning(
                            f'Params [{k:s}] will not be optimized.')
        if logger:
            logger.info(
                f'Filtered {len(optim_params_filter)} {param_filter} params.')
    elif only_requires_grad:
        # only optimize for params with requires_grad = True:
        for net in networks:
            optim_params += filter(
                lambda p: p.requires_grad, net.parameters())
        # for net in networks:
        #     for k, v in net.named_parameters():
        #         if v.requires_grad:
        #             optim_params.append(v)
        #         else:
        #             logger.warning(
        #                 f'Params [{k:s}] will not be optimized.')
    else:
        # if optimizing all params (no filter):
        optim_params = itertools.chain(
            *[net.parameters() for net in networks])

    return optim_params, optim_params_filter


def config_optimizer(train_opt: dict, name: str, net=None,
    optim_params=None):

    if name not in ["G", "D"]:
        raise NotImplementedError(f"Invalid optimizer name: {name}")

    if optim_params is None:
        optim_params = []

    if not optim_params:
        optim_params, _ = get_optim_params(net, True)

    # TODO: remove
    # print("params:", name, len(optim_params))

    optim = train_opt.get('optim_' + name, 'adam')
    lr = train_opt.get('lr_' + name,  1e-4)
    wd = train_opt.get('weight_decay_' + name, 0)
    eps = train_opt.get('eps_' + name, 1e-8)
    nesterov = train_opt.get('nesterov_' + name, False)
    delta = train_opt.get('delta_' + name, 0.1)
    wd_ratio = train_opt.get('wd_ratio_' + name, 0.1)
    beta1 = train_opt.get('beta1_' + name, 0.9)
    momentum = train_opt.get('momentum_' + name, beta1)
    beta2 = train_opt.get('beta2_' + name, 0.999)

    if optim == 'sgd':
        optimizer = torch.optim.SGD(optim_params, lr=lr,
            weight_decay=wd, momentum=momentum)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(optim_params, lr=lr,
            weight_decay=wd, eps=eps)
    elif optim == 'sgdp':
        optimizer = SGDP(optim_params, lr=lr,
            weight_decay=wd, momentum=momentum,
            dampening=train_opt.get('dampening_' + name, 0),
            nesterov=nesterov,
            eps=eps, delta=delta, wd_ratio=wd_ratio)
    elif optim == 'adamp':
        optimizer = AdamP(optim_params, lr=lr,
            weight_decay=wd, betas=(beta1, beta2),
            nesterov=nesterov, eps=eps,
            delta=delta, wd_ratio=wd_ratio)
    elif optim == 'ranger':
        optimizer = Ranger(optim_params, lr=lr,
            weight_decay=wd, betas=(beta1, beta2),
            eps=eps, alpha=train_opt.get('alpha_' + name, 0.5),
            k=train_opt.get('k_' + name, 6),
            N_sma_threshhold=train_opt.get('N_sma_threshhold_' + name, 5),
            use_gc=train_opt.get('use_gc_' + name, True),
            gc_conv_only=train_opt.get('gc_conv_only_' + name, False),
            gc_loc=train_opt.get('gc_loc_' + name, True))
    elif optim == 'madgrad':
        optimizer = MADGRAD(optim_params, lr=lr,
            eps=eps, momentum=momentum, weight_decay=wd,
            decay_type=train_opt.get('decay_type_' + name, 'AdamW'))
    else: # default to 'adam'
        optimizer = torch.optim.Adam(optim_params, lr=lr,
            weight_decay=wd, betas=(beta1, beta2))

    return optimizer


def get_optimizers(cri_gan=None, netD=None, netG=None, train_opt=None,
    logger=None, optimizers=None, optim_paramsG=None, optim_paramsD=None):
    """Interface function to maintain previous functionality, to be
    removed when all models are updated."""

    # G
    # optimizer_G = get_optimizer_G(train_opt, netG, optim_paramsG)
    optimizer_G = config_optimizer(train_opt, "G", netG, optim_paramsG)
    optimizers.append(optimizer_G)

    # D
    if cri_gan:
        # optimizer_D = get_optimizer_G(train_opt, netD, optim_paramsD)
        optimizer_D = config_optimizer(train_opt, "D", netD, optim_paramsD)
        optimizers.append(optimizer_D)
        return optimizers, optimizer_G, optimizer_D

    return optimizers, optimizer_G


def config_optimizer_filter(train_opt: dict, name: str, net=None,
    optim_params=None, param_filter=None):
    """Equivalent to config_optimizer() but for cases where a
    parameter filter is used to create separate parameter groups."""

    if name not in ["G", "D"]:
        raise NotImplementedError(f"Invalid optimizer name: {name}")

    if param_filter:
        assert isinstance(param_filter, str)
        param_filter = '.{}.'.format(param_filter)  # '.RRDB.'

    # optimizers
    if optim_params is None:
        optim_params = []

    if not optim_params:
        optim_params, optim_params_filter = get_optim_params(
            net, True, param_filter)

    # TODO: remove
    # print("Params:", name, len(optim_params))
    # print("Filter:", name, len(optim_params_filter))

    optim = train_opt.get('optim_' + name, 'adam')
    lr = train_opt.get('lr_' + name,  1e-4)
    lr_filter = train_opt.get('lr_filter_', lr)
    wd = train_opt.get('weight_decay_' + name, 0)
    eps = train_opt.get('eps_' + name, 1e-8)
    nesterov = train_opt.get('nesterov_' + name, False)
    delta = train_opt.get('delta_' + name, 0.1)
    wd_ratio = train_opt.get('wd_ratio_' + name, 0.1)
    dampening = train_opt.get('dampening_' + name, 0)
    beta1 = train_opt.get('beta1_' + name, 0.9)
    momentum = train_opt.get('momentum_' + name, beta1)
    beta2 = train_opt.get('beta2_' + name, 0.999)

    # TODO: configure other independent variables for the
    # filtered parameters besides lr_filter

    if optim == 'sgd':
        optimizer = torch.optim.SGD(
            [
                {"params": optim_params,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": wd},
                {"params": optim_params_filter,
                "lr": lr_filter,
                "momentum": momentum,
                "weight_decay": wd}
            ]
        )
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            [
                {"params": optim_params,
                "lr": lr,
                "eps": eps,
                "weight_decay": wd},
                {"params": optim_params_filter,
                "lr": lr_filter,
                "eps": eps,
                "weight_decay": wd}
            ]
        )
    elif optim == 'sgdp':
        optimizer = SGDP(
            [
                {"params": optim_params,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": wd,
                "dampening": dampening,
                "nesterov": nesterov,
                "eps": eps, 
                "delta": delta,
                "wd_ratio": wd_ratio
                },
                {"params": optim_params_filter,
                "lr": lr_filter,
                "momentum": momentum,
                "weight_decay": wd,
                "dampening": dampening,
                "nesterov": nesterov,
                "eps": eps, 
                "delta": delta,
                "wd_ratio": wd_ratio}
            ]
        )
    elif optim == 'adamp':
        optimizer = AdamP(
            [
                {"params": optim_params,
                "lr": lr,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": wd,
                "nesterov": nesterov,
                "eps": eps,
                "delta": delta,
                "wd_ratio": wd_ratio
                },
                {"params": optim_params_filter,
                "lr": lr_filter,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": wd,
                "nesterov": nesterov,
                "eps": eps,
                "delta": delta,
                "wd_ratio": wd_ratio}
            ]
        )
    elif optim == 'ranger':
        optimizer = Ranger([
                {"params": optim_params,
                "lr": lr,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": wd,
                "eps": eps,
                "alpha": train_opt.get('alpha_' + name, 0.5),
                "k": train_opt.get('k_' + name, 6),
                "N_sma_threshhold": train_opt.get('N_sma_threshhold_' + name, 5),
                "use_gc": train_opt.get('use_gc_' + name, True),
                "gc_conv_only": train_opt.get('gc_conv_only_' + name, False),
                "gc_loc": train_opt.get('gc_loc_' + name, True)
                },
                {"params": optim_params_filter,
                "lr": lr_filter,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": wd,
                "eps": eps,
                "alpha": train_opt.get('alpha_' + name, 0.5),
                "k": train_opt.get('k_' + name, 6),
                "N_sma_threshhold": train_opt.get('N_sma_threshhold_' + name, 5),
                "use_gc": train_opt.get('use_gc_' + name, True),
                "gc_conv_only": train_opt.get('gc_conv_only_' + name, False),
                "gc_loc": train_opt.get('gc_loc_' + name, True) }
            ])
    elif optim == 'madgrad':
        optimizer = MADGRAD(
            [
                {"params": optim_params,
                "lr": lr,
                "eps": eps,
                "momentum": momentum,
                "weight_decay": wd,
                "decay_type": train_opt.get('decay_type_' + name, 'AdamW'),
                },
                {"params": optim_params_filter,
                "lr": lr_filter,
                "eps": eps,
                "momentum": momentum,
                "weight_decay": wd,
                "decay_type": train_opt.get('decay_type_' + name, 'AdamW'),}
            ]
        )
    else: # default to 'adam'
        optimizer = torch.optim.Adam(
            [
                {"params": optim_params,
                "lr": lr,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": wd},
                {"params": optim_params_filter,
                "lr": lr_filter,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": wd}
            ]
        )

    return optimizer


def get_optimizers_filter(cri_gan=None, netD=None, netG=None,
    train_opt=None, logger=None, optimizers=None, param_filter=None,
    optim_paramsG=None, optim_paramsD=None):
    """Interface function to maintain previous functionality, to be
    removed when all models are updated."""

    # G
    optimizer_G = config_optimizer_filter(
        train_opt, "G", netG, optim_paramsG, param_filter)
    optimizers.append(optimizer_G)

    return optimizers, optimizer_G
