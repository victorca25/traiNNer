#different optimizers can be used here, can be changed in the options 

import torch
from models.modules.optimizers.adamp import AdamP, SGDP
from models.modules.optimizers.ranger import Ranger
from models.modules.optimizers.madgrad import MADGRAD


def get_optimizers(cri_gan=None, netD=None, netG=None, train_opt=None,
    logger=None, optimizers=None, optim_paramsG=[], optim_paramsD=[]):
    # optimizers
    # G
    if not optim_paramsG:
        optim_paramsG = []
        for k, v in netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                optim_paramsG.append(v)
            else:
                logger.warning('Params [{:s}] will not be optimized.'.format(k))

    lr_G = train_opt.get('lr_G',  1e-4)
    wd_G = train_opt.get('weight_decay_G', 0)
    optim_G = train_opt.get('optim_G', 'adam')
    eps_G = train_opt.get('eps_G', 1e-8)
    nesterov_G = train_opt.get('nesterov_G', False)
    delta_G = train_opt.get('delta_G', 0.1)
    wd_ratio_G = train_opt.get('wd_ratio_G', 0.1)
    dampening_G = train_opt.get('dampening_G', 0)
    beta1_G = train_opt.get('beta1_G', 0.9)
    momentum_G = train_opt.get('momentum_G', beta1_G)
    beta2_G = train_opt.get('beta2_G', 0.999)

    if optim_G == 'sgd':
        optimizer_G = torch.optim.SGD(optim_paramsG, lr=lr_G,
            weight_decay=wd_G, momentum=momentum_G)
    elif optim_G == 'rmsprop':
        optimizer_G = torch.optim.RMSprop(optim_paramsG, lr=lr_G,
            weight_decay=wd_G, eps=eps_G)
    elif optim_G == 'sgdp':
        optimizer_G = SGDP(optim_paramsG, lr=lr_G,
            weight_decay=wd_G, momentum=momentum_G,
            dampening=dampening_G, nesterov=nesterov_G,
            eps=eps_G, delta=delta_G, wd_ratio=wd_ratio_G)
    elif optim_G == 'adamp':
        optimizer_G = AdamP(optim_paramsG, lr=lr_G,
            weight_decay=wd_G, betas=(beta1_G, beta2_G),
            nesterov=nesterov_G, eps=eps_G, 
            delta=delta_G, wd_ratio=wd_ratio_G)
    elif optim_G == 'ranger':
        optimizer_G = Ranger(optim_paramsG, lr=lr_G,
            weight_decay=wd_G, betas=(beta1_G, beta2_G),
            eps=eps_G, alpha=train_opt.get('alpha_G', 0.5),
            k=train_opt.get('k_G', 6),
            N_sma_threshhold=train_opt.get('N_sma_threshhold_G', 5),
            use_gc=train_opt.get('use_gc_G', True),
            gc_conv_only=train_opt.get('gc_conv_only_G', False),
            gc_loc=train_opt.get('gc_loc_G', True))
    elif optim_G == 'madgrad':
        optimizer_G = MADGRAD(optim_paramsG, lr=lr_G,
            eps=eps_G, momentum=momentum_G, weight_decay=wd_G,
            decay_type=train_opt.get('decay_type_G', 'AdamW'))
    else: # default to 'adam'
        optimizer_G = torch.optim.Adam(optim_paramsG, lr=lr_G,
            weight_decay=wd_G, betas=(beta1_G, beta2_G))
    optimizers.append(optimizer_G)

    # D
    if cri_gan:
        if not optim_paramsD:
            optim_paramsD = netD.parameters()

        lr_D = train_opt.get('lr_D',  1e-4)
        wd_D = train_opt.get('weight_decay_D', 0)
        optim_D = train_opt.get('optim_D', 'adam')
        eps_D = train_opt.get('eps_D', 1e-8)
        nesterov_D = train_opt.get('nesterov_D', False)
        delta_D = train_opt.get('delta_D', 0.1)
        wd_ratio_D = train_opt.get('wd_ratio_D', 0.1)
        dampening_D = train_opt.get('dampening_D', 0)
        beta1_D = train_opt.get('beta1_D', 0.9)
        momentum_D = train_opt.get('momentum_D', beta1_G)
        beta2_D = train_opt.get('beta2_D', 0.999)

        if optim_D == 'sgd':
            optimizer_D = torch.optim.SGD(optim_paramsD, lr=lr_D,
                weight_decay=wd_D, momentum=momentum_D)
        elif optim_D == 'rmsprop':
            optimizer_D = torch.optim.RMSprop(optim_paramsD, lr=lr_D,
                weight_decay=wd_D, eps=eps_D)
        elif optim_D == 'sgdp':
            optimizer_D = SGDP(optim_paramsD, lr=lr_D,
                weight_decay=wd_D, momentum=momentum_D,
                dampening=dampening_D, nesterov=nesterov_D,
                eps=eps_D, delta=delta_D, wd_ratio=wd_ratio_D)
        elif optim_D == 'adamp':
            optimizer_D = AdamP(optim_paramsD, lr=lr_D,
                weight_decay=wd_D, betas=(beta1_D, beta2_D),
                nesterov=nesterov_D, eps=eps_D, delta=delta_D, wd_ratio=wd_ratio_D)
        elif optim_D == 'ranger':
            optimizer_D = Ranger(optim_paramsD, lr=lr_D,
                weight_decay=wd_D, betas=(beta1_D, beta2_D),
                eps=eps_D, alpha=train_opt.get('alpha_D', 0.5),
                k=train_opt.get('k_D', 6),
                N_sma_threshhold=train_opt.get('N_sma_threshhold_D', 5),
                use_gc=train_opt.get('use_gc_D', True),
                gc_conv_only=train_opt.get('gc_conv_only_D', False),
                gc_loc=train_opt.get('gc_loc_D', True))
        elif optim_D == 'madgrad':
            optimizer_D = MADGRAD(optim_paramsD, lr=lr_D,
                eps=eps_D, momentum=momentum_D, weight_decay=wd_D,
                decay_type=train_opt.get('decay_type_D', 'AdamW'))
        else: # default to 'adam'
            optimizer_D = torch.optim.Adam(optim_paramsD, lr=lr_D,
                weight_decay=wd_D, betas=(beta1_D, beta2_D))
        optimizers.append(optimizer_D)
        return optimizers, optimizer_G, optimizer_D

    return optimizers, optimizer_G

def get_optimizers_filter(cri_gan=None, netD=None, netG=None, train_opt=None, logger=None, optimizers=None, param_filter=None):
    if param_filter:
        assert isinstance(param_filter, str)
        param_filter = '.{}.'.format(param_filter)  # '.RRDB.'

    # optimizers
    # G
    optim_params_filter = []
    optim_params_other = []
    for k, v in netG.named_parameters():  # can optimize for a part of the model
        # print(k, v.requires_grad)
        if v.requires_grad:
            if param_filter in k:
                optim_params_filter.append(v)
                # print('opt', k)
            else:
                optim_params_other.append(v)
        else:
            logger.warning('Params [{:s}] will not be optimized.'.format(k))
    logger.info('Filtered {} {} params.'.format(len(optim_params_filter), param_filter))

    lr_G = train_opt.get('lr_G',  1e-4)
    lr_filter = train_opt.get('lr_filter', lr_G)
    wd_G = train_opt.get('weight_decay_G', 0)
    optim_G = train_opt.get('optim_G', 'adam')
    eps_G = train_opt.get('eps_G', 1e-8)
    nesterov_G = train_opt.get('nesterov_G', False)
    delta_G = train_opt.get('delta_G', 0.1)
    wd_ratio_G = train_opt.get('wd_ratio_G', 0.1)
    dampening_G = train_opt.get('dampening_G', 0)
    beta1_G = train_opt.get('beta1_G', 0.9)
    momentum_G = train_opt.get('momentum_G', beta1_G)
    beta2_G = train_opt.get('beta2_G', 0.999)

    if optim_G == 'sgd':
        optimizer_G = torch.optim.SGD(
            [
                {"params": optim_params_other,
                "lr": lr_G,
                "momentum": momentum_G,
                "weight_decay": wd_G},
                {"params": optim_params_filter,
                "lr": lr_filter,
                "momentum": momentum_G,
                "weight_decay": wd_G}
            ]
        )
    elif optim_G == 'rmsprop':
        optimizer_G = torch.optim.RMSprop(
            [
                {"params": optim_params_other,
                "lr": lr_G,
                "eps": train_opt['eps_G'],
                "weight_decay": wd_G},
                {"params": optim_params_filter,
                "lr": lr_filter,
                "eps": train_opt['eps_G'],
                "weight_decay": wd_G}
            ]
        )
    elif optim_G == 'sgdp':
        optimizer_G = SGDP(
            [
                {"params": optim_params_other,
                "lr": lr_G,
                "momentum": momentum_G,
                "weight_decay": wd_G,
                "dampening": dampening_G,
                "nesterov": nesterov_G,
                "eps": eps_G, 
                "delta": delta_G,
                "wd_ratio": wd_ratio_G
                },
                {"params": optim_params_filter,
                "lr": lr_filter,
                "momentum": momentum_G,
                "weight_decay": wd_G,
                "dampening": dampening_G,
                "nesterov": nesterov_G,
                "eps": eps_G, 
                "delta": delta_G,
                "wd_ratio": wd_ratio_G}
            ]
        )
    elif optim_G == 'adamp':
        optimizer_G = AdamP(
            [
                {"params": optim_params_other,
                "lr": lr_G,
                "beta1": beta1_G,
                "beta2": beta2_G,
                "weight_decay": wd_G,
                "nesterov": nesterov_G,
                "eps": eps_G,
                "delta": delta_G,
                "wd_ratio": wd_ratio_G
                },
                {"params": optim_params_filter,
                "lr": lr_filter,
                "beta1": beta1_G,
                "beta2": beta2_G,
                "weight_decay": wd_G,
                "nesterov": nesterov_G,
                "eps": eps_G,
                "delta": delta_G,
                "wd_ratio": wd_ratio_G}
            ]
        )
    elif optim_G == 'ranger':
        optimizer_G = Ranger([
                {"params": optim_params_other,
                "lr": lr_G,
                "beta1": beta1_G,
                "beta2": beta2_G,
                "weight_decay": wd_G,
                "eps": eps_G,
                "alpha": train_opt.get('alpha_G', 0.5),
                "k": train_opt.get('k_G', 6),
                "N_sma_threshhold": train_opt.get('N_sma_threshhold_G', 5),
                "use_gc": train_opt.get('use_gc_G', True),
                "gc_conv_only": train_opt.get('gc_conv_only_G', False),
                "gc_loc": train_opt.get('gc_loc_G', True)
                },
                {"params": optim_params_filter,
                "lr": lr_filter,
                "beta1": beta1_G,
                "beta2": beta2_G,
                "weight_decay": wd_G,
                "eps": eps_G,
                "alpha": train_opt.get('alpha_G', 0.5),
                "k": train_opt.get('k_G', 6),
                "N_sma_threshhold": train_opt.get('N_sma_threshhold_G', 5),
                "use_gc": train_opt.get('use_gc_G', True),
                "gc_conv_only": train_opt.get('gc_conv_only_G', False),
                "gc_loc": train_opt.get('gc_loc_G', True) }
            ])
    elif optim_G == 'madgrad':
        optimizer_G = MADGRAD(
            [
                {"params": optim_params_other,
                "lr": lr_G,
                "eps": eps_G,
                "momentum": momentum_G,
                "weight_decay": wd_G,
                "decay_type": train_opt.get('decay_type_G', 'AdamW'),
                },
                {"params": optim_params_filter,
                "lr": lr_filter,
                "eps": eps_G,
                "momentum": momentum_G,
                "weight_decay": wd_G,
                "decay_type": train_opt.get('decay_type_G', 'AdamW'),}
            ]
        )
    else: # default to 'adam'
        optimizer_G = torch.optim.Adam(
            [
                {"params": optim_params_other,
                "lr": lr_G,
                "beta1": beta1_G,
                "beta2": beta2_G,
                "weight_decay": wd_G},
                {"params": optim_params_filter,
                "lr": lr_filter,
                "beta1": beta1_G,
                "beta2": beta2_G,
                "weight_decay": wd_G}
            ]
        )
    optimizers.append(optimizer_G)

    return optimizers, optimizer_G
