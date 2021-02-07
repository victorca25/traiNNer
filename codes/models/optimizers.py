#different optimizers can be used here, can be changed in the options 

import torch


def get_optimizers(cri_gan=None, netD=None, netG=None, train_opt=None, logger=None, optimizers=None):
    # optimizers
    # G
    optim_params = []
    for k, v in netG.named_parameters():  # can optimize for a part of the model
        if v.requires_grad:
            optim_params.append(v)
        else:
            logger.warning('Params [{:s}] will not be optimized.'.format(k))
    
    wd_G = train_opt.get('weight_decay_G', 0)
    optim_G = train_opt.get('optim_G', 'adam')
    
    if optim_G == 'sgd':
        optimizer_G = torch.optim.SGD(optim_params, lr=train_opt['lr_G'], \
            weight_decay=wd_G, momentum=(train_opt['momentum_G']))
    elif optim_G == 'rmsprop':
        optimizer_G = torch.optim.RMSprop(optim_params, lr=train_opt['lr_G'], \
            weight_decay=wd_G, eps=(train_opt['eps_G']))
    else: # default to 'adam'
        optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
            weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
    optimizers.append(optimizer_G)
    
    # D
    if cri_gan:
        wd_D = train_opt.get('weight_decay_D', 0)
        optim_D = train_opt.get('optim_D', 'adam')
        
        if optim_D == 'sgd':
            optimizer_D = torch.optim.SGD(netD.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, momentum=(train_opt['momentum_D']))
        elif optim_D == 'rmsprop':
            optimizer_D = torch.optim.RMSprop(netD.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, eps=(train_opt['eps_D']))
        else: # default to 'adam'
            optimizer_D = torch.optim.Adam(netD.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
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

    wd_G = train_opt.get('weight_decay_G', 0)
    optim_G = train_opt.get('optim_G', 'adam')
    
    if optim_G == 'sgd':
        optimizer_G = torch.optim.SGD(
            [
                {"params": optim_params_other,
                "lr": train_opt['lr_G'],
                "momentum": train_opt['eps_G'],
                "weight_decay": wd_G},
                {"params": optim_params_filter,
                "lr": train_opt.get('lr_filter', train_opt['lr_G']),
                "momentum": train_opt['momentum_G'],
                "weight_decay": wd_G}
            ]
        )
    elif optim_G == 'rmsprop':
        optimizer_G = torch.optim.RMSprop(
            [
                {"params": optim_params_other,
                "lr": train_opt['lr_G'],
                "eps": train_opt['eps_G'],
                "weight_decay": wd_G},
                {"params": optim_params_filter,
                "lr": train_opt.get('lr_filter', train_opt['lr_G']),
                "eps": train_opt['eps_G'],
                "weight_decay": wd_G}
            ]
        )
    else: # default to 'adam'
        optimizer_G = torch.optim.Adam(
            [
                {"params": optim_params_other,
                "lr": train_opt['lr_G'],
                "beta1": train_opt['beta1_G'],
                "beta2": train_opt.get('beta2_G', 0.999),
                "weight_decay": wd_G},
                {"params": optim_params_filter,
                "lr": train_opt.get('lr_filter', train_opt['lr_G']),
                "beta1": train_opt['beta1_G'],
                "beta2": train_opt.get('beta2_G', 0.999),
                "weight_decay": wd_G}
            ]
        )
    optimizers.append(optimizer_G)

    return optimizers, optimizer_G
