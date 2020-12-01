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
            logger.warning('Params [{:s}] will not optimize.'.format(k))
    
    wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
    optim_G = train_opt['optim_G'] if train_opt['optim_G'] else 'adam'
    
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
        wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
        optim_D = train_opt['optim_D'] if train_opt['optim_D'] else 'adam'
        
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
