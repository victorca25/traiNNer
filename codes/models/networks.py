import functools
import logging
import inspect
import torch
import torch.nn as nn
from torch.nn import init

#import models.modules.sft_arch as sft_arch
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != "DiscConvBlock": #ASRResNet's DiscConvBlock causes confusion
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


def define_optim(opt, params, network_label, **kwargs):
    optim_type = opt['optim_{}'.format(network_label)]
    if optim_type is None: # default to Adam if not specified
        optim_type = 'Adam'

    # get optimizer
    optim = None
    if hasattr(torch.optim, optim_type):
        optim = getattr(torch.optim, optim_type)
    else:
        try:
            import torch_optimizer
            if hasattr(torch_optimizer, optim_type):
                optim = getattr(torch_optimizer, optim_type)
        except ModuleNotFoundError:
            pass
    if optim is None:
        raise NotImplementedError('Optimizer type [{:s}] is not recognized.'.format(optim_type))

    # read arguments from opt
    args = {}
    parameters = inspect.signature(optim).parameters
    for arg in parameters.keys():
        if arg == 'params':
            args['params'] = params
        elif arg in kwargs:
            args[arg] = kwargs[arg]
        elif arg == 'betas':
            if 'beta1_{}'.format(network_label) in opt or 'beta2_{}'.format(network_label) in opt:
                beta1 = opt['beta1_{}'.format(network_label)] or 0.9
                beta2 = opt['beta2_{}'.format(network_label)] or 0.999
                args[arg] = (beta1, beta2)
        else:
            value = opt['{}_{}'.format(arg, network_label)]
            if value is not None:
                args[arg] = value
            elif arg == 'lr':
                raise TypeError('Optimizer missing required argument: {:s}_{:s}'.format(arg, network_label))
    return optim(**args)


####################
# define network
####################


# Generator
def define_G(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    
    if opt_net['net_act']: # If set, use a different activation function
        act_type = opt_net['net_act']
    else: # Use networks defaults
        if which_model == 'sr_resnet':
            act_type = 'relu'
        elif which_model == 'RRDB_net':
            act_type = 'leakyrelu'
    
    if which_model == 'sr_resnet':  # SRResNet
        from models.modules.architectures import SRResNet_arch
        netG = SRResNet_arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type=act_type, mode=opt_net['mode'], upsample_mode='pixelshuffle', \
            convtype=opt_net['convtype'])
    elif which_model == 'sft_arch':  # SFT-GAN
        from models.modules.architectures import sft_arch
        netG = sft_arch.SFT_Net()
    elif which_model == 'RRDB_net':  # RRDB
        from models.modules.architectures import RRDBNet_arch
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type=act_type, mode=opt_net['mode'], upsample_mode='upconv', convtype=opt_net['convtype'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    if opt['is_train']:
        init_weights(netG, init_type='kaiming', scale=0.1)
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG


# Discriminator
def define_D(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']
    
    if which_model == 'dis_acd':  # sft-gan, Auxiliary Classifier Discriminator
        from models.modules.architectures import sft_arch
        netD = sft_arch.ACD_VGG_BN_96()
    elif which_model == 'discriminator_vgg_96':
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_96(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'])
    elif which_model == 'discriminator_vgg_128_SN':
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_128_SN()
    elif which_model == 'discriminator_vgg_128':
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_128(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'])
    elif which_model == 'discriminator_vgg_192' or which_model == 'discriminator_192':
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_192(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'])
    elif which_model == 'discriminator_vgg_256' or which_model == 'discriminator_256':
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_256(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'])
    elif which_model == 'discriminator_vgg': # General adaptative case
        from models.modules.architectures import discriminators
        try:
            size = int(opt['datasets']['train']['HR_size'])
            netD = discriminators.Discriminator_VGG(size=size, in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
                norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'])
        except ValueError:
            raise ValueError('VGG Discriminator size [{:s}] could not be parsed from the HR patch size. Check that the image patch size is either a power of 2 or 3 multiplied by a power of 2.'.format(vgg_size))
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    """
    elif which_model.startswith('discriminator_vgg_'): # User-defined case
        models.modules.architectures import discriminators
        vgg_size = which_model[18:]
        try:
            size = int(vgg_size)
            netD = discriminators.Discriminator_VGG(size=size, in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
                norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'])
        except ValueError:
            raise ValueError('VGG Discriminator size [{:s}] could not be parsed.'.format(vgg_size))
    #"""
    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD


def define_F(opt, use_bn=False):
    from models.modules.architectures import perceptual
    
    feat_network = 'vgg' #opt['feat_network'] #can be configurable option 
    
    gpu_ids = opt['gpu_ids']
    if opt['datasets']['train']['znorm']:
        z_norm = opt['datasets']['train']['znorm']
    else:
        z_norm = False
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    
    if feat_network == 'resnet': #ResNet
        netF = perceptual.ResNet101FeatureExtractor(use_input_norm=True, device=device)
    else: #VGG network (default)
        netF = perceptual.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
            use_input_norm=True, device=device)
    
    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF
