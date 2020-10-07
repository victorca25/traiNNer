import functools
import logging
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
        elif which_model == 'ppon':
            act_type = 'leakyrelu'
    
    if which_model == 'sr_resnet':  # SRResNet
        from models.modules.architectures import SRResNet_arch
        netG = SRResNet_arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type=act_type, mode=opt_net['mode'], upsample_mode='pixelshuffle', \
            convtype=opt_net['convtype'], finalact=opt_net['finalact'])
    elif which_model == 'sft_arch':  # SFT-GAN
        from models.modules.architectures import sft_arch
        netG = sft_arch.SFT_Net()
    elif which_model == 'RRDB_net':  # RRDB
        from models.modules.architectures import RRDBNet_arch
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type=act_type, mode=opt_net['mode'], upsample_mode='upconv', convtype=opt_net['convtype'], \
            finalact=opt_net['finalact'], gaussian_noise=opt_net['gaussian'], plus=opt_net['plus'])
    elif which_model == 'ppon':
        from models.modules.architectures import PPON_arch
        netG = PPON_arch.PPON(in_nc=opt_net['in_nc'], nf=opt_net['nf'], nb=opt_net['nb'], out_nc=opt_net['out_nc'], 
            upscale=opt_net['scale'], act_type=act_type) #(in_nc=3, nf=64, nb=24, out_nc=3)
    elif which_model == 'asr_cnn':
        from models.modules.architectures import ASRResNet_arch
        netG = ASRResNet_arch.ASRCNN(upscale_factor=opt_net['scale'], spectral_norm = True, self_attention = True, max_pool=True, poolsize = 4, finalact='tanh')
    elif which_model == 'asr_resnet':
        from models.modules.architectures import ASRResNet_arch
        netG = ASRResNet_arch.ASRResNet(scale_factor=opt_net['scale'], spectral_norm = True, self_attention = True, max_pool=True, poolsize = 4)
    elif which_model == 'abpn_net':
        from models.modules.architectures import ABPN_arch
        netG = ABPN_arch.ABPN_v5(input_dim=3, dim=32)
        # netG = ABPN_arch.ABPN_v5(input_dim=opt_net['in_nc'], dim=opt_net['out_nc'])
    elif which_model == 'pan_net': #PAN
        from models.modules.architectures import PAN_arch
        netG = PAN_arch.PAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                            nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
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
    which_model_G = opt_net['which_model_G']
    
    if which_model_G == 'ppon':
        model_G = 'PPON'
    else:
        model_G = 'ESRGAN'
    
    if which_model == 'dis_acd':  # sft-gan, Auxiliary Classifier Discriminator
        from models.modules.architectures import sft_arch
        netD = sft_arch.ACD_VGG_BN_96()
    elif which_model == 'discriminator_vgg_96':
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_96(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'], arch=model_G)
    elif which_model == 'discriminator_vgg_128_SN':
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_128_SN()
    elif which_model == 'discriminator_vgg_128':
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_128(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'], arch=model_G)
    elif which_model == 'discriminator_vgg_192' or which_model == 'discriminator_192': #vic in PPON its called Discriminator_192, instead of BasicSR's Discriminator_VGG_192
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_192(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'], arch=model_G)
    elif which_model == 'discriminator_vgg_256' or which_model == 'discriminator_256':
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_256(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'], arch=model_G)
    elif which_model == 'discriminator_vgg': # General adaptative case
        from models.modules.architectures import discriminators
        try:
            size = int(opt['datasets']['train']['HR_size'])
            netD = discriminators.Discriminator_VGG(size=size, in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
                norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'], arch=model_G)
        except ValueError:
            raise ValueError('VGG Discriminator size [{:s}] could not be parsed from the HR patch size. Check that the image patch size is either a power of 2 or 3 multiplied by a power of 2.'.format(vgg_size))
    elif which_model == 'adiscriminator':
        from models.modules.architectures import ASRResNet_arch
        netD = ASRResNet_arch.ADiscriminator(spectral_norm=opt_net['spectral_norm'], self_attention=opt_net['self_attention'], \
            max_pool=opt_net['max_pool'], poolsize=opt_net['poolsize'])
    elif which_model == 'adiscriminator_s':
        from models.modules.architectures import ASRResNet_arch
        netD = ASRResNet_arch.ADiscriminator_S(spectral_norm=opt_net['spectral_norm'], self_attention=opt_net['self_attention'], \
            max_pool=opt_net['max_pool'], poolsize=opt_net['poolsize'] )
    elif which_model == 'discriminator_vgg_128_fea': #VGG-like discriminator with features extraction
        from models.modules.architectures import discriminators
        netD = discriminators.Discriminator_VGG_128_fea(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], \
            convtype=opt_net['convtype'], arch=model_G, spectral_norm=opt_net['spectral_norm'], self_attention=opt_net['self_attention'], \
            max_pool=opt_net['max_pool'], poolsize=opt_net['poolsize'])
    elif which_model == 'patchgan' or which_model == 'NLayerDiscriminator':
        from models.modules.architectures import discriminators
        netD = discriminators.NLayerDiscriminator(input_nc=opt_net['in_nc'], ndf=opt_net['nf'], n_layers=opt_net['nlayer'])
    elif which_model == 'pixelgan' or which_model == 'PixelDiscriminator':
        from models.modules.architectures import discriminators
        netD = discriminators.PixelDiscriminator(input_nc=opt_net['in_nc'], ndf=opt_net['nf'])
    elif which_model == 'multiscale':
        from models.modules.architectures import discriminators
        netD = discriminators.MultiscaleDiscriminator(input_nc=opt_net['in_nc'], ndf=opt_net['nf'], \
            n_layers=opt_net['nlayer'], num_D=opt_net['num_D'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    """
    elif which_model.startswith('discriminator_vgg_'): # User-defined case
        models.modules.architectures import discriminators
        vgg_size = which_model[18:]
        try:
            size = int(vgg_size)
            netD = discriminators.Discriminator_VGG(size=size, in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
                norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], convtype=opt_net['convtype'], arch=model_G)
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
            use_input_norm=True, device=device, z_norm=z_norm)
    
    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF
