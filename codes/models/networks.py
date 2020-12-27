import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from options.options import opt_get

#import models.modules.sft_arch as sft_arch
logger = logging.getLogger('base')
####################
# initialize networks
####################

def weights_init_normal(m, bias_fill=0, mean=0.0, std=0.02):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1 and classname != "DiscConvBlock": #ASRResNet's DiscConvBlock causes confusion
    # elif classname.find('Linear') != -1:
    if hasattr(m, 'weight') and isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # init.normal_(m.weight.data, 0.0, std)
        init.normal_(m.weight, mean=mean, std=std)
        if m.bias is not None:
            m.bias.data.fill_(bias_fill)
    # elif classname.find('BatchNorm2d') != -1:
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        init.normal_(m.weight.data, mean=1.0, std=std)  # BN also uses norm
        if hasattr(m, 'bias') and m.bias is not None:
            # init.constant_(m.bias.data, 0.0)
            m.bias.data.fill_(bias_fill)

def weights_init_xavier(m, scale=1, bias_fill=0, **kwargs):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1 and classname != "DiscConvBlock": #ASRResNet's DiscConvBlock causes confusion
    # elif classname.find('Linear') != -1:
    if hasattr(m, 'weight') and isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # init.xavier_normal_(m.weight.data, gain=gain)
        init.xavier_normal_(m.weight, **kwargs)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.fill_(bias_fill)
    # elif classname.find('BatchNorm2d') != -1:
    # elif isinstance(m, _BatchNorm):
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        # init.constant_(m.weight.data, 1.0)
        init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            # init.constant_(m.bias.data, 0.0)
            m.bias.data.fill_(bias_fill)

def weights_init_kaiming(m, scale=1, bias_fill=0, **kwargs):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1 and classname != "DiscConvBlock": #ASRResNet's DiscConvBlock causes confusion
    # elif classname.find('Linear') != -1:
    if hasattr(m, 'weight') and isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        init.kaiming_normal_(m.weight, **kwargs)
        m.weight.data *= scale
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(bias_fill)
    # elif classname.find('BatchNorm2d') != -1:
    # elif isinstance(m, _BatchNorm):
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        # init.constant_(m.weight.data, 1.0)
        init.constant_(m.weight, 1)
        if m.bias is not None:
            # init.constant_(m.bias.data, 0.0)
            m.bias.data.fill_(bias_fill)

def weights_init_orthogonal(m, bias_fill=0, **kwargs):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    # elif classname.find('Linear') != -1:
    if hasattr(m, 'weight') and isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # init.orthogonal_(m.weight.data, gain=1)
        init.orthogonal_(m.weight.data, **kwargs)
        if m.bias is not None:
            m.bias.data.fill_(bias_fill)
    # elif classname.find('BatchNorm2d') != -1:
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        # init.constant_(m.weight.data, 1.0)
        init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            # init.constant_(m.bias.data, 0.0)
            m.bias.data.fill_(bias_fill)

def init_weights(net, init_type='kaiming', scale=1, std=0.02, gain=0.02):
    '''Initialize network weights.
    To initialize a network: 
        1. register CPU/GPU device (with multi-GPU support)
        2. initialize the network weights
    Parameters:
        net (network)        -- the network to be initialized
        init_type (str)      -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        scale (float)        -- scaling factor for kaiming.
        gain (float)         -- scaling factor for xavier.
        std (float)          -- scaling factor for normal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    'kaiming' is used in the ESRGAN paper, 'normal' in the original pix2pix and CycleGAN paper.
    kaiming and xavier might work better for some applications.
    '''
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    if init_type == 'xavier':
        weights_init_xavier_ = functools.partial(weights_init_xavier, gain=gain)
        net.apply(weights_init_xavier_)
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
def define_G(opt, step=0):
    '''Create a generator
    Returns a generator
    The generator is usually initialized with <init_weights>.
    '''
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    init_type = opt_net.get('init_type', 'kaiming')
    init_scale = opt_net.get('init_scale', 0.1)
    
    if opt_net['net_act']: # If set, use a different activation function
        act_type = opt_net['net_act']
    else: # Use networks defaults
        if which_model == 'sr_resnet':
            act_type = 'relu'
        elif which_model == 'RRDB_net':
            act_type = 'leakyrelu'
        elif which_model == 'ppon':
            act_type = 'leakyrelu'
        else:
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
            finalact=opt_net['finalact'], gaussian_noise=opt_net['gaussian'], plus=opt_net['plus'], nr=opt_net['nr'])
    elif which_model == 'MRRDB_net':  # Modified RRDB
        from models.modules.architectures import RRDBNet_arch
        netG = RRDBNet_arch.MRRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], gc=opt_net['gc'])
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
                            nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'],
                            self_attention=opt_net.get('self_attention', False), 
                            double_scpa=opt_net.get('double_scpa', False),
                            ups_inter_mode=opt_net.get('ups_inter_mode', 'nearest'))
    elif which_model == 'sofvsr_net':
        from models.modules.architectures import SOFVSR_arch
        netG = SOFVSR_arch.SOFVSR(scale=opt_net['scale'],n_frames=opt_net.get('n_frames', 3),
                                  channels=opt_net.get('channels', 320), img_ch=opt_net.get('img_ch', 1), 
                                  SR_net=opt_net.get('SR_net', 'sofvsr'), 
                                  sr_nf=opt_net.get('sr_nf', 64), sr_nb=opt_net.get('sr_nb', 23), 
                                  sr_gc=opt_net.get('sr_gc', 32), sr_unf=opt_net.get('sr_unf', 24),
                                  sr_gaussian_noise=opt_net.get('sr_gaussian_noise', 64), 
                                  sr_plus=opt_net.get('sr_plus', False), sr_sa=opt_net.get('sr_sa', True),
                                  sr_upinter_mode=opt_net.get('sr_upinter_mode', 'nearest'))
    elif which_model == 'sr3d_net':
        from models.modules.architectures import SR3DNet_arch
        netG = SR3DNet_arch.SR3DNet(scale=opt['scale'], in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], 
                                    nf=opt_net['nf'])
    elif which_model == 'rife_net':
        from models.modules.architectures import RIFE_arch
        netG = RIFE_arch.RIFE()
    elif which_model == 'SRFlowNet':
        from models.modules.architectures import SRFlowNet_arch
        netG = SRFlowNet_arch.SRFlowNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                nf=opt_net['nf'], nb=opt_net['nb'], scale=opt['scale'], K=opt_net['flow']['K'], opt=opt, step=step)
    elif which_model == 'unet_net':
        from models.modules.architectures import UNet_arch
        netG = UNet_arch.UnetGenerator(input_nc=opt_net['in_nc'], output_nc=opt_net['out_nc'], 
                            num_downs=opt_net['num_downs'], ngf=opt_net['ngf'], 
                            norm_type=opt_net['norm_type'], use_dropout=opt_net['use_dropout'],
                            upsample_mode=opt_net['upsample_mode'])
    elif which_model == 'resnet_net':
        from models.modules.architectures import ResNet_arch
        netG = ResNet_arch.ResnetGenerator(input_nc=opt_net['in_nc'], output_nc=opt_net['out_nc'], 
                            n_blocks=opt_net['n_blocks'], ngf=opt_net['ngf'], 
                            norm_type=opt_net['norm_type'], use_dropout=opt_net['use_dropout'],
                            upsample_mode=opt_net['upsample_mode'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    if opt['is_train'] and which_model != 'MRRDB_net':
        # Note: MRRDB_net initializes the modules during init, no need to initialize again here
        init_weights(netG, init_type=init_type, scale=init_scale)
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG


# Discriminator
def define_D(opt):
    '''Create a discriminator
    Returns a discriminator
    Some of the available types of discriminators:
        vgg_*: discriminators based on a VGG-like network architecture.
            The ones with '_fea' in the name also allow to extract feature 
            maps from the discriminator to use for feature losses. 
        patchgan: PatchGAN classifier described in the original pix2pix paper.
            It can classify whether 70×70 overlapping patches are real or fake.
            Such a patch-level discriminator architecture has fewer parameters
            than a full-image discriminator and can work on arbitrarily-sized images
            in a fully convolutional fashion.
            [n_layers]: With this option, you can specify the number of conv layers 
            in the discriminator with the parameter <n_layers_D> 
            (default=3 as used in basic (PatchGAN).)
        multiscale: can create multiple patchgan discriminators that operate at 
            different scales. Each one at half the scale of the previous. Must 
            coordinate with the LR_size. 
        pixelgan: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
            It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator is usually initialized with <init_weights>.
    '''
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']
    which_model_G = opt_net['which_model_G']
    init_type = opt_net.get('init_type', 'kaiming')
    init_scale = opt_net.get('init_scale', 1)
    
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
            raise ValueError('VGG Discriminator size could not be parsed from the HR patch size. Check that the image patch size is either a power of 2 or 3 multiplied by a power of 2.')
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
    elif which_model == 'discriminator_vgg_fea': #VGG-like discriminator with features extraction
        from models.modules.architectures import discriminators
        try:
            size = int(opt['datasets']['train']['HR_size'])
            netD = discriminators.Discriminator_VGG_fea(size=size, in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
                norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'], \
                convtype=opt_net['convtype'], arch=model_G, spectral_norm=opt_net['spectral_norm'], self_attention=opt_net['self_attention'], \
                max_pool=opt_net['max_pool'], poolsize=opt_net['poolsize'])
        except ValueError:
            raise ValueError('VGG Discriminator size could not be parsed from the HR patch size. Check that the image patch size is either a power of 2 or 3 multiplied by a power of 2.')
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
    init_weights(netD, init_type=init_type, scale=init_scale)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD


def define_F(opt, use_bn=False):
    '''Create a feature extraction network for feature losses
    '''
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


####################
# model coversions and validation for 
# network loading
####################

def normal2mod(state_dict):
    if 'model.0.weight' in state_dict:
        try:
            logger.info('Converting and loading an RRDB model to modified RRDB')
        except:
            print('Converting and loading an RRDB model to modified RRDB')
        crt_net = {}
        items = []

        for k, v in state_dict.items():
            items.append(k)

        # # directly copy
        # for k, v in crt_net.items():
        #     if k in state_dict and state_dict[k].size() == v.size():
        #         crt_net[k] = state_dict[k]
        #         items.remove(k)

        crt_net['conv_first.weight'] = state_dict['model.0.weight']
        crt_net['conv_first.bias'] = state_dict['model.0.bias']

        for k in items.copy():
            if 'RDB' in k:
                ori_k = k.replace('model.1.sub.', 'RRDB_trunk.')
                if '.0.weight' in k:
                    ori_k = ori_k.replace('.0.weight', '.weight')
                elif '.0.bias' in k:
                    ori_k = ori_k.replace('.0.bias', '.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net['trunk_conv.weight'] = state_dict['model.1.sub.23.weight']
        crt_net['trunk_conv.bias'] = state_dict['model.1.sub.23.bias']
        crt_net['upconv1.weight'] = state_dict['model.3.weight']
        crt_net['upconv1.bias'] = state_dict['model.3.bias']
        crt_net['upconv2.weight'] = state_dict['model.6.weight']
        crt_net['upconv2.bias'] = state_dict['model.6.bias']
        crt_net['HRconv.weight'] = state_dict['model.8.weight']
        crt_net['HRconv.bias'] = state_dict['model.8.bias']
        crt_net['conv_last.weight'] = state_dict['model.10.weight']
        crt_net['conv_last.bias'] = state_dict['model.10.bias']
        state_dict = crt_net

    return state_dict

def mod2normal(state_dict):
    if 'conv_first.weight' in state_dict:
        try:
            logger.info('Converting and loading a modified RRDB model to normal RRDB')
        except:
            print('Converting and loading a modified RRDB model to normal RRDB')
        crt_net = {}
        items = []
        for k, v in state_dict.items():
            items.append(k)

        crt_net['model.0.weight'] = state_dict['conv_first.weight']
        crt_net['model.0.bias'] = state_dict['conv_first.bias']

        for k in items.copy():
            if 'RDB' in k:
                ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
                if '.weight' in k:
                    ori_k = ori_k.replace('.weight', '.0.weight')
                elif '.bias' in k:
                    ori_k = ori_k.replace('.bias', '.0.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net['model.1.sub.23.weight'] = state_dict['trunk_conv.weight']
        crt_net['model.1.sub.23.bias'] = state_dict['trunk_conv.bias']
        crt_net['model.3.weight'] = state_dict['upconv1.weight']
        crt_net['model.3.bias'] = state_dict['upconv1.bias']
        crt_net['model.6.weight'] = state_dict['upconv2.weight']
        crt_net['model.6.bias'] = state_dict['upconv2.bias']
        crt_net['model.8.weight'] = state_dict['HRconv.weight']
        crt_net['model.8.bias'] = state_dict['HRconv.bias']
        crt_net['model.10.weight'] = state_dict['conv_last.weight']
        crt_net['model.10.bias'] = state_dict['conv_last.bias']
        state_dict = crt_net
    return state_dict

def model_val(opt_net=None, state_dict=None, model_type=None):
    if model_type == 'G':
        model = opt_get(opt_net, ['network_G', 'which_model_G'])
        if model == 'RRDB_net': # tonormal
            return mod2normal(state_dict)
        elif model == 'MRRDB_net': # tomod
            return normal2mod(state_dict)
        else:
            return state_dict
    elif model_type == 'D':
        # no particular Discriminator validation at the moment
        # model = opt_get(opt_net, ['network_G', 'which_model_D'])
        return state_dict
    else:
        # if model_type not provided, return unchanged 
        # (can do other validations here)
        return state_dict
