import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from options.options import opt_get

logger = logging.getLogger('base')

####################
# initialize networks
####################

def weights_init_normal(m, bias_fill=0, mean=0.0, std=0.02):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.normal_(m.weight.data, mean=mean, std=std)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(bias_fill)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        init.normal_(m.weight.data, mean=1.0, std=std)  # BN also uses norm
        if hasattr(m, 'bias') and m.bias is not None:
            # init.constant_(m.bias.data, 0.0)
            m.bias.data.fill_(bias_fill)

def weights_init_xavier(m, scale=1, bias_fill=0, **kwargs):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        # init.xavier_normal_(m.weight.data, gain=gain)
        init.xavier_normal_(m.weight, **kwargs)
        m.weight.data *= scale
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(bias_fill)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        # init.constant_(m.weight.data, 1.0)
        init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            # init.constant_(m.bias.data, 0.0)
            m.bias.data.fill_(bias_fill)

def weights_init_kaiming(m, scale=1, bias_fill=0, **kwargs):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        # init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        init.kaiming_normal_(m.weight, **kwargs)
        m.weight.data *= scale
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(bias_fill)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        # init.constant_(m.weight.data, 1.0)
        init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            # init.constant_(m.bias.data, 0.0)
            m.bias.data.fill_(bias_fill)

def weights_init_orthogonal(m, bias_fill=0, **kwargs):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        # init.orthogonal_(m.weight.data, gain=1)
        init.orthogonal_(m.weight.data, **kwargs)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(bias_fill)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        # init.constant_(m.weight.data, 1.0)
        init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            # init.constant_(m.bias.data, 0.0)
            m.bias.data.fill_(bias_fill)


def init_weights(net, init_type='kaiming', scale=1, std=0.02, gain=0.02):
    """Initialize network weights.
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
    """
    logger.info(f'Initialization method [{init_type:s}]')
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
        raise NotImplementedError(f'initialization method [{init_type:s}] not implemented')


####################
# define network
####################

def get_network(opt, step=0, selector=None):
    """ Fetch the network and configuration from the options file
    based on the <selector>
    """

    gpu_ids = opt['gpu_ids']
    opt_net = opt[selector]
    kind = opt_net.get('type').lower()

    opt_net_pass = opt_net.copy()
    init_type = opt_net_pass.pop('init_type', 'kaiming')
    init_scale = opt_net_pass.pop('init_scale', 0.1)
    opt_net_pass.pop('strict')
    opt_net_pass.pop('type')

    # generators
    if kind == 'sr_resnet':
        from models.modules.architectures import SRResNet_arch
        net = SRResNet_arch.SRResNet
    elif kind == 'sft_arch':
        from models.modules.architectures import sft_arch
        net = sft_arch.SFT_Net
    elif kind == 'rrdb_net':  # ESRGAN
        from models.modules.architectures import RRDBNet_arch
        net = RRDBNet_arch.RRDBNet
    elif kind == 'mrrdb_net':  # Modified ESRGAN
        from models.modules.architectures import RRDBNet_arch
        net = RRDBNet_arch.MRRDBNet
    elif kind == 'ppon':
        from models.modules.architectures import PPON_arch
        net = PPON_arch.PPON
    elif kind == 'asr_cnn':
        from models.modules.architectures import ASRResNet_arch
        net = ASRResNet_arch.ASRCNN
    elif kind == 'asr_resnet':
        from models.modules.architectures import ASRResNet_arch
        net = ASRResNet_arch.ASRResNet
    elif kind == 'abpn_net':
        from models.modules.architectures import ABPN_arch
        net = ABPN_arch.ABPN_v5
    elif kind == 'pan_net':
        from models.modules.architectures import PAN_arch
        net = PAN_arch.PAN
    elif kind == 'a2n_net':
        from models.modules.architectures import PAN_arch
        net = PAN_arch.AAN
    elif kind == 'sofvsr_net':
        from models.modules.architectures import SOFVSR_arch
        net = SOFVSR_arch.SOFVSR
    elif kind == 'sr3d_net':
        from models.modules.architectures import SR3DNet_arch
        net = SR3DNet_arch.SR3DNet
    elif kind == 'rife_net':
        from models.modules.architectures import RIFE_arch
        net = RIFE_arch.RIFE
    elif kind == 'srflow_net':
        from models.modules.architectures import SRFlowNet_arch
        net = SRFlowNet_arch.SRFlowNet
        n_opt_pass = {}
        for kop, vop in opt_net_pass.items():
            if kop in ['in_nc', 'out_nc', 'nf', 'nb', 'scale', 'K']:
                n_opt_pass[kop] = vop
        n_opt_pass['opt'] = opt
        n_opt_pass['step'] = step
        opt_net_pass = n_opt_pass
    elif kind == 'wbcunet_net':
        from models.modules.architectures import WBCNet_arch
        net = WBCNet_arch.UnetGeneratorWBC
    elif kind == 'unet_net':
        from models.modules.architectures import UNet_arch
        net = UNet_arch.UnetGenerator
    elif kind == 'resnet_net':
        from models.modules.architectures import ResNet_arch
        net = ResNet_arch.ResnetGenerator
    elif kind == 'dvd_net':
        from models.modules.architectures import DVDNet_arch
        net = DVDNet_arch.DVDNet
    elif kind == 'edvr_net':
        from models.modules.architectures import EDVR_arch
        net = EDVR_arch.EDVR
    # discriminators
    elif kind == 'dis_acd':  # sft-gan, Auxiliary Classifier Discriminator
        from models.modules.architectures import sft_arch
        net = sft_arch.ACD_VGG_BN_96
    elif kind == 'discriminator_vgg_96':
        from models.modules.architectures import discriminators
        net = discriminators.Discriminator_VGG_96
    elif kind == 'discriminator_vgg_128_sn':
        from models.modules.architectures import discriminators
        net = discriminators.Discriminator_VGG_128_SN
    elif kind == 'discriminator_vgg_128':
        from models.modules.architectures import discriminators
        net = discriminators.Discriminator_VGG_128
    elif kind in ('discriminator_vgg_192', 'discriminator_192'):
        from models.modules.architectures import discriminators
        net = discriminators.Discriminator_VGG_192
    elif kind in ('discriminator_vgg_256', 'discriminator_256'):
        from models.modules.architectures import discriminators
        net = discriminators.Discriminator_VGG_256
    elif kind == 'discriminator_vgg': # General adaptative case
        from models.modules.architectures import discriminators
        net = discriminators.Discriminator_VGG
    elif kind == 'adiscriminator':
        from models.modules.architectures import ASRResNet_arch
        net = ASRResNet_arch.ADiscriminator
    elif kind == 'adiscriminator_s':
        from models.modules.architectures import ASRResNet_arch
        net = ASRResNet_arch.ADiscriminator_S
    elif kind == 'discriminator_vgg_128_fea': #VGG-like discriminator with features extraction
        from models.modules.architectures import discriminators
        net = discriminators.Discriminator_VGG_128_fea
    elif kind == 'discriminator_vgg_fea': #VGG-like discriminator with features extraction
        from models.modules.architectures import discriminators
        net = discriminators.Discriminator_VGG_fea
    elif kind in ('patchgan', 'nlayerdiscriminator'):
        from models.modules.architectures import discriminators
        net = discriminators.NLayerDiscriminator
    elif kind in ('pixelgan', 'pixeldiscriminator'):
        from models.modules.architectures import discriminators
        net = discriminators.PixelDiscriminator
    elif kind == 'multiscale':
        from models.modules.architectures import discriminators
        net = discriminators.MultiscaleDiscriminator
    elif kind == 'unet':
        from models.modules.architectures import discriminators
        net = discriminators.UNetDiscriminator
    else:
        raise NotImplementedError('Model [{:s}] not recognized'.format(kind))
    """
    elif kind.startswith('discriminator_vgg_'): # User-defined case
        models.modules.architectures import discriminators
        vgg_size = kind[18:]
        try:
            opt_net_pass['size'] = int(vgg_size)
            net = discriminators.Discriminator_VGG
        except ValueError:
            raise ValueError(f'VGG Discriminator size [{vgg_size:s}] could not be parsed.')
    #"""

    net = net(**opt_net_pass)

    if opt['is_train'] and kind != 'mrrdb_net':
        # TODO: Note: MRRDB_net initializes the modules during init, no need to initialize again here for now
        init_weights(net, init_type=init_type, scale=init_scale)

    if gpu_ids:
        assert torch.cuda.is_available()
        net = nn.DataParallel(net)
    return net


def define_network(opt, step=0, net_name='G'):
    """ Format the network to be fetched based on the <net_name>
    *net_name in G, D, G_A, G_B, D_A, D_B, E, F, etc
    """
    selector = f'network_{net_name}'
    return get_network(opt, step, selector)


# Generator
def define_G(opt, step=0, net_name='G'):
    """Create a generator
    Returns a generator
    The generator is usually initialized with <init_weights>.

    Important: it is assumed all the required parameters are parsed with the
    <options> module and can be passed directly to each generator with
    (**opt_net_pass) after removing unneeded keys

    TODO: currently only an interface to maintain compatibility with few
    changes, will not be needed later
    """
    return define_network(opt=opt, step=step, net_name=net_name)


# Discriminator
def define_D(opt, net_name='D'):
    """Create a discriminator
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

    Important: it is assumed all the required parameters are parsed with the
    <options> module and can be passed directly to each discriminator with
    (**opt_net_pass) after removing unneeded keys

    TODO: currently only an interface to maintain compatibility with few
    changes, will not be needed later
    """
    return define_network(opt=opt, net_name=net_name)


def define_F(opt):
    """Create a feature extraction network for feature losses.
    """
    from models.modules.architectures import perceptual

    gpu_ids = opt['gpu_ids']
    z_norm = opt['datasets']['train'].get('znorm', False)

    # TODO: move version validation to defaults.py
    perc_opts = opt['train'].get("perceptual_opt")
    if perc_opts:
        net = perc_opts.get('feature_network', 'vgg19')
        w_l_p = perc_opts.get('perceptual_layers', {'conv5_4': 1})
        w_l_s = perc_opts.get('style_layers', {})
        remove_pooling = perc_opts.get('remove_pooling', False)
        use_input_norm = perc_opts.get('use_input_norm', True)
        requires_grad = perc_opts.get('requires_grad', False)
        change_padding = perc_opts.get('change_padding', False)
        load_path = perc_opts.get('pretrained_path', None)
    else:
        net = opt['train'].get('feature_network', 'vgg19')
        w_l_p = {'conv5_4': 1}
        w_l_s = {}
        remove_pooling = False
        use_input_norm = True
        requires_grad = False
        change_padding = False
        load_path = None

    feature_weight = opt['train'].get('feature_weight', 0)
    style_weight = opt['train'].get('style_weight', 0)

    w_l = w_l_p.copy()
    w_l.update(w_l_s)
    listen_list = list(w_l.keys())

    if 'resnet' in net:
        # ResNet
        netF = perceptual.ResNet101FeatureExtractor(
            use_input_norm=use_input_norm, device=device, z_norm=z_norm)
    else:
        # VGG network (default)
        netF = perceptual.FeatureExtractor(
            listen_list=listen_list, net=net,
            use_input_norm=use_input_norm, z_norm=z_norm,
            requires_grad=requires_grad, remove_pooling=remove_pooling,
            pooling_stride=2, change_padding=change_padding,
            load_path=load_path)

    if gpu_ids:
        assert torch.cuda.is_available()
        netF = nn.DataParallel(netF)

    return netF


# Additional auxiliary networks
def define_ext(opt, net_name=None):
    """Create additional auxiliary networks."""

    if net_name == 'locnet':
        from models.modules.adatarget.atg import LocNet
        if "network_Loc" in opt:
            p_size = opt["network_Loc"]["p_size"]
            s_size = opt["network_Loc"]["s_size"]
        else:
            p_size = 7
            s_size = 9
        net_ext = LocNet(p_size=p_size, s_size=s_size)
        init_type = 'kaiming'
        init_scale = 1
        # Note: original inits BN with: m.weight.data.normal_(1.0, 0.02)

    if opt['is_train']:
        init_weights(
            net_ext, init_type=init_type, scale=init_scale)

    return net_ext


####################
# model conversions and validation for
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
        model = opt_get(opt_net, ['network_G', 'type']).lower()
        if model in ('rrdb_net', 'esrgan'): # tonormal
            return mod2normal(state_dict)
        elif model == 'mrrdb_net' or model == 'srflow_net': # tomod
            return normal2mod(state_dict)
        return state_dict
    elif model_type == 'D':
        # no particular Discriminator validation at the moment
        # model = opt_get(opt_net, ['network_G', 'type']).lower()
        return state_dict
    # if model_type not provided, return unchanged
    # (can do other validations here)
    return state_dict

def cem2normal(state_dict):
    if str(list(state_dict.keys())[0]).startswith('generated_image_model'):
        try:
            logger.info('Unwrapping the Generator model from CEM')
        except:
            print('Unwrapping the Generator model from CEM')
        crt_net = {}
        items = []

        for k, v in state_dict.items():
            items.append(k)

        for k in items.copy():
            if 'generated_image_model.module.' in k:
                ori_k = k.replace('generated_image_model.module.', '')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        state_dict = crt_net

    return state_dict
