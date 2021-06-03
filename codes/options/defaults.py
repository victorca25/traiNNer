""" pre-flight check, will fill default configurations when needed """


def get_network_G_config(network_G, scale, crop_size):
    kind_G = None
    if isinstance(network_G, str):
        kind_G = network_G.lower()
        network_G = {}
    elif isinstance(network_G, dict):
        if 'which_model_G' in network_G:
            which_model = 'which_model_G'
        elif 'type' in network_G:
            which_model = 'type'
        kind_G = network_G[which_model].lower()

    full_network_G = {}
    full_network_G['strict'] = network_G.pop('strict', False) # True | False: whether to load the model in strict mode or not

    # SR networks
    if kind_G in ('rrdb_net', 'esrgan', 'evsrgan', 'esrgan-lite'):
        # ESRGAN (or EVSRGAN):
        full_network_G['type'] = "rrdb_net" # RRDB_net (original ESRGAN arch)
        full_network_G['norm_type'] = network_G.pop('norm_type', None)  # "instance" normalization, "batch" normalization or no norm
        full_network_G['mode'] = network_G.pop('mode', "CNA")  # CNA: conv->norm->act, NAC: norm->act->conv
        if kind_G == 'esrgan-lite':
            full_network_G['nf'] = network_G.pop('nf', 32)  # number of filters in the first conv layer
            full_network_G['nb'] = network_G.pop('nb', 12)  # number of RRDB blocks
        else:
            full_network_G['nf'] = network_G.pop('nf', 64)  # number of filters in the first conv layer
            full_network_G['nb'] = network_G.pop('nb', 23)  # number of RRDB blocks
        full_network_G['nr'] = network_G.pop('nr', 3)  #  number of residual layers in each RRDB block
        full_network_G['in_nc'] = network_G.pop('in_nc', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['out_nc'] = network_G.pop('out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        full_network_G['gc'] = network_G.pop('gc', 32)  #
        if kind_G == 'evsrgan':
            full_network_G['convtype'] = network_G.pop('convtype', "Conv3D")  # Conv3D for video
        else:
            full_network_G['convtype'] = network_G.pop('convtype', "Conv2D")  # Conv2D | PartialConv2D | DeformConv2D | Conv3D
        full_network_G['act_type'] = network_G.pop('net_act', None) or network_G.pop('act_type', "leakyrelu")  # swish | leakyrelu
        full_network_G['gaussian_noise'] = network_G.pop('gaussian', True)  # add gaussian noise in the net latent # True | False
        full_network_G['plus'] = network_G.pop('plus', False)  # use the ESRGAN+ modifications # true | false
        full_network_G['finalact'] = network_G.pop('finalact', None)  # Activation function, ie use "tanh" to make outputs fit in [-1, 1] range. Default = None. Coordinate with znorm.
        full_network_G['upscale'] = network_G.pop('scale', scale)
        full_network_G['upsample_mode'] = network_G.pop('upsample_mode', "upconv") # the type of upsample to use
    elif kind_G in ('mrrdb_net', 'mesrgan'):
        # ESRGAN modified arch:
        full_network_G['type'] = "mrrdb_net" # MRRDB_net (modified/"new" arch) | sr_resnet
        full_network_G['in_nc'] = network_G.pop('in_nc', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['out_nc'] = network_G.pop('out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        full_network_G['nf'] = network_G.pop('nf', 64)  # number of filters in the first conv layer
        full_network_G['nb'] = network_G.pop('nb', 23)  # number of RRDB blocks
        full_network_G['gc'] = network_G.pop('gc', 32)  #
    elif 'ppon' in kind_G:
        # PPON:
        full_network_G['type'] = "ppon"  # RRDB_net (original ESRGAN arch)
        full_network_G['in_nc'] = network_G.pop('in_nc', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['out_nc'] = network_G.pop('out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        full_network_G['nf'] = network_G.pop('nf', 64)  # number of filters in the first conv layer
        full_network_G['nb'] = network_G.pop('nb', 23)  # number of RRDB blocks
        full_network_G['upscale'] = network_G.pop('scale', scale)
        full_network_G['act_type'] = network_G.pop('net_act', None) or network_G.pop('act_type', "leakyrelu")  # swish | leakyrelu
    elif 'asr_cnn' in kind_G:
        full_network_G['type'] = "asr_cnn"  # ASRCNN
        full_network_G['upscale_factor'] = network_G.pop('scale', scale)
        full_network_G['spectral_norm'] = network_G.pop('spectral_norm', True)
        full_network_G['self_attention'] = network_G.pop('self_attention', True)
        full_network_G['spectral_norm'] = network_G.pop('spectral_norm', True)
        full_network_G['max_pool'] = network_G.pop('max_pool', True)
        full_network_G['poolsize'] = network_G.pop('poolsize', 4)
        full_network_G['finalact'] = network_G.pop('finalact', 'tanh')
    elif 'asr_resnet' in kind_G:
        full_network_G['type'] = "asr_resnet"  # ASRResNet
        full_network_G['scale_factor'] = network_G.pop('scale', scale)
        full_network_G['spectral_norm'] = network_G.pop('spectral_norm', True)
        full_network_G['self_attention'] = network_G.pop('self_attention', True)
        full_network_G['spectral_norm'] = network_G.pop('spectral_norm', True)
        full_network_G['max_pool'] = network_G.pop('max_pool', True)
        full_network_G['poolsize'] = network_G.pop('poolsize', 4)
    elif kind_G in ('sr_resnet', 'srresnet', 'srgan'):
        # SRGAN:
        full_network_G['type'] = "sr_resnet"  # SRResNet
        full_network_G['in_nc'] = network_G.pop('in_nc', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['out_nc'] = network_G.pop('out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        full_network_G['nf'] = network_G.pop('nf', 64)  # number of filters in the first conv layer
        full_network_G['nb'] = network_G.pop('nb', 16)  # number of RRDB blocks
        full_network_G['upscale'] = network_G.pop('scale', scale)
        full_network_G['norm_type'] = network_G.pop('norm_type', None)  # "instance" normalization, "batch" normalization or no norm
        full_network_G['act_type'] = network_G.pop('net_act', None) or network_G.pop('act_type', "leakyrelu")  # swish | relu | leakyrelu
        full_network_G['mode'] = network_G.pop('mode', "CNA")  # CNA: conv->norm->act, NAC: norm->act->conv
        full_network_G['upsample_mode'] = network_G.pop('upsample_mode', "pixelshuffle") # the type of upsample to use
        full_network_G['convtype'] = network_G.pop('convtype', "Conv2D")  # Conv2D | PartialConv2D | DeformConv2D | Conv3D
        full_network_G['finalact'] = network_G.pop('finalact', None)  # Activation function, ie use "tanh" to make outputs fit in [-1, 1] range. Default = None. Coordinate with znorm.
        full_network_G['res_scale'] = network_G.pop('res_scale', 1)
    #TODO: msrresnet
    elif kind_G in ('sft_arch', 'sft_net'):
        full_network_G['type'] = "sft_arch"  # SFT-GAN
    elif kind_G in ('pan_net', 'pan'):
        # PAN:
        full_network_G['type'] = "pan_net"  # PAN_net
        full_network_G['in_nc'] = network_G.pop('in_nc', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['out_nc'] = network_G.pop('out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        full_network_G['nf'] = network_G.pop('nf', 40)  # number of filters in each conv layer
        full_network_G['unf'] = network_G.pop('unf', 24)  # number of filters during upscale
        full_network_G['nb'] = network_G.pop('nb', 16)  # number of blocks
        full_network_G['scale'] = network_G.pop('scale', scale)
        full_network_G['self_attention'] = network_G.pop('self_attention', False)
        full_network_G['double_scpa'] = network_G.pop('double_scpa', False)
        full_network_G['ups_inter_mode'] = network_G.pop('ups_inter_mode', "nearest")
    elif kind_G in ('abpn_net', 'abpn'):
        full_network_G['type'] = "abpn_net"  # ABPN_net
        full_network_G['input_dim'] = network_G.pop('in_nc', None) or network_G.pop('input_dim', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['dim'] = network_G.pop('dim', 32)
    # SRFlow
    elif kind_G in ('srflow_net', 'srflow'):
        # SRFLOW:
        full_network_G['type'] = "srflow_net"  # SRFlow_net
        full_network_G['in_nc'] = network_G.pop('in_nc', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['out_nc'] = network_G.pop('out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        full_network_G['nf'] = network_G.pop('nf', 64)  # number of filters in the first conv layer
        full_network_G['nb'] = network_G.pop('nb', 23)  # number of RRDB blocks
        full_network_G['gc'] = network_G.pop('gc', 32)  #
        full_network_G['scale'] = network_G.pop('scale', scale)
        full_network_G['upscale'] = full_network_G['scale']
        flow_config = network_G['flow'] if ('flow' in network_G) else {}
        full_network_G['K'] = flow_config.pop('K', 16) if flow_config else 16
        #Note: the network also needs opt and step, below options not used as network parameters, but for GLOW
        full_network_G['train_RRDB'] = network_G.pop('train_RRDB', False)  # if RRDB network will be trained
        full_network_G['train_RRDB_delay'] = network_G.pop('train_RRDB_delay', 0.5)  # at what % of training will RRDB start training
        full_network_G['flow'] = {}
        full_network_G['flow']['K'] = flow_config.pop('K', 16)
        full_network_G['flow']['L'] = flow_config.pop('L', 3)
        full_network_G['flow']['noInitialInj'] = flow_config.pop('noInitialInj', True)
        full_network_G['flow']['coupling'] = flow_config.pop('coupling', "CondAffineSeparatedAndCond")
        full_network_G['flow']['additionalFlowNoAffine'] = flow_config.pop('additionalFlowNoAffine', 2)
        full_network_G['flow']['fea_up0'] = flow_config.pop('fea_up0', True)
        if 'split' in flow_config:
            full_network_G['flow']['split'] = {
                "enable": flow_config['split'].pop('enable', True)}
        else:
            full_network_G['flow']['split'] = {
                "enable": True}
        if 'augmentation' in flow_config:
            full_network_G['flow']['augmentation'] = {
                "noiseQuant": flow_config['augmentation'].pop('noiseQuant', True)}
        else:
            full_network_G['flow']['augmentation'] = {
                "noiseQuant": True}
        if 'stackRRDB' in flow_config:
            full_network_G['flow']['stackRRDB'] = {
                "blocks": flow_config['stackRRDB'].pop('blocks', [ 1, 8, 15, 22 ]),
                "concat": flow_config['stackRRDB'].pop('concat', True)}
        else:
            full_network_G['flow']['stackRRDB'] = {
                "blocks": [ 1, 8, 15, 22 ],
                "concat": True}
    # image to image translation
    elif 'unet' in kind_G:
        #UNET:
        full_network_G['type'] = "unet_net"
        full_network_G['input_nc'] = network_G.pop('in_nc', 3) # # of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['output_nc'] = network_G.pop('out_nc', 3) # # of output image channels: 3 for RGB and 1 for grayscale
        if kind_G == 'unet_128':
            full_network_G['num_downs'] = network_G.pop('num_downs', 7) # for 'unet_128' (for 128x128 input images)
        elif kind_G == 'unet_256':
            full_network_G['num_downs'] = network_G.pop('num_downs', 8) # for 'unet_256' (for 256x256 input images)
        else:
            full_network_G['num_downs'] = network_G.pop('num_downs', 8) #7 for 'unet_128' (for 128x128 input images) | 8 for 'unet_256' (for 256x256 input images)
        # check valid crop size for UNET
        if full_network_G['num_downs'] == 7:
            assert crop_size == 128, f'Invalid crop size {crop_size} for UNET config, must be 128'
        elif full_network_G['num_downs'] == 8:
            assert crop_size == 256, f'Invalid crop size {crop_size} for UNET config, must be 256'
        elif full_network_G['num_downs'] == 9:
            assert crop_size == 512, f'Invalid crop size {crop_size} for UNET config, must be 512'
        full_network_G['ngf'] = network_G.pop('ngf', 64) # # of gen filters in the last conv layer
        full_network_G['norm_type'] = network_G.pop('norm_type', "batch") # "instance" normalization or "batch" normalization
        full_network_G['use_dropout'] = network_G.pop('use_dropout', False) # whether to use dropout or not
        #TODO: add:
        # full_network_G['dropout_prob'] = network_G.pop('dropout_prob', 0.5) # the default dropout probability
        full_network_G['upsample_mode'] = network_G.pop('upsample_mode', "deconv") # deconv | upconv # the type of upsample to use, deconvolution or upsample+convolution
    elif 'resnet' in kind_G and kind_G != 'sr_resnet':
        #RESNET:
        full_network_G['type'] = "resnet_net"
        full_network_G['input_nc'] = network_G.pop('in_nc', 3) # # of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['output_nc'] = network_G.pop('out_nc', 3) # # of output image channels: 3 for RGB and 1 for grayscale
        if kind_G == 'resnet_6blocks':
            full_network_G['n_blocks'] = network_G.pop('n_blocks', 6)  # 6 for resnet_6blocks (with 6 Resnet blocks) and
        elif kind_G == 'resnet_9blocks':
            full_network_G['n_blocks'] = network_G.pop('n_blocks', 9)  # 9 for resnet_9blocks (with 9 Resnet blocks)
        else:
            full_network_G['n_blocks'] = network_G.pop('n_blocks', 9)  # 6 for resnet_6blocks (with 6 Resnet blocks) and 9 for resnet_9blocks (with 9 Resnet blocks)
        full_network_G['ngf'] = network_G.pop('ngf', 64)  # num. of gen filters in the last conv layer
        full_network_G['norm_type'] = network_G.pop('norm_type', "instance") # "instance" normalization or "batch" normalization
        full_network_G['use_dropout'] = network_G.pop('use_dropout', False) # whether to use dropout or not
        #TODO: add:
        # full_network_G['dropout_prob'] = network_G.pop('dropout_prob', 0.5) # the default dropout probability
        full_network_G['upsample_mode'] = network_G.pop('upsample_mode', "deconv") # deconv | upconv # the type of upsample to use, deconvolution or upsample+convolution
        full_network_G['padding_type'] = network_G.pop('padding_type', "reflect")
    # video networks
    elif kind_G in ('sofvsr_net', 'sofvsr'):
        full_network_G['type'] = "sofvsr_net"  # RRDB_net (original ESRGAN arch)
        full_network_G['n_frames'] = network_G.pop('n_frames', 3)  # number of frames the network will use to estimate the central frame (n-1)/2. Must coincide with "num_frames" in the dataset.
        full_network_G['channels'] = network_G.pop('channels', 320)  # feature extraction layer with 320 kernels of size 3 × 3
        full_network_G['scale'] = network_G.pop('scale', scale)
        full_network_G['img_ch'] = network_G.pop('in_nc', 3) or network_G.pop('img_ch', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        # for SR network:
        full_network_G['SR_net'] = network_G.pop('SR_net', "rrdb")  # sofvsr | rrdb | pan
        full_network_G['sr_nf'] = network_G.pop('sr_nf', 64)  # for rrdb or pan # number of filters in the first conv layer
        full_network_G['sr_nb'] = network_G.pop('sr_nb', 23)  # for rrdb or pan # number of RRDB blocks
        full_network_G['sr_gc'] = network_G.pop('sr_gc', 32)  # for rrdb
        full_network_G['sr_unf'] = network_G.pop('sr_unf', 24)  # for pan # number of filters during upscale
        full_network_G['sr_gaussian_noise'] = network_G.pop('sr_gaussian_noise', True)  # for rrdb # add gaussian noise in the net latent # True | False
        full_network_G['sr_plus'] = network_G.pop('sr_plus', False)  # for rrdb # use the ESRGAN+ modifications # true | false
        full_network_G['sr_sa'] = network_G.pop('sr_sa', True)  # for pan # self_attention
        full_network_G['sr_upinter_mode'] = network_G.pop('sr_upinter_mode', "nearest")  # for pan
        # unused options for RRDB:
        # full_network_G['sr_norm_type'] = network_G.pop('sr_norm_type', None)  # "instance" normalization, "batch" normalization or no norm
        # full_network_G['sr_mode'] = network_G.pop('sr_mode', "CNA")  # CNA: conv->norm->act, NAC: norm->act->conv
        # full_network_G['sr_nr'] = network_G.pop('sr_nr', 3)  #  number of residual layers in each RRDB block
        # full_network_G['sr_out_nc'] = network_G.pop('sr_out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        # full_network_G['sr_group'] = network_G.pop('sr_group', 1)  #
        # full_network_G['sr_convtype'] = network_G.pop('sr_convtype', "Conv2D")  # Conv2D | PartialConv2D | DeformConv2D | Conv3D
        # full_network_G['sr_act_type'] = network_G.pop('sr_net_act', None) or network_G.pop('sr_act_type', "leakyrelu")  # swish | leakyrelu
        # full_network_G['sr_finalact'] = network_G.pop('sr_finalact', None)  # Activation function, ie use "tanh" to make outputs fit in [-1, 1] range. Default = None. Coordinate with znorm.
        # full_network_G['sr_upsample_mode'] = network_G.pop('sr_upsample_mode', "upconv") # the type of upsample to use
    elif kind_G in ('sr3d_net', 'sr3d'):
        # SR3D:
        full_network_G['type'] = "sr3d_net"  # SR3DNet
        full_network_G['in_nc'] = network_G.pop('in_nc', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['out_nc'] = network_G.pop('out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        full_network_G['nf'] = network_G.pop('nf', 64)  # number of filters in the conv layers
        full_network_G['nb'] = network_G.pop('nb', 23)  # number of Conv3D  blocks
        full_network_G['scale'] = network_G.pop('scale', scale)
        full_network_G['n_frames'] = network_G.pop('n_frames', 5)  # number of frames the network will use to estimate the central frame (n-1)/2. Must coincide with "num_frames" in the dataset.
    elif kind_G in ('edvr_net', 'edvr'):
        # EDVR:
        full_network_G['type'] = "edvr_net"  # EDVR
        full_network_G['num_in_ch'] = network_G.pop('in_nc', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['num_out_ch'] = network_G.pop('out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        full_network_G['num_feat'] = network_G.pop('nf', 64)  # number of features (M=64, L=128)
        full_network_G['num_frame'] = network_G.pop('n_frames', 5)  # number of frames the network will use to estimate the central frame (n-1)/2. Must coincide with "num_frames" in the dataset.
        full_network_G['upscale'] = network_G.pop('scale', scale)
        full_network_G['deformable_groups'] = network_G.pop('deformable_groups', 8)  # number of deformable offset groups in the deformable layers
        full_network_G['num_extract_block'] = network_G.pop('n_extract_block', 5)  # number of extract blocks
        full_network_G['num_reconstruct_block'] = network_G.pop('n_reconstruct_block', 10)  # number of reconstruction blocks (M=10, L=40)
        full_network_G['center_frame_idx'] = network_G.pop('center_frame_idx', None)  # fix center frame, if None will use num_frame // 2
        full_network_G['with_predeblur'] = network_G.pop('predeblur', False)  # use pre-deblur
        full_network_G['with_tsa'] = network_G.pop('tsa', True)  # use Temporal Spatial Attention
        full_network_G['upsample_mode'] = network_G.pop('upsample_mode', "pixelshuffle")  # pixelshuffle | upconv
        full_network_G['add_rrdb'] = network_G.pop('add_rrdb', False)  # adds RRDB blocks before upsample step to improve SR
        full_network_G['nb'] = network_G.pop('nb', 23)  # number of blocks, only applies to add_rrdb's RRDB blocks
    elif kind_G in ('rife_net', 'rife'):
        full_network_G['type'] = "rife_net"  # RIFE
    elif kind_G == 'dvd_net':
        full_network_G['type'] = "dvd_net"  # DVD
        full_network_G['in_nc'] = network_G.pop('in_nc', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['out_nc'] = network_G.pop('out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        full_network_G['nf'] = network_G.pop('nf', 64)  # number of filters in the conv layers
    else:
        raise NotImplementedError(f'Generator model [{kind_G:s}] not recognized')

    #TODO: check if any options in network_G went unprocessed
    if bool(network_G):
        print(network_G)

    return full_network_G


def get_network_D_config(network_D, scale, crop_size, model_G):
    
    # Note: in PPON they used 100 features in the linear classifier for
    # VGG-like discriminator instead of 128 like in BasicSR. Not important, but can review.
    if model_G == 'ppon':
        model_G = 'PPON'
    else:
        model_G = 'ESRGAN'

    kind_D = None
    if isinstance(network_D, str):
        kind_D = network_D.lower()
        network_D = {}
    elif isinstance(network_D, dict):
        if 'which_model_D' in network_D:
            which_model = 'which_model_D'
        elif 'type' in network_D:
            which_model = 'type'
        kind_D = network_D[which_model].lower()

    full_network_D = {}
    full_network_D['strict'] = network_D.pop('strict', True) # True | False: whether to load the model in strict mode or not

    if kind_D == 'dis_acd':
        # sft-gan, Auxiliary Classifier Discriminator
        full_network_D['type'] = network_D.pop('type', "dis_acd")
    elif kind_D == 'discriminator_vgg_128_sn':
        # TODO: will be replaced by regular discriminator_vgg with optional spectral norm
        full_network_D['type'] = network_D.pop('type', "discriminator_vgg_128_SN")
    elif kind_D in ('adiscriminator', 'adiscriminator_s'):
        # TODO: replace with discriminator_vgg_fea
        full_network_D['type'] = network_D.pop('type', "adiscriminator")
        full_network_D['spectral_norm'] = network_D.pop('spectral_norm', True)
        full_network_D['self_attention'] = network_D.pop('self_attention', True)
        full_network_D['max_pool'] = network_D.pop('max_pool', False)
        full_network_D['poolsize'] = network_D.pop('poolsize', 4)
    elif 'discriminator_vgg_' in kind_D or kind_D == 'discriminator_192' or kind_D == 'discriminator_256' or kind_D == 'discriminator_vgg':
        # 'discriminator_vgg_96', 'discriminator_vgg_128', 'discriminator_vgg_192' or 'discriminator_192', 'discriminator_vgg_256' or 'discriminator_256'
        full_network_D['type'] = network_D.pop('type', "discriminator_vgg")  # VGG-like discriminator
        full_network_D['in_nc'] = network_D.pop('in_nc', 3)  # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_D['base_nf'] = network_D.pop('nf', 64)  # num. of features in conv layers
        full_network_D['norm_type'] = network_D.pop('norm_type', "batch")  # "instance" normalization, "batch" normalization or no norm
        full_network_D['mode'] = network_D.pop('mode', "CNA")  # CNA: conv->norm->act, NAC: norm->act->conv
        full_network_D['act_type'] = network_D.pop('net_act', None) or network_D.pop('act_type', "leakyrelu")  # swish | leakyrelu
        full_network_D['convtype'] = network_D.pop('convtype', "Conv2D")
        full_network_D['arch'] = network_D.pop('G_arch', model_G)
        if kind_D in ['discriminator_vgg', 'discriminator_vgg_fea']:
            full_network_D['size'] = network_D.pop('D_size', crop_size)
        if "_fea" in kind_D:
            # feature extraction/maching: 'discriminator_vgg_128_fea', 'discriminator_vgg_fea'
            # TODO: these options are not currently enabled in the networks
            full_network_D['spectral_norm'] = network_D.pop('spectral_norm', False)
            full_network_D['self_attention'] = network_D.pop('self_attention', False)
            full_network_D['max_pool'] = network_D.pop('max_pool', False)
            full_network_D['poolsize'] = network_D.pop('poolsize', 4)
    elif kind_D in ['patchgan', 'nlayerdiscriminator', 'multiscale', 'pixelgan', 'pixeldiscriminator']:
        if kind_D in ('patchgan', 'nlayerdiscriminator'):
            full_network_D['type'] = 'patchgan'
        elif kind_D == 'multiscale':
            full_network_D['type'] = 'multiscale'
        elif kind_D in ('pixelgan', 'pixeldiscriminator'):
            full_network_D['type'] = 'pixelgan'
        full_network_D['input_nc'] = network_D.pop('in_nc', 3)  # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_D['ndf'] = network_D.pop('nf', 64)  # num. of features in conv layers
        if kind_D in ['patchgan', 'nlayerdiscriminator', 'multiscale']:
            full_network_D['n_layers'] = network_D.pop('n_layers', None) or network_D.pop('nlayer', 3)
        if kind_D in ['patchgan'] or kind_D in ['nlayerdiscriminator']:
            full_network_D['patch'] = network_D.pop('patch_output', True)  # discriminator will return full result as image patch
            full_network_D['use_spectral_norm'] = network_D.pop('spectral_norm', None) or network_D.pop('use_spectral_norm', False)
        if kind_D == 'multiscale':
            full_network_D['num_D'] = network_D.pop('num_D', 3)  # number of discriminators (scales)
    else:
        raise NotImplementedError(f'Discriminator model [{kind_D:s}] not recognized')

    #TODO: add check for vgg_# to validate the crop size matches the discriminator patch size
    # with: vgg_size = kind[18:] and int(vgg_size)

    #TODO: check if any options in network_D went unprocessed
    if bool(network_D):
        print(network_D)

    return full_network_D

def get_network_defaults(opt):
    scale = opt.get('scale', 1)
    crop_size = int(opt['datasets']['train']['crop_size'])

    #TODO: could check dataset type to match model, not needed

    #TODO: can check model type and validate networks (sr, video, i2i, etc)

    # network_G:
    network_G = opt.pop('network_G', None)
    network_G = get_network_G_config(network_G, scale, crop_size)
    model_G = network_G['type']
    opt['network_G'] = network_G

    # network_D:
    # fetch crop_size (if HR_size used, crop_size should have been injected already)
    # Note: VGG Discriminator image patch size should be either a power of 2 number or 3 multiplied by a power of 2.
    if opt.get('network_D', None):
        network_D = opt.pop('network_D', None)
        network_D = get_network_D_config(network_D, scale, crop_size, model_G)
        opt['network_D'] = network_D
        # opt.update(network_D)

    return opt




def main():
    from options import NoneDict, dict_to_nonedict
    opt = {}
    opt['network_G'] = 'ESRGAN'
    opt['network_D'] = 'patchgan'

    opt['datasets'] = {} 
    opt['datasets']['train'] = {}
    opt['datasets']['train']['crop_size'] = 128
    opt = dict_to_nonedict(opt)

    opt = get_network_defaults(opt)
    print(opt)


if __name__ == '__main__':
    main()
