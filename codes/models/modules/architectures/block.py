from collections import OrderedDict

import torch
import torch.nn as nn
from models.modules.architectures.convolutions.partialconv2d import PartialConv2d #TODO
from models.modules.architectures.convolutions.deformconv2d import DeformConv2d
from models.networks import weights_init_normal, weights_init_xavier, weights_init_kaiming, weights_init_orthogonal


####################
# Basic blocks
####################

# Swish activation funtion
def swish_func(x, beta=1.0, inplace=False):
    """
    "Swish: a Self-Gated Activation Function"
    Searching for Activation Functions (https://arxiv.org/abs/1710.05941)

    If beta=1 applies the Sigmoid Linear Unit (SiLU) function element-wise
    If beta=0, Swish becomes the scaled linear function (identity
      activation) f(x) = x/2
    As beta -> ∞, the sigmoid component converges to approach a 0-1 function
      (unit step), and multiplying that by x gives us f(x)=2max(0,x), which
      is the ReLU multiplied by a constant factor of 2, so Swish becomes like
      the ReLU function.

    Including beta, Swish can be loosely viewed as a smooth function that
      nonlinearly interpolate between identity (linear) and ReLU function.
      The degree of interpolation can be controlled by the model if beta is
      set as a trainable parameter.

    Alt: 1.78718727865 * (x * sigmoid(x) - 0.20662096414)
    """

    if inplace:
        # In-place implementation, may consume less GPU memory:
        result = x.clone()
        torch.sigmoid_(beta*x)
        x *= result
        return x
    # Normal out-of-place implementation:
    return x * torch.sigmoid(beta * x)

# Swish module
class Swish(nn.Module):

    __constants__ = ['beta', 'slope', 'inplace']

    def __init__(self, beta=1.0, slope=1.67653251702, inplace=False):
        """
        Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
        """
        super(Swish, self).__init__()
        self.inplace = inplace
        # self.beta = beta # user-defined beta parameter, non-trainable
        # self.beta = beta * torch.nn.Parameter(torch.ones(1)) # learnable beta parameter, create a tensor out of beta
        self.beta = torch.nn.Parameter(torch.tensor(beta)) # learnable beta parameter, create a tensor out of beta
        self.beta.requiresGrad = True # set requiresGrad to true to make it trainable

        self.slope = slope / 2 # user-defined "slope", non-trainable
        # self.slope = slope * torch.nn.Parameter(torch.ones(1)) # learnable slope parameter, create a tensor out of slope
        # self.slope = torch.nn.Parameter(torch.tensor(slope)) # learnable slope parameter, create a tensor out of slope
        # self.slope.requiresGrad = True # set requiresGrad to true to true to make it trainable

    def forward(self, x):
        """
        # Disabled, using inplace causes:
        # "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
        if self.inplace:
            input.mul_(torch.sigmoid(self.beta*input))
            return 2 * self.slope * input
        else:
            return 2 * self.slope * swish_func(input, self.beta)
        """
        return 2 * self.slope * swish_func(x, self.beta, self.inplace)


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1, beta=1.0):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    # beta: for swish
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type in ('leakyrelu', 'lrelu'):
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'tanh':  # [-1, 1] range output
        layer = nn.Tanh()
    elif act_type == 'sigmoid':  # [0, 1] range output
        layer = nn.Sigmoid()
    elif act_type == 'swish':
        layer = Swish(beta=beta, inplace=inplace)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class Identity(nn.Module):
    def __init__(self, *kwargs):
        super(Identity, self).__init__()

    def forward(self, x, *kwargs):
        return x


def norm(norm_type, nc):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
        # norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
        # norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    # elif norm_type == 'layer':
    #     return lambda num_features: nn.GroupNorm(1, num_features)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def add_spectral_norm(module, use_spectral_norm=False):
    """ Add spectral norm to any module passed if use_spectral_norm = True,
    else, returns the original module without change
    """
    if use_spectral_norm:
        return nn.utils.spectral_norm(module)
    return module


def pad(pad_type, padding):
    """
    helper selecting padding layer
    if padding is 'zero', can be done with conv layers
    """
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    elif pad_type == 'zero':
        layer = nn.ZeroPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        return 'Identity .. \n|' + self.sub.__repr__().replace('\n', '\n|')


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        return 'Identity + \n|' + self.sub.__repr__().replace('\n', '\n|')


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1,
    bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA',
    convtype='Conv2D', spectral_norm=False):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    if convtype=='PartialConv2D':
        c = PartialConv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias, groups=groups)
    elif convtype=='DeformConv2D':
        c = DeformConv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias, groups=groups)
    elif convtype=='Conv3D':
        c = nn.Conv3d(in_nc, out_nc, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias, groups=groups)
    else:
        # default case is standard 'Conv2D':
        c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias, groups=groups)  # normal conv2d

    if spectral_norm:
        c = nn.utils.spectral_norm(c)

    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block. (block)
        num_basic_block (int): number of blocks. (n_layers)
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class Mean(nn.Module):
  def __init__(self, dim: list, keepdim=False):
    super().__init__()
    self.dim = dim
    self.keepdim = keepdim

  def forward(self, x):
    return torch.mean(x, self.dim, self.keepdim)


####################
# initialize modules
####################

@torch.no_grad()
def default_init_weights(module_list, init_type='kaiming', scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.
    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        init_type (str): the type of initialization in: 'normal', 'kaiming'
            or 'orthogonal'
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1. (for 'kaiming')
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function:
            mean and/or std for 'normal'.
            a and/or mode for 'kaiming'
            gain for 'orthogonal' and xavier
    """

    # TODO
    # logger.info('Initialization method [{:s}]'.format(init_type))
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if init_type == 'normal':
                weights_init_normal(m, bias_fill=bias_fill, **kwargs)
            if init_type == 'xavier':
                weights_init_xavier(m, scale=scale, bias_fill=bias_fill, **kwargs)
            elif init_type == 'kaiming':
                weights_init_kaiming(m, scale=scale, bias_fill=bias_fill, **kwargs)
            elif init_type == 'orthogonal':
                weights_init_orthogonal(m, bias_fill=bias_fill)
            else:
                raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))



####################
# Scaling
####################

class Upsample(nn.Module):
    r"""Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """

    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super(Upsample, self).__init__()
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.size = size
        self.align_corners = align_corners
        # self.interp = nn.functional.interpolate

    def forward(self, x):
        return nn.functional.interpolate(
            x, size=self.size, scale_factor=self.scale_factor,
            mode=self.mode, align_corners=self.align_corners)
        # return self.interp(x, size=self.size,
        #     scale_factor=self.scale_factor, mode=self.mode,
        #     align_corners=self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2,
    kernel_size=3, stride=1, bias=True, pad_type='zero',
    norm_type=None, act_type='relu', convtype='Conv2D'):
    """ Pixel shuffle layer.
    (Real-Time Single Image and Video Super-Resolution Using an
    Efficient Sub-Pixel Convolutional Neural Network, CVPR17).
    """
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias,
                        pad_type=pad_type, norm_type=None, act_type=None, convtype=convtype)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_block(in_nc, out_nc, upscale_factor=2, kernel_size=3,
    stride=1, bias=True, pad_type='zero', norm_type=None,
    act_type='relu', mode='nearest', convtype='Conv2D'):
    """ Upconv layer described in
    https://distill.pub/2016/deconv-checkerboard/.
    Example to replace deconvolutions:
        - from: nn.ConvTranspose2d(in_nc, out_nc, kernel_size=4, stride=2, padding=1)
        - to: upconv_block(in_nc, out_nc,kernel_size=3, stride=1, act_type=None)
    """
    upscale_factor = (1, upscale_factor, upscale_factor) if convtype == 'Conv3D' else upscale_factor
    upsample = Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias,
        pad_type=pad_type, norm_type=norm_type, act_type=act_type,
        convtype=convtype)
    return sequential(upsample, conv)


class DepthToSpace(nn.Module):
    """ PixelShuffle / DepthToSpace / unsqueeze2d.
    Rearranges data from depth into blocks of spatial data. This is
    the reverse transformation of SpaceToDepth. More specifically,
    this op outputs a copy of the input tensor where values from the
    depth dimension are moved in spatial blocks to the height and width
    dimensions.

    Args:
        block_size (int): indicates the  input block size and how the
            data is moved. In SR its equivalent to the scale factor.
        form: select tensorflow ('tf') or pytorch ('pt') style shuffle.
    """
    def __init__(self, block_size:int=2, form:str='pt'):
        super().__init__()
        self.bs = block_size
        self.form = form

    def forward(self, x):
        if self.form == 'tf':
            return depth_to_space_tf(x, self.bs)
        return depth_to_space(x, self.bs)

    def extra_repr(self):
        return f"block_size={self.bs}"


def depth_to_space(x, bs:int=2):
    """ Pixel shuffle (PyTorch).
    Equivalent to torch.nn.PixelShuffle().
    Args:
        x (Tensor): Input tensor (b, c, h, w).
        bs: block_size, scale factor.
    Returns:
        Tensor: tensor after pixel shuffle.
    """
    assert bs >= 1 and isinstance(bs, int)
    if bs == 1:
        return x

    b, c, h, w = x.size()
    if c % (bs ** 2) != 0:
        raise ValueError("The tensor channels must be divisible by "
                         "(bs ** 2).")
    new_d = -1  # c // (bs ** 2)
    new_h = h * bs
    new_w = w * bs

    # (b, c//bs^2, bs, bs, h, w)
    x = x.view(b, new_d, bs, bs, h, w)
    # (b, c//bs^2, h, bs, w, bs)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    # (b, c//bs^2, h * bs, w * bs)
    return x.view(b, new_d, new_h, new_w)


def depth_to_space_tf(x, bs:int=2):
    """ Pixel shuffle (TensorFlow).
    Equivalent to:
        https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
    Args:
        x (Tensor): Input tensor (b, c, h, w).
        bs: block_size, scale factor.
    Returns:
        Tensor: tensor after pixel shuffle.
    """
    assert bs >= 1 and isinstance(bs, int)
    if bs == 1:
        return x

    b, c, h, w = x.size()
    if c % (bs ** 2) != 0:
        raise ValueError("The tensor channels must be divisible by "
                         "(bs ** 2).")
    new_d = -1  # c // (bs ** 2)
    new_h = h * bs
    new_w = w * bs

    # (b, bs, bs, c//bs^2, h, w)
    x = x.view(b, bs, bs, new_d, h, w)
    # (b, c//bs^2, h, bs, w, bs)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
    # (b, c//bs^2, h * bs, w * bs)
    return x.view(b, new_d, new_h, new_w)


class SpaceToDepth(nn.Module):
    """ PixelUnshuffle / SpaceToDepth / squeeze2d.
    Rearranges blocks of spatial data, into depth. This operation
    outputs a copy of the input tensor where values from the height
    and width dimensions are moved to the depth dimension.

    Args:
        block_size: indicates the input block size, where
            non-overlapping blocks of size block_size x block size are
            rearranged into depth at each location. In SR its equivalent
            to the downscale factor.
        form: select tensorflow ('tf') or pytorch ('pt') style unshuffle.
    """
    def __init__(self, block_size:int=2, form:str='pt'):
        super().__init__()
        self.bs = block_size
        self.form = form

    def forward(self, x):
        if self.form == 'tf':
            return space_to_depth_tf(x, self.bs)
        return space_to_depth(x, self.bs)
    
    def extra_repr(self):
        return f"block_size={self.bs}"


def space_to_depth(x, bs:int=2):
    """ Pixel unshuffle (PyTorch).
    This is the inverse of torch.nn.PixelShuffle().
    Equivalent to nn.PixelUnshuffle().
    Args:
        x (Tensor): Input tensor (b, c, h, w).
        bs: block_size, scale factor.
    Returns:
        Tensor: tensor after pixel unshuffle.
    """
    assert bs >= 1 and isinstance(bs, int)
    if bs == 1:
        return x

    b, c, h, w = x.size()
    assert h % bs == 0 and w % bs == 0, "{}".format((h, w, bs))
    new_d = -1  # c * (bs**2)
    new_h = h // bs
    new_w = w // bs

    # (b, c, h//bs, bs, w//bs, bs)
    x = x.view(b, c, new_h, bs, new_w, bs)
    # (b, c, bs, bs, h//bs, w//bs)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    # (b, c*bs^2, h//bs, w//bs)
    return x.view(b, new_d, new_h, new_w)


def space_to_depth_tf(x, bs:int=2):
    """ Pixel unshuffle (TensorFlow).
    Equivalent to:
        https://www.tensorflow.org/api_docs/python/tf/nn/space_to_depth
    Args:
        x (Tensor): Input tensor (b, c, h, w).
        bs: block_size, scale factor.
    Returns:
        Tensor: tensor after pixel unshuffle.
    """
    assert bs >= 1 and isinstance(bs, int)
    if bs == 1:
        return x

    b, c, h, w = x.size()
    assert h % bs == 0 and w % bs == 0, "{}".format((h, w, bs))
    new_d = -1  # c * (bs**2)
    new_h = h // bs
    new_w = w // bs

    # (b, c, h//bs, bs, w//bs, bs)
    x = x.view(b, c, new_h, bs, new_w, bs)
    # (b, bs, bs, c, h//bs, w//bs)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    # (b, c*bs^2, h//bs, w//bs)
    return x.view(b, new_d, new_h, new_w)


# PPON
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation, groups=groups)



####################
# ESRGANplus
####################

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float).to(torch.device('cuda'))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# TODO: Not used:
# https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/models/base_model.py
class minibatch_std_concat_layer(nn.Module):
    def __init__(self, averaging='all'):
        super(minibatch_std_concat_layer, self).__init__()
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode'%self.averaging
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def forward(self, x):
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)
        vals = self.adjusted_std(x, dim=0, keepdim=True)
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)
        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = mean(vals, axis=[2,3], keepdim=True)             # torch.mean(torch.mean(vals, 2, keepdim=True), 3, keepdim=True)
        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = mean(x, [0,2,3], keepdim=True)                   # torch.mean(torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:                                                           # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1]/self.n, self.shape[2], self.shape[3])
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)


####################
# Useful blocks
####################

class SelfAttentionBlock(nn.Module):
    """
        Implementation of Self attention Block according to paper
        'Self-Attention Generative Adversarial Networks'
        (https://arxiv.org/abs/1805.08318)
        Flexible Self Attention (FSA) layer according to paper
        Efficient Super Resolution For Large-Scale Images Using
            Attentional GAN (https://arxiv.org/pdf/1812.04821.pdf)
            The FSA layer borrows the self attention layer from SAGAN,
            and wraps it with a max-pooling layer to reduce the size
            of the feature maps and enable large-size images to fit in memory.
        Used in Generator and Discriminator Networks.
    """

    def __init__(self, in_dim, max_pool=False, poolsize = 4, spectral_norm=False, ret_attention=False): #in_dim = in_feature_maps
        super(SelfAttentionBlock,self).__init__()

        self.in_dim = in_dim
        self.max_pool = max_pool
        self.poolsize = poolsize
        self.ret_attention = ret_attention

        if self.max_pool:
            self.pooled = nn.MaxPool2d(kernel_size=self.poolsize, stride=self.poolsize) #kernel_size=4, stride=4
            # Note: can test using strided convolutions instead of MaxPool2d! :
            #upsample_block_num = int(math.log(scale_factor, 2))
            #self.pooled = nn.Conv2d .... strided conv
            # upsample_o = [UpconvBlock(in_channels=in_dim, out_channels=in_dim, upscale_factor=2, mode='bilinear', act_type='leakyrelu') for _ in range(upsample_block_num)]
            ## upsample_o.append(nn.Conv2d(nf, in_nc, kernel_size=9, stride=1, padding=4))
            ## self.upsample_o = nn.Sequential(*upsample_o)

            # self.upsample_o = B.Upsample(scale_factor=self.poolsize, mode='bilinear', align_corners=False)

        self.conv_f = add_spectral_norm(
            nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8,
                kernel_size=1, padding=0),
            use_spectral_norm=spectral_norm)  # query_conv
        self.conv_g = add_spectral_norm(
            nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8,
                kernel_size=1, padding=0),
            use_spectral_norm=spectral_norm)  # key_conv
        self.conv_h = add_spectral_norm(
            nn.Conv1d(in_channels=in_dim, out_channels=in_dim,
                kernel_size=1, padding=0),
            use_spectral_norm=spectral_norm)  # value_conv

        self.gamma = nn.Parameter(torch.zeros(1)) # Trainable interpolation parameter
        self.softmax  = nn.Softmax(dim = -1)

    def forward(self,input):
        """
            inputs :
                input : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        if self.max_pool:  # Downscale with Max Pool
            x = self.pooled(input)
        else:
            x = input

        batch_size, C, width, height = x.size()

        N = width * height
        x = x.view(batch_size, -1, N)
        f = self.conv_f(x)  # proj_query  # B X CX(N)
        g = self.conv_g(x)  # proj_key    # B X C x (*W*H)
        h = self.conv_h(x)  # proj_value  # B X C X N

        # energy, transpose check
        s = torch.bmm(f.permute(0, 2, 1), g)
        # get probabilities
        # beta #attention # BX (N) X (N)
        attention = self.softmax(s)

        out = torch.bmm(h, attention.permute(0,2,1))
        out = out.view(batch_size, C, width, height)

        if self.max_pool:
            # Upscale to original size
            # out = self.upsample_o(out)
            out = Upsample(
                size=(input.shape[2],input.shape[3]),
                mode='bicubic', align_corners=False)(out)

        # add original input
        out = self.gamma*out + input

        if self.ret_attention:
            return out, attention
        else:
            return out

