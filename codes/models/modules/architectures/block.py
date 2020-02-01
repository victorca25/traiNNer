from collections import OrderedDict

import torch
import torch.nn as nn
from models.modules.architectures.convolutions.partialconv2d import PartialConv2d


####################
# Basic blocks
####################

# Swish activation funtion
def swish_func(x, beta=1.0):
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

    # In-place implementation, may consume less GPU memory:
    """ 
    result = x.clone()
    torch.sigmoid_(beta*x)
    x *= result
    return x
    #"""

    # Normal out-of-place implementation:
    # """
    return x * torch.sigmoid(beta * x)
    # """


# Swish module
class Swish(nn.Module):
    __constants__ = ["beta", "slope", "inplace"]

    def __init__(self, beta=1.0, slope=1.67653251702, inplace=False):
        """
        Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
        """
        super().__init__()
        self.inplace = inplace
        # self.beta = beta # user-defined beta parameter, non-trainable
        # self.beta = beta * torch.nn.Parameter(torch.ones(1)) # learnable beta parameter, create a tensor out of beta
        self.beta = torch.nn.Parameter(
            torch.tensor(beta)
        )  # learnable beta parameter, create a tensor out of beta
        self.beta.requiresGrad = True  # set requiresGrad to true to make it trainable

        self.slope = slope / 2  # user-defined "slope", non-trainable
        # self.slope = slope * torch.nn.Parameter(torch.ones(1)) # learnable slope parameter, create a tensor out of slope
        # self.slope = torch.nn.Parameter(torch.tensor(slope)) # learnable slope parameter, create a tensor out of slope
        # self.slope.requiresGrad = True # set requiresGrad to true to true to make it trainable

    def forward(self, input):
        """
        # Disabled, using inplace causes:
        # "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
        if self.inplace:
            input.mul_(torch.sigmoid(self.beta*input))
            return 2 * self.slope * input
        else:
            return 2 * self.slope * swish_func(input, self.beta)
        """
        return 2 * self.slope * swish_func(input, self.beta)


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1, beta=1.0):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    # beta: for swish
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "leakyrelu" or act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == "Tanh" or act_type == "tanh":  # [-1, 1] range output
        layer = nn.Tanh()
    elif act_type == "sigmoid":  # [0, 1] range output
        layer = nn.Sigmoid()
    elif act_type == "swish":
        layer = Swish(beta=beta, inplace=inplace)
    else:
        raise NotImplementedError(
            "activation layer [{:s}] is not found".format(act_type)
        )
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == "batch":
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == "instance":
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(
            "normalization layer [{:s}] is not found".format(norm_type)
        )
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == "reflect":
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == "replicate":
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError(
            "padding layer [{:s}] is not implemented".format(pad_type)
        )
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
        tmpstr = "Identity .. \n|"
        modstr = self.sub.__repr__().replace("\n", "\n|")
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = "Identity + \n|"
        modstr = self.sub.__repr__().replace("\n", "\n|")
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(
        in_nc,
        out_nc,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        pad_type="zero",
        norm_type=None,
        act_type="relu",
        mode="CNA",
        convtype="Conv2D",
        spectral_norm=False,
):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ["CNA", "NAC", "CNAC"], "Wrong conv mode [{:s}]".format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != "zero" else None
    padding = padding if pad_type == "zero" else 0

    if convtype == "PartialConv2D":
        c = PartialConv2d(
            in_nc,
            out_nc,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
    else:  # default case is standard 'Conv2D':
        c = nn.Conv2d(
            in_nc,
            out_nc,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )  # normal conv2d

    if spectral_norm:
        c = nn.utils.spectral_norm(c)

    a = act(act_type) if act_type else None
    if "CNA" in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == "NAC":
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


####################
# Useful blocks
####################


class ResNetBlock(nn.Module):
    """
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    """

    def __init__(
            self,
            in_nc,
            mid_nc,
            out_nc,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            pad_type="zero",
            norm_type=None,
            act_type="relu",
            mode="CNA",
            res_scale=1,
            convtype="Conv2D",
    ):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(
            in_nc,
            mid_nc,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            convtype,
        )
        if mode == "CNA":
            act_type = None
        if mode == "CNAC":  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(
            mid_nc,
            out_nc,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            convtype,
        )
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(
            self,
            nc,
            kernel_size=3,
            gc=32,
            stride=1,
            bias=True,
            pad_type="zero",
            norm_type=None,
            act_type="leakyrelu",
            mode="CNA",
            convtype="Conv2D",
            spectral_norm=False,
    ):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(
            nc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )
        self.conv2 = conv_block(
            nc + gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )
        self.conv3 = conv_block(
            nc + 2 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )
        self.conv4 = conv_block(
            nc + 3 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )
        if mode == "CNA":
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(
            nc + 4 * gc,
            nc,
            3,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=last_act,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """

    def __init__(
            self,
            nc,
            kernel_size=3,
            gc=32,
            stride=1,
            bias=True,
            pad_type="zero",
            norm_type=None,
            act_type="leakyrelu",
            mode="CNA",
            convtype="Conv2D",
            spectral_norm=False,
    ):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(
            nc,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            convtype,
            spectral_norm=spectral_norm,
        )
        self.RDB2 = ResidualDenseBlock_5C(
            nc,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            convtype,
            spectral_norm=spectral_norm,
        )
        self.RDB3 = ResidualDenseBlock_5C(
            nc,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            convtype,
            spectral_norm=spectral_norm,
        )

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


# PPON
class _ResBlock_32(nn.Module):
    def __init__(self, nc=64):
        super(_ResBlock_32, self).__init__()
        self.c1 = conv_layer(nc, nc, 3, 1, 1)
        self.d1 = conv_layer(nc, nc // 2, 3, 1, 1)  # rate=1
        self.d2 = conv_layer(nc, nc // 2, 3, 1, 2)  # rate=2
        self.d3 = conv_layer(nc, nc // 2, 3, 1, 3)  # rate=3
        self.d4 = conv_layer(nc, nc // 2, 3, 1, 4)  # rate=4
        self.d5 = conv_layer(nc, nc // 2, 3, 1, 5)  # rate=5
        self.d6 = conv_layer(nc, nc // 2, 3, 1, 6)  # rate=6
        self.d7 = conv_layer(nc, nc // 2, 3, 1, 7)  # rate=7
        self.d8 = conv_layer(nc, nc // 2, 3, 1, 8)  # rate=8
        self.act = act("lrelu")
        self.c2 = conv_layer(nc * 4, nc, 1, 1, 1)  # 256-->64

    def forward(self, input):
        output1 = self.act(self.c1(input))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)
        d5 = self.d5(output1)
        d6 = self.d6(output1)
        d7 = self.d7(output1)
        d8 = self.d8(output1)

        add1 = d1 + d2
        add2 = add1 + d3
        add3 = add2 + d4
        add4 = add3 + d5
        add5 = add4 + d6
        add6 = add5 + d7
        add7 = add6 + d8

        combine = torch.cat([d1, add1, add2, add3, add4, add5, add6, add7], 1)
        output2 = self.c2(self.act(combine))
        output = input + output2.mul(0.2)

        return output


class RRBlock_32(nn.Module):
    def __init__(self):
        super(RRBlock_32, self).__init__()
        self.RB1 = _ResBlock_32()
        self.RB2 = _ResBlock_32()
        self.RB3 = _ResBlock_32()

    def forward(self, input):
        out = self.RB1(input)
        out = self.RB2(out)
        out = self.RB3(out)
        return out.mul(0.2) + input


####################
# Upsampler
####################


class Upsample(nn.Module):
    # To prevent warning: nn.Upsample is deprecated
    # https://discuss.pytorch.org/t/which-function-is-better-for-upsampling-upsampling-or-interpolate/21811/8
    # From: https://pytorch.org/docs/stable/_modules/torch/nn/modules/upsampling.html#Upsample
    # Alternative: https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588/2?u=ptrblck

    def __init__(
            self, size=None, scale_factor=None, mode="nearest", align_corners=None
    ):
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
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        # return self.interp(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = "scale_factor=" + str(self.scale_factor)
        else:
            info = "size=" + str(self.size)
        info += ", mode=" + self.mode
        return info


def pixelshuffle_block(
        in_nc,
        out_nc,
        upscale_factor=2,
        kernel_size=3,
        stride=1,
        bias=True,
        pad_type="zero",
        norm_type=None,
        act_type="relu",
        convtype="Conv2D",
):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """
    conv = conv_block(
        in_nc,
        out_nc * (upscale_factor ** 2),
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=None,
        act_type=None,
        convtype=convtype,
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(
        in_nc,
        out_nc,
        upscale_factor=2,
        kernel_size=3,
        stride=1,
        bias=True,
        pad_type="zero",
        norm_type=None,
        act_type="relu",
        mode="nearest",
        convtype="Conv2D",
):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    # upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    upsample = Upsample(
        scale_factor=upscale_factor, mode=mode
    )  # Updated to prevent the "nn.Upsample is deprecated" Warning
    conv = conv_block(
        in_nc,
        out_nc,
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=norm_type,
        act_type=act_type,
        convtype=convtype,
    )
    return sequential(upsample, conv)


# PPON
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=padding,
        bias=True,
        dilation=dilation,
        groups=groups,
    )
