import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair, _single

import torchvision.ops as O


class DeformConv2d(nn.Module):
    """
    A Custom Deformable Convolution layer that acts similarly to a regular Conv2d layer.
    """
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(DeformConv2d, self).__init__()

        self.conv_offset = nn.Conv2d(in_nc, 2 * (kernel_size**2), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

        self.dcn_conv = O.DeformConv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        offset = self.conv_offset(x)
        return self.dcn_conv(x, offset=offset)

class ModulatedDeformConv(nn.Module):
    """A Modulated Deformable Conv layer.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        deformable_groups (int): Number of offset groups.
        bias (bool or str): Same as nn.Conv2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        stride_h, stride_w = _pair(self.stride)
        pad_h, pad_w = _pair(self.padding)
        dil_h, dil_w = _pair(self.dilation)
        _, n_in_channels, _, _ = x.shape
        n_weight_grps = n_in_channels // self.weight.shape[1]
        return torch.ops.torchvision.deform_conv2d(x, self.weight, offset, mask, self.bias, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, n_weight_grps, self.deformable_groups, True)

class ModulatedDeformConvPack(ModulatedDeformConv):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        deformable_groups (int): Number of offset groups.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(self.in_channels, self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, 
            stride=_pair(self.stride), padding=_pair(self.padding), dilation=_pair(self.dilation), bias=True)
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        stride_h, stride_w = _pair(self.stride)
        pad_h, pad_w = _pair(self.padding)
        dil_h, dil_w = _pair(self.dilation)
        _, n_in_channels, _, _ = x.shape
        n_weight_grps = n_in_channels // self.weight.shape[1]
        return torch.ops.torchvision.deform_conv2d(x, self.weight, offset, mask, self.bias, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, n_weight_grps, self.deformable_groups, True)

class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        # TODO: In the official EDVR, this is just a warning. Here I've made it actually interrupt training.
        # Not sure what to do with it, or if it is even necessary, so I'm leaving it off for now.
        # offset_absmean = torch.mean(torch.abs(offset))
        # if offset_absmean > 50:
        #     raise ValueError(f'Offset abs mean is {offset_absmean}, larger than 50.')

        stride_h, stride_w = _pair(self.stride)
        pad_h, pad_w = _pair(self.padding)
        dil_h, dil_w = _pair(self.dilation)
        _, n_in_channels, _, _ = x.shape
        n_weight_grps = n_in_channels // self.weight.shape[1]
        return torch.ops.torchvision.deform_conv2d(x, self.weight, offset, mask, self.bias, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, n_weight_grps, self.deformable_groups, True)
